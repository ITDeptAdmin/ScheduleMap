# scripts/sync_mec_to_csv.py
"""
Sync Modern Events Calendar (MEC) events to a GitHub-tracked CSV.

Environment variables (required):
- MEC_BASE_URL: e.g. https://staging12.ramusa.org/wp-json/mec/v1.0
- MEC_TOKEN: MEC API key (sent as header: mec-token)
- CSV_PATH: output CSV filename in repo, e.g. "Clinic Master Schedule TEST.csv"

What it does:
- Fetches upcoming events from MEC REST API
- Fetches each single event for full details + occurrences (dates[])
- Writes a CSV matching your schedule page's expected columns
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

CSV_HEADER = [
    "canceled",
    "lat",
    "lng",
    "address",
    "city",
    "state",
    "title",
    "facility",
    "telehealth",
    "start_time",
    "stop_time",
    "start_date",
    "end_date",
    "parking_date",
    "parking_time",
    "medical",
    "dental",
    "vision",
    "dentures",
    "url",
]


def _env(name: str) -> str:
    val = (os.getenv(name) or "").strip()
    if not val:
        raise SystemExit(f"Missing env var: {name}")
    return val


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")
    return _normalize_ws(s)


def _ymd_to_mdy(ymd: str) -> str:
    y, m, d = ymd.split("-")
    return f"{int(m)}/{int(d)}/{int(y)}"


def _safe_float_str(v: Any) -> str:
    try:
        return f"{float(v):.10f}".rstrip("0").rstrip(".")
    except Exception:
        return ""


def _get_json(
    base_url: str, token: str, path: str, params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    headers = {"mec-token": token}
    params = params or {}

    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code >= 400:
        # Some environments accept params as JSON body for GET
        r = requests.get(url, headers=headers, json=params, timeout=30)

    r.raise_for_status()
    return r.json()


def _iter_event_ids(base_url: str, token: str) -> List[int]:
    """
    Walk MEC pagination from /events/ starting at today.
    """
    ids: List[int] = []
    next_date = datetime.utcnow().date().isoformat()
    next_offset = 0
    seen: set[int] = set()

    while True:
        payload = {
            "limit": 50,
            "order": "ASC",
            "offset": next_offset,
            "start_date": next_date,
            "include_past_events": 0,
            "include_ongoing_events": 1,
        }
        data = _get_json(base_url, token, "/events/", payload)

        events_by_day = data.get("events") or {}
        if isinstance(events_by_day, dict):
            for _day, evs in events_by_day.items():
                if not isinstance(evs, list):
                    continue
                for ev in evs:
                    if not isinstance(ev, dict):
                        continue
                    raw_id = ev.get("ID", ev.get("id"))
                    try:
                        i = int(raw_id)
                    except Exception:
                        continue
                    if i not in seen:
                        seen.add(i)
                        ids.append(i)

        pagination = data.get("pagination") or {}
        if not pagination.get("has_more_events"):
            break

        next_date = str(pagination.get("next_date") or next_date)
        next_offset = int(pagination.get("next_offset") or 0)

    return ids


def _list_names(d: Any) -> List[str]:
    if not isinstance(d, dict):
        return []
    out: List[str] = []
    for v in d.values():
        if isinstance(v, dict):
            name = str(v.get("name") or "").strip()
            if name:
                out.append(name)
    return out


def _first_location(single: Dict[str, Any]) -> Dict[str, str]:
    """
    MEC single-event JSON contains locations under data.mec.locations, keyed by id.
    """
    data = single.get("data") if isinstance(single.get("data"), dict) else {}
    mec = data.get("mec") if isinstance(data.get("mec"), dict) else {}
    locations = mec.get("locations") if isinstance(mec.get("locations"), dict) else {}

    if not isinstance(locations, dict) or not locations:
        return {"facility": "", "address": "", "city": "", "state": "", "lat": "", "lng": ""}

    loc_obj = next(iter(locations.values()))
    if not isinstance(loc_obj, dict):
        return {"facility": "", "address": "", "city": "", "state": "", "lat": "", "lng": ""}

    return {
        "facility": str(loc_obj.get("name") or "").strip(),
        "address": str(loc_obj.get("address") or "").strip(),
        "city": str(loc_obj.get("city") or "").strip(),
        "state": str(loc_obj.get("state") or "").strip(),
        "lat": _safe_float_str(loc_obj.get("latitude")),
        "lng": _safe_float_str(loc_obj.get("longitude")),
    }


def _detect_telehealth(categories: List[str]) -> str:
    return "Yes" if any("telehealth" in c.lower() for c in categories) else "No"


def _detect_services(categories: List[str], tags: List[str], content_text: str) -> Dict[str, str]:
    """
    Prefer explicit categories/tags, but fall back to content text.
    Also supports basic negation like "No Dental" or "No Vision services".
    """
    blob = " ".join([*categories, *tags, content_text]).lower()

    def negated(word: str) -> bool:
        return bool(re.search(rf"\bno\s+{re.escape(word)}\b", blob)) or bool(
            re.search(rf"\bno\s+{re.escape(word)}\s+services\b", blob)
        )

    services = {
        "medical": "Yes" if ("medical" in blob) and not negated("medical") else "",
        "dental": "Yes" if ("dental" in blob) and not negated("dental") else "",
        "vision": "Yes" if ("vision" in blob) and not negated("vision") else "",
        "dentures": "Yes" if ("denture" in blob) and not negated("dentures") else "",
    }

    # If explicitly "medical services only" then clear others.
    if re.search(r"\bmedical\s+services\s+only\b", blob):
        services["medical"] = "Yes"
        services["dental"] = ""
        services["vision"] = ""
        services["dentures"] = ""

    return services


def _detect_canceled(data: Dict[str, Any]) -> str:
    post = data.get("post") if isinstance(data.get("post"), dict) else {}
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}

    post_status = str(post.get("post_status") or "").lower()
    mec_status = str(meta.get("mec_event_status") or "").lower()
    cancelled_reason = str(meta.get("mec_cancelled_reason") or "").strip().lower()

    if "cancel" in post_status:
        return "Yes"
    if "cancel" in mec_status:
        return "Yes"
    if cancelled_reason:
        return "Yes"
    return "No"


def _extract_content_text(data: Dict[str, Any]) -> str:
    post = data.get("post") if isinstance(data.get("post"), dict) else {}
    html = str(post.get("post_content") or data.get("content") or "")
    return _strip_html(html)


def _build_rows_from_event(single: Dict[str, Any]) -> List[Dict[str, str]]:
    data = single.get("data") if isinstance(single.get("data"), dict) else {}
    mec = data.get("mec") if isinstance(data.get("mec"), dict) else {}

    title = str(data.get("title") or "").strip()
    url = str(mec.get("permalink") or data.get("permalink") or "").strip()

    categories = _list_names(mec.get("categories"))
    tags = _list_names(mec.get("tags"))

    telehealth = _detect_telehealth(categories)
    content_text = _extract_content_text(data)
    services = _detect_services(categories, tags, content_text)

    loc = _first_location(single)
    canceled = _detect_canceled(data)

    dates = single.get("dates")
    if not isinstance(dates, list):
        return []

    def time_fmt(x: Dict[str, Any]) -> str:
        hour = str(x.get("hour") or "").strip()
        minutes = str(x.get("minutes") or "").strip()
        ampm = str(x.get("ampm") or "").strip().lower()
        if not hour or not minutes or not ampm:
            return ""
        minutes = minutes.zfill(2)
        return f"{int(hour)}:{minutes} {ampm}"

    rows: List[Dict[str, str]] = []
    for occ in dates:
        if not isinstance(occ, dict):
            continue
        start = occ.get("start") if isinstance(occ.get("start"), dict) else {}
        end = occ.get("end") if isinstance(occ.get("end"), dict) else {}

        start_ymd = str(start.get("date") or "").strip()
        end_ymd = str(end.get("date") or start_ymd).strip()
        if not start_ymd:
            continue

        start_mdy = _ymd_to_mdy(start_ymd)
        end_mdy = _ymd_to_mdy(end_ymd)

        allday = str(occ.get("allday") or "0") == "1"
        hide_time = str(occ.get("hide_time") or "0") == "1"

        start_time = "" if (allday or hide_time) else time_fmt(start)
        stop_time = "" if (allday or hide_time) else time_fmt(end)

        rows.append(
            {
                "canceled": canceled,
                "lat": loc["lat"],
                "lng": loc["lng"],
                "address": loc["address"],
                "city": loc["city"],
                "state": loc["state"],
                "title": title,
                "facility": loc["facility"],
                "telehealth": telehealth,
                "start_time": start_time,
                "stop_time": stop_time,
                "start_date": start_mdy,
                "end_date": end_mdy,
                "parking_date": "",
                "parking_time": "",
                "medical": services["medical"],
                "dental": services["dental"],
                "vision": services["vision"],
                "dentures": services["dentures"],
                "url": url,
            }
        )

    def sort_key(r: Dict[str, str]) -> Tuple[int, str]:
        try:
            m, d, y = [int(x) for x in r["start_date"].split("/")]
            return (y * 10000 + m * 100 + d, r.get("title", ""))
        except Exception:
            return (99999999, r.get("title", ""))

    return sorted(rows, key=sort_key)


def main() -> None:
    base_url = _env("MEC_BASE_URL")
    token = _env("MEC_TOKEN")
    csv_path = _env("CSV_PATH")

    ids = _iter_event_ids(base_url, token)

    all_rows: List[Dict[str, str]] = []
    for eid in ids:
        single = _get_json(base_url, token, f"/events/{eid}", params={})
        all_rows.extend(_build_rows_from_event(single))

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k, "") for k in CSV_HEADER})


if __name__ == "__main__":
    main()
