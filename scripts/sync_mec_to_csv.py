#!/usr/bin/env python3
# scripts/sync_mec_to_csv.py
"""
Sync Modern Events Calendar (MEC) events to a GitHub-tracked CSV.

Required env vars:
- MEC_BASE_URL: e.g. https://www.staging12.ramusa.org/wp-json/mec/v1.0
- MEC_TOKEN: MEC API key (sent as header: mec-token)
- CSV_PATH: output CSV filename in repo, e.g. "Clinic Master Schedule TEST.csv"

Optional env vars (recommended):
- MEC_START: YYYY-MM-DD (default: 2026-01-01)
- MEC_END:   YYYY-MM-DD (default: 2027-12-31)
- MEC_LIMIT: int (default: 200)
- MAX_SPAN_DAYS: int (default: 10)  # skips suspiciously long occurrences from repeat rules
- MEC_DEBUG: "1" to print basic diagnostics (no secrets)

What it does:
- Fetches event IDs from /events using start/end/limit
- Fetches full details from /events/{id}
- Writes one CSV row per occurrence in data.dates[]
- Fills:
  - location: facility/address/city/state/lat/lng
  - site: "City, State" (if column exists)
  - dates: start_date/end_date
  - times: start_time/stop_time (blank for all-day)
  - services: medical/dental/vision/dentures from custom fields (fallback to content scan)
  - telehealth: from custom field "Clinic Type" (fallback to content scan)
  - parking_date/parking_time from custom fields
  - canceled: from meta.mec_event_status containing "cancel"
- Preserves existing CSV header/order if CSV already exists
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Tuple

import requests


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[()]", "", s)
    return s


def _strip_html(html: str) -> str:
    if not html:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;?", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _read_existing_header(csv_path: str) -> Optional[List[str]]:
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            return header if header else None
        except StopIteration:
            return None


def _header_index_map(header: List[str]) -> Dict[str, str]:
    return {_norm(h): h for h in header}


def _set_by_alias(row: Dict[str, str], header_map: Dict[str, str], aliases: List[str], value: Any) -> None:
    v = "" if value is None else str(value)
    for a in aliases:
        key = _norm(a)
        if key in header_map:
            row[header_map[key]] = v
            return


def _find_header_contains(header: List[str], *needles: str) -> Optional[str]:
    needles_n = [n.lower() for n in needles]
    for h in header:
        hn = h.lower()
        if all(n in hn for n in needles_n):
            return h
    return None


def _yn(val: bool) -> str:
    return "Yes" if val else "No"


def _mec_get(base_url: str, token: str, path: str, params: Optional[dict] = None) -> Any:
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    r = requests.get(url, headers={"mec-token": token}, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def _flatten_events_payload(payload: Any) -> List[Dict[str, Any]]:
    """
    Handles MEC shapes:
    - [ {...}, {...} ]
    - { "events": [ ... ] }
    - { "events": { "2026-01-21": [ ... ], ... } }
    - { "2026-01-21": [ ... ], ... }
    """
    out: List[Dict[str, Any]] = []

    def add_list(lst: Any) -> None:
        if isinstance(lst, list):
            for item in lst:
                if isinstance(item, dict):
                    out.append(item)

    if isinstance(payload, list):
        add_list(payload)
        return out

    if isinstance(payload, dict):
        if "events" in payload:
            ev = payload.get("events")
            if isinstance(ev, list):
                add_list(ev)
                return out
            if isinstance(ev, dict):
                for v in ev.values():
                    add_list(v)
                return out

        for v in payload.values():
            add_list(v)
        return out

    return out


def _extract_event_ids_from_list_payload(payload: Any) -> List[int]:
    events = _flatten_events_payload(payload)
    ids: List[int] = []
    for e in events:
        for k in ("ID", "id", "post_id", "event_id"):
            if k in e and e.get(k) is not None:
                try:
                    ids.append(int(e.get(k)))
                    break
                except Exception:
                    continue
    return sorted(set(ids))


def _extract_custom_fields(event_data: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    fields = event_data.get("fields", [])
    if not isinstance(fields, list):
        return out

    for f in fields:
        if not isinstance(f, dict):
            continue
        label = _norm(str(f.get("label", "")))
        value = f.get("value", "")
        value_s = "" if value is None else str(value).strip()

        if label == "services":
            out["services"] = value_s
        elif label == "clinic type":
            out["clinic_type"] = value_s
        elif label == "parking_date":
            out["parking_date"] = value_s
        elif label.startswith("parking_time"):
            out["parking_time"] = value_s

    return out


def _services_flags(services_value: str, fallback_text: str = "") -> Dict[str, bool]:
    text = services_value.strip()
    source = text if text else fallback_text

    def has(word: str) -> bool:
        return word.lower() in source.lower()

    return {
        "medical": has("medical"),
        "dental": has("dental"),
        "vision": has("vision"),
        "dentures": has("denture"),
    }


def _clinic_type_flags(clinic_type_value: str, fallback_text: str = "") -> Dict[str, bool]:
    text = clinic_type_value.strip()
    source = text if text else fallback_text

    def has(word: str) -> bool:
        return word.lower() in source.lower()

    return {
        "telehealth": has("telehealth"),
        "popup": has("popup"),
    }


def _pick_first_dict_value(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(d, dict) or not d:
        return None
    first_key = sorted(d.keys(), key=lambda x: str(x))[0]
    val = d.get(first_key)
    return val if isinstance(val, dict) else None


def _extract_location(event_data: Dict[str, Any]) -> Dict[str, str]:
    loc = _pick_first_dict_value(event_data.get("locations", {})) or {}
    return {
        "facility": str(loc.get("name", "")).strip(),
        "address": str(loc.get("address", "")).strip(),
        "city": str(loc.get("city", "")).strip(),
        "state": str(loc.get("state", "")).strip(),
        "lat": str(loc.get("latitude", "")).strip(),
        "lng": str(loc.get("longitude", "")).strip(),
        "opening_hour": str(loc.get("opening_hour", "")).strip(),
        "postal_code": str(loc.get("postal_code", "")).strip(),
    }


def _parse_ymd(s: str) -> Optional[date]:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _format_time_from_parts(hour: Any, minutes: Any, ampm: Any) -> str:
    """
    MEC often returns: hour="1", minutes="0", ampm="PM"
    Output: "1:00 pm"
    """
    if hour in (None, "", 0, "0"):
        return ""
    try:
        h = int(str(hour).strip())
    except Exception:
        return ""

    try:
        m = int(str(minutes).strip()) if minutes not in (None, "") else 0
    except Exception:
        m = 0

    ap = str(ampm or "").strip().lower()
    if ap not in {"am", "pm"}:
        return ""

    return f"{h}:{m:02d} {ap}"


def _extract_occurrences(event_data: Dict[str, Any], max_span_days: int) -> List[Dict[str, str]]:
    """
    Returns list of occurrence dicts:
    - start_date, end_date, start_time, stop_time
    Skips suspicious long spans (repeat artifacts) beyond max_span_days.
    """
    dates = event_data.get("dates", [])
    out: List[Dict[str, str]] = []

    if not isinstance(dates, list):
        return out

    for d in dates:
        if not isinstance(d, dict):
            continue

        s_obj = d.get("start") or {}
        e_obj = d.get("end") or {}

        s_date = str(s_obj.get("date") or "").strip()
        e_date = str(e_obj.get("date") or s_date or "").strip()

        if not s_date:
            continue

        # all-day / hide_time may appear as strings "0"/"1"
        allday = str(d.get("allday") or "0").strip() in {"1", "true", "True", "yes", "Yes"}
        hide_time = str(d.get("hide_time") or "0").strip() in {"1", "true", "True", "yes", "Yes"}

        start_time = "" if (allday or hide_time) else _format_time_from_parts(
            s_obj.get("hour"), s_obj.get("minutes"), s_obj.get("ampm")
        )
        stop_time = "" if (allday or hide_time) else _format_time_from_parts(
            e_obj.get("hour"), e_obj.get("minutes"), e_obj.get("ampm")
        )

        # skip repeat artifacts that create huge ranges
        sd = _parse_ymd(s_date)
        ed = _parse_ymd(e_date)
        if sd and ed:
            span = (ed - sd).days
            if span > max_span_days:
                continue

        out.append(
            {
                "start_date": s_date,
                "end_date": e_date,
                "start_time": start_time,
                "stop_time": stop_time,
            }
        )

    # de-dupe identical occurrences
    seen = set()
    uniq: List[Dict[str, str]] = []
    for o in out:
        key = (o["start_date"], o["end_date"], o["start_time"], o["stop_time"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(o)

    return uniq


def _extract_canceled(event_data: Dict[str, Any]) -> bool:
    """
    MEC stores status in meta.mec_event_status (e.g., EventScheduled / EventCancelled).
    We'll treat anything containing 'cancel' as canceled.
    """
    meta = event_data.get("meta", {})
    status = ""
    if isinstance(meta, dict):
        status = str(meta.get("mec_event_status") or meta.get("event_status") or "").strip()
    if not status:
        status = str(event_data.get("event_status") or "").strip()

    return "cancel" in status.lower()


def _build_rows_for_event(detail: Dict[str, Any], header: List[str], max_span_days: int) -> List[Dict[str, str]]:
    data = detail.get("data", {})
    post = data.get("post", {}) if isinstance(data, dict) else {}
    content_html = post.get("post_content") or post.get("post_content_filtered") or ""
    content_text = _strip_html(str(content_html))

    loc = _extract_location(data)
    fields = _extract_custom_fields(data)

    svc = _services_flags(fields.get("services", ""), fallback_text=content_text)
    ctype = _clinic_type_flags(fields.get("clinic_type", ""), fallback_text=content_text)

    parking_date = fields.get("parking_date", "").strip()
    parking_time = fields.get("parking_time", "").strip()

    canceled = _extract_canceled(data)

    title = str(data.get("title") or post.get("post_title") or "").strip()

    url = str(data.get("permalink") or "").strip()
    if not url:
        url = str(post.get("guid") or "").strip()

    occurrences = _extract_occurrences(data, max_span_days=max_span_days)
    if not occurrences:
        occurrences = [{"start_date": "", "end_date": "", "start_time": "", "stop_time": ""}]

    header_map = _header_index_map(header)
    rows: List[Dict[str, str]] = []

    for occ in occurrences:
        row = {h: "" for h in header}

        _set_by_alias(row, header_map, ["title"], title)
        _set_by_alias(row, header_map, ["facility"], loc["facility"])
        _set_by_alias(row, header_map, ["address"], loc["address"])
        _set_by_alias(row, header_map, ["city"], loc["city"])
        _set_by_alias(row, header_map, ["state"], loc["state"])
        _set_by_alias(row, header_map, ["lat", "latitude"], loc["lat"])
        _set_by_alias(row, header_map, ["lng", "longitude", "lon"], loc["lng"])
        _set_by_alias(row, header_map, ["url", "link", "permalink"], url)

        # your CSV has "site" as "City, State"
        site_value = ", ".join([p for p in [loc["city"], loc["state"]] if p])
        _set_by_alias(row, header_map, ["site"], site_value)

        _set_by_alias(row, header_map, ["start_date", "start date"], occ["start_date"])
        _set_by_alias(row, header_map, ["end_date", "end date"], occ["end_date"])
        _set_by_alias(row, header_map, ["start_time", "start time"], occ["start_time"])
        _set_by_alias(row, header_map, ["stop_time", "stop time", "end_time", "end time"], occ["stop_time"])

        _set_by_alias(row, header_map, ["parking_date", "parking date"], parking_date)
        _set_by_alias(row, header_map, ["parking_time", "parking time"], parking_time)

        _set_by_alias(row, header_map, ["medical"], _yn(svc["medical"]))
        _set_by_alias(row, header_map, ["dental"], _yn(svc["dental"]))
        _set_by_alias(row, header_map, ["vision"], _yn(svc["vision"]))
        _set_by_alias(row, header_map, ["dentures"], _yn(svc["dentures"]))

        # your CSV has telehealth as Yes/No already
        _set_by_alias(row, header_map, ["telehealth"], _yn(ctype["telehealth"]))

        # canceled column exists in your CSV
        _set_by_alias(row, header_map, ["canceled", "cancelled"], _yn(canceled))

        # flexible matches in case of minor header variations
        ccol = _find_header_contains(header, "cancel")
        if ccol and not row.get(ccol):
            row[ccol] = _yn(canceled)

        rows.append(row)

    return rows


def main() -> None:
    base_url = os.environ.get("MEC_BASE_URL", "").strip()
    token = os.environ.get("MEC_TOKEN", "").strip()
    csv_path = os.environ.get("CSV_PATH", "").strip()

    start = os.environ.get("MEC_START", "2026-01-01").strip()
    end = os.environ.get("MEC_END", "2027-12-31").strip()
    limit = os.environ.get("MEC_LIMIT", "200").strip()
    max_span_days = int(os.environ.get("MAX_SPAN_DAYS", "10").strip() or "10")

    debug = os.environ.get("MEC_DEBUG", "").strip() in {"1", "true", "yes", "y"}

    if not base_url or not token or not csv_path:
        raise SystemExit("Missing env vars. Need MEC_BASE_URL, MEC_TOKEN, CSV_PATH.")

    header = _read_existing_header(csv_path)
    if not header:
        header = [
            "canceled",
            "lat",
            "lng",
            "address",
            "city",
            "state",
            "site",
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

    list_params = {"start": start, "end": end, "limit": limit}
    if debug:
        print(f"Fetching events list with params: start={start} end={end} limit={limit}")

    list_payload = _mec_get(base_url, token, "events", params=list_params)
    event_ids = _extract_event_ids_from_list_payload(list_payload)

    if not event_ids:
        raise SystemExit("No events returned from MEC /events endpoint. Try widening MEC_START/MEC_END.")

    if debug:
        print(f"Found {len(event_ids)} unique event IDs")

    all_rows: List[Dict[str, str]] = []
    for eid in event_ids:
        detail = _mec_get(base_url, token, f"events/{eid}")
        all_rows.extend(_build_rows_for_event(detail, header, max_span_days=max_span_days))

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"Wrote {len(all_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
