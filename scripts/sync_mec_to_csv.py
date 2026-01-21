#!/usr/bin/env python3
# scripts/sync_mec_to_csv.py
"""
Sync Modern Events Calendar (MEC) events to a GitHub-tracked CSV.

Required env vars:
- MEC_BASE_URL: e.g. https://www.ramusa.org/wp-json/mec/v1.0
- MEC_TOKEN: MEC API key (sent as header: mec-token)
- CSV_PATH: output CSV filename in repo, e.g. "Clinic Master Schedule for git.csv"

Optional env vars:
- MEC_START: YYYY-MM-DD (default: 2010-01-01)
- MEC_END:   YYYY-MM-DD (default: 2030-12-31)
- MEC_LIMIT: int (default: 200)
- MAX_PAGES: int (default: 50)   # defensive paging attempt on /events list endpoint
- MAX_SPAN_DAYS: int (default: 10)  # skips suspiciously long occurrences from repeat rules
- MEC_DEBUG: "1" to print diagnostics
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
        r = csv.reader(f)
        try:
            header = next(r)
            return header if header else None
        except StopIteration:
            return None


def _header_map(header: List[str]) -> Dict[str, str]:
    return {_norm(h): h for h in header}


def _set_by_alias(row: Dict[str, str], hmap: Dict[str, str], aliases: List[str], value: Any) -> None:
    v = "" if value is None else str(value)
    for a in aliases:
        k = _norm(a)
        if k in hmap:
            row[hmap[k]] = v
            return


def _find_header_contains(header: List[str], *needles: str) -> Optional[str]:
    needles = [n.lower() for n in needles]
    for h in header:
        hl = h.lower()
        if all(n in hl for n in needles):
            return h
    return None


def _yn(b: bool) -> str:
    return "Yes" if b else "No"


def _mec_get(session: requests.Session, base_url: str, token: str, path: str, params: Optional[dict] = None, debug: bool = False) -> Any:
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    r = session.get(url, headers={"mec-token": token}, params=params, timeout=60)
    if debug:
        print(f"[mec] GET {r.url} -> {r.status_code}")
    try:
        r.raise_for_status()
    except Exception:
        # print a short snippet to help diagnose 500/401 responses
        snippet = (r.text or "")[:500]
        print(f"[mec] ERROR body (first 500 chars): {snippet}")
        raise
    return r.json()


def _flatten_events_payload(payload: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    def add_list(lst: Any) -> None:
        if isinstance(lst, list):
            for it in lst:
                if isinstance(it, dict):
                    out.append(it)

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


def _extract_event_ids(list_payload: Any) -> List[int]:
    items = _flatten_events_payload(list_payload)
    ids: List[int] = []
    for e in items:
        for k in ("ID", "id", "post_id", "event_id"):
            if k in e and e.get(k) is not None:
                try:
                    ids.append(int(e.get(k)))
                    break
                except Exception:
                    continue
    return sorted(set(ids))


def _fetch_all_event_ids(
    session: requests.Session,
    base_url: str,
    token: str,
    start: str,
    end: str,
    limit: int,
    max_pages: int,
    debug: bool,
) -> List[int]:
    """
    Defensive approach:
    - Call /events with given start/end/limit
    - Also try paging using page=, paged=, offset=
      (some MEC installs implement one of these, some ignore them).
    - Stop when IDs stop changing or we get empty results.
    """
    base_params = {"start": start, "end": end, "limit": str(limit)}

    # First request (no paging)
    first = _mec_get(session, base_url, token, "events", params=base_params, debug=debug)
    ids0 = _extract_event_ids(first)
    if debug:
        print(f"[mec] IDs (no paging): {len(ids0)}")

    all_ids = set(ids0)

    # Try common paging styles
    paging_styles: List[Tuple[str, int]] = [
        ("page", 2),
        ("paged", 2),
        ("offset", limit),  # offset increments by limit
    ]

    for style, start_val in paging_styles:
        if debug:
            print(f"[mec] Trying paging style: {style}")
        prev_snapshot = sorted(all_ids)

        val = start_val
        for i in range(2, max_pages + 1):
            params = dict(base_params)

            if style in ("page", "paged"):
                params[style] = str(i)
            else:  # offset
                params[style] = str(val)
                val += limit

            payload = _mec_get(session, base_url, token, "events", params=params, debug=debug)
            ids = _extract_event_ids(payload)
            if debug:
                print(f"[mec] {style}={params[style]} -> {len(ids)} IDs")

            if not ids:
                break

            before = len(all_ids)
            all_ids.update(ids)

            # If this paging param is ignored, results won't change; bail quickly
            if len(all_ids) == before and sorted(all_ids) == prev_snapshot:
                # no new IDs; assume paging param not supported or no more pages
                break

            prev_snapshot = sorted(all_ids)

    return sorted(all_ids)


def _pick_first_dict_value(d: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(d, dict) or not d:
        return None
    first_key = sorted(d.keys(), key=lambda x: str(x))[0]
    v = d.get(first_key)
    return v if isinstance(v, dict) else None


def _extract_location(event_data: Dict[str, Any]) -> Dict[str, str]:
    # Common MEC structure: locations is a dict-of-dicts
    loc = _pick_first_dict_value(event_data.get("locations", {})) or {}

    # Fallbacks some installs expose:
    # - event_data["location"] or event_data["data"]["location"] etc.
    # We'll try to be resilient.
    def pick(*keys: str) -> str:
        for k in keys:
            v = loc.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
        return ""

    return {
        "facility": str(loc.get("name", "")).strip(),
        "address": str(loc.get("address", "")).strip(),
        "city": str(loc.get("city", "")).strip(),
        "state": str(loc.get("state", "")).strip(),
        "lat": pick("latitude", "lat"),
        "lng": pick("longitude", "lng", "lon"),
    }


def _extract_custom_fields(event_data: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    fields = event_data.get("fields", [])
    if not isinstance(fields, list):
        return out

    for f in fields:
        if not isinstance(f, dict):
            continue
        label = _norm(str(f.get("label", "")))
        value = "" if f.get("value") is None else str(f.get("value")).strip()

        if label == "services":
            out["services"] = value
        elif label == "clinic type":
            out["clinic_type"] = value
        elif label == "parking_date":
            out["parking_date"] = value
        elif label.startswith("parking_time"):
            out["parking_time"] = value

    return out


def _services_flags(services_value: str, fallback_text: str) -> Dict[str, bool]:
    src = services_value.strip() or fallback_text

    def has(w: str) -> bool:
        return w.lower() in src.lower()

    return {
        "medical": has("medical"),
        "dental": has("dental"),
        "vision": has("vision"),
        "dentures": has("denture"),
    }


def _clinic_type_flags(clinic_type_value: str, fallback_text: str) -> Dict[str, bool]:
    src = clinic_type_value.strip() or fallback_text

    def has(w: str) -> bool:
        return w.lower() in src.lower()

    return {"telehealth": has("telehealth"), "popup": has("popup")}


def _parse_ymd(s: str) -> Optional[date]:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _format_time(hour: Any, minutes: Any, ampm: Any) -> str:
    if hour in (None, "", "0", 0):
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


def _detail_dates_blocks(detail: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []

    d0 = detail.get("dates")
    if isinstance(d0, list):
        for it in d0:
            if isinstance(it, dict):
                blocks.append(it)

    d1 = detail.get("date")
    if isinstance(d1, dict):
        blocks.append(d1)

    data = detail.get("data")
    if isinstance(data, dict) and isinstance(data.get("dates"), list) and not blocks:
        for it in data["dates"]:
            if isinstance(it, dict):
                blocks.append(it)

    return blocks


def _extract_occurrences(detail: Dict[str, Any], max_span_days: int) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    blocks = _detail_dates_blocks(detail)

    for b in blocks:
        s_obj = b.get("start") or {}
        e_obj = b.get("end") or {}

        s_date = str(s_obj.get("date") or "").strip()
        e_date = str(e_obj.get("date") or s_date or "").strip()
        if not s_date:
            continue

        allday = str(b.get("allday") or "0").strip().lower() in {"1", "true", "yes"}
        hide_time = str(b.get("hide_time") or "0").strip().lower() in {"1", "true", "yes"}

        start_time = "" if (allday or hide_time) else _format_time(s_obj.get("hour"), s_obj.get("minutes"), s_obj.get("ampm"))
        stop_time = "" if (allday or hide_time) else _format_time(e_obj.get("hour"), e_obj.get("minutes"), e_obj.get("ampm"))

        sd = _parse_ymd(s_date)
        ed = _parse_ymd(e_date)
        if sd and ed and (ed - sd).days > max_span_days:
            continue

        out.append(
            {
                "start_date": s_date,
                "end_date": e_date,
                "start_time": start_time,
                "stop_time": stop_time,
            }
        )

    # de-dupe
    seen = set()
    uniq: List[Dict[str, str]] = []
    for o in out:
        k = (o["start_date"], o["end_date"], o["start_time"], o["stop_time"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(o)

    return uniq


def _extract_canceled(event_data: Dict[str, Any]) -> bool:
    meta = event_data.get("meta", {})
    status = ""
    if isinstance(meta, dict):
        status = str(meta.get("mec_event_status") or meta.get("event_status") or "").strip()
    if not status:
        status = str(event_data.get("event_status") or "").strip()
    return "cancel" in status.lower()


def _build_rows_for_event(detail: Dict[str, Any], header: List[str], max_span_days: int, debug: bool) -> List[Dict[str, str]]:
    data = detail.get("data", {})
    post = data.get("post", {}) if isinstance(data, dict) else {}
    content_html = post.get("post_content") or post.get("post_content_filtered") or ""
    content_text = _strip_html(str(content_html))

    loc = _extract_location(data)
    fields = _extract_custom_fields(data)
    svc = _services_flags(fields.get("services", ""), content_text)
    ctype = _clinic_type_flags(fields.get("clinic_type", ""), content_text)
    canceled = _extract_canceled(data)

    parking_date = fields.get("parking_date", "").strip()
    parking_time = fields.get("parking_time", "").strip()

    title = str(data.get("title") or post.get("post_title") or "").strip()
    url = str(data.get("permalink") or "").strip() or str(post.get("guid") or "").strip()

    occurrences = _extract_occurrences(detail, max_span_days=max_span_days)
    if debug and not occurrences:
        eid = data.get("ID") or data.get("id") or post.get("ID")
        print(f"[warn] No occurrences parsed for event {eid} title={title!r}. Keys(detail)={list(detail.keys())}")

    if not occurrences:
        occurrences = [{"start_date": "", "end_date": "", "start_time": "", "stop_time": ""}]

    hmap = _header_map(header)
    rows: List[Dict[str, str]] = []

    for occ in occurrences:
        row = {h: "" for h in header}

        _set_by_alias(row, hmap, ["canceled", "cancelled"], _yn(canceled))
        _set_by_alias(row, hmap, ["lat"], loc["lat"])
        _set_by_alias(row, hmap, ["lng", "lon", "longitude"], loc["lng"])
        _set_by_alias(row, hmap, ["address"], loc["address"])
        _set_by_alias(row, hmap, ["city"], loc["city"])
        _set_by_alias(row, hmap, ["state"], loc["state"])
        _set_by_alias(row, hmap, ["facility"], loc["facility"])
        _set_by_alias(row, hmap, ["title"], title)
        _set_by_alias(row, hmap, ["url"], url)

        site_value = ", ".join([p for p in [loc["city"], loc["state"]] if p])
        _set_by_alias(row, hmap, ["site"], site_value)

        _set_by_alias(row, hmap, ["telehealth"], _yn(ctype["telehealth"]))

        _set_by_alias(row, hmap, ["start_date", "start date"], occ["start_date"])
        _set_by_alias(row, hmap, ["end_date", "end date"], occ["end_date"])
        _set_by_alias(row, hmap, ["start_time", "start time"], occ["start_time"])
        _set_by_alias(row, hmap, ["stop_time", "stop time", "end_time", "end time"], occ["stop_time"])

        _set_by_alias(row, hmap, ["parking_date", "parking date"], parking_date)
        _set_by_alias(row, hmap, ["parking_time", "parking time"], parking_time)

        _set_by_alias(row, hmap, ["medical"], _yn(svc["medical"]))
        _set_by_alias(row, hmap, ["dental"], _yn(svc["dental"]))
        _set_by_alias(row, hmap, ["vision"], _yn(svc["vision"]))
        _set_by_alias(row, hmap, ["dentures"], _yn(svc["dentures"]))

        # extra safety for slight header variants
        for col, val in [
            (_find_header_contains(header, "start", "date"), occ["start_date"]),
            (_find_header_contains(header, "end", "date"), occ["end_date"]),
            (_find_header_contains(header, "start", "time"), occ["start_time"]),
            (_find_header_contains(header, "stop", "time"), occ["stop_time"]),
        ]:
            if col and not row.get(col):
                row[col] = val

        rows.append(row)

    return rows


def main() -> None:
    base_url = os.environ.get("MEC_BASE_URL", "").strip()
    token = os.environ.get("MEC_TOKEN", "").strip()
    csv_path = os.environ.get("CSV_PATH", "").strip()

    start = os.environ.get("MEC_START", "2010-01-01").strip()
    end = os.environ.get("MEC_END", "2030-12-31").strip()
    limit = int((os.environ.get("MEC_LIMIT", "200").strip() or "200"))
    max_pages = int((os.environ.get("MAX_PAGES", "50").strip() or "50"))
    max_span_days = int((os.environ.get("MAX_SPAN_DAYS", "10").strip() or "10"))
    debug = os.environ.get("MEC_DEBUG", "").strip().lower() in {"1", "true", "yes", "y"}

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

    with requests.Session() as session:
        if debug:
            print(f"[cfg] base_url={base_url}")
            print(f"[cfg] start={start} end={end} limit={limit} max_pages={max_pages}")

        event_ids = _fetch_all_event_ids(
            session=session,
            base_url=base_url,
            token=token,
            start=start,
            end=end,
            limit=limit,
            max_pages=max_pages,
            debug=debug,
        )

        if not event_ids:
            raise SystemExit("No events returned from MEC /events endpoint. Try widening MEC_START/MEC_END.")

        if debug:
            print(f"[mec] Total unique event IDs: {len(event_ids)}")

        all_rows: List[Dict[str, str]] = []
        for eid in event_ids:
            detail = _mec_get(session, base_url, token, f"events/{eid}", debug=debug)
            all_rows.extend(_build_rows_for_event(detail, header, max_span_days=max_span_days, debug=debug))

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
