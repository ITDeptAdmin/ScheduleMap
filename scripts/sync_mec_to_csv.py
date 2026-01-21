#!/usr/bin/env python3
# scripts/sync_mec_to_csv.py
"""
Sync Modern Events Calendar (MEC) events to a GitHub-tracked CSV/TSV.

Required env vars:
- MEC_BASE_URL: e.g. https://www.ramusa.org/wp-json/mec/v1.0
- MEC_TOKEN: MEC API key (sent as header: mec-token)
- CSV_PATH: output filename in repo, e.g. "Clinic Master Schedule for git.csv"

Optional env vars:
- MEC_START: YYYY-MM-DD (default: 2010-01-01)
- MEC_END:   YYYY-MM-DD (default: 2030-12-31)
- MEC_LIMIT: int (default: 200)
- MAX_PAGES: int (default: 50)      # best-effort paging attempts
- MAX_SPAN_DAYS: int (default: 10)  # skip suspiciously long occurrences
- CSV_DELIMITER: "auto" (default), "tab", or "comma"
- MEC_DEBUG: "1" to print diagnostics
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

# --- US state normalization (full name -> USPS) ---
_STATE_ABBR = {
    "alabama": "AL","alaska": "AK","arizona": "AZ","arkansas": "AR","california": "CA",
    "colorado": "CO","connecticut": "CT","delaware": "DE","district of columbia": "DC",
    "florida": "FL","georgia": "GA","hawaii": "HI","idaho": "ID","illinois": "IL","indiana": "IN",
    "iowa": "IA","kansas": "KS","kentucky": "KY","louisiana": "LA","maine": "ME","maryland": "MD",
    "massachusetts": "MA","michigan": "MI","minnesota": "MN","mississippi": "MS","missouri": "MO",
    "montana": "MT","nebraska": "NE","nevada": "NV","new hampshire": "NH","new jersey": "NJ",
    "new mexico": "NM","new york": "NY","north carolina": "NC","north dakota": "ND","ohio": "OH",
    "oklahoma": "OK","oregon": "OR","pennsylvania": "PA","rhode island": "RI","south carolina": "SC",
    "south dakota": "SD","tennessee": "TN","texas": "TX","utah": "UT","vermont": "VT","virginia": "VA",
    "washington": "WA","west virginia": "WV","wisconsin": "WI","wyoming": "WY",
    "puerto rico": "PR",
}

def _to_state_abbr(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if len(s) == 2 and s.isalpha():
        return s.upper()
    k = s.lower()
    return _STATE_ABBR.get(k, s)

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

def _parse_ymd(s: str) -> Optional[date]:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

def _ymd_to_mdy(s: str) -> str:
    d = _parse_ymd(s)
    if not d:
        return (s or "").strip()
    return f"{d.month}/{d.day}/{d.year}"

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

def _mec_get(base_url: str, token: str, path: str, params: Optional[dict] = None, debug: bool = False) -> Any:
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    r = requests.get(url, headers={"mec-token": token}, params=params, timeout=60)
    if debug:
        q = ""
        if params:
            q = "?" + "&".join([f"{k}={v}" for k, v in params.items()])
        print(f"[mec] GET {url}{q} -> {r.status_code}")
    r.raise_for_status()
    return r.json()

# --- delimiter/header detection (preserve what the Schedule page expects) ---
def _detect_existing_format(csv_path: str, forced: str) -> Tuple[Optional[List[str]], str]:
    """
    Returns (header_or_none, delimiter_to_use).
    forced: "auto" | "tab" | "comma"
    """
    if forced == "tab":
        delim = "\t"
    elif forced == "comma":
        delim = ","
    else:
        delim = ","  # default; may be overridden by file sniff

    if not os.path.exists(csv_path):
        return None, delim

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        if forced == "auto":
            # very simple reliable sniff: if tabs are common, treat as TSV
            if sample.count("\t") > sample.count(","):
                delim = "\t"
            else:
                delim = ","

        reader = csv.reader(f, delimiter=delim)
        try:
            header = next(reader)
            return (header if header else None), delim
        except StopIteration:
            return None, delim

# --- payload parsing: recursively find any dict that has an ID-ish field ---
_ID_KEYS = ("ID", "id", "post_id", "event_id")

def _maybe_id(d: Any) -> Optional[int]:
    if not isinstance(d, dict):
        return None
    for k in _ID_KEYS:
        if k in d and d.get(k) is not None:
            try:
                return int(d.get(k))
            except Exception:
                return None
    return None

def _collect_ids_recursive(payload: Any, out: Set[int]) -> None:
    if isinstance(payload, dict):
        eid = _maybe_id(payload)
        if eid is not None:
            out.add(eid)
        for v in payload.values():
            _collect_ids_recursive(v, out)
    elif isinstance(payload, list):
        for it in payload:
            _collect_ids_recursive(it, out)

def _extract_event_ids(payload: Any) -> List[int]:
    s: Set[int] = set()
    _collect_ids_recursive(payload, s)
    return sorted(s)

def _pick_first_dict_value(d: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(d, dict) or not d:
        return None
    first_key = sorted(d.keys(), key=lambda x: str(x))[0]
    v = d.get(first_key)
    return v if isinstance(v, dict) else None

def _extract_location(event_data: Dict[str, Any]) -> Dict[str, str]:
    loc = _pick_first_dict_value(event_data.get("locations", {})) or {}
    state = _to_state_abbr(str(loc.get("state", "")).strip())
    return {
        "facility": str(loc.get("name", "")).strip(),
        "address": str(loc.get("address", "")).strip(),
        "city": str(loc.get("city", "")).strip(),
        "state": state,
        "lat": str(loc.get("latitude", "")).strip(),
        "lng": str(loc.get("longitude", "")).strip(),
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
        elif label in {"clinic type", "clinictype"}:
            out["clinic_type"] = value
        elif label in {"parking_date", "parking date"}:
            out["parking_date"] = value
        elif label in {"parking_time", "parking time"} or label.startswith("parking_time"):
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

    return {"telehealth": has("telehealth"), "popup": has("pop-up") or has("popup")}

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
    return f"{h}:{m:02d} {ap.upper()}"

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
                "start_date": _ymd_to_mdy(s_date),
                "end_date": _ymd_to_mdy(e_date),
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
        print(f"[warn] No occurrences parsed for event {eid} title={title!r}")

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

        # safety for slight header variants
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

def _sort_key(row: Dict[str, str]) -> Tuple[int, str, str]:
    # old sheet uses M/D/YYYY; sort lexically is wrong -> parse it
    s = (row.get("start_date") or "").strip()
    if not s:
        return (1, "9999-12-31", (row.get("title") or "").lower())
    try:
        d = datetime.strptime(s, "%m/%d/%Y").date()
        return (0, d.isoformat(), (row.get("title") or "").lower())
    except Exception:
        return (0, s, (row.get("title") or "").lower())

def main() -> None:
    base_url = os.environ.get("MEC_BASE_URL", "").strip()
    token = os.environ.get("MEC_TOKEN", "").strip()
    csv_path = os.environ.get("CSV_PATH", "").strip()

    start = os.environ.get("MEC_START", "2010-01-01").strip()
    end = os.environ.get("MEC_END", "2030-12-31").strip()
    limit = os.environ.get("MEC_LIMIT", "200").strip()
    max_pages = int((os.environ.get("MAX_PAGES", "50") or "50").strip())
    max_span_days = int((os.environ.get("MAX_SPAN_DAYS", "10") or "10").strip())
    debug = (os.environ.get("MEC_DEBUG", "") or "").strip().lower() in {"1", "true", "yes", "y"}
    forced_delim = (os.environ.get("CSV_DELIMITER", "auto") or "auto").strip().lower()

    if not base_url or not token or not csv_path:
        raise SystemExit("Missing env vars. Need MEC_BASE_URL, MEC_TOKEN, CSV_PATH.")

    header, delim = _detect_existing_format(csv_path, forced_delim)
    if debug:
        print(f"[cfg] base_url={base_url}")
        print(f"[cfg] start={start} end={end} limit={limit} max_pages={max_pages} max_span_days={max_span_days}")
        print(f"[cfg] csv_path={csv_path} delimiter={'TAB' if delim == chr(9) else 'COMMA'} (forced={forced_delim})")

    # If the file doesn't exist yet, use a safe default header (but your existing header will be preserved when present)
    if not header:
        header = [
            "canceled","lat","lng","address","city","state","title","facility","telehealth",
            "start_time","stop_time","start_date","end_date","parking_date","parking_time",
            "medical","dental","vision","dentures","url",
        ]

    # --- list events with best-effort paging ---
    base_params = {"start": start, "end": end, "limit": limit}

    ids: Set[int] = set()

    # page 1
    payload = _mec_get(base_url, token, "events", params=base_params, debug=debug)
    page_ids = _extract_event_ids(payload)
    ids.update(page_ids)
    if debug:
        print(f"[mec] page=1 extracted unique IDs: {len(page_ids)} (total unique now {len(ids)})")

    # try pages 2..N
    for page in range(2, max_pages + 1):
        p = dict(base_params)
        p["page"] = str(page)
        payload = _mec_get(base_url, token, "events", params=p, debug=debug)
        new_ids = _extract_event_ids(payload)
        before = len(ids)
        ids.update(new_ids)
        if debug:
            print(f"[mec] page={page} extracted unique IDs: {len(new_ids)} (total unique now {len(ids)})")
        # stop if nothing new (paging ignored or end reached)
        if len(ids) == before:
            break

    event_ids = sorted(ids)
    if not event_ids:
        raise SystemExit("No events returned from MEC /events endpoint. Check token + endpoint + date params.")

    if debug:
        print(f"[mec] Total unique event IDs: {len(event_ids)}")

    # --- build rows ---
    all_rows: List[Dict[str, str]] = []
    for eid in event_ids:
        detail = _mec_get(base_url, token, f"events/{eid}", params=base_params, debug=debug)
        all_rows.extend(_build_rows_for_event(detail, header, max_span_days=max_span_days, debug=debug))

    # sort for sanity (helps you eyeball missing stuff)
    all_rows.sort(key=_sort_key)

    # --- write preserving delimiter ---
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore", delimiter=delim)
        w.writeheader()
        w.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {csv_path}")

if __name__ == "__main__":
    main()
