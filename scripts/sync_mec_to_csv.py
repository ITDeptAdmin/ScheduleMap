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
- MAX_PAGES: int (default: 50)   # best-effort paging attempts
- MAX_SPAN_DAYS: int (default: 10)
- CSV_DELIMITER: "tab" or "comma" (default: comma)
- MEC_DEBUG: "1" to print diagnostics
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime, date
from typing import Any, Dict, List, Optional

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


def _csv_delimiter_from_env() -> str:
    v = (os.environ.get("CSV_DELIMITER", "") or "").strip().lower()
    return "\t" if v in {"tab", "tsv"} else ","


def _read_existing_header(csv_path: str, delimiter: str) -> Optional[List[str]]:
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        r = csv.reader(f, delimiter=delimiter)
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


def _looks_like_event(item: Dict[str, Any]) -> bool:
    # Prevent grabbing random dicts that happen to have an "ID" key.
    keys = set(item.keys())
    eventish = {"title", "permalink", "date", "dates", "start", "end", "locations", "data", "post"}
    return bool(keys & eventish)


def _extract_event_id(item: Dict[str, Any]) -> Optional[int]:
    # Prefer the keys that are most likely the MEC event post ID.
    for k in ("event_id", "post_id", "ID", "id"):
        if k in item and item.get(k) is not None:
            try:
                return int(item.get(k))
            except Exception:
                return None
    return None


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

        out.append({"start_date": s_date, "end_date": e_date, "start_time": start_time, "stop_time": stop_time})

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


def _extract_event_ids(list_payload: Any) -> List[int]:
    items = _flatten_events_payload(list_payload)
    ids: List[int] = []
    for it in items:
        if not _looks_like_event(it):
            continue
        eid = _extract_event_id(it)
        if eid is not None:
            ids.append(eid)
    return sorted(set(ids))


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

    delimiter = _csv_delimiter_from_env()
    delim_label = "TAB" if delimiter == "\t" else "COMMA"

    if not base_url or not token or not csv_path:
        raise SystemExit("Missing env vars. Need MEC_BASE_URL, MEC_TOKEN, CSV_PATH.")

    if debug:
        print(f"[cfg] base_url={base_url}")
        print(f"[cfg] start={start} end={end} limit={limit} max_pages={max_pages} max_span_days={max_span_days}")
        print(f"[cfg] csv_path={csv_path} delimiter={delim_label}")

    header = _read_existing_header(csv_path, delimiter=delimiter)
    if not header:
        header = [
            "canceled","lat","lng","address","city","state","site","facility","telehealth",
            "start_time","stop_time","start_date","end_date","parking_date","parking_time",
            "medical","dental","vision","dentures","url",
        ]

    base_params = {"start": start, "end": end, "limit": limit}

    # Page 1
    payload = _mec_get(base_url, token, "events", params=base_params, debug=debug)
    all_ids = set(_extract_event_ids(payload))
    if debug:
        print(f"[mec] page=1 extracted unique IDs: {len(all_ids)}")

    # Best-effort paging (many MEC installs ignore page params; harmless to try)
    for page in range(2, max_pages + 1):
        p = dict(base_params)
        p["page"] = str(page)
        payload = _mec_get(base_url, token, "events", params=p, debug=debug)
        ids = set(_extract_event_ids(payload))
        before = len(all_ids)
        all_ids |= ids
        if debug:
            print(f"[mec] page={page} -> extracted {len(ids)} IDs (total unique now {len(all_ids)})")
        # If adding pages doesn't increase IDs, paging is probably ignored
        if len(all_ids) == before:
            break

    event_ids = sorted(all_ids)
    if not event_ids:
        raise SystemExit("No events returned from MEC /events endpoint. Check token + endpoint + date params.")

    if debug:
        print(f"[mec] Total unique event IDs: {len(event_ids)}")

    all_rows: List[Dict[str, str]] = []
    skipped_404 = 0
    skipped_other = 0

    for eid in event_ids:
        try:
            detail = _mec_get(base_url, token, f"events/{eid}", params=None, debug=debug)
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 404:
                skipped_404 += 1
                if debug:
                    print(f"[skip] events/{eid} -> 404")
                continue
            skipped_other += 1
            if debug:
                print(f"[skip] events/{eid} -> HTTP {status}")
            continue
        except Exception:
            skipped_other += 1
            if debug:
                print(f"[skip] events/{eid} -> exception")
            continue

        all_rows.extend(_build_rows_for_event(detail, header, max_span_days=max_span_days, debug=debug))

    # Optional: sort by start_date for nicer diffs/readability
    def _sort_key(row: Dict[str, str]):
        s = (row.get("start_date") or "").strip()
        return (s == "", s, (row.get("title") or "").lower())

    all_rows.sort(key=_sort_key)

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore", delimiter=delimiter)
        w.writeheader()
        w.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {csv_path} (skipped 404={skipped_404}, skipped_other={skipped_other})")


if __name__ == "__main__":
    main()
