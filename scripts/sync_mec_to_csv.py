#!/usr/bin/env python3
# scripts/sync_mec_to_csv.py
"""
Sync Modern Events Calendar (MEC) events to a GitHub-tracked CSV.

Fixes included in this version:
- Uses MEC REST "All Events" pagination correctly: start_date + offset, advancing via pagination.next_date/next_offset.
- Retries on HTTP 202/429/5xx.
- Services flags come from custom field "Services"; if missing, falls back to taxonomy/title text.
- Extracts occurrences robustly:
    * Builds per-occurrence hints from the /events list response (deep-scans each list item for start/end blocks).
    * Falls back to scanning the event detail payload for occurrences.
- Per-event row cleanup to eliminate duplicates and bad "range" rows:
    * If any single-day rows exist, drop multi-day range rows for that event.
    * For duplicates on same date(s), keep the row that contains times (prefer timeful over timeless).
- Writes comma CSV by default (set CSV_DELIMITER=tab for TSV).

NEW in this version:
- Adds "title" and "clinic_type" columns (matching your "raw file" export format).
- Blanks start_time/end_time for "all-day" events.
- Optionally forces POPUP clinics (telehealth=No) to be treated as all-day (blank times),
  because your UI doesn’t want to show 8:00 AM–6:00 PM for popup clinics.
  Control with env var POPUP_ALLDAY (default: 1).

Required env vars:
- MEC_BASE_URL  e.g. https://www.ramusa.org/wp-json/mec/v1.0
- MEC_TOKEN     API key (header mec-token)
- CSV_PATH      output file path in repo

Optional env vars:
- MEC_START       YYYY-MM-DD (default: 2010-01-01)
- MEC_END         YYYY-MM-DD (default: 2030-12-31)
- MEC_LIMIT       int (default: 200)
- MAX_PAGES       int (default: 5000)  # safety cap for pagination loops
- MAX_SPAN_DAYS   int (default: 10)    # skip weird huge multi-day spans
- CSV_DELIMITER   "comma" or "tab" (default: comma)
- DATE_FORMAT     "mdy" or "ymd" (default: mdy)
- INCLUDE_PAST    "1" include past events (default: 1)
- INCLUDE_ONGOING "1" include ongoing events (default: 1)
- POPUP_ALLDAY    "1" blank times for popup clinics (telehealth=No) (default: 1)
- MEC_DEBUG       "1" to print diagnostics
"""

from __future__ import annotations

import csv
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests


_STATE_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "district of columbia": "DC",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN",
    "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO",
    "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA",
    "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "puerto rico": "PR",
}

# Matches your "raw file looks like this" header order
DEFAULT_HEADER: List[str] = [
    "canceled",
    "lat",
    "lng",
    "address",
    "city",
    "state",
    "site",
    "title",
    "facility",
    "clinic_type",
    "telehealth",
    "start_time",
    "end_time",
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


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[()]", "", s)
    return s


def _to_state_abbr(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if len(s) == 2 and s.isalpha():
        return s.upper()
    return _STATE_ABBR.get(s.lower(), s)


def _yn(v: bool) -> str:
    return "Yes" if v else "No"


def _get_delimiter() -> str:
    v = (os.environ.get("CSV_DELIMITER", "comma") or "comma").strip().lower()
    return "\t" if v in {"tab", "tsv"} else ","


def _parse_ymd(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def _format_date(s: str, date_format: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if date_format == "ymd":
        return s
    d = _parse_ymd(s)
    if not d:
        return s
    return f"{d.month}/{d.day}/{d.year}"


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


def _coerce_text(v: Any) -> str:
    """Turn MEC field values into a useful string (handles list/dict)."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        parts = []
        for it in v:
            t = _coerce_text(it)
            if t:
                parts.append(t)
        return ", ".join(parts).strip()
    if isinstance(v, dict):
        parts = []
        for k, it in v.items():
            t = _coerce_text(it)
            if t:
                parts.append(t)
            else:
                if isinstance(it, (int, float)) and it:
                    parts.append(str(k))
        return ", ".join([p for p in parts if p]).strip()
    return str(v).strip()


@dataclass(frozen=True)
class Cfg:
    base_url: str
    token: str
    csv_path: str
    start: str
    end: str
    limit: int
    max_pages: int
    max_span_days: int
    delimiter: str
    date_format: str
    include_past: bool
    include_ongoing: bool
    popup_allday: bool
    debug: bool


def _debug(cfg: Cfg, msg: str) -> None:
    if cfg.debug:
        print(msg)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "ram-mec-sync/2.3",
            "Accept": "application/json,text/plain,*/*",
        }
    )
    return s


def _resp_snippet(resp: requests.Response, limit: int = 300) -> str:
    try:
        txt = resp.text or ""
    except Exception:
        return ""
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt[:limit]


def _mec_get(
    cfg: Cfg,
    sess: requests.Session,
    path: str,
    params: Optional[Dict[str, str]] = None,
    *,
    allow_404: bool = False,
    retries: int = 6,
) -> requests.Response:
    """
    Retries on:
      - 202 (accepted/processing)
      - 429 rate-limit
      - 5xx transient errors
    """
    url = cfg.base_url.rstrip("/") + "/" + path.lstrip("/")
    headers = {"mec-token": cfg.token}

    resp: requests.Response
    for attempt in range(retries + 1):
        resp = sess.get(url, headers=headers, params=params, timeout=60)
        _debug(cfg, f"[mec] GET {resp.url} -> {resp.status_code}")

        if allow_404 and resp.status_code == 404:
            return resp

        if resp.status_code in {202, 429, 500, 502, 503, 504} and attempt < retries:
            backoff = min(30, 2 ** attempt)
            _debug(cfg, f"[mec] retryable {resp.status_code}; body='{_resp_snippet(resp)}'; sleeping {backoff}s")
            time.sleep(backoff)
            continue

        return resp

    return resp


def _extract_event_id(item: Dict[str, Any]) -> Optional[int]:
    for k in ("ID", "id", "post_id", "event_id"):
        if k in item and item.get(k) is not None:
            try:
                return int(item.get(k))
            except Exception:
                return None
    d = item.get("data")
    if isinstance(d, dict):
        for k in ("ID", "id", "post_id", "event_id"):
            if k in d and d.get(k) is not None:
                try:
                    return int(d.get(k))
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


def _deep_find_label_value_pairs(obj: Any) -> Iterable[Tuple[str, Any]]:
    """Yield (label, value) for any dict containing keys like label/value anywhere in the payload."""
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if "label" in cur and ("value" in cur or "values" in cur):
                label = _coerce_text(cur.get("label"))
                value = cur.get("value", cur.get("values"))
                if label:
                    yield (label, value)
            for v in cur.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            for it in cur:
                if isinstance(it, (dict, list)):
                    stack.append(it)


def _extract_custom_fields(event_data: Dict[str, Any], whole_detail: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}

    def consider(label_raw: str, value_any: Any) -> None:
        label = _norm(label_raw)
        value = _coerce_text(value_any)

        if label == "services":
            out["services"] = value
        elif label in {"clinic type", "clinictype"}:
            out["clinic_type"] = value
        elif label in {"parking_date", "parking date"}:
            out["parking_date"] = value
        elif label in {"parking_time", "parking time"} or label.startswith("parking_time"):
            out["parking_time"] = value

    fields = event_data.get("fields", [])
    if isinstance(fields, list):
        for f in fields:
            if isinstance(f, dict):
                consider(_coerce_text(f.get("label")), f.get("value"))

    need_any = any(k not in out or not out.get(k) for k in ("services", "clinic_type", "parking_date", "parking_time"))
    if need_any:
        for lbl, val in _deep_find_label_value_pairs(whole_detail):
            consider(lbl, val)

    return out


def _extract_taxonomy_text(event_data: Dict[str, Any]) -> str:
    chunks: List[str] = []

    def take_names(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, dict):
            for v in obj.values():
                take_names(v)
        elif isinstance(obj, list):
            for it in obj:
                take_names(it)
        else:
            s = str(obj).strip()
            if s and len(s) <= 120:
                chunks.append(s)

    for key in ("categories", "category", "tags", "tag", "labels", "label", "terms", "term"):
        if key in event_data:
            take_names(event_data.get(key))

    title = str(event_data.get("title") or "").strip()
    if title:
        chunks.append(title)

    txt = " ".join(chunks)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _services_flags(services_value: str, fallback_tax_text: str) -> Dict[str, bool]:
    src = (services_value or "").strip() or (fallback_tax_text or "").strip()

    def has(w: str) -> bool:
        return w.lower() in src.lower()

    return {
        "medical": has("medical"),
        "dental": has("dental"),
        "vision": has("vision"),
        "dentures": has("denture"),
    }


def _clinic_type_flags(clinic_type_value: str, fallback_tax_text: str) -> Dict[str, bool]:
    src = (clinic_type_value or "").strip() or (fallback_tax_text or "").strip()

    def has(w: str) -> bool:
        return w.lower() in src.lower()

    return {"telehealth": has("telehealth")}


def _extract_canceled(event_data: Dict[str, Any]) -> bool:
    meta = event_data.get("meta", {})
    status = ""
    if isinstance(meta, dict):
        status = str(meta.get("mec_event_status") or meta.get("event_status") or "").strip()
    if not status:
        status = str(event_data.get("event_status") or "").strip()
    return "cancel" in status.lower()


# ---------------- Occurrence extraction ----------------

_YMD_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _find_occurrence_blocks_anywhere(obj: Any) -> Iterable[Dict[str, Any]]:
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if "start" in cur and "end" in cur:
                s = cur.get("start")
                e = cur.get("end")
                if isinstance(s, dict) and isinstance(e, dict) and ("date" in s or "date" in e):
                    yield cur
            for v in cur.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            for it in cur:
                if isinstance(it, (dict, list)):
                    stack.append(it)


def _extract_occurrences_from_detail(detail: Dict[str, Any], max_span_days: int) -> List[Dict[str, Any]]:
    occs_raw: List[Dict[str, Any]] = []

    for b in _find_occurrence_blocks_anywhere(detail):
        s_obj = b.get("start") or {}
        e_obj = b.get("end") or {}

        s_date = str(s_obj.get("date") or "").strip()
        e_date = str(e_obj.get("date") or s_date or "").strip()
        if not s_date:
            continue

        sd = _parse_ymd(s_date)
        ed = _parse_ymd(e_date)
        if sd and ed and (ed - sd).days > max_span_days:
            continue

        occs_raw.append(
            {
                "start": s_obj,
                "end": e_obj,
                "allday": b.get("allday"),
                "hide_time": b.get("hide_time"),
            }
        )

    # strong dedupe
    seen: Set[Tuple[str, str, str, str, str, str, str]] = set()
    occs: List[Dict[str, Any]] = []
    for o in occs_raw:
        s = o.get("start") or {}
        e = o.get("end") or {}
        key = (
            str(s.get("date") or ""),
            str(e.get("date") or ""),
            str(s.get("hour") or ""),
            str(s.get("minutes") or ""),
            str(s.get("ampm") or ""),
            str(e.get("hour") or ""),
            str(e.get("minutes") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        occs.append(o)

    return occs


def _extract_occ_hint_from_list_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    hints: List[Dict[str, Any]] = []

    # 1) top-level start/end
    if isinstance(item.get("start"), dict) and isinstance(item.get("end"), dict):
        s = item["start"]
        e = item["end"]
        if "date" in s or "date" in e:
            hints.append({"start": s, "end": e, "allday": item.get("allday"), "hide_time": item.get("hide_time")})

    # 2) deep scan in item (many installs nest occurrence blocks here)
    for b in _find_occurrence_blocks_anywhere(item):
        s = b.get("start") or {}
        if isinstance(s, dict) and str(s.get("date") or "").strip():
            hints.append(
                {
                    "start": b.get("start") or {},
                    "end": b.get("end") or {},
                    "allday": b.get("allday"),
                    "hide_time": b.get("hide_time"),
                }
            )

    # 3) alternate keys
    for k1, k2 in (("start_date", "end_date"), ("date", "date")):
        sd = _coerce_text(item.get(k1))
        ed = _coerce_text(item.get(k2)) if k2 != k1 else sd
        if _YMD_RE.match(sd):
            hints.append({"start": {"date": sd}, "end": {"date": ed or sd}, "allday": "1", "hide_time": "1"})

    # 4) timestamp-only variant
    ts = item.get("timestamp")
    if ts is not None:
        try:
            ts_i = int(ts)
            dt = datetime.utcfromtimestamp(ts_i)
            ymd = dt.strftime("%Y-%m-%d")
            hints.append({"start": {"date": ymd}, "end": {"date": ymd}, "allday": "1", "hide_time": "1"})
        except Exception:
            pass

    # dedupe within item
    seen: Set[Tuple[str, str, str, str, str, str, str]] = set()
    uniq: List[Dict[str, Any]] = []
    for h in hints:
        s = h.get("start") or {}
        e = h.get("end") or {}
        key = (
            str(s.get("date") or ""),
            str(e.get("date") or ""),
            str(s.get("hour") or ""),
            str(s.get("minutes") or ""),
            str(s.get("ampm") or ""),
            str(e.get("hour") or ""),
            str(e.get("minutes") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(h)

    return uniq


def _occ_to_row_dates_times(occ: Dict[str, Any], cfg: Cfg, *, force_all_day: bool) -> Dict[str, str]:
    s_obj = occ.get("start") or {}
    e_obj = occ.get("end") or {}
    s_date = str(s_obj.get("date") or "").strip()
    e_date = str(e_obj.get("date") or s_date or "").strip()

    allday = str(occ.get("allday") or "0").strip().lower() in {"1", "true", "yes"}
    hide_time = str(occ.get("hide_time") or "0").strip().lower() in {"1", "true", "yes"}

    is_all_day = force_all_day or allday or hide_time

    start_time = "" if is_all_day else _format_time(s_obj.get("hour"), s_obj.get("minutes"), s_obj.get("ampm"))
    end_time = "" if is_all_day else _format_time(e_obj.get("hour"), e_obj.get("minutes"), e_obj.get("ampm"))

    return {
        "start_date": _format_date(s_date, cfg.date_format),
        "end_date": _format_date(e_date, cfg.date_format),
        "start_time": start_time,
        "end_time": end_time,
        "_start_ymd": s_date,
        "_end_ymd": e_date,
        "_is_all_day": "1" if is_all_day else "0",
    }


def _ymd_in_range(d: str, start: str, end: str) -> bool:
    dd = _parse_ymd(d)
    if not dd:
        return True
    ds = _parse_ymd(start)
    de = _parse_ymd(end)
    if not ds or not de:
        return True
    return ds <= dd <= de


def _flatten_all_events_response(payload: Any) -> List[Dict[str, Any]]:
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


def _safe_site(city: str, state: str) -> str:
    parts = [p for p in [city.strip(), state.strip()] if p]
    return ", ".join(parts)


def _fetch_event_detail(cfg: Cfg, sess: requests.Session, eid: int) -> Optional[Dict[str, Any]]:
    resp = _mec_get(cfg, sess, f"events/{eid}", params=None, allow_404=True)
    if resp.status_code == 404:
        _debug(cfg, f"[warn] /events/{eid} -> 404 (skipping)")
        return None
    if resp.status_code != 200:
        raise SystemExit(f"MEC /events/{eid} failed: HTTP {resp.status_code} url={resp.url} body='{_resp_snippet(resp)}'")
    return resp.json()


def _iter_all_occurrences(cfg: Cfg, sess: requests.Session) -> Tuple[Set[int], Dict[int, List[Dict[str, Any]]]]:
    start_date = cfg.start
    offset = 0
    seen_guard: Set[Tuple[str, int]] = set()

    all_ids: Set[int] = set()
    occ_map: Dict[int, List[Dict[str, Any]]] = {}

    params_base: Dict[str, str] = {
        "limit": str(cfg.limit),
        "include_past_events": "1" if cfg.include_past else "0",
        "include_ongoing_events": "1" if cfg.include_ongoing else "0",
    }

    pages = 0
    while True:
        pages += 1
        if pages > cfg.max_pages:
            _debug(cfg, f"[mec] hit MAX_PAGES={cfg.max_pages}, stopping pagination safety cap")
            break

        key = (start_date, offset)
        if key in seen_guard:
            _debug(cfg, f"[mec] pagination loop detected at start_date={start_date} offset={offset}, stopping")
            break
        seen_guard.add(key)

        params = dict(params_base)
        params["start_date"] = start_date
        params["offset"] = str(offset)

        resp = _mec_get(cfg, sess, "events", params=params)
        if resp.status_code != 200:
            raise SystemExit(f"MEC /events failed: HTTP {resp.status_code} url={resp.url} body='{_resp_snippet(resp)}'")

        payload = resp.json()
        items = _flatten_all_events_response(payload)

        ids_this: Set[int] = set()
        hints_added = 0

        for it in items:
            if not isinstance(it, dict):
                continue
            eid = _extract_event_id(it)
            if eid is None:
                continue

            ids_this.add(eid)
            all_ids.add(eid)

            hints = _extract_occ_hint_from_list_item(it)
            for h in hints:
                s_obj = h.get("start") or {}
                s_date = str(s_obj.get("date") or "").strip()
                if s_date and not _ymd_in_range(s_date, cfg.start, cfg.end):
                    continue
                occ_map.setdefault(eid, []).append(h)
                hints_added += 1

        _debug(
            cfg,
            f"[mec] list page {pages}: start_date={start_date} offset={offset} "
            f"-> items={len(items)} ids={len(ids_this)} hints_added={hints_added} unique_total={len(all_ids)}"
        )

        pag = payload.get("pagination") if isinstance(payload, dict) else None
        next_date = str(pag.get("next_date") or "").strip() if isinstance(pag, dict) else ""
        next_offset = pag.get("next_offset") if isinstance(pag, dict) else None

        if not next_date or next_offset is None:
            break

        if _parse_ymd(next_date) and _parse_ymd(cfg.end) and _parse_ymd(next_date) > _parse_ymd(cfg.end):
            break

        try:
            offset = int(next_offset)
        except Exception:
            break
        start_date = next_date

    # de-dupe hints per event
    for eid, occs in list(occ_map.items()):
        seen: Set[Tuple[str, str, str, str, str, str, str]] = set()
        uniq: List[Dict[str, Any]] = []
        for o in occs:
            s = o.get("start") or {}
            e = o.get("end") or {}
            k = (
                str(s.get("date") or ""),
                str(e.get("date") or ""),
                str(s.get("hour") or ""),
                str(s.get("minutes") or ""),
                str(s.get("ampm") or ""),
                str(e.get("hour") or ""),
                str(e.get("minutes") or ""),
            )
            if k in seen:
                continue
            seen.add(k)
            uniq.append(o)
        occ_map[eid] = uniq

    return all_ids, occ_map


def _postprocess_event_rows(rows: List[Dict[str, str]], cfg: Cfg) -> List[Dict[str, str]]:
    """
    1) If the event has ANY single-day rows (start_ymd == end_ymd),
       drop all multi-day rows for that event.

    2) Collapse duplicates for the same date(s):
       keep the row that has times (prefer timeful over timeless).
    """
    if not rows:
        return rows

    def has_time(r: Dict[str, str]) -> bool:
        return bool((r.get("start_time") or "").strip() or (r.get("end_time") or "").strip())

    # 1) drop range rows when daily rows exist
    daily_exists = any((r.get("_start_ymd") or "") and (r.get("_start_ymd") == r.get("_end_ymd")) for r in rows)
    if daily_exists:
        before = len(rows)
        rows = [r for r in rows if (r.get("_start_ymd") == r.get("_end_ymd"))]
        _debug(cfg, f"[clean] dropped range rows: {before} -> {len(rows)}")

    # 2) collapse timeless duplicates, preferring timeful
    best: Dict[Tuple[str, str, str, str, str], Dict[str, str]] = {}
    for r in rows:
        key = (
            (r.get("url") or "").strip().lower(),
            (r.get("facility") or "").strip().lower(),
            (r.get("_start_ymd") or "").strip(),
            (r.get("_end_ymd") or "").strip(),
            (r.get("telehealth") or "").strip(),
        )
        cur = best.get(key)
        if cur is None:
            best[key] = r
            continue
        if has_time(r) and not has_time(cur):
            best[key] = r
            continue
        if has_time(r) == has_time(cur):
            score_r = sum(1 for k in ("parking_date", "parking_time") if (r.get(k) or "").strip())
            score_c = sum(1 for k in ("parking_date", "parking_time") if (cur.get(k) or "").strip())
            if score_r > score_c:
                best[key] = r

    out = list(best.values())
    if cfg.debug:
        _debug(cfg, f"[clean] collapsed duplicates by date: {len(rows)} -> {len(out)}")
    return out


def _build_rows_for_event(cfg: Cfg, detail: Dict[str, Any], occ_hints: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    data = detail.get("data", {})
    if not isinstance(data, dict):
        data = {}

    loc = _extract_location(data)
    fields = _extract_custom_fields(data, whole_detail=detail)

    tax_text = _extract_taxonomy_text(data)
    svc = _services_flags(fields.get("services", ""), tax_text)
    ctype = _clinic_type_flags(fields.get("clinic_type", ""), tax_text)

    canceled = _extract_canceled(data)
    parking_date = fields.get("parking_date", "").strip()
    parking_time = fields.get("parking_time", "").strip()

    event_title = str(data.get("title") or "").strip()
    url = str(data.get("permalink") or "").strip()

    # Determine clinic type (what your UI calls these)
    is_telehealth = bool(ctype["telehealth"])
    clinic_type_label = "Telehealth Clinic" if is_telehealth else "Popup Clinic"

    # What should appear as the “title” in the UI?
    # Prefer facility name, otherwise fall back to MEC event title
    display_title = loc["facility"].strip() or event_title

    # Occurrences:
    occs = occ_hints if occ_hints else _extract_occurrences_from_detail(detail, max_span_days=cfg.max_span_days)

    if cfg.debug:
        if fields.get("services"):
            _debug(cfg, f"[svc] services_field='{fields.get('services')}' title='{event_title}'")
        else:
            _debug(cfg, f"[svc] services_field=MISSING title='{event_title}' (taxonomy fallback: '{tax_text}')")
        if not occs:
            eid = data.get("ID") or data.get("id")
            _debug(cfg, f"[warn] No occurrences parsed for event {eid} title={event_title!r}")

    if not occs:
        occs = [{"start": {"date": ""}, "end": {"date": ""}, "allday": "1", "hide_time": "1"}]

    rows: List[Dict[str, str]] = []
    for occ in occs:
        # Force popup clinics to be "all day" (blank times) if POPUP_ALLDAY=1
        force_all_day = (cfg.popup_allday and not is_telehealth)
        dt = _occ_to_row_dates_times(occ, cfg, force_all_day=force_all_day)

        row: Dict[str, str] = {h: "" for h in DEFAULT_HEADER}
        row["canceled"] = _yn(canceled)
        row["lat"] = loc["lat"]
        row["lng"] = loc["lng"]
        row["address"] = loc["address"]
        row["city"] = loc["city"]
        row["state"] = loc["state"]
        row["site"] = _safe_site(loc["city"], loc["state"])

        row["title"] = display_title
        row["facility"] = loc["facility"].strip() or display_title

        row["clinic_type"] = clinic_type_label
        row["telehealth"] = _yn(is_telehealth)

        row["start_time"] = dt["start_time"]
        row["end_time"] = dt["end_time"]
        row["start_date"] = dt["start_date"]
        row["end_date"] = dt["end_date"]

        row["parking_date"] = parking_date
        row["parking_time"] = parking_time

        row["medical"] = _yn(svc["medical"])
        row["dental"] = _yn(svc["dental"])
        row["vision"] = _yn(svc["vision"])
        row["dentures"] = _yn(svc["dentures"])
        row["url"] = url

        # internal fields used for cleanup/dedupe
        row["_start_ymd"] = dt["_start_ymd"]
        row["_end_ymd"] = dt["_end_ymd"]
        row["_title"] = event_title
        row["_is_all_day"] = dt["_is_all_day"]

        rows.append(row)

    rows = _postprocess_event_rows(rows, cfg)
    return rows


def _dedupe_rows_global(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    """Final safety net across all events."""
    seen: Set[Tuple[str, str, str, str, str, str, str]] = set()
    out: List[Dict[str, str]] = []
    for r in rows:
        key = (
            (r.get("url") or "").strip().lower(),
            (r.get("facility") or "").strip().lower(),
            (r.get("_start_ymd") or r.get("start_date") or "").strip(),
            (r.get("_end_ymd") or r.get("end_date") or "").strip(),
            (r.get("start_time") or "").strip().lower(),
            (r.get("end_time") or "").strip().lower(),
            (r.get("telehealth") or "").strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _write_csv(cfg: Cfg, rows: Sequence[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(cfg.csv_path) or ".", exist_ok=True)
    with open(cfg.csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=DEFAULT_HEADER,
            extrasaction="ignore",
            delimiter=cfg.delimiter,
            quoting=csv.QUOTE_MINIMAL,
        )
        w.writeheader()
        w.writerows(rows)


def _load_cfg() -> Cfg:
    base_url = os.environ.get("MEC_BASE_URL", "").strip()
    token = os.environ.get("MEC_TOKEN", "").strip()
    csv_path = os.environ.get("CSV_PATH", "").strip()

    start = os.environ.get("MEC_START", "2010-01-01").strip()
    end = os.environ.get("MEC_END", "2030-12-31").strip()
    limit = int((os.environ.get("MEC_LIMIT", "200") or "200").strip())
    max_pages = int((os.environ.get("MAX_PAGES", "5000") or "5000").strip())
    max_span_days = int((os.environ.get("MAX_SPAN_DAYS", "10") or "10").strip())

    debug = (os.environ.get("MEC_DEBUG", "") or "").strip().lower() in {"1", "true", "yes", "y"}
    popup_allday = (os.environ.get("POPUP_ALLDAY", "1") or "1").strip().lower() in {"1", "true", "yes", "y"}

    delimiter = _get_delimiter()
    date_format = (os.environ.get("DATE_FORMAT", "mdy") or "mdy").strip().lower()
    if date_format not in {"mdy", "ymd"}:
        date_format = "mdy"

    include_past = (os.environ.get("INCLUDE_PAST", "1") or "1").strip().lower() in {"1", "true", "yes", "y"}
    include_ongoing = (os.environ.get("INCLUDE_ONGOING", "1") or "1").strip().lower() in {"1", "true", "yes", "y"}

    if not base_url or not token or not csv_path:
        raise SystemExit("Missing env vars. Need MEC_BASE_URL, MEC_TOKEN, CSV_PATH.")

    if _parse_ymd(start) is None or _parse_ymd(end) is None:
        raise SystemExit("MEC_START and MEC_END must be YYYY-MM-DD")

    return Cfg(
        base_url=base_url,
        token=token,
        csv_path=csv_path,
        start=start,
        end=end,
        limit=limit,
        max_pages=max_pages,
        max_span_days=max_span_days,
        delimiter=delimiter,
        date_format=date_format,
        include_past=include_past,
        include_ongoing=include_ongoing,
        popup_allday=popup_allday,
        debug=debug,
    )


def main() -> None:
    cfg = _load_cfg()
    delimiter_label = "TAB" if cfg.delimiter == "\t" else "COMMA"

    print(f"[cfg] base_url={cfg.base_url}")
    print(f"[cfg] start={cfg.start} end={cfg.end} limit={cfg.limit} max_pages={cfg.max_pages} max_span_days={cfg.max_span_days}")
    print(f"[cfg] include_past={int(cfg.include_past)} include_ongoing={int(cfg.include_ongoing)} popup_allday={int(cfg.popup_allday)}")
    print(f"[cfg] csv_path={cfg.csv_path} delimiter={delimiter_label} date_format={cfg.date_format}")

    sess = _session()

    event_ids, occ_map = _iter_all_occurrences(cfg, sess)
    _debug(cfg, f"[mec] Total unique event IDs: {len(event_ids)} (with occurrence hints for {len(occ_map)})")

    all_rows: List[Dict[str, str]] = []
    skipped_404 = 0

    for eid in sorted(event_ids):
        detail = _fetch_event_detail(cfg, sess, eid)
        if detail is None:
            skipped_404 += 1
            continue
        rows = _build_rows_for_event(cfg, detail, occ_hints=occ_map.get(eid))
        all_rows.extend(rows)

    before = len(all_rows)
    all_rows = _dedupe_rows_global(all_rows)
    after = len(all_rows)

    # strip internal fields before writing
    for r in all_rows:
        r.pop("_start_ymd", None)
        r.pop("_end_ymd", None)
        r.pop("_title", None)
        r.pop("_is_all_day", None)

    _write_csv(cfg, all_rows)
    print(f"Wrote {len(all_rows)} rows to {cfg.csv_path} (skipped_404={skipped_404}, deduped={before - after})")


if __name__ == "__main__":
    main()
