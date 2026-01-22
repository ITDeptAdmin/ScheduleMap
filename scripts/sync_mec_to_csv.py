#!/usr/bin/env python3
# scripts/sync_mec_to_csv.py
"""
Sync Modern Events Calendar (MEC) events to a GitHub-tracked CSV.

Fixes (stable for live site):
- Retries on HTTP 202/429/5xx (prevents random Action failures).
- Services / Clinic Type / Parking fields are extracted via:
    1) data["fields"] when present
    2) deep-scan anywhere in the event JSON when missing
  (MEC sometimes omits data["fields"] even when filled in WP).
- Recurrence robustness:
    - Extract occurrences from event detail (start/end blocks)
    - ALSO collect occurrence "hints" from /events list endpoint by date windows
      and merge them (fixes "missing 1-3 dates" in repeating telehealth series).
- Writes comma CSV by default (your schedule JS requires commas).

Required env vars:
- MEC_BASE_URL
- MEC_TOKEN
- CSV_PATH

Optional env vars:
- MEC_START (default 2010-01-01)
- MEC_END (default 2030-12-31)
- MEC_LIMIT (default 200)
- MAX_PAGES (default 50)
- MAX_SPAN_DAYS (default 10)  # skip multi-day blocks longer than this in detail parsing
- MEC_WINDOW_DAYS (default 365)  # how big each /events list window is
- CSV_DELIMITER: "comma" or "tab" (default comma)
- DATE_FORMAT: "mdy" or "ymd" (default mdy)
- MEC_DEBUG: "1"
"""

from __future__ import annotations

import csv
import os
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
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

DEFAULT_HEADER: List[str] = [
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


def _truthy(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on", "checked"}


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
    window_days: int
    delimiter: str
    date_format: str
    debug: bool


def _debug(cfg: Cfg, msg: str) -> None:
    if cfg.debug:
        print(msg)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "ram-mec-sync/1.4",
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
    Retries on 202/429/5xx.
    """
    url = cfg.base_url.rstrip("/") + "/" + path.lstrip("/")
    headers = {"mec-token": cfg.token}

    for attempt in range(retries + 1):
        resp = sess.get(url, headers=headers, params=params, timeout=60)
        _debug(cfg, f"[mec] GET {resp.url} -> {resp.status_code}")

        if allow_404 and resp.status_code == 404:
            return resp

        if resp.status_code in {202, 429, 500, 502, 503, 504} and attempt < retries:
            backoff = min(30, 2**attempt)
            _debug(cfg, f"[mec] retryable {resp.status_code}; body='{_resp_snippet(resp)}'; sleeping {backoff}s")
            time.sleep(backoff)
            continue

        return resp

    return resp


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


def _extract_event_id(item: Dict[str, Any]) -> Optional[int]:
    for k in ("ID", "id", "post_id", "event_id"):
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


def _coerce_field_value_to_text(field: Dict[str, Any]) -> str:
    """
    MEC checkbox fields can return value as:
      - list of labels
      - dict of label->0/1
      - string of ids/indexes like "1,3"
    Normalize to a string of labels.
    """
    value = field.get("value")

    if isinstance(value, (list, tuple, set)):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return " ".join(parts).strip()

    if isinstance(value, dict):
        selected = [str(k).strip() for k, v in value.items() if _truthy(v) and str(k).strip()]
        if selected:
            return " ".join(selected).strip()
        vals = [str(v).strip() for v in value.values() if str(v).strip()]
        return " ".join(vals).strip()

    s = "" if value is None else str(value).strip()
    if not s:
        return ""

    options = field.get("options") or field.get("option") or field.get("values") or field.get("items")
    opts_list: List[str] = []
    if isinstance(options, list):
        for opt in options:
            if isinstance(opt, dict):
                label = (opt.get("label") or opt.get("title") or opt.get("name") or opt.get("value") or "").strip()
                if label:
                    opts_list.append(label)
            else:
                label = str(opt).strip()
                if label:
                    opts_list.append(label)

    if opts_list and re.fullmatch(r"[\d,\s]+", s):
        idxs: List[int] = []
        for tok in re.split(r"[,\s]+", s):
            if not tok:
                continue
            try:
                idxs.append(int(tok))
            except Exception:
                pass

        mapped: List[str] = []
        for i in idxs:
            if 1 <= i <= len(opts_list):
                mapped.append(opts_list[i - 1])
            elif 0 <= i < len(opts_list):
                mapped.append(opts_list[i])

        if mapped:
            return " ".join(mapped).strip()

    return s


def _find_field_dicts_anywhere(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    MEC sometimes nests custom fields outside data["fields"].
    We scan any dict that looks like {label: ..., value: ...}.
    """
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if "label" in cur and "value" in cur and isinstance(cur.get("label"), (str, int, float)):
                yield cur
            for v in cur.values():
                if isinstance(v, (dict, list)):
                    stack.append(v)
        elif isinstance(cur, list):
            for it in cur:
                if isinstance(it, (dict, list)):
                    stack.append(it)


def _extract_custom_fields(detail: Dict[str, Any]) -> Dict[str, str]:
    """
    Preferred: detail["data"]["fields"]
    Fallback: deep-scan whole detail JSON for any field dicts.
    """
    out: Dict[str, str] = {}

    data = detail.get("data", {})
    if isinstance(data, dict):
        fields = data.get("fields", [])
        if isinstance(fields, list):
            for f in fields:
                if not isinstance(f, dict):
                    continue
                label = _norm(str(f.get("label", "")))
                value_text = _coerce_field_value_to_text(f)
                if label == "services":
                    out["services"] = value_text
                elif label in {"clinic type", "clinictype"}:
                    out["clinic_type"] = value_text
                elif label in {"parking_date", "parking date"}:
                    out["parking_date"] = value_text
                elif label in {"parking_time", "parking time"} or label.startswith("parking_time"):
                    out["parking_time"] = value_text

    if out.get("services") and out.get("clinic_type") and ("parking_date" in out or "parking_time" in out):
        return out

    for f in _find_field_dicts_anywhere(detail):
        label = _norm(str(f.get("label", "")))
        if label not in {"services", "clinic type", "clinictype", "parking_date", "parking date", "parking_time", "parking time"}:
            continue

        value_text = _coerce_field_value_to_text(f)
        if label == "services" and not out.get("services"):
            out["services"] = value_text
        elif label in {"clinic type", "clinictype"} and not out.get("clinic_type"):
            out["clinic_type"] = value_text
        elif label in {"parking_date", "parking date"} and "parking_date" not in out:
            out["parking_date"] = value_text
        elif (label in {"parking_time", "parking time"} or label.startswith("parking_time")) and "parking_time" not in out:
            out["parking_time"] = value_text

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
            if s and len(s) <= 80:
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


def _services_flags(services_value: str, safe_fallback_text: str) -> Dict[str, bool]:
    src = (services_value or "").strip() or (safe_fallback_text or "").strip()
    s = src.lower()

    def has_word(pattern: str) -> bool:
        return re.search(pattern, s) is not None

    return {
        "medical": has_word(r"\bmedical\b"),
        "dental": has_word(r"\bdental\b"),
        "vision": has_word(r"\bvision\b"),
        "dentures": has_word(r"\bdenture(s)?\b"),
    }


def _clinic_type_flags(clinic_type_value: str, safe_fallback_text: str) -> Dict[str, bool]:
    src = (clinic_type_value or "").strip() or (safe_fallback_text or "").strip()
    s = src.lower()
    # treat "popup" as NOT telehealth, "telehealth" as telehealth
    return {"telehealth": "telehealth" in s and "popup" not in s}


def _extract_canceled(event_data: Dict[str, Any]) -> bool:
    meta = event_data.get("meta", {})
    status = ""
    if isinstance(meta, dict):
        status = str(meta.get("mec_event_status") or meta.get("event_status") or "").strip()
    if not status:
        status = str(event_data.get("event_status") or "").strip()
    return "cancel" in status.lower()


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


def _extract_occurrences_from_detail(detail: Dict[str, Any], max_span_days: int, date_format: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []

    for b in _find_occurrence_blocks_anywhere(detail):
        s_obj = b.get("start") or {}
        e_obj = b.get("end") or {}

        s_date = str(s_obj.get("date") or "").strip()
        e_date = str(e_obj.get("date") or s_date or "").strip()
        if not s_date:
            continue

        allday = str(b.get("allday") or "0").strip().lower() in {"1", "true", "yes"}
        hide_time = str(b.get("hide_time") or "0").strip().lower() in {"1", "true", "yes"}

        start_time = "" if (allday or hide_time) else _format_time(s_obj.get("hour"), s_obj.get("minutes"), s_obj.get("ampm"))
        end_time = "" if (allday or hide_time) else _format_time(e_obj.get("hour"), e_obj.get("minutes"), e_obj.get("ampm"))

        sd = _parse_ymd(s_date)
        ed = _parse_ymd(e_date)
        if sd and ed and (ed - sd).days > max_span_days:
            continue

        out.append(
            {
                "start_date": _format_date(s_date, date_format),
                "end_date": _format_date(e_date, date_format),
                "start_time": start_time,
                "end_time": end_time,
                "_raw_start": s_date,
                "_raw_end": e_date,
            }
        )

    seen: Set[Tuple[str, str, str, str]] = set()
    uniq: List[Dict[str, str]] = []
    for o in out:
        k = (o["start_date"], o["end_date"], o["start_time"], o["end_time"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(o)

    return uniq


def _safe_site(city: str, state: str) -> str:
    parts = [p for p in [city.strip(), state.strip()] if p]
    return ", ".join(parts)


def _extract_occurrence_hint_from_list_item(item: Dict[str, Any], cfg: Cfg) -> Optional[Tuple[str, str, str, str]]:
    """
    Best-effort: derive (raw_start_ymd, raw_end_ymd, start_time, end_time) from /events list item.
    We only need dates; times are optional (empty ok).
    """
    def pick_ymd(v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
            return s
        return None

    raw_start = None
    raw_end = None

    for key in ("start_date", "date", "start"):
        if key in item:
            v = item.get(key)
            if isinstance(v, dict):
                raw_start = pick_ymd(v.get("date")) or raw_start
            else:
                raw_start = pick_ymd(v) or raw_start

    for key in ("end_date", "end"):
        if key in item:
            v = item.get(key)
            if isinstance(v, dict):
                raw_end = pick_ymd(v.get("date")) or raw_end
            else:
                raw_end = pick_ymd(v) or raw_end

    if raw_start and not raw_end:
        raw_end = raw_start

    if not raw_start:
        return None

    start_time = str(item.get("start_time") or "").strip()
    end_time = str(item.get("end_time") or item.get("stop_time") or "").strip()
    return (raw_start, raw_end or raw_start, start_time, end_time)


def _collect_event_ids_and_occurrence_hints(cfg: Cfg, sess: requests.Session) -> Tuple[List[int], Dict[int, Set[Tuple[str, str, str, str]]]]:
    """
    Pull IDs via date-windowed /events calls and collect occurrence hints per event.
    This prevents:
      - /events returning only a subset (like 46 IDs)
      - missing repeat dates that the detail endpoint doesn't list
    """
    start_d = _parse_ymd(cfg.start)
    end_d = _parse_ymd(cfg.end)
    if not start_d or not end_d:
        raise SystemExit("MEC_START/MEC_END must be YYYY-MM-DD")

    occ_hints: Dict[int, Set[Tuple[str, str, str, str]]] = {}
    all_ids: Set[int] = set()

    window_days = max(7, min(366, cfg.window_days))
    cur = start_d

    def fetch_page(params: Dict[str, str]) -> List[Dict[str, Any]]:
        resp = _mec_get(cfg, sess, "events", params=params)
        if resp.status_code != 200:
            raise SystemExit(f"MEC /events failed: HTTP {resp.status_code} url={resp.url} body='{_resp_snippet(resp)}'")
        return _flatten_events_payload(resp.json())

    while cur <= end_d:
        win_end = min(end_d, cur + timedelta(days=window_days - 1))
        win_start_s = cur.isoformat()
        win_end_s = win_end.isoformat()

        base_params = {"start": win_start_s, "end": win_end_s, "limit": str(cfg.limit)}

        # detect paging key inside the window
        items_p1 = fetch_page(dict(base_params))
        items_p2 = fetch_page({**base_params, "page": "2"}) if cfg.max_pages >= 2 else []
        uses_page = bool({i for i in [_extract_event_id(x) for x in items_p2] if i} - {i for i in [_extract_event_id(x) for x in items_p1] if i})

        page_key = "page"
        if not uses_page:
            items_pd1 = fetch_page(dict(base_params))
            items_pd2 = fetch_page({**base_params, "paged": "2"}) if cfg.max_pages >= 2 else []
            uses_paged = bool({i for i in [_extract_event_id(x) for x in items_pd2] if i} - {i for i in [_extract_event_id(x) for x in items_pd1] if i})
            page_key = "paged" if uses_paged else "page"

        _debug(cfg, f"[mec] window {win_start_s}..{win_end_s} paging={page_key}")

        seen_in_window: Set[int] = set()
        for page in range(1, cfg.max_pages + 1):
            params = dict(base_params)
            if page > 1:
                params[page_key] = str(page)

            items = fetch_page(params)
            if not items and page == 1:
                break

            ids_this_page: List[int] = []
            for it in items:
                eid = _extract_event_id(it)
                if eid is None:
                    continue
                ids_this_page.append(eid)
                all_ids.add(eid)
                seen_in_window.add(eid)

                hint = _extract_occurrence_hint_from_list_item(it, cfg)
                if hint:
                    occ_hints.setdefault(eid, set()).add(hint)

            _debug(cfg, f"[mec] window page={page} -> items={len(items)} ids={len(set(ids_this_page))} unique_total={len(all_ids)}")

            # stop rules
            if len(items) < cfg.limit:
                break

        cur = win_end + timedelta(days=1)

    return sorted(all_ids), occ_hints


def _fetch_event_detail(cfg: Cfg, sess: requests.Session, eid: int) -> Optional[Dict[str, Any]]:
    params = {"start": cfg.start, "end": cfg.end, "limit": str(cfg.limit)}

    resp = _mec_get(cfg, sess, f"events/{eid}", params=params, allow_404=True)
    if resp.status_code == 404:
        _debug(cfg, f"[warn] /events/{eid} with params -> 404, retrying without params")
        resp2 = _mec_get(cfg, sess, f"events/{eid}", params=None, allow_404=True)
        if resp2.status_code == 404:
            _debug(cfg, f"[warn] /events/{eid} -> 404 (skipping)")
            return None
        if resp2.status_code != 200:
            raise SystemExit(f"MEC /events/{eid} failed: HTTP {resp2.status_code} url={resp2.url} body='{_resp_snippet(resp2)}'")
        return resp2.json()

    if resp.status_code != 200:
        raise SystemExit(f"MEC /events/{eid} failed: HTTP {resp.status_code} url={resp.url} body='{_resp_snippet(resp)}'")
    return resp.json()


def _merge_occurrences(
    cfg: Cfg,
    occ_from_detail: List[Dict[str, str]],
    occ_hints: Set[Tuple[str, str, str, str]],
) -> List[Dict[str, str]]:
    """
    Merge /events-list hints (raw ymd) into detail-derived occurrences.
    If hint date isn't present in detail occurrences, add it.
    Times from hint are used only if detail time missing.
    """
    existing_raw: Set[Tuple[str, str]] = set()
    for o in occ_from_detail:
        rs = o.get("_raw_start") or ""
        re_ = o.get("_raw_end") or ""
        if rs:
            existing_raw.add((rs, re_ or rs))

    merged = list(occ_from_detail)

    # Choose a default time from first detail occurrence (if any)
    default_start_time = ""
    default_end_time = ""
    for o in occ_from_detail:
        if o.get("start_time") or o.get("end_time"):
            default_start_time = o.get("start_time") or ""
            default_end_time = o.get("end_time") or ""
            break

    for (raw_s, raw_e, st, et) in sorted(occ_hints):
        if (raw_s, raw_e) in existing_raw:
            continue
        merged.append(
            {
                "start_date": _format_date(raw_s, cfg.date_format),
                "end_date": _format_date(raw_e, cfg.date_format),
                "start_time": st or default_start_time,
                "end_time": et or default_end_time,
                "_raw_start": raw_s,
                "_raw_end": raw_e,
            }
        )

    # de-dupe final by formatted fields
    seen: Set[Tuple[str, str, str, str]] = set()
    out: List[Dict[str, str]] = []
    for o in merged:
        k = (o.get("start_date", ""), o.get("end_date", ""), o.get("start_time", ""), o.get("end_time", ""))
        if k in seen:
            continue
        seen.add(k)
        out.append(o)

    # sort by raw start date if present
    def sort_key(o: Dict[str, str]) -> str:
        return o.get("_raw_start") or o.get("start_date") or ""

    out.sort(key=sort_key)
    return out


def _build_rows_for_event(cfg: Cfg, detail: Dict[str, Any], hints: Set[Tuple[str, str, str, str]]) -> List[Dict[str, str]]:
    data = detail.get("data", {})
    if not isinstance(data, dict):
        data = {}

    loc = _extract_location(data)
    fields = _extract_custom_fields(detail)

    safe_tax_text = _extract_taxonomy_text(data)
    svc = _services_flags(fields.get("services", ""), safe_tax_text)
    ctype = _clinic_type_flags(fields.get("clinic_type", ""), safe_tax_text)

    canceled = _extract_canceled(data)
    parking_date = (fields.get("parking_date") or "").strip()
    parking_time = (fields.get("parking_time") or "").strip()

    title = str(data.get("title") or "").strip()
    url = str(data.get("permalink") or "").strip()

    occ_detail = _extract_occurrences_from_detail(detail, max_span_days=cfg.max_span_days, date_format=cfg.date_format)
    if cfg.debug and fields.get("services"):
        _debug(cfg, f"[svc] services_field='{fields.get('services')}' title='{title}'")
    if cfg.debug and not fields.get("services"):
        _debug(cfg, f"[svc] services_field=MISSING title='{title}' (will rely on deep-scan + taxonomy fallback)")

    occurrences = _merge_occurrences(cfg, occ_detail, hints)

    if cfg.debug and not occurrences:
        eid = data.get("ID") or data.get("id")
        _debug(cfg, f"[warn] No occurrences parsed for event {eid} title={title!r}")

    if not occurrences:
        occurrences = [{"start_date": "", "end_date": "", "start_time": "", "end_time": ""}]

    rows: List[Dict[str, str]] = []
    for occ in occurrences:
        row: Dict[str, str] = {h: "" for h in DEFAULT_HEADER}
        row["canceled"] = _yn(canceled)
        row["lat"] = loc["lat"]
        row["lng"] = loc["lng"]
        row["address"] = loc["address"]
        row["city"] = loc["city"]
        row["state"] = loc["state"]
        row["site"] = _safe_site(loc["city"], loc["state"])
        row["facility"] = loc["facility"]
        row["telehealth"] = _yn(ctype["telehealth"])
        row["start_time"] = occ.get("start_time", "")
        row["end_time"] = occ.get("end_time", "")
        row["start_date"] = occ.get("start_date", "")
        row["end_date"] = occ.get("end_date", "")
        row["parking_date"] = parking_date
        row["parking_time"] = parking_time
        row["medical"] = _yn(svc["medical"])
        row["dental"] = _yn(svc["dental"])
        row["vision"] = _yn(svc["vision"])
        row["dentures"] = _yn(svc["dentures"])
        row["url"] = url
        rows.append(row)

    return rows


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
    max_pages = int((os.environ.get("MAX_PAGES", "50") or "50").strip())
    max_span_days = int((os.environ.get("MAX_SPAN_DAYS", "10") or "10").strip())
    window_days = int((os.environ.get("MEC_WINDOW_DAYS", "365") or "365").strip())

    debug = (os.environ.get("MEC_DEBUG", "") or "").strip().lower() in {"1", "true", "yes", "y"}

    delimiter = _get_delimiter()
    date_format = (os.environ.get("DATE_FORMAT", "mdy") or "mdy").strip().lower()
    if date_format not in {"mdy", "ymd"}:
        date_format = "mdy"

    if not base_url or not token or not csv_path:
        raise SystemExit("Missing env vars. Need MEC_BASE_URL, MEC_TOKEN, CSV_PATH.")

    return Cfg(
        base_url=base_url,
        token=token,
        csv_path=csv_path,
        start=start,
        end=end,
        limit=limit,
        max_pages=max_pages,
        max_span_days=max_span_days,
        window_days=window_days,
        delimiter=delimiter,
        date_format=date_format,
        debug=debug,
    )


def main() -> None:
    cfg = _load_cfg()
    delimiter_label = "TAB" if cfg.delimiter == "\t" else "COMMA"

    print(f"[cfg] base_url={cfg.base_url}")
    print(f"[cfg] start={cfg.start} end={cfg.end} limit={cfg.limit} max_pages={cfg.max_pages} max_span_days={cfg.max_span_days} window_days={cfg.window_days}")
    print(f"[cfg] csv_path={cfg.csv_path} delimiter={delimiter_label} date_format={cfg.date_format}")

    if cfg.delimiter != ",":
        print("[warn] Your schedule JS requires commas (`txt.includes(',')`). Prefer CSV_DELIMITER=comma.")

    sess = _session()

    # Windowed list scan: fixes "only 46 IDs" and captures repeating occurrence hints
    event_ids, occ_hints_map = _collect_event_ids_and_occurrence_hints(cfg, sess)
    _debug(cfg, f"[mec] Total unique event IDs: {len(event_ids)} (with occurrence hints for {len(occ_hints_map)})")

    all_rows: List[Dict[str, str]] = []
    skipped_404 = 0

    for eid in event_ids:
        detail = _fetch_event_detail(cfg, sess, eid)
        if detail is None:
            skipped_404 += 1
            continue
        hints = occ_hints_map.get(eid, set())
        all_rows.extend(_build_rows_for_event(cfg, detail, hints))

    _write_csv(cfg, all_rows)
    print(f"Wrote {len(all_rows)} rows to {cfg.csv_path} (skipped_404={skipped_404})")


if __name__ == "__main__":
    main()
