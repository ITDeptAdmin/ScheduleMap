#!/usr/bin/env python3
# scripts/sync_mec_to_csv.py
"""
Sync Modern Events Calendar (MEC) events to a GitHub-tracked CSV.

Key behavior:
- Writes comma CSV by default (your schedule JS requires commas).
- Expands occurrences into rows.
- Skips 404s gracefully.
- Services (medical/dental/vision/dentures) are derived ONLY from:
  1) MEC custom field "Services" (preferred)
  2) structured taxonomies (categories/tags/labels) as fallback
  NOT from full HTML body text (avoids false positives from boilerplate).

Required env vars:
- MEC_BASE_URL
- MEC_TOKEN
- CSV_PATH

Optional env vars:
- MEC_START (default 2010-01-01)
- MEC_END (default 2030-12-31)
- MEC_LIMIT (default 200)
- MAX_PAGES (default 50)
- MAX_SPAN_DAYS (default 10)
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
    debug: bool


def _debug(cfg: Cfg, msg: str) -> None:
    if cfg.debug:
        print(msg)


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "ram-mec-sync/1.1"})
    return s


def _mec_get(
    cfg: Cfg,
    sess: requests.Session,
    path: str,
    params: Optional[Dict[str, str]] = None,
    *,
    allow_404: bool = False,
    retries: int = 4,
) -> requests.Response:
    url = cfg.base_url.rstrip("/") + "/" + path.lstrip("/")
    headers = {"mec-token": cfg.token}

    for attempt in range(retries + 1):
        try:
            resp = sess.get(url, headers=headers, params=params, timeout=60)
            _debug(cfg, f"[mec] GET {resp.url} -> {resp.status_code}")

            if allow_404 and resp.status_code == 404:
                return resp

            if resp.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                backoff = 2 ** attempt
                _debug(cfg, f"[mec] retryable {resp.status_code}; sleeping {backoff}s")
                time.sleep(backoff)
                continue

            return resp
        except Exception as e:
            if attempt < retries:
                backoff = 2 ** attempt
                _debug(cfg, f"[mec] exception {type(e).__name__}: {e}; sleeping {backoff}s")
                time.sleep(backoff)
                continue
            raise
    raise RuntimeError("unreachable")


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


def _extract_event_ids(payload: Any) -> List[int]:
    items = _flatten_events_payload(payload)
    ids: List[int] = []
    for it in items:
        eid = _extract_event_id(it)
        if eid is not None:
            ids.append(eid)
    return sorted(set(ids))


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


def _extract_taxonomy_text(event_data: Dict[str, Any]) -> str:
    """
    Safely extracts names from structured taxonomies (categories/tags/labels).
    Avoids scanning full HTML/event description which often contains boilerplate.
    """
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

    # Also include the title (often contains "Dental", "Vision", etc.)
    title = str(event_data.get("title") or "").strip()
    if title:
        chunks.append(title)

    txt = " ".join(chunks)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt


def _services_flags(services_value: str, safe_fallback_text: str) -> Dict[str, bool]:
    """
    IMPORTANT:
    - We do NOT use HTML body text because it can contain generic lists of services.
    - We only use custom field 'Services' or structured taxonomies (categories/tags/labels) as fallback.
    """
    src = (services_value or "").strip()
    if not src:
        src = (safe_fallback_text or "").strip()

    def has(w: str) -> bool:
        return w.lower() in src.lower()

    return {
        "medical": has("medical"),
        "dental": has("dental"),
        "vision": has("vision"),
        "dentures": has("denture"),
    }


def _clinic_type_flags(clinic_type_value: str, safe_fallback_text: str) -> Dict[str, bool]:
    src = (clinic_type_value or "").strip() or (safe_fallback_text or "").strip()

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


def _extract_occurrences(detail: Dict[str, Any], max_span_days: int, date_format: str) -> List[Dict[str, str]]:
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


def _build_rows_for_event(cfg: Cfg, detail: Dict[str, Any]) -> List[Dict[str, str]]:
    data = detail.get("data", {})
    if not isinstance(data, dict):
        data = {}

    loc = _extract_location(data)
    fields = _extract_custom_fields(data)

    safe_tax_text = _extract_taxonomy_text(data)
    svc = _services_flags(fields.get("services", ""), safe_tax_text)
    ctype = _clinic_type_flags(fields.get("clinic_type", ""), safe_tax_text)

    canceled = _extract_canceled(data)
    parking_date = fields.get("parking_date", "").strip()
    parking_time = fields.get("parking_time", "").strip()

    title = str(data.get("title") or "").strip()
    url = str(data.get("permalink") or "").strip()

    occurrences = _extract_occurrences(detail, max_span_days=cfg.max_span_days, date_format=cfg.date_format)
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
        row["start_time"] = occ["start_time"]
        row["end_time"] = occ["end_time"]
        row["start_date"] = occ["start_date"]
        row["end_date"] = occ["end_date"]
        row["parking_date"] = parking_date
        row["parking_time"] = parking_time
        row["medical"] = _yn(svc["medical"])
        row["dental"] = _yn(svc["dental"])
        row["vision"] = _yn(svc["vision"])
        row["dentures"] = _yn(svc["dentures"])
        row["url"] = url

        rows.append(row)

    return rows


def _iter_event_ids(cfg: Cfg, sess: requests.Session) -> List[int]:
    base_params = {"start": cfg.start, "end": cfg.end, "limit": str(cfg.limit)}

    def fetch_page(page_param_name: str, page: int) -> List[int]:
        params = dict(base_params)
        if page > 1:
            params[page_param_name] = str(page)
        resp = _mec_get(cfg, sess, "events", params=params)
        if resp.status_code != 200:
            raise SystemExit(f"MEC /events failed: HTTP {resp.status_code} url={resp.url}")
        return _extract_event_ids(resp.json())

    ids_page1 = fetch_page("page", 1)
    ids_page2 = fetch_page("page", 2) if cfg.max_pages >= 2 else []
    uses_page = bool(set(ids_page2) - set(ids_page1))

    if not uses_page:
        ids_paged1 = fetch_page("paged", 1)
        ids_paged2 = fetch_page("paged", 2) if cfg.max_pages >= 2 else []
        uses_paged = bool(set(ids_paged2) - set(ids_paged1))
        page_key = "paged" if uses_paged else "page"
        _debug(cfg, f"[mec] paging mode detected: {page_key}")
    else:
        page_key = "page"
        _debug(cfg, "[mec] paging mode detected: page")

    all_ids: Set[int] = set()
    for page in range(1, cfg.max_pages + 1):
        params = dict(base_params)
        if page > 1:
            params[page_key] = str(page)

        resp = _mec_get(cfg, sess, "events", params=params)
        if resp.status_code != 200:
            raise SystemExit(f"MEC /events failed: HTTP {resp.status_code} url={resp.url}")

        ids = _extract_event_ids(resp.json())
        before = len(all_ids)
        all_ids.update(ids)

        _debug(cfg, f"[mec] {page_key}={page} -> got {len(ids)} ids (unique now {len(all_ids)})")

        if page > 1 and len(all_ids) == before:
            break
        if len(ids) < cfg.limit:
            break

    out = sorted(all_ids)
    if not out:
        raise SystemExit("No events returned from MEC /events endpoint. Check token + endpoint + date params.")
    return out


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
            raise SystemExit(f"MEC /events/{eid} failed: HTTP {resp2.status_code} url={resp2.url}")
        return resp2.json()

    if resp.status_code != 200:
        raise SystemExit(f"MEC /events/{eid} failed: HTTP {resp.status_code} url={resp.url}")
    return resp.json()


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
        delimiter=delimiter,
        date_format=date_format,
        debug=debug,
    )


def main() -> None:
    cfg = _load_cfg()

    delimiter_label = "TAB" if cfg.delimiter == "\t" else "COMMA"
    print(f"[cfg] base_url={cfg.base_url}")
    print(f"[cfg] start={cfg.start} end={cfg.end} limit={cfg.limit} max_pages={cfg.max_pages} max_span_days={cfg.max_span_days}")
    print(f"[cfg] csv_path={cfg.csv_path} delimiter={delimiter_label} date_format={cfg.date_format}")
    if cfg.delimiter != ",":
        print("[warn] Your schedule JS requires commas (`txt.includes(',')`). Use CSV_DELIMITER=comma unless you also change JS.")

    sess = _session()

    event_ids = _iter_event_ids(cfg, sess)
    _debug(cfg, f"[mec] Total unique event IDs: {len(event_ids)}")

    all_rows: List[Dict[str, str]] = []
    skipped_404 = 0

    for eid in event_ids:
        detail = _fetch_event_detail(cfg, sess, eid)
        if detail is None:
            skipped_404 += 1
            continue
        all_rows.extend(_build_rows_for_event(cfg, detail))

    _write_csv(cfg, all_rows)
    print(f"Wrote {len(all_rows)} rows to {cfg.csv_path} (skipped_404={skipped_404})")


if __name__ == "__main__":
    main()
