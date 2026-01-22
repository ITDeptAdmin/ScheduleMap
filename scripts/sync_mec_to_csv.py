#!/usr/bin/env python3
"""
sync_mec_to_csv.py

Fetch events from a MEC-like API and write a normalized CSV.

Rule:
  - If an event is marked "All-day", the CSV start_time/end_time are blank.

Env vars:
  MEC_BASE_URL   (required)
  MEC_TOKEN      (optional)
  CSV_PATH       (required)
  MEC_START      default: 2010-01-01
  MEC_END        default: 2030-12-31
  MEC_LIMIT      default: 200
  MAX_PAGES      default: 50
  MAX_SPAN_DAYS  default: 10
  CSV_DELIMITER  default: comma  (comma|tab|pipe|semicolon)
  DATE_FORMAT    default: mdy    (mdy|ymd)
  MEC_DEBUG      default: 0/1
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


# ----------------------------
# Config & logging
# ----------------------------

def env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise SystemExit(f"Missing required env var: {name}")
    return v

def to_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")

def debug_enabled() -> bool:
    return to_bool(os.getenv("MEC_DEBUG", "0"))

def log(msg: str) -> None:
    print(msg, flush=True)

def delimiter_from_env(v: str) -> str:
    m = v.strip().lower()
    if m in ("comma", ","):
        return ","
    if m in ("tab", "\\t", "tsv"):
        return "\t"
    if m in ("pipe", "|"):
        return "|"
    if m in ("semicolon", ";"):
        return ";"
    raise SystemExit(f"Unknown CSV_DELIMITER: {v}")


# ----------------------------
# Date/time parsing/formatting
# ----------------------------

def parse_iso_date(s: str) -> date:
    return datetime.strptime(s.strip(), "%Y-%m-%d").date()

def parse_any_date(s: str) -> Optional[date]:
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None

def format_date(d: date, mode: str) -> str:
    if mode == "ymd":
        return d.strftime("%Y-%m-%d")
    if mode == "mdy":
        return f"{d.month}/{d.day}/{d.year}"
    raise SystemExit(f"Unknown DATE_FORMAT: {mode}")

def normalize_time_str(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    if re.search(r"\b(AM|PM)\b", s, flags=re.IGNORECASE):
        s = re.sub(r"\b(am|pm)\b", lambda m: m.group(1).upper(), s, flags=re.IGNORECASE)
        return s
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            t = datetime.strptime(s, fmt).time()
            # %-I may not work on Windows; keep it safe:
            hour = int(t.strftime("%I"))
            minute = t.strftime("%M")
            ampm = t.strftime("%p")
            return f"{hour}:{minute} {ampm}"
        except Exception:
            pass
    return s


# ----------------------------
# MEC API client
# ----------------------------

@dataclass
class Cfg:
    base_url: str
    token: str
    csv_path: str
    start: date
    end: date
    limit: int
    max_pages: int
    max_span_days: int
    csv_delimiter: str
    date_format: str
    include_past: bool = True
    include_ongoing: bool = True


def _looks_like_event_list(x: Any) -> bool:
    if not isinstance(x, list) or not x:
        return False
    if not isinstance(x[0], dict):
        return False
    # Many event items include one of these
    keys = set(x[0].keys())
    return bool(keys.intersection({"id", "ID", "event_id", "title", "start_date", "date_start"}))

def _find_event_list_anywhere(obj: Any, max_depth: int = 6) -> Optional[List[Dict[str, Any]]]:
    """
    Recursively search for a list of dicts that looks like an event list.
    This fixes the '200 but items=0' case when payload is nested.
    """
    if max_depth < 0:
        return None

    if _looks_like_event_list(obj):
        return obj  # type: ignore[return-value]

    if isinstance(obj, dict):
        # common containers first
        for k in ("events", "items", "data", "results", "posts"):
            if k in obj:
                found = _find_event_list_anywhere(obj[k], max_depth=max_depth - 1)
                if found is not None:
                    return found

        # otherwise scan all values
        for v in obj.values():
            found = _find_event_list_anywhere(v, max_depth=max_depth - 1)
            if found is not None:
                return found

    if isinstance(obj, list):
        for v in obj:
            found = _find_event_list_anywhere(v, max_depth=max_depth - 1)
            if found is not None:
                return found

    return None


class MecClient:
    def __init__(self, cfg: Cfg):
        self.cfg = cfg
        self.sess = requests.Session()
        self.sess.headers.update({"Accept": "application/json"})
        if cfg.token:
            self.sess.headers.update({"Authorization": f"Bearer {cfg.token}"})

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = self.cfg.base_url.rstrip("/") + path
        r = self.sess.get(url, params=params, timeout=60)
        if debug_enabled():
            q = ""
            if params:
                q = "?" + "&".join([f"{k}={params[k]}" for k in params])
            log(f"[mec] GET {url}{q} -> {r.status_code}")
        r.raise_for_status()
        return r.json()

    def list_events_page(self, start_date: date, offset: int) -> List[Dict[str, Any]]:
        params = {
            "limit": self.cfg.limit,
            "include_past_events": 1 if self.cfg.include_past else 0,
            "include_ongoing_events": 1 if self.cfg.include_ongoing else 0,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "offset": offset,
        }
        data = self._get("/events", params=params)

        found = _find_event_list_anywhere(data)
        if found is None:
            if debug_enabled():
                if isinstance(data, dict):
                    log(f"[mec] Could not locate event list in response. Top-level keys: {list(data.keys())}")
                else:
                    log(f"[mec] Could not locate event list in response. Type: {type(data)}")
            return []

        return found

    def get_event(self, event_id: int) -> Dict[str, Any]:
        data = self._get(f"/events/{event_id}")
        # common wrappers
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], dict):
                return data["data"]
        return data if isinstance(data, dict) else {}


# ----------------------------
# Extraction helpers
# ----------------------------

def get_nested(d: Dict[str, Any], *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def first_non_empty(*vals: Any) -> Any:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        return v
    return None

def str_or_empty(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()

def parse_services(event: Dict[str, Any]) -> Tuple[str, str, str, str]:
    raw = first_non_empty(
        event.get("services"),
        event.get("services_field"),
        get_nested(event, "meta", "services"),
        get_nested(event, "meta", "services_field"),
    )

    text = ""
    if isinstance(raw, list):
        text = ", ".join([str(x) for x in raw])
    else:
        text = str(raw) if raw is not None else ""

    if not text.strip():
        text = " ".join([
            str_or_empty(event.get("title")),
            str_or_empty(event.get("name")),
            str_or_empty(event.get("taxonomy")),
            str_or_empty(get_nested(event, "meta", "taxonomy")),
        ])

    t = text.lower()
    medical = "Yes" if "medical" in t else "No"
    dental = "Yes" if "dental" in t else "No"
    vision = "Yes" if "vision" in t else "No"
    dentures = "Yes" if "denture" in t else "No"
    return medical, dental, vision, dentures

def extract_location(event: Dict[str, Any]) -> Tuple[str, str, str, str, str, str, str, str]:
    venue = first_non_empty(
        event.get("venue"),
        event.get("location"),
        get_nested(event, "meta", "venue"),
        get_nested(event, "meta", "location"),
    )
    if not isinstance(venue, dict):
        venue = {}

    lat = first_non_empty(event.get("lat"), venue.get("lat"), venue.get("latitude"), get_nested(event, "meta", "lat"))
    lng = first_non_empty(event.get("lng"), venue.get("lng"), venue.get("longitude"), get_nested(event, "meta", "lng"))

    address = first_non_empty(
        event.get("address"),
        venue.get("address"),
        venue.get("street"),
        get_nested(event, "meta", "address"),
    )

    city = first_non_empty(event.get("city"), venue.get("city"), get_nested(event, "meta", "city"))
    state = first_non_empty(event.get("state"), venue.get("state"), get_nested(event, "meta", "state"))

    site = first_non_empty(event.get("site"), venue.get("site"), (f"{city}, {state}" if city and state else ""))
    facility = first_non_empty(event.get("facility"), venue.get("name"), event.get("venue_name"), event.get("title"))

    tel = first_non_empty(
        event.get("telehealth"),
        get_nested(event, "meta", "telehealth"),
        get_nested(event, "meta", "is_telehealth"),
        event.get("is_telehealth"),
    )
    telehealth = "Yes" if (tel is True or (isinstance(tel, str) and tel.strip().lower() in ("1","yes","true","on"))) else "No"

    return (
        str_or_empty(lat),
        str_or_empty(lng),
        str_or_empty(address),
        str_or_empty(city),
        str_or_empty(state),
        str_or_empty(site),
        str_or_empty(facility),
        telehealth,
    )

def extract_url(event: Dict[str, Any]) -> str:
    return str_or_empty(first_non_empty(event.get("url"), event.get("permalink"), get_nested(event, "meta", "url")))

def extract_canceled(event: Dict[str, Any]) -> str:
    status = str_or_empty(first_non_empty(event.get("status"), event.get("event_status"), get_nested(event, "meta", "status"))).lower()
    if "cancel" in status:
        return "Yes"
    c = first_non_empty(event.get("canceled"), event.get("cancelled"), get_nested(event, "meta", "canceled"))
    if c is True or (isinstance(c, str) and c.strip().lower() in ("1","yes","true","on")):
        return "Yes"
    return "No"

def extract_parking(event: Dict[str, Any]) -> Tuple[str, str]:
    pd = first_non_empty(event.get("parking_date"), get_nested(event, "meta", "parking_date"))
    pt = first_non_empty(event.get("parking_time"), get_nested(event, "meta", "parking_time"))
    return str_or_empty(pd), str_or_empty(pt)


# ----------------------------
# All-day detection & time extraction
# ----------------------------

def is_all_day(event: Dict[str, Any]) -> bool:
    for k in ("all_day", "allDay", "allday"):
        if event.get(k) is True:
            return True

    meta = first_non_empty(event.get("meta"), event.get("mec"), {}) or {}
    if isinstance(meta, dict):
        for k in ("all_day", "allDay", "allday", "mec_allday", "mec_all_day"):
            v = meta.get(k)
            if v is True:
                return True
            if isinstance(v, str) and v.strip().lower() in ("1", "true", "yes", "on"):
                return True

    return False

def extract_times(event: Dict[str, Any]) -> Tuple[str, str]:
    st = first_non_empty(
        event.get("start_time"),
        get_nested(event, "meta", "start_time"),
        get_nested(event, "mec", "start_time"),
    )
    et = first_non_empty(
        event.get("end_time"),
        get_nested(event, "meta", "end_time"),
        get_nested(event, "mec", "end_time"),
    )

    if not st or not et:
        sched = first_non_empty(
            event.get("time"),
            get_nested(event, "meta", "time"),
            get_nested(event, "meta", "hourly_schedule"),
        )
        if isinstance(sched, str) and "-" in sched:
            left, right = [x.strip() for x in sched.split("-", 1)]
            st = st or left
            et = et or right

    return normalize_time_str(str_or_empty(st)), normalize_time_str(str_or_empty(et))


# ----------------------------
# Occurrence expansion & cleaning
# ----------------------------

def extract_event_dates(event: Dict[str, Any]) -> Tuple[Optional[date], Optional[date]]:
    sd = first_non_empty(event.get("start_date"), get_nested(event, "meta", "start_date"), event.get("date_start"))
    ed = first_non_empty(event.get("end_date"), get_nested(event, "meta", "end_date"), event.get("date_end"))

    d1 = parse_any_date(str_or_empty(sd))
    d2 = parse_any_date(str_or_empty(ed))
    return d1, d2

def extract_occurrence_dates(event: Dict[str, Any]) -> List[Tuple[date, date]]:
    occ = first_non_empty(event.get("occurrences"), get_nested(event, "meta", "occurrences"))
    out: List[Tuple[date, date]] = []

    if isinstance(occ, list) and occ:
        for item in occ:
            if not isinstance(item, dict):
                continue
            sd = parse_any_date(str_or_empty(first_non_empty(item.get("start_date"), item.get("start"), item.get("date_start"))))
            ed = parse_any_date(str_or_empty(first_non_empty(item.get("end_date"), item.get("end"), item.get("date_end"))))
            if sd and not ed:
                ed = sd
            if sd and ed:
                out.append((sd, ed))

    if not out:
        sd, ed = extract_event_dates(event)
        if sd and not ed:
            ed = sd
        if sd and ed:
            out.append((sd, ed))

    return out

def daterange(d1: date, d2: date) -> Iterable[date]:
    cur = d1
    while cur <= d2:
        yield cur
        cur += timedelta(days=1)

def expand_to_rows(event: Dict[str, Any], cfg: Cfg) -> List[Dict[str, str]]:
    canceled = extract_canceled(event)
    lat, lng, address, city, state, site, facility, telehealth = extract_location(event)
    url = extract_url(event)
    parking_date, parking_time = extract_parking(event)
    medical, dental, vision, dentures = parse_services(event)

    all_day = is_all_day(event)
    start_time, end_time = extract_times(event)

    # RULE: all-day => blank times
    if all_day:
        start_time = ""
        end_time = ""

    rows: List[Dict[str, str]] = []
    for (sd, ed) in extract_occurrence_dates(event):
        if (ed - sd).days > cfg.max_span_days:
            if debug_enabled():
                log(f"[warn] span too long ({sd} -> {ed}), clamping to {cfg.max_span_days} days for url={url}")
            ed = sd + timedelta(days=cfg.max_span_days)

        for d in daterange(sd, ed):
            rows.append({
                "canceled": canceled,
                "lat": lat,
                "lng": lng,
                "address": address,
                "city": city,
                "state": state,
                "site": site,
                "facility": facility,
                "telehealth": telehealth,
                "start_time": start_time,
                "end_time": end_time,
                "start_date": format_date(d, cfg.date_format),
                "end_date": format_date(d, cfg.date_format),
                "parking_date": parking_date,
                "parking_time": parking_time,
                "medical": medical,
                "dental": dental,
                "vision": vision,
                "dentures": dentures,
                "url": url,
                "_all_day": "1" if all_day else "0",
            })

    return rows

def drop_range_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    if not rows:
        return rows

    by_url: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_url.setdefault(r.get("url", ""), []).append(r)

    out: List[Dict[str, str]] = []
    for u, group in by_url.items():
        if len(group) == 1:
            out.append(group[0])
            continue

        def keyfunc(rr: Dict[str, str]) -> Tuple[int, int, int]:
            d = parse_any_date(rr["start_date"]) or date.min
            return (d.year, d.month, d.day)

        out.append(max(group, key=keyfunc))

    return out

def collapse_duplicates_by_date(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    def score(r: Dict[str, str]) -> int:
        return sum(1 for k, v in r.items() if not k.startswith("_") and str(v).strip() != "")

    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for r in rows:
        grouped.setdefault((r.get("url", ""), r.get("start_date", "")), []).append(r)

    out: List[Dict[str, str]] = []
    for group in grouped.values():
        out.append(max(group, key=score))
    return out

def enforce_all_day_override(rows: List[Dict[str, str]]) -> None:
    for r in rows:
        if r.get("_all_day") == "1":
            r["start_time"] = ""
            r["end_time"] = ""


# ----------------------------
# Main sync logic
# ----------------------------

def collect_unique_event_ids(client: MecClient, cfg: Cfg) -> List[int]:
    start_date = cfg.start
    offset = 0
    seen_pages = set()

    unique_ids: List[int] = []
    seen_ids = set()

    for page in range(1, cfg.max_pages + 1):
        key = (start_date.isoformat(), offset)
        if key in seen_pages:
            log(f"[mec] pagination loop detected at start_date={start_date} offset={offset}, stopping")
            break
        seen_pages.add(key)

        items = client.list_events_page(start_date=start_date, offset=offset)

        ids_this_page: List[int] = []
        max_date: Optional[date] = None

        for it in items:
            if not isinstance(it, dict):
                continue
            eid = it.get("id") or it.get("event_id") or it.get("ID")
            try:
                eid_int = int(eid)
            except Exception:
                continue
            ids_this_page.append(eid_int)
            if eid_int not in seen_ids:
                seen_ids.add(eid_int)
                unique_ids.append(eid_int)

            d = parse_any_date(str_or_empty(first_non_empty(it.get("start_date"), it.get("date_start"), get_nested(it, "meta", "start_date"))))
            if d and (max_date is None or d > max_date):
                max_date = d

        log(f"[mec] list page {page}: start_date={start_date.isoformat()} offset={offset} -> items={len(items)} ids={len(set(ids_this_page))} unique_total={len(unique_ids)}")

        if not items:
            break

        # advance pagination
        if max_date is None or max_date <= start_date:
            offset += len(items)
        else:
            start_date = max_date
            offset = 0

    return unique_ids

def write_csv(path: str, rows: List[Dict[str, str]], delimiter: str) -> None:
    fieldnames = [
        "canceled","lat","lng","address","city","state","site","facility","telehealth",
        "start_time","end_time","start_date","end_date","parking_date","parking_time",
        "medical","dental","vision","dentures","url"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter=delimiter)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def main() -> int:
    cfg = Cfg(
        base_url=env("MEC_BASE_URL"),
        token=os.getenv("MEC_TOKEN", ""),
        csv_path=env("CSV_PATH"),
        start=parse_iso_date(os.getenv("MEC_START", "2010-01-01")),
        end=parse_iso_date(os.getenv("MEC_END", "2030-12-31")),
        limit=int(os.getenv("MEC_LIMIT", "200")),
        max_pages=int(os.getenv("MAX_PAGES", "50")),
        max_span_days=int(os.getenv("MAX_SPAN_DAYS", "10")),
        csv_delimiter=delimiter_from_env(os.getenv("CSV_DELIMITER", "comma")),
        date_format=os.getenv("DATE_FORMAT", "mdy").strip().lower(),
    )

    log(f"[cfg] base_url={cfg.base_url}")
    log(f"[cfg] start={cfg.start.isoformat()} end={cfg.end.isoformat()} limit={cfg.limit} max_pages={cfg.max_pages} max_span_days={cfg.max_span_days}")
    log(f"[cfg] include_past={1 if cfg.include_past else 0} include_ongoing={1 if cfg.include_ongoing else 0}")
    log(f"[cfg] csv_path={cfg.csv_path} delimiter=COMMA date_format={cfg.date_format}")

    client = MecClient(cfg)

    ids = collect_unique_event_ids(client, cfg)
    log(f"[mec] Total unique event IDs: {len(ids)}")

    all_rows: List[Dict[str, str]] = []
    skipped_404 = 0

    for eid in ids:
        try:
            event = client.get_event(eid)
        except requests.HTTPError as e:
            if getattr(e.response, "status_code", None) == 404:
                skipped_404 += 1
                continue
            raise

        rows = expand_to_rows(event, cfg)

        before = len(rows)
        rows = drop_range_rows(rows)
        if debug_enabled():
            log(f"[clean] dropped range rows: {before} -> {len(rows)}")

        before = len(rows)
        rows = collapse_duplicates_by_date(rows)
        if debug_enabled():
            log(f"[clean] collapsed duplicates by date: {before} -> {len(rows)}")

        enforce_all_day_override(rows)
        all_rows.extend(rows)

    all_rows = collapse_duplicates_by_date(all_rows)
    enforce_all_day_override(all_rows)

    write_csv(cfg.csv_path, all_rows, cfg.csv_delimiter)
    log(f"Wrote {len(all_rows)} rows to {cfg.csv_path} (skipped_404={skipped_404})")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
