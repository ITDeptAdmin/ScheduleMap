#!/usr/bin/env python3
"""
sync_mec_to_csv.py

Fetch events from a MEC (Modern Events Calendar) JSON API and write a normalized CSV.

Key behaviors (based on your requirements / logs):
- Paginates /events using start_date + offset + limit
- Collects unique event IDs plus "occurrence hints" (dates seen on list pages)
- Fetches /events/{id} for each ID and emits rows:
    * If we have occurrence hints -> one row per hint date (single-day rows)
    * Else -> one row from the event's own start/end date
- For multi-day events (start_date != end_date) that are "all-day", we collapse to ONE row
  on the end_date (matches your "dropped range rows: 3 -> 1" behavior).
- If event is all-day: start_time/end_time are BLANK in CSV.
- Collapses duplicates by date within an event (keeps the "best" row with times/services/etc).
- Adds safe retry for transient 5xx responses and avoids the offset=1 @ ancient dates issue
  by falling back to event-detail date inference when list items don't include usable dates.

Env vars:
  MEC_BASE_URL      (required)  e.g. https://example.com/wp-json/mec/v1
  MEC_TOKEN         (optional)  token if your endpoint requires auth
  CSV_PATH          (optional)  default: Clinic Master Schedule for git.csv
  MEC_START         (optional)  default: 2010-01-01
  MEC_END           (optional)  default: 2030-12-31
  MEC_LIMIT         (optional)  default: 200
  MAX_PAGES         (optional)  default: 50
  CSV_DELIMITER     (optional)  comma|tab|pipe  default: comma
  DATE_FORMAT       (optional)  mdy|ymd        default: mdy
  MEC_DEBUG         (optional)  1 enables debug logs

CSV columns:
canceled,lat,lng,address,city,state,site,facility,telehealth,start_time,end_time,
start_date,end_date,parking_date,parking_time,medical,dental,vision,dentures,url
"""

from __future__ import annotations

import csv
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


# -------------------------
# Logging / debug
# -------------------------

def debug_enabled() -> bool:
    return os.getenv("MEC_DEBUG", "").strip() not in ("", "0", "false", "False")


def log(msg: str) -> None:
    print(msg, file=sys.stderr)


# -------------------------
# Config
# -------------------------

def parse_date_ymd(s: str, default: date) -> date:
    s = (s or "").strip()
    if not s:
        return default
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return default


def parse_int(s: str, default: int) -> int:
    try:
        return int(str(s).strip())
    except Exception:
        return default


def parse_delimiter(name: str) -> str:
    name = (name or "").strip().lower()
    if name in ("comma", ",", "csv"):
        return ","
    if name in ("tab", "\\t", "tsv"):
        return "\t"
    if name in ("pipe", "|"):
        return "|"
    return ","


@dataclass
class Cfg:
    base_url: str
    token: str
    csv_path: str
    delimiter: str
    date_format: str  # "mdy" or "ymd"
    start: date
    end: date
    limit: int
    max_pages: int
    include_past: bool
    include_ongoing: bool


def load_cfg() -> Cfg:
    base_url = os.getenv("MEC_BASE_URL", "").strip()
    if not base_url:
        raise SystemExit("MEC_BASE_URL is required")

    token = os.getenv("MEC_TOKEN", "").strip()
    csv_path = os.getenv("CSV_PATH", "Clinic Master Schedule for git.csv").strip()

    delimiter = parse_delimiter(os.getenv("CSV_DELIMITER", "comma"))
    date_format = (os.getenv("DATE_FORMAT", "mdy") or "mdy").strip().lower()
    if date_format not in ("mdy", "ymd"):
        date_format = "mdy"

    start = parse_date_ymd(os.getenv("MEC_START", "2010-01-01"), date(2010, 1, 1))
    end = parse_date_ymd(os.getenv("MEC_END", "2030-12-31"), date(2030, 12, 31))

    limit = parse_int(os.getenv("MEC_LIMIT", "200"), 200)
    max_pages = parse_int(os.getenv("MAX_PAGES", "50"), 50)

    # based on your logs you always include these
    include_past = True
    include_ongoing = True

    if debug_enabled():
        log(f"[cfg] base_url=***")
        log(
            f"[cfg] start={start.isoformat()} end={end.isoformat()} "
            f"limit={limit} max_pages={max_pages}"
        )
        log(
            f"[cfg] include_past={1 if include_past else 0} "
            f"include_ongoing={1 if include_ongoing else 0}"
        )
        log(
            f"[cfg] csv_path={csv_path} delimiter="
            f"{'COMMA' if delimiter==',' else delimiter!r} date_format={date_format}"
        )

    return Cfg(
        base_url=base_url,
        token=token,
        csv_path=csv_path,
        delimiter=delimiter,
        date_format=date_format,
        start=start,
        end=end,
        limit=limit,
        max_pages=max_pages,
        include_past=include_past,
        include_ongoing=include_ongoing,
    )


# -------------------------
# Helpers
# -------------------------

def str_or_empty(x: Any) -> str:
    return "" if x is None else str(x)


def get_nested(d: Any, *keys: str) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        if k not in cur:
            return None
        cur = cur[k]
    return cur


def first_nonempty(*vals: Any) -> str:
    for v in vals:
        s = str_or_empty(v).strip()
        if s:
            return s
    return ""


def parse_any_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None

    # normalize common date-time strings
    # Accept: YYYY-MM-DD, YYYY-MM-DD HH:MM:SS, ISO-ish
    m = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except Exception:
            pass

    # Accept: M/D/YYYY or MM/DD/YYYY
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        mm, dd, yyyy = m.groups()
        try:
            return date(int(yyyy), int(mm), int(dd))
        except Exception:
            return None

    return None


def parse_any_time(s: str) -> str:
    """Return a normalized display time like '8:00 AM' or ''."""
    s = (s or "").strip()
    if not s:
        return ""

    # already looks like "8:00 AM"
    if re.search(r"\bAM\b|\bPM\b", s, re.IGNORECASE):
        return s

    # "08:00" or "8:00"
    m = re.match(r"^(\d{1,2}):(\d{2})$", s)
    if m:
        hh = int(m.group(1))
        mm = m.group(2)
        ampm = "AM"
        if hh == 0:
            hh = 12
        elif hh == 12:
            ampm = "PM"
        elif hh > 12:
            hh -= 12
            ampm = "PM"
        return f"{hh}:{mm} {ampm}"

    # "0800"
    m = re.match(r"^(\d{2})(\d{2})$", s)
    if m:
        hh = int(m.group(1))
        mm = m.group(2)
        ampm = "AM"
        if hh == 0:
            hh = 12
        elif hh == 12:
            ampm = "PM"
        elif hh > 12:
            hh -= 12
            ampm = "PM"
        return f"{hh}:{mm} {ampm}"

    return s


def format_date(d: date, fmt: str) -> str:
    if fmt == "ymd":
        return d.strftime("%Y-%m-%d")
    # mdy (your CSV)
    return f"{d.month}/{d.day}/{d.year}"


def truthy(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str_or_empty(x).strip().lower()
    return s in ("1", "true", "yes", "y", "on")


# -------------------------
# MEC client
# -------------------------

class MecClient:
    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg
        self.sess = requests.Session()
        self.sess.headers.update({"Accept": "application/json"})

        # Try to be compatible with common token styles.
        if cfg.token:
            self.sess.headers.update(
                {
                    "Authorization": f"Bearer {cfg.token}",
                    "X-API-KEY": cfg.token,
                    "X-Auth-Token": cfg.token,
                }
            )

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = self.cfg.base_url.rstrip("/") + path

        last_exc: Optional[Exception] = None
        for attempt in range(1, 4):  # 3 tries total
            try:
                r = self.sess.get(url, params=params, timeout=60)

                if debug_enabled():
                    q = ""
                    if params:
                        # (good enough for logs)
                        q = "?" + "&".join([f"{k}={params[k]}" for k in params])
                    log(f"[mec] GET ***/{path.lstrip('/')} {q} -> {r.status_code}")

                # retry transient 5xx
                if r.status_code >= 500 and attempt < 3:
                    time.sleep(1)
                    continue

                r.raise_for_status()
                return r.json()

            except Exception as e:
                last_exc = e
                if attempt < 3:
                    time.sleep(1)
                    continue
                raise

        raise last_exc or RuntimeError("Unknown request failure")

    def list_events_page(self, start_date: date, offset: int) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {
            "limit": self.cfg.limit,
            "start_date": start_date.isoformat(),
            "offset": offset,
            "include_past_events": 1 if self.cfg.include_past else 0,
            "include_ongoing_events": 1 if self.cfg.include_ongoing else 0,
        }
        data = self._get("/events", params=params)

        # Different MEC installs return shapes like:
        # { "events": [...] } or { "data": [...] } or directly [...]
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            for k in ("events", "data", "items", "results"):
                v = data.get(k)
                if isinstance(v, list):
                    return [x for x in v if isinstance(x, dict)]
        return []

    def get_event(self, event_id: int) -> Dict[str, Any]:
        data = self._get(f"/events/{event_id}", params=None)
        if isinstance(data, dict):
            return data
        return {}


# -------------------------
# List-page date extraction
# -------------------------

def extract_list_item_start_date(it: Dict[str, Any]) -> Optional[date]:
    """
    Try hard to pull a start date from list-page event items, which often have
    different shapes than /events/{id}.
    """
    candidates = [
        it.get("start_date"),
        it.get("date_start"),
        it.get("start"),
        it.get("startDate"),
        it.get("start_date_local"),
        get_nested(it, "meta", "start_date"),
        get_nested(it, "mec", "start_date"),
        get_nested(it, "data", "start_date"),
        # sometimes in "date" / "occurrence"
        it.get("date"),
        it.get("occurrence"),
        get_nested(it, "meta", "date"),
    ]
    for c in candidates:
        d = parse_any_date(str_or_empty(c))
        if d:
            return d
    return None


def extract_list_item_occurrence_date(it: Dict[str, Any]) -> Optional[date]:
    """
    If the list item represents an occurrence, it may include an occurrence date.
    """
    candidates = [
        it.get("occurrence_date"),
        it.get("occurrence"),
        it.get("date"),
        get_nested(it, "meta", "occurrence_date"),
        get_nested(it, "meta", "date"),
        get_nested(it, "data", "occurrence_date"),
    ]
    for c in candidates:
        d = parse_any_date(str_or_empty(c))
        if d:
            return d
    return None


# -------------------------
# Event detail parsing
# -------------------------

def extract_event_dates(ev: Dict[str, Any]) -> Tuple[date, date]:
    """
    Return (start_date, end_date) from event detail. Defaults to today if missing.
    """
    today = date.today()
    sd = parse_any_date(first_nonempty(
        ev.get("start_date"),
        ev.get("date_start"),
        get_nested(ev, "data", "start_date"),
        get_nested(ev, "mec", "start_date"),
        get_nested(ev, "meta", "start_date"),
        get_nested(ev, "dates", "start"),
    )) or today

    ed = parse_any_date(first_nonempty(
        ev.get("end_date"),
        ev.get("date_end"),
        get_nested(ev, "data", "end_date"),
        get_nested(ev, "mec", "end_date"),
        get_nested(ev, "meta", "end_date"),
        get_nested(ev, "dates", "end"),
    )) or sd

    return sd, ed


def is_all_day(ev: Dict[str, Any]) -> bool:
    candidates = [
        ev.get("all_day"),
        ev.get("allday"),
        ev.get("allDay"),
        get_nested(ev, "data", "all_day"),
        get_nested(ev, "mec", "all_day"),
        get_nested(ev, "meta", "all_day"),
    ]
    return any(truthy(c) for c in candidates)


def extract_times(ev: Dict[str, Any]) -> Tuple[str, str]:
    st = first_nonempty(
        ev.get("start_time"),
        ev.get("time_start"),
        get_nested(ev, "data", "start_time"),
        get_nested(ev, "mec", "start_time"),
        get_nested(ev, "meta", "start_time"),
    )
    et = first_nonempty(
        ev.get("end_time"),
        ev.get("time_end"),
        get_nested(ev, "data", "end_time"),
        get_nested(ev, "mec", "end_time"),
        get_nested(ev, "meta", "end_time"),
    )
    return parse_any_time(st), parse_any_time(et)


def extract_url(ev: Dict[str, Any]) -> str:
    return first_nonempty(
        ev.get("url"),
        ev.get("permalink"),
        ev.get("link"),
        get_nested(ev, "data", "url"),
        get_nested(ev, "data", "permalink"),
        get_nested(ev, "meta", "url"),
    )


def extract_cancelled(ev: Dict[str, Any]) -> bool:
    # Try typical status fields
    status = first_nonempty(
        ev.get("status"),
        ev.get("event_status"),
        get_nested(ev, "data", "status"),
        get_nested(ev, "meta", "status"),
    ).lower()
    if "cancel" in status:
        return True
    if truthy(ev.get("canceled")) or truthy(ev.get("cancelled")):
        return True
    return False


def extract_location(ev: Dict[str, Any]) -> Dict[str, str]:
    """
    Normalize location fields used in your CSV:
    lat,lng,address,city,state,site,facility,telehealth
    """
    # Venue can be a nested object or string
    venue = get_nested(ev, "venue") or get_nested(ev, "location") or get_nested(ev, "data", "venue") or {}
    if isinstance(venue, str):
        venue = {"name": venue}

    lat = first_nonempty(
        ev.get("lat"), ev.get("latitude"),
        get_nested(venue, "lat"), get_nested(venue, "latitude"),
        get_nested(ev, "data", "lat"), get_nested(ev, "data", "latitude"),
    )
    lng = first_nonempty(
        ev.get("lng"), ev.get("lon"), ev.get("longitude"),
        get_nested(venue, "lng"), get_nested(venue, "lon"), get_nested(venue, "longitude"),
        get_nested(ev, "data", "lng"), get_nested(ev, "data", "longitude"),
    )

    address = first_nonempty(
        ev.get("address"),
        get_nested(venue, "address"),
        get_nested(ev, "data", "address"),
        get_nested(ev, "meta", "address"),
    )
    city = first_nonempty(
        ev.get("city"),
        get_nested(venue, "city"),
        get_nested(ev, "data", "city"),
    )
    state = first_nonempty(
        ev.get("state"),
        get_nested(venue, "state"),
        get_nested(ev, "data", "state"),
    )

    # "site" appears to be like "East Bend, NC"
    site = first_nonempty(
        ev.get("site"),
        get_nested(ev, "data", "site"),
        get_nested(ev, "meta", "site"),
        first_nonempty(city, "").strip() + (", " + state.strip() if city.strip() and state.strip() else ""),
    )

    # facility name
    facility = first_nonempty(
        ev.get("facility"),
        get_nested(venue, "name"),
        get_nested(ev, "data", "facility"),
        get_nested(ev, "meta", "facility"),
        ev.get("title"),
        ev.get("post_title"),
    )

    telehealth = truthy(first_nonempty(
        ev.get("telehealth"),
        get_nested(ev, "data", "telehealth"),
        get_nested(ev, "meta", "telehealth"),
    ))

    return {
        "lat": lat,
        "lng": lng,
        "address": address,
        "city": city,
        "state": state,
        "site": site,
        "facility": facility,
        "telehealth": "Yes" if telehealth else "No",
    }


def extract_parking(ev: Dict[str, Any]) -> Tuple[str, str]:
    pd = first_nonempty(
        ev.get("parking_date"),
        get_nested(ev, "data", "parking_date"),
        get_nested(ev, "meta", "parking_date"),
    )
    pt = first_nonempty(
        ev.get("parking_time"),
        get_nested(ev, "data", "parking_time"),
        get_nested(ev, "meta", "parking_time"),
    )
    return pd.strip(), pt.strip()


def parse_services(ev: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Return "Yes"/"No" for (medical, dental, vision, dentures)
    """
    # Your logs show a "services_field" sometimes; also sometimes missing and you used taxonomy fallback.
    services_field = first_nonempty(
        ev.get("services"),
        ev.get("services_field"),
        get_nested(ev, "data", "services"),
        get_nested(ev, "meta", "services"),
        get_nested(ev, "acf", "services"),
        get_nested(ev, "taxonomy"),
        get_nested(ev, "taxonomies"),
    )

    blob = str_or_empty(services_field).lower()

    # Taxonomy fallback often embeds words
    medical = "yes" if "medical" in blob else "no"
    dental = "yes" if "dental" in blob else "no"
    vision = "yes" if "vision" in blob else "no"
    dentures = "yes" if "denture" in blob else "no"

    return (
        "Yes" if medical == "yes" else "No",
        "Yes" if dental == "yes" else "No",
        "Yes" if vision == "yes" else "No",
        "Yes" if dentures == "yes" else "No",
    )


# -------------------------
# Pagination + occurrence hints
# -------------------------

def collect_unique_event_ids_with_hints(client: MecClient, cfg: Cfg) -> Tuple[List[int], Dict[int, List[date]]]:
    start_date = cfg.start
    offset = 0
    seen_pages = set()

    unique_ids: List[int] = []
    seen_ids = set()
    hints: Dict[int, List[date]] = {}

    for page in range(1, cfg.max_pages + 1):
        key = (start_date.isoformat(), offset)
        if key in seen_pages:
            log(f"[mec] pagination loop detected at start_date={start_date} offset={offset}, stopping")
            break
        seen_pages.add(key)

        items = client.list_events_page(start_date=start_date, offset=offset)

        ids_this_page: List[int] = []
        max_date: Optional[date] = None
        last_id: Optional[int] = None
        hints_added = 0

        for it in items:
            eid = it.get("id") or it.get("event_id") or it.get("ID")
            try:
                eid_int = int(eid)
            except Exception:
                continue

            last_id = eid_int
            ids_this_page.append(eid_int)

            if eid_int not in seen_ids:
                seen_ids.add(eid_int)
                unique_ids.append(eid_int)

            # occurrence hint
            occ = extract_list_item_occurrence_date(it) or extract_list_item_start_date(it)
            if occ:
                hints.setdefault(eid_int, [])
                if occ not in hints[eid_int]:
                    hints[eid_int].append(occ)
                    hints_added += 1

            d = extract_list_item_start_date(it)
            if d and (max_date is None or d > max_date):
                max_date = d

        if debug_enabled():
            log(
                f"[mec] list page {page}: start_date={start_date.isoformat()} offset={offset} "
                f"-> items={len(items)} ids={len(set(ids_this_page))} hints_added={hints_added} unique_total={len(unique_ids)}"
            )
        else:
            log(
                f"[mec] list page {page}: start_date={start_date.isoformat()} offset={offset} "
                f"-> items={len(items)} ids={len(set(ids_this_page))} unique_total={len(unique_ids)}"
            )

        if not items:
            break

        # If list items don't include a usable max_date, fetch event detail for the last ID
        # so we can advance start_date and avoid fragile offset paging on ancient dates.
        if max_date is None and last_id is not None:
            try:
                ev = client.get_event(last_id)
                ev_sd, _ = extract_event_dates(ev)
                max_date = ev_sd
                if debug_enabled():
                    log(f"[mec] inferred max_date from event detail id={last_id}: {max_date}")
            except Exception:
                max_date = None

        # advance pagination
        if max_date is not None and max_date > start_date:
            start_date = max_date
            offset = 0
        else:
            offset += len(items)

        # Safety bump for very old dates if offset paging causes API trouble
        if start_date <= cfg.start and offset > 0 and len(items) < cfg.limit:
            start_date = start_date + timedelta(days=1)
            offset = 0
            if debug_enabled():
                log(f"[mec] safety bump: advancing start_date to {start_date} to avoid offset paging on very old start_date")

    # Normalize hints list ordering
    for eid in list(hints.keys()):
        hints[eid] = sorted(set(hints[eid]))

    return unique_ids, hints


# -------------------------
# Row model + cleaning
# -------------------------

CSV_HEADERS = [
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


def row_quality_score(r: Dict[str, str]) -> int:
    """
    Used for collapsing duplicates by date: keep the most informative row.
    """
    score = 0
    for k in ("start_time", "end_time", "address", "facility", "url"):
        if str_or_empty(r.get(k)).strip():
            score += 1
    for k in ("medical", "dental", "vision", "dentures"):
        if r.get(k) == "Yes":
            score += 1
    return score


def drop_range_rows(rows: List[Dict[str, str]], all_day_flag: bool) -> List[Dict[str, str]]:
    """
    If an event spans multiple days and is all-day, your desired behavior is a SINGLE row.
    We keep the row whose start_date=end_date=end_date (i.e., end day).
    """
    if not rows:
        return rows
    if not all_day_flag:
        return rows

    # If any rows have differing start/end, keep only the one on the latest date.
    # (Assumes rows are single-day rows already.)
    def row_date(r: Dict[str, str]) -> Optional[date]:
        return parse_any_date(r.get("start_date", ""))

    dated = [(row_date(r), r) for r in rows]
    dated = [(d, r) for (d, r) in dated if d is not None]
    if not dated:
        return rows
    latest = max(dated, key=lambda x: x[0])[1]
    return [latest]


def collapse_duplicates_by_date(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Keep one row per (start_date) within an event.
    """
    best: Dict[str, Dict[str, str]] = {}
    for r in rows:
        key = str_or_empty(r.get("start_date")).strip()
        if not key:
            continue
        if key not in best or row_quality_score(r) > row_quality_score(best[key]):
            best[key] = r
    # preserve date order
    out = [best[k] for k in sorted(best.keys(), key=lambda s: parse_any_date(s) or date(1900, 1, 1))]
    return out


# -------------------------
# Build rows from an event
# -------------------------

def build_rows_for_event(ev: Dict[str, Any], cfg: Cfg, occurrence_hints: Optional[List[date]] = None) -> List[Dict[str, str]]:
    cancelled = extract_cancelled(ev)
    loc = extract_location(ev)
    parking_date, parking_time = extract_parking(ev)
    medical, dental, vision, dentures = parse_services(ev)
    url = extract_url(ev)

    sd, ed = extract_event_dates(ev)
    all_day_flag = is_all_day(ev)

    start_time, end_time = extract_times(ev)
    if all_day_flag:
        # your requirement: if "All-day Event" is checked, leave times blank
        start_time, end_time = "", ""

    # If event is multi-day and all-day, collapse to one row on end date
    if all_day_flag and sd != ed:
        sd = ed

    rows: List[Dict[str, str]] = []

    def make_row(d: date) -> Dict[str, str]:
        return {
            "canceled": "Yes" if cancelled else "No",
            "lat": loc["lat"],
            "lng": loc["lng"],
            "address": loc["address"],
            "city": loc["city"],
            "state": loc["state"],
            "site": loc["site"],
            "facility": loc["facility"],
            "telehealth": loc["telehealth"],
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
        }

    if occurrence_hints:
        # one row per occurrence date
        for d in sorted(set(occurrence_hints)):
            # keep within global window
            if d < cfg.start or d > cfg.end:
                continue
            rows.append(make_row(d))
    else:
        # one row based on sd/ed (already collapsed for all-day multi-day)
        if sd < cfg.start or sd > cfg.end:
            return []
        rows.append(make_row(sd))

    # Clean: drop range rows (defensive), then collapse duplicates by date
    before = len(rows)
    rows = drop_range_rows(rows, all_day_flag=all_day_flag)
    if debug_enabled() and len(rows) != before:
        log(f"[clean] dropped range rows: {before} -> {len(rows)}")

    before = len(rows)
    rows = collapse_duplicates_by_date(rows)
    if debug_enabled() and len(rows) != before:
        log(f"[clean] collapsed duplicates by date: {before} -> {len(rows)}")

    return rows


# -------------------------
# CSV write
# -------------------------

def write_csv(path: str, delimiter: str, rows: Iterable[Dict[str, str]]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADERS, delimiter=delimiter, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
            count += 1
    return count


# -------------------------
# Main
# -------------------------

def main() -> int:
    cfg = load_cfg()
    client = MecClient(cfg)

    ids, hints = collect_unique_event_ids_with_hints(client, cfg)
    log(f"[mec] Total unique event IDs: {len(ids)} (with occurrence hints for {len(hints)})")

    all_rows: List[Dict[str, str]] = []
    skipped_404 = 0

    for eid in ids:
        try:
            ev = client.get_event(eid)
        except requests.HTTPError as e:
            # If a missing event appears, skip it.
            if getattr(e.response, "status_code", None) == 404:
                skipped_404 += 1
                continue
            raise

        # Build rows. If we have hints for this event, use them.
        occ = hints.get(eid)
        rows = build_rows_for_event(ev, cfg, occurrence_hints=occ)
        all_rows.extend(rows)

    # Sort output by start_date (parsed)
    def sort_key(r: Dict[str, str]) -> Tuple[date, str]:
        d = parse_any_date(r.get("start_date", "")) or date(1900, 1, 1)
        return (d, r.get("site", ""))

    all_rows.sort(key=sort_key)

    count = write_csv(cfg.csv_path, cfg.delimiter, all_rows)
    log(f"Wrote {count} rows to {cfg.csv_path} (skipped_404={skipped_404})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
