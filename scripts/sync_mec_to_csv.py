#!/usr/bin/env python3
"""
sync_mec_to_csv.py

Robust MEC -> CSV sync.

Key behaviors:
- Calls /events with start_date+offset pagination and collects unique event IDs plus "occurrence hints".
- Fetches /events/{id} for each ID and emits one row per occurrence date (or a single row if none).
- If event is all-day: start_time/end_time are BLANK in CSV (per your requirement).
- If event is multi-day AND all-day: collapse to ONE row on end_date.
- Collapses duplicates by date within an event (keeps the most informative row).
- Handles MEC API 500s on /events by bumping start_date forward until the API responds.

Env vars:
  MEC_BASE_URL  (required)  e.g. https://example.com/wp-json/mec/v1
  MEC_TOKEN     (optional)  token if endpoint requires auth
  CSV_PATH      (optional)  default: Clinic Master Schedule for git.csv
  MEC_START     (optional)  default: 2010-01-01
  MEC_END       (optional)  default: 2030-12-31
  MEC_LIMIT     (optional)  default: 200
  MAX_PAGES     (optional)  default: 50
  CSV_DELIMITER (optional)  comma|tab|pipe  default: comma
  DATE_FORMAT   (optional)  mdy|ymd         default: mdy
  MEC_DEBUG     (optional)  1 enables debug logs
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
# Logging
# -------------------------

def _debug() -> bool:
    return os.getenv("MEC_DEBUG", "").strip() not in ("", "0", "false", "False")

def log(msg: str) -> None:
    print(msg, file=sys.stderr)


# -------------------------
# Config
# -------------------------

def parse_int(s: str, default: int) -> int:
    try:
        return int(str(s).strip())
    except Exception:
        return default

def parse_date_ymd(s: str, default: date) -> date:
    s = (s or "").strip()
    if not s:
        return default
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return default

def parse_delimiter(name: str) -> str:
    name = (name or "").strip().lower()
    if name in ("comma", ",", "csv", "com", "c"):
        return ","
    if name in ("tab", "\\t", "tsv", "t"):
        return "\t"
    if name in ("pipe", "|", "p"):
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

    # When MEC /events returns 500 for an old start_date, bump forward by this many days and retry.
    bump_days_on_500: int = 180
    max_bumps_on_500: int = 60  # 60 * 180d ~= 30 years max movement


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

    include_past = True
    include_ongoing = True

    log("[cfg] base_url=***")
    log(f"[cfg] start={start.isoformat()} end={end.isoformat()} limit={limit} max_pages={max_pages}")
    log(f"[cfg] include_past={1 if include_past else 0} include_ongoing={1 if include_ongoing else 0}")
    log(f"[cfg] csv_path={csv_path} delimiter={'COMMA' if delimiter==',' else repr(delimiter)} date_format={date_format}")

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

def truthy(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    s = str_or_empty(x).strip().lower()
    return s in ("1", "true", "yes", "y", "on")

def first_nonempty(*vals: Any) -> str:
    for v in vals:
        s = str_or_empty(v).strip()
        if s:
            return s
    return ""

def get_nested(d: Any, *keys: str) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur

def parse_any_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None

    m = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except Exception:
            pass

    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", s)
    if m:
        mm, dd, yyyy = m.groups()
        try:
            return date(int(yyyy), int(mm), int(dd))
        except Exception:
            return None

    return None

def parse_any_time(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    if re.search(r"\bAM\b|\bPM\b", s, re.IGNORECASE):
        return s
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
    return s

def format_date(d: date, fmt: str) -> str:
    if fmt == "ymd":
        return d.strftime("%Y-%m-%d")
    return f"{d.month}/{d.day}/{d.year}"


# -------------------------
# MEC client
# -------------------------

class MecClient:
    def __init__(self, cfg: Cfg) -> None:
        self.cfg = cfg
        self.sess = requests.Session()
        self.sess.headers.update({"Accept": "application/json"})
        if cfg.token:
            # Keep this minimal; too many auth headers can confuse some setups.
            self.sess.headers.update({"Authorization": f"Bearer {cfg.token}"})

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = self.cfg.base_url.rstrip("/") + path

        last_exc: Optional[Exception] = None
        for attempt in range(1, 4):
            try:
                r = self.sess.get(url, params=params, timeout=60)
                if _debug():
                    q = ""
                    if params:
                        q = " ?" + "&".join([f"{k}={params[k]}" for k in params])
                    log(f"[mec] GET ***/{path.lstrip('/')} {q} -> {r.status_code}")

                # Retry transient 5xx
                if r.status_code >= 500 and attempt < 3:
                    time.sleep(1)
                    continue

                r.raise_for_status()

                # Some MEC setups return invalid JSON occasionally; fall back to text.
                try:
                    return r.json()
                except Exception:
                    return {"_raw": r.text}

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
        return data if isinstance(data, dict) else {}


# -------------------------
# Extract from list page items
# -------------------------

def extract_list_item_start_date(it: Dict[str, Any]) -> Optional[date]:
    candidates = [
        it.get("start_date"),
        it.get("date_start"),
        it.get("start"),
        it.get("startDate"),
        it.get("date"),
        it.get("occurrence"),
        get_nested(it, "meta", "start_date"),
        get_nested(it, "meta", "date"),
    ]
    for c in candidates:
        d = parse_any_date(str_or_empty(c))
        if d:
            return d
    return None

def extract_list_item_occurrence_date(it: Dict[str, Any]) -> Optional[date]:
    candidates = [
        it.get("occurrence_date"),
        it.get("occurrence"),
        it.get("date"),
        get_nested(it, "meta", "occurrence_date"),
        get_nested(it, "meta", "date"),
    ]
    for c in candidates:
        d = parse_any_date(str_or_empty(c))
        if d:
            return d
    return None


# -------------------------
# Extract from event detail
# -------------------------

def extract_event_dates(ev: Dict[str, Any]) -> Tuple[date, date]:
    today = date.today()
    sd = parse_any_date(first_nonempty(
        ev.get("start_date"),
        ev.get("date_start"),
        get_nested(ev, "data", "start_date"),
        get_nested(ev, "meta", "start_date"),
        get_nested(ev, "dates", "start"),
    )) or today

    ed = parse_any_date(first_nonempty(
        ev.get("end_date"),
        ev.get("date_end"),
        get_nested(ev, "data", "end_date"),
        get_nested(ev, "meta", "end_date"),
        get_nested(ev, "dates", "end"),
    )) or sd

    return sd, ed

def is_all_day(ev: Dict[str, Any]) -> bool:
    return any(truthy(x) for x in (
        ev.get("all_day"),
        ev.get("allday"),
        ev.get("allDay"),
        get_nested(ev, "data", "all_day"),
        get_nested(ev, "meta", "all_day"),
    ))

def extract_times(ev: Dict[str, Any]) -> Tuple[str, str]:
    st = first_nonempty(
        ev.get("start_time"),
        ev.get("time_start"),
        get_nested(ev, "data", "start_time"),
        get_nested(ev, "meta", "start_time"),
    )
    et = first_nonempty(
        ev.get("end_time"),
        ev.get("time_end"),
        get_nested(ev, "data", "end_time"),
        get_nested(ev, "meta", "end_time"),
    )
    return parse_any_time(st), parse_any_time(et)

def extract_url(ev: Dict[str, Any]) -> str:
    return first_nonempty(ev.get("url"), ev.get("permalink"), ev.get("link"), get_nested(ev, "data", "url"))

def extract_cancelled(ev: Dict[str, Any]) -> bool:
    status = first_nonempty(ev.get("status"), ev.get("event_status"), get_nested(ev, "data", "status")).lower()
    if "cancel" in status:
        return True
    if truthy(ev.get("canceled")) or truthy(ev.get("cancelled")):
        return True
    return False

def extract_location(ev: Dict[str, Any]) -> Dict[str, str]:
    venue = get_nested(ev, "venue") or get_nested(ev, "location") or get_nested(ev, "data", "venue") or {}
    if isinstance(venue, str):
        venue = {"name": venue}

    lat = first_nonempty(ev.get("lat"), ev.get("latitude"), get_nested(venue, "lat"), get_nested(venue, "latitude"))
    lng = first_nonempty(ev.get("lng"), ev.get("lon"), ev.get("longitude"), get_nested(venue, "lng"), get_nested(venue, "longitude"))

    address = first_nonempty(ev.get("address"), get_nested(venue, "address"), get_nested(ev, "data", "address"))
    city = first_nonempty(ev.get("city"), get_nested(venue, "city"), get_nested(ev, "data", "city"))
    state = first_nonempty(ev.get("state"), get_nested(venue, "state"), get_nested(ev, "data", "state"))

    site = first_nonempty(
        ev.get("site"),
        get_nested(ev, "data", "site"),
        (city + (", " + state if city and state else "")) if (city or state) else "",
    )

    facility = first_nonempty(
        ev.get("facility"),
        get_nested(venue, "name"),
        get_nested(ev, "data", "facility"),
        ev.get("title"),
        ev.get("post_title"),
    )

    telehealth = truthy(first_nonempty(ev.get("telehealth"), get_nested(ev, "data", "telehealth")))
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
    pd = first_nonempty(ev.get("parking_date"), get_nested(ev, "data", "parking_date"), get_nested(ev, "meta", "parking_date"))
    pt = first_nonempty(ev.get("parking_time"), get_nested(ev, "data", "parking_time"), get_nested(ev, "meta", "parking_time"))
    return pd.strip(), pt.strip()

def parse_services(ev: Dict[str, Any]) -> Tuple[str, str, str, str]:
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
    medical = "Yes" if "medical" in blob else "No"
    dental = "Yes" if "dental" in blob else "No"
    vision = "Yes" if "vision" in blob else "No"
    dentures = "Yes" if "denture" in blob else "No"
    return medical, dental, vision, dentures


# -------------------------
# Pagination + recovery from 500
# -------------------------

def collect_unique_event_ids_with_hints(client: MecClient, cfg: Cfg) -> Tuple[List[int], Dict[int, List[date]]]:
    start_date = cfg.start
    offset = 0

    seen_pages = set()
    unique_ids: List[int] = []
    seen_ids = set()
    hints: Dict[int, List[date]] = {}

    page = 0
    bumps = 0

    while page < cfg.max_pages:
        page += 1
        key = (start_date.isoformat(), offset)
        if key in seen_pages:
            log(f"[mec] pagination loop detected at start_date={start_date} offset={offset}, stopping")
            break
        seen_pages.add(key)

        # ---- Attempt list call; if MEC returns 500, bump start_date forward and retry ----
        try:
            items = client.list_events_page(start_date=start_date, offset=offset)
        except requests.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            if status == 500:
                bumps += 1
                if bumps > cfg.max_bumps_on_500:
                    log(f"[mec] too many 500s while listing events; last start_date={start_date} offset={offset}")
                    raise

                new_start = start_date + timedelta(days=cfg.bump_days_on_500)
                log(f"[mec] 500 from /events at start_date={start_date} offset={offset}. Bumping start_date -> {new_start} and retrying.")
                start_date = new_start
                offset = 0
                # do not count this as a real pagination "page"
                page -= 1
                continue
            raise

        ids_this_page: List[int] = []
        max_date: Optional[date] = None
        hints_added = 0

        for it in items:
            eid = it.get("id") or it.get("event_id") or it.get("ID")
            try:
                eid_int = int(eid)
            except Exception:
                continue

            ids_this_page.append(eid_int)
            if eid_int not in seen_ids:
                seen_ids.add(eid_int)
                unique_ids.append(eid_int)

            occ = extract_list_item_occurrence_date(it) or extract_list_item_start_date(it)
            if occ:
                hints.setdefault(eid_int, [])
                if occ not in hints[eid_int]:
                    hints[eid_int].append(occ)
                    hints_added += 1

            d = extract_list_item_start_date(it)
            if d and (max_date is None or d > max_date):
                max_date = d

        log(
            f"[mec] list page {page}: start_date={start_date.isoformat()} offset={offset} "
            f"-> items={len(items)} ids={len(set(ids_this_page))} hints_added={hints_added} unique_total={len(unique_ids)}"
        )

        if not items:
            break

        # advance pagination
        if max_date is not None and max_date > start_date:
            start_date = max_date
            offset = 0
        else:
            offset += len(items)

    # Normalize hint ordering
    for eid in list(hints.keys()):
        hints[eid] = sorted(set(hints[eid]))

    return unique_ids, hints


# -------------------------
# Rows / cleaning
# -------------------------

CSV_HEADERS = [
    "canceled","lat","lng","address","city","state","site","facility","telehealth",
    "start_time","end_time","start_date","end_date","parking_date","parking_time",
    "medical","dental","vision","dentures","url",
]

def row_quality_score(r: Dict[str, str]) -> int:
    score = 0
    for k in ("start_time","end_time","address","facility","url"):
        if str_or_empty(r.get(k)).strip():
            score += 1
    for k in ("medical","dental","vision","dentures"):
        if r.get(k) == "Yes":
            score += 1
    return score

def collapse_duplicates_by_date(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    best: Dict[str, Dict[str, str]] = {}
    for r in rows:
        key = str_or_empty(r.get("start_date")).strip()
        if not key:
            continue
        if key not in best or row_quality_score(r) > row_quality_score(best[key]):
            best[key] = r
    def sk(s: str) -> date:
        return parse_any_date(s) or date(1900,1,1)
    return [best[k] for k in sorted(best.keys(), key=sk)]


def build_rows_for_event(ev: Dict[str, Any], cfg: Cfg, occurrence_hints: Optional[List[date]]) -> List[Dict[str, str]]:
    cancelled = extract_cancelled(ev)
    loc = extract_location(ev)
    parking_date, parking_time = extract_parking(ev)
    medical, dental, vision, dentures = parse_services(ev)
    url = extract_url(ev)

    sd, ed = extract_event_dates(ev)
    all_day_flag = is_all_day(ev)

    start_time, end_time = extract_times(ev)

    # IMPORTANT: all-day => blank times
    if all_day_flag:
        start_time, end_time = "", ""

    # IMPORTANT: multi-day all-day => single row on end_date
    if all_day_flag and sd != ed:
        sd = ed
        ed = ed

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

    rows: List[Dict[str, str]] = []
    if occurrence_hints:
        for d in sorted(set(occurrence_hints)):
            if d < cfg.start or d > cfg.end:
                continue
            rows.append(make_row(d))
    else:
        if sd < cfg.start or sd > cfg.end:
            return []
        rows.append(make_row(sd))

    rows = collapse_duplicates_by_date(rows)
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
            if getattr(e.response, "status_code", None) == 404:
                skipped_404 += 1
                continue
            raise

        rows = build_rows_for_event(ev, cfg, occurrence_hints=hints.get(eid))
        all_rows.extend(rows)

    # Sort final output by parsed date then site
    def sort_key(r: Dict[str, str]) -> Tuple[date, str]:
        d = parse_any_date(r.get("start_date", "")) or date(1900, 1, 1)
        return (d, r.get("site", ""))

    all_rows.sort(key=sort_key)

    count = write_csv(cfg.csv_path, cfg.delimiter, all_rows)
    log(f"Wrote {count} rows to {cfg.csv_path} (skipped_404={skipped_404})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
