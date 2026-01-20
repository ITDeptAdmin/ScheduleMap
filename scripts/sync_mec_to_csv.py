#!/usr/bin/env python3
"""
Sync Modern Events Calendar (MEC) events to a GitHub-tracked CSV.

Required env vars:
- MEC_BASE_URL: e.g. https://www.staging12.ramusa.org/wp-json/mec/v1.0
- MEC_TOKEN: MEC API key (sent as header: mec-token)
- CSV_PATH: output CSV filename in repo, e.g. "Clinic Master Schedule TEST.csv"

What this version fixes:
- Pulls facility/address/city/state/lat/lng from MEC "locations" block
- Pulls Services / Clinic Type / parking_date / parking_time from MEC "fields"
- Uses fallback keyword parsing ONLY if fields are missing (older events)
- Preserves your existing CSV header if the file already exists
"""

from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests


# ----------------------------
# Helpers
# ----------------------------

def norm(s: str) -> str:
    """Normalize keys/labels for matching."""
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[()]", "", s)
    return s

def strip_html(html: str) -> str:
    if not html:
        return ""
    # very light strip; good enough for fallbacks
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.I)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;?", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def pick_first_dict_value(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(d, dict) or not d:
        return None
    # keys may be strings like "260"
    first_key = sorted(d.keys(), key=lambda x: str(x))[0]
    val = d.get(first_key)
    return val if isinstance(val, dict) else None

def read_existing_header(csv_path: str) -> Optional[List[str]]:
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
            return header if header else None
        except StopIteration:
            return None

def header_index_map(header: List[str]) -> Dict[str, str]:
    """Map normalized header -> original header."""
    return {norm(h): h for h in header}

def set_by_alias(row: Dict[str, str], header_map: Dict[str, str], aliases: List[str], value: str) -> None:
    """Set a value into the CSV row using any matching header alias."""
    v = "" if value is None else str(value)
    for a in aliases:
        key = norm(a)
        if key in header_map:
            row[header_map[key]] = v
            return

def find_header_contains(header: List[str], *needles: str) -> Optional[str]:
    """Find a header column that contains all needles (case-insensitive)."""
    needles_n = [n.lower() for n in needles]
    for h in header:
        hn = h.lower()
        if all(n in hn for n in needles_n):
            return h
    return None

def yn(val: bool) -> str:
    return "Yes" if val else "No"


# ----------------------------
# MEC API calls
# ----------------------------

def mec_get(base_url: str, token: str, path: str, params: Optional[dict] = None) -> Any:
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    r = requests.get(url, headers={"mec-token": token}, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def fetch_event_ids(base_url: str, token: str) -> List[int]:
    """
    MEC endpoint usually returns upcoming events. We keep it simple:
    GET /events
    """
    data = mec_get(base_url, token, "events")
    # common shapes:
    # - list of events
    # - dict with "events": [...]
    events = data.get("events") if isinstance(data, dict) else data
    ids: List[int] = []
    if isinstance(events, list):
        for e in events:
            try:
                ids.append(int(e.get("ID") or e.get("id")))
            except Exception:
                continue
    return sorted(set(ids))

def fetch_event_detail(base_url: str, token: str, event_id: int) -> Dict[str, Any]:
    return mec_get(base_url, token, f"events/{event_id}")


# ----------------------------
# Field extraction
# ----------------------------

def extract_custom_fields(event_data: Dict[str, Any]) -> Dict[str, str]:
    """
    event_data is the "data" object inside the MEC event response.
    fields example:
      {"fields":[
        {"label":"Services","value":"Medical, Dental, Vision"},
        {"label":"Clinic Type","value":"Popup"},
        {"label":"parking_date","value":"2026-11-06"},
        {"label":"parking_time (type it in)","value":"no later than 11:59 p.m...."}
      ]}
    """
    out: Dict[str, str] = {}
    fields = event_data.get("fields", [])
    if not isinstance(fields, list):
        return out

    for f in fields:
        if not isinstance(f, dict):
            continue
        label = norm(str(f.get("label", "")))
        value = f.get("value", "")
        if value is None:
            value = ""
        value_s = str(value).strip()

        if label == "services":
            out["services"] = value_s
        elif label == "clinic type":
            out["clinic_type"] = value_s
        elif label == "parking_date":
            out["parking_date"] = value_s
        elif label.startswith("parking_time"):
            out["parking_time"] = value_s

    return out

def services_flags(services_value: str, fallback_text: str = "") -> Dict[str, bool]:
    """
    Use Services field if present. Only fall back to text scan if empty.
    """
    text = services_value.strip()
    source = text if text else fallback_text.lower()

    def has(word: str) -> bool:
        return word.lower() in source.lower()

    return {
        "medical": has("medical"),
        "dental": has("dental"),
        "vision": has("vision"),
        "dentures": has("denture"),
    }

def clinic_type_flags(clinic_type_value: str, fallback_text: str = "") -> Dict[str, bool]:
    """
    Use Clinic Type field if present. Only fall back if empty.
    """
    text = clinic_type_value.strip()
    source = text if text else fallback_text.lower()

    def has(word: str) -> bool:
        return word.lower() in source.lower()

    return {
        "popup": has("popup"),
        "telehealth": has("telehealth"),
    }

def extract_location(event_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Prefer event_data.locations[<id>] since your snippet shows it includes everything.
    """
    loc = pick_first_dict_value(event_data.get("locations", {})) or {}
    return {
        "facility": str(loc.get("name", "")).strip(),
        "address": str(loc.get("address", "")).strip(),
        "city": str(loc.get("city", "")).strip(),
        "state": str(loc.get("state", "")).strip(),
        "lat": str(loc.get("latitude", "")).strip(),
        "lng": str(loc.get("longitude", "")).strip(),
        "opening_hour": str(loc.get("opening_hour", "")).strip(),
    }

def extract_dates(event_data: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Return list of (start_date, end_date) occurrences.
    Your response typically has top-level "dates": [...]
    """
    dates = event_data.get("dates", [])
    out: List[Tuple[str, str]] = []
    if not isinstance(dates, list):
        return out
    for d in dates:
        if not isinstance(d, dict):
            continue
        start = (d.get("start") or {}).get("date")
        end = (d.get("end") or {}).get("date")
        if start:
            out.append((str(start), str(end or start)))
    return out


# ----------------------------
# CSV building
# ----------------------------

def build_rows_for_event(event_id: int, detail: Dict[str, Any], header: List[str]) -> List[Dict[str, str]]:
    """
    Build one CSV row per occurrence date range.
    """
    data = detail.get("data", {})
    post = data.get("post", {}) if isinstance(data, dict) else {}
    content_html = post.get("post_content") or post.get("post_content_filtered") or ""
    content_text = strip_html(str(content_html))

    # location + custom fields
    loc = extract_location(data)
    fields = extract_custom_fields(data)

    svc = services_flags(fields.get("services", ""), fallback_text=content_text)
    ctype = clinic_type_flags(fields.get("clinic_type", ""), fallback_text=content_text)

    parking_date = fields.get("parking_date", "").strip()
    parking_time = fields.get("parking_time", "").strip()

    title = str(data.get("title") or post.get("post_title") or "").strip()
    url = str(post.get("guid") or "").strip()
    # some MEC installs provide permalink elsewhere; try common alternatives
    if not url:
        url = str(data.get("permalink") or data.get("url") or "").strip()

    occurrences = extract_dates(data)
    if not occurrences:
        # still output one row if no dates found
        occurrences = [("", "")]

    header_map = header_index_map(header)
    rows: List[Dict[str, str]] = []

    for (start_date, end_date) in occurrences:
        row = {h: "" for h in header}

        # title/city fields vary by your CSV; set common ones if present
        set_by_alias(row, header_map, ["title", "event", "clinic", "name"], title)
        set_by_alias(row, header_map, ["facility"], loc["facility"])
        set_by_alias(row, header_map, ["address"], loc["address"])
        set_by_alias(row, header_map, ["city"], loc["city"])
        set_by_alias(row, header_map, ["state"], loc["state"])
        set_by_alias(row, header_map, ["lat", "latitude"], loc["lat"])
        set_by_alias(row, header_map, ["lng", "longitude", "lon"], loc["lng"])
        set_by_alias(row, header_map, ["opening_hour", "opening hour"], loc["opening_hour"])
        set_by_alias(row, header_map, ["url", "link", "permalink"], url)

        # dates
        set_by_alias(row, header_map, ["start_date", "start date", "date start"], start_date)
        set_by_alias(row, header_map, ["end_date", "end date", "date end"], end_date)

        # parking
        # if your CSV headers are different, we also try "contains" matching
        set_by_alias(row, header_map, ["parking_date", "parking date"], parking_date)
        set_by_alias(row, header_map, ["parking_time", "parking time"], parking_time)

        # service flags (Yes/No)
        set_by_alias(row, header_map, ["medical"], yn(svc["medical"]))
        set_by_alias(row, header_map, ["dental"], yn(svc["dental"]))
        set_by_alias(row, header_map, ["vision"], yn(svc["vision"]))
        set_by_alias(row, header_map, ["dentures"], yn(svc["dentures"]))

        # clinic type flags (Yes/No)
        # your CSV might have "POP-UP CLINIC" etc; try contains match too
        set_by_alias(row, header_map, ["popup", "pop-up", "pop up"], yn(ctype["popup"]))
        set_by_alias(row, header_map, ["telehealth"], yn(ctype["telehealth"]))

        # More flexible header matches if your CSV uses longer names
        popup_col = find_header_contains(header, "pop")
        tele_col = find_header_contains(header, "telehealth")
        if popup_col and not row.get(popup_col):
            row[popup_col] = yn(ctype["popup"])
        if tele_col and not row.get(tele_col):
            row[tele_col] = yn(ctype["telehealth"])

        # parking flexible match
        pdate_col = find_header_contains(header, "parking", "date")
        ptime_col = find_header_contains(header, "parking", "time")
        if pdate_col and not row.get(pdate_col):
            row[pdate_col] = parking_date
        if ptime_col and not row.get(ptime_col):
            row[ptime_col] = parking_time

        rows.append(row)

    return rows


def main() -> None:
    base_url = os.environ.get("MEC_BASE_URL", "").strip()
    token = os.environ.get("MEC_TOKEN", "").strip()
    csv_path = os.environ.get("CSV_PATH", "").strip()

    if not base_url or not token or not csv_path:
        raise SystemExit("Missing env vars. Need MEC_BASE_URL, MEC_TOKEN, CSV_PATH.")

    header = read_existing_header(csv_path)
    if not header:
        # Fallback header if the file doesn't exist yet
        header = [
            "title", "facility", "address", "city", "state", "lat", "lng",
            "medical", "dental", "vision", "dentures",
            "telehealth", "popup",
            "parking_date", "parking_time",
            "start_date", "end_date",
            "url",
        ]

    event_ids = fetch_event_ids(base_url, token)
    if not event_ids:
        raise SystemExit("No events returned from MEC /events endpoint.")

    all_rows: List[Dict[str, str]] = []
    for eid in event_ids:
        detail = fetch_event_detail(base_url, token, eid)
        all_rows.extend(build_rows_for_event(eid, detail, header))

    # Write CSV
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r)

    print(f"Wrote {len(all_rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
