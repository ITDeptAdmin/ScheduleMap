#!/usr/bin/env python3
# scripts/compare_schedule_csv.py
"""
Read-only comparison of a candidate schedule CSV against a baseline CSV
(and optionally the public website CSV). Used by the diagnostic workflow;
never writes to the repository, MEC, or the website.

Telehealth rows are compared per occurrence (event URL + start date); Popup
rows per event URL (one ranged row per event). Removals are split into
expired (end date before --today) and current/future.

Usage:
  python scripts/compare_schedule_csv.py --baseline A.csv --candidate B.csv \
      [--public C.csv] [--reconciliation R.json] [--sync-exit N] [--today YYYY-MM-DD] \
      [--expect-absent slug@YYYY-MM-DD,...] [--expect-present slug,...] \
      --out-md report.md --out-json report.json
Exit code is 0 unless an expectation fails or inputs are unreadable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

COMPARE_FIELDS = (
    "canceled", "lat", "lng", "address", "city", "state", "site", "title", "facility",
    "clinic_type", "start_time", "end_time", "start_date", "end_date",
    "parking_date", "parking_time", "medical", "dental", "vision", "dentures",
)


def parse_date(s: str) -> Optional[date]:
    s = (s or "").strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def load(path: str) -> Tuple[List[str], List[Dict[str, str]], str]:
    with open(path, "rb") as fb:
        digest = hashlib.sha256(fb.read().replace(b"\r\n", b"\n")).hexdigest()
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows, digest


def slug(url: str) -> str:
    return (url or "").strip().rstrip("/").rsplit("/", 1)[-1].lower()


def key(r: Dict[str, str]) -> Tuple[str, str]:
    """Occurrence key: Telehealth = (slug, start date); Popup = (slug, '')."""
    if (r.get("telehealth") or "").strip().lower() == "yes":
        d = parse_date(r.get("start_date", ""))
        return (slug(r.get("url", "")), d.isoformat() if d else r.get("start_date", ""))
    return (slug(r.get("url", "")), "")


def summarize(rows: List[Dict[str, str]]) -> Dict[str, int]:
    return {
        "rows": len(rows),
        "telehealth": sum(1 for r in rows if r.get("telehealth") == "Yes"),
        "popup": sum(1 for r in rows if r.get("telehealth") == "No"),
        "canceled": sum(1 for r in rows if r.get("canceled") == "Yes"),
        "events": len({slug(r.get("url", "")) for r in rows}),
    }


def describe(r: Dict[str, str]) -> str:
    rng = r.get("start_date", "")
    if r.get("end_date") and r.get("end_date") != r.get("start_date"):
        rng += "-" + r["end_date"]
    times = f" {r.get('start_time')}-{r.get('end_time')}" if r.get("start_time") else ""
    return f"{r.get('clinic_type', '')} | {rng}{times} | {r.get('facility', '')} | {r.get('url', '')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--public")
    ap.add_argument("--reconciliation")
    ap.add_argument("--sync-exit", type=int, default=0)
    ap.add_argument("--today")
    ap.add_argument("--expect-absent", default="")
    ap.add_argument("--expect-present", default="")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args()

    today = parse_date(a.today) if a.today else date.today()
    b_head, base, b_sha = load(a.baseline)
    c_head, cand, c_sha = load(a.candidate)
    candidate_written = a.sync_exit == 0

    b_map = {key(r): r for r in base}
    c_map = {key(r): r for r in cand}
    removed = [b_map[k] for k in b_map if k not in c_map]
    added = [c_map[k] for k in c_map if k not in b_map]
    changed = []
    for k in b_map.keys() & c_map.keys():
        diffs = {f: (b_map[k].get(f, ""), c_map[k].get(f, "")) for f in COMPARE_FIELDS if b_map[k].get(f, "") != c_map[k].get(f, "")}
        if diffs:
            changed.append((c_map[k], diffs))

    def expired(r: Dict[str, str]) -> bool:
        end = parse_date(r.get("end_date", "")) or parse_date(r.get("start_date", ""))
        return bool(end and end < today)

    removed_expired = [r for r in removed if expired(r)]
    removed_current = [r for r in removed if not expired(r)]

    checks: List[Tuple[str, bool]] = []
    for tok in [t.strip() for t in a.expect_absent.split(",") if t.strip()]:
        s, _, d = tok.partition("@")
        present = any(slug(r.get("url", "")) == s.lower() and (parse_date(r.get("start_date", "")) or date.min).isoformat() == d for r in cand)
        checks.append((f"{s} {d} absent from candidate", not present))
    for s in [t.strip().lower() for t in a.expect_present.split(",") if t.strip()]:
        checks.append((f"{s} present in candidate", any(slug(r.get("url", "")) == s for r in cand)))
    checks.append(("candidate header matches baseline header", c_head == b_head))

    public = None
    if a.public and os.path.isfile(a.public) and os.path.getsize(a.public) > 0:
        try:
            p_head, prow, p_sha = load(a.public)
            public = {"rows": len(prow), "sha256": p_sha, "matches_baseline": p_sha == b_sha, "matches_candidate": p_sha == c_sha}
        except Exception as exc:  # report, do not fail the comparison
            public = {"error": exc.__class__.__name__}

    recon = None
    if a.reconciliation and os.path.isfile(a.reconciliation):
        with open(a.reconciliation, encoding="utf-8") as f:
            recon = json.load(f)

    report = {
        "today": today.isoformat(),
        "sync_exit_code": a.sync_exit,
        "candidate_written": candidate_written,
        "baseline": {**summarize(base), "sha256": b_sha},
        "candidate": {**summarize(cand), "sha256": c_sha},
        "public": public,
        "added": [describe(r) for r in added],
        "removed_expired": [describe(r) for r in removed_expired],
        "removed_current_or_future": [describe(r) for r in removed_current],
        "changed": [{"row": describe(r), "fields": d} for r, d in changed],
        "checks": [{"check": n, "ok": ok} for n, ok in checks],
        "sync_reconciliation": recon,
    }
    with open(a.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    L: List[str] = ["# MEC -> CSV candidate comparison (read-only diagnostic)", ""]
    if not candidate_written:
        L += [f"**Production validation FAILED (exit {a.sync_exit}). The sync refused to write, so "
              "`candidate.csv` is still a copy of the baseline; see `sync.log` and the reconciliation below.**", ""]
    L += [f"Today (site date used for expiry): {today.isoformat()}", "",
          "| | rows | Telehealth | Popup | canceled | events |", "|---|---|---|---|---|---|"]
    for name in ("baseline", "candidate"):
        s = report[name]
        L.append(f"| {name} | {s['rows']} | {s['telehealth']} | {s['popup']} | {s['canceled']} | {s['events']} |")
    if public:
        L += ["", f"Public CSV: {public}"]
    for title, items in (("Added", report["added"]), ("Removed - expired (allowed)", report["removed_expired"]),
                         ("Removed - current/future", report["removed_current_or_future"])):
        L += ["", f"## {title} ({len(items)})"] + [f"- {x}" for x in items]
    L += ["", f"## Changed in place ({len(changed)})"]
    for r, d in changed:
        L.append(f"- {describe(r)}: " + "; ".join(f"{k}: {v[0]!r} -> {v[1]!r}" for k, v in d.items()))
    if recon:
        L += ["", "## Sync reconciliation (from sync_mec_to_csv.py)", "",
              f"pruned expired: {recon.get('pruned_expired')}; missing current/future: {len(recon.get('missing', []))}"]
        L += [f"- {m['status']}: {m['start_date']} {m['start_time']} {m['facility']} {m['url']} -- {m['reason']}" for m in recon.get("missing", [])]
    L += ["", "## Checks"] + [f"- {'PASS' if ok else 'FAIL'}: {n}" for n, ok in checks]
    with open(a.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if all(ok for _, ok in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
