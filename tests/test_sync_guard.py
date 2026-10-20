"""Focused checks for the occurrence-loss reconciliation and existing validations.

Runs scripts/sync_mec_to_csv.py main() end-to-end against a fake in-memory MEC/WP REST
backend (no network, no credentials). The fake mirrors how the RAM schedule plugin
writes MEC fields:
  - plugin full save:    mec_in_days, _ram_telehealth_mec_in_days, mec_not_in_days (removed)
  - cancel/manual path:  mec_in_days only
Usage (from the repository root; needs only git and Python 3.9+):
    python tests/test_sync_guard.py [--reference-ref ce758a5]

--reference-ref is the commit whose scripts/sync_mec_to_csv.py (pre-fix) and
"Clinic Master Schedule for git.csv" are used for the byte-identical output check
and the historical Oct 20, 2026 reconciliation fixture. Exit code 0 = all passed.
"""
import argparse
import csv
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
import types
from contextlib import redirect_stdout
from datetime import date

_ap = argparse.ArgumentParser()
_ap.add_argument("--reference-ref", default="ce758a5eeedf9f6cc70b24f169ac6d33b53606f3")
_args = _ap.parse_args()
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCHED = os.path.join(REPO, "scripts", "sync_mec_to_csv.py")
_REF_DIR = tempfile.mkdtemp(prefix="sync_ref_")


def _git_show(path, out_name):
    out = os.path.join(_REF_DIR, out_name)
    data = subprocess.run(["git", "-C", REPO, "show", f"{_args.reference_ref}:{path}"],
                          check=True, capture_output=True).stdout
    with open(out, "wb") as fh:
        fh.write(data)
    return out


ORIGINAL = _git_show("scripts/sync_mec_to_csv.py", "sync_reference.py")
COMMITTED = _git_show("Clinic Master Schedule for git.csv", "reference.csv")
try:
    import requests  # noqa: F401
except ImportError:  # network is fully faked; a stub module is enough
    _stub = types.ModuleType("requests")
    _stub.Session = object
    _stub.Response = object
    sys.modules["requests"] = _stub


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sync = load(PATCHED, "sync_patched")
orig = load(ORIGINAL, "sync_orig")
TODAY = date(2026, 10, 20)
sync._today_site = lambda: TODAY
orig_build = sync._build_rows_for_event


# ---------------- fake backend ----------------
class Resp:
    def __init__(self, code, payload=None, headers=None, url=""):
        self.status_code, self._p, self.headers, self.url = code, payload, headers or {}, url
        self.text = "" if payload is None else str(payload)[:200]

    def json(self):
        return self._p


def popup(eid, slug, start, end, canceled=False, facility=None):
    return {"type": "popup", "eid": eid, "slug": slug, "start": start, "end": end,
            "canceled": canceled, "facility": facility or slug.title()}


def tele(eid, slug, dates, *, plugin=None, not_in=(), facility=None, times=("09-00-AM", "03-00-PM"), table=None):
    """dates = current mec_in_days; plugin = _ram_telehealth_mec_in_days (None -> same as dates);
    table = MEC events-table data.mec.days (None -> same as dates; report-only evidence)."""
    return {"type": "tele", "eid": eid, "slug": slug, "dates": list(dates),
            "plugin": list(dates) if plugin is None else list(plugin), "not_in": list(not_in),
            "table": None if table is None else list(table),
            "canceled": False, "facility": facility or slug.title(), "times": times}


def in_days(dates, times):
    return ",".join(f"{d}:{d}:{times[0]}:{times[1]}" for d in dates)


def detail_payload(ev):
    loc = {"1": {"name": ev["facility"], "address": "1 Main St", "city": "Knoxville",
                 "state": "Tennessee", "latitude": "35.9", "longitude": "-83.9"}}
    data = {"ID": ev["eid"], "title": ev["slug"], "permalink": f"https://www.ramusa.org/events/{ev['slug']}/",
            "locations": loc,
            "meta": {"mec_event_status": "EventCancelled" if ev["canceled"] else "EventScheduled"}}
    if ev["type"] == "tele":
        data["fields"] = [{"label": "Services", "value": "Medical"}, {"label": "Clinic Type", "value": "Telehealth"}]
        data["meta"]["mec_in_days"] = in_days(ev["dates"], ev["times"])
        data["meta"]["_ram_telehealth_mec_in_days"] = in_days(ev["plugin"], ev["times"])
        data["meta"]["mec_not_in_days"] = ",".join(ev["not_in"])
        data["mec"] = {"days": in_days(ev["dates"] if ev.get("table") is None else ev["table"], ev["times"])}
    else:
        data["fields"] = [{"label": "Services", "value": "Medical, Dental, Vision"}]
    return {"data": data}


def list_items(ev):
    if ev["type"] == "popup":  # MEC lists each day of a multi-day popup as an occurrence
        return [{"ID": ev["eid"], "start": {"date": ev["start"]}, "end": {"date": ev["start"]}},
                {"ID": ev["eid"], "start": {"date": ev["end"]}, "end": {"date": ev["end"]}}]
    return [{"ID": ev["eid"], "start": {"date": d}, "end": {"date": d}} for d in ev["dates"]]


class Backend:
    def __init__(self, events, *, wp_status=200, wp_total_delta=0, mec_list_hide=(), detail_404=()):
        self.events = {e["eid"]: e for e in events}
        self.wp_status, self.wp_total_delta = wp_status, wp_total_delta
        self.mec_list_hide, self.detail_404 = set(mec_list_hide), set(detail_404)

    def get(self, url, headers=None, params=None, timeout=None):
        if "/wp/v2/mec-events" in url:
            if self.wp_status != 200:
                return Resp(self.wp_status, url=url)
            if params and params.get("page") != "1":
                return Resp(400, url=url)
            ids = [{"id": e} for e in self.events]
            return Resp(200, ids, {"X-WP-TotalPages": "1", "X-WP-Total": str(len(ids) + self.wp_total_delta)}, url=url)
        path = url.split("/mec/v1.0/", 1)[1]
        if path == "events":
            items = [it for e in self.events.values() if e["eid"] not in self.mec_list_hide for it in list_items(e)]
            return Resp(200, {"events": items, "pagination": {}}, url=url)
        eid = int(path.split("/")[1])
        if eid in self.detail_404 or eid not in self.events:
            return Resp(404, url=url)
        return Resp(200, detail_payload(self.events[eid]), url=url)


PROD_ENV = {"MIN_ROW_COUNT": "1", "MAX_UNEXPLAINED_FUTURE_ROW_LOSS": "2", "MAX_FUTURE_ROW_LOSS": "15"}


def run(mod, backend, csv_path, extra_env=None):
    env = {"MEC_BASE_URL": "https://example.test/wp-json/mec/v1.0", "MEC_TOKEN": "x", "CSV_PATH": csv_path,
           "CSV_DELIMITER": "comma", "DATE_FORMAT": "mdy", "REQUIRE_WP_REST_DISCOVERY": "1",
           "MEC_DEBUG": "", **PROD_ENV}
    env.update(extra_env or {})
    saved = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        os.environ[k] = v
    mod._session = lambda: backend
    buf, code = io.StringIO(), 0
    try:
        with redirect_stdout(buf):
            mod.main()
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if isinstance(exc.code, str):
            buf.write(exc.code + chr(10))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return code, buf.getvalue()


def read_rows(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


# ---------------- data ----------------
TH_B = [f"2026-{m:02d}-{d:02d}" for m, d in [(11, 2), (11, 9), (11, 16), (11, 23), (12, 7), (12, 14), (12, 21)]]
BASE_EVENTS = [
    popup(1, "pop-past", "2026-03-01", "2026-03-02"),
    popup(2, "pop-future", "2026-12-11", "2026-12-12"),
    popup(3, "pop-future-2", "2027-03-13", "2027-03-14"),
    tele(10, "rio-townsend-telehealth", ["2026-10-06", "2026-11-03", "2026-12-01"], times=("09-30-AM", "02-30-PM")),
    tele(11, "care-cuts", TH_B),
    tele(12, "sevier-co-food-ministries-telehealth", ["2026-10-22", "2026-11-19", "2026-12-17"]),
    tele(13, "church-street", ["2026-10-15", "2026-11-05"]),
    # display duplicate of care-cuts on 11/02 under a second event (collapsed by global dedupe)
    tele(14, "care-cuts-alias", ["2026-11-02"], facility="Care-Cuts"),
]

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"\n      {detail}" if detail and not cond else ""))


def ev_with(changes, base=BASE_EVENTS):
    out = []
    for e in base:
        if e["eid"] in changes and changes[e["eid"]] is None:
            continue
        e = dict(e)
        e.update(changes.get(e["eid"], {}))
        out.append(e)
    return out


def scenario(name, events_now, expect_code, *, backend_kw=None, env=None, expect_in_out=(), extra=None, build=None):
    tmp = tempfile.mkdtemp()
    try:
        p = os.path.join(tmp, "schedule.csv")
        code0, out0 = run(sync, Backend(BASE_EVENTS), p, {"MAX_UNEXPLAINED_FUTURE_ROW_LOSS": "", "MAX_FUTURE_ROW_LOSS": ""})
        assert code0 == 0, out0
        before = open(p, "rb").read()
        if build:
            sync._build_rows_for_event = build
        try:
            code, out = run(sync, Backend(events_now, **(backend_kw or {})), p, env)
        finally:
            sync._build_rows_for_event = orig_build
        after = open(p, "rb").read()
        ok = code == expect_code and ((after == before) if expect_code else True)
        missing_text = [t for t in expect_in_out if t not in out]
        check(name, ok and not missing_text,
              f"code={code} csv_unchanged={after == before} missing_text={missing_text}\n" + out[-1800:])
        if extra:
            extra(out, read_rows(p))
        return out
    finally:
        shutil.rmtree(tmp)


# 1. Legitimate past-date pruning (plugin save removes passed dates from both stores).
scenario("past-date pruning passes with no manual work",
         ev_with({10: {"dates": ["2026-11-03", "2026-12-01"], "plugin": ["2026-11-03", "2026-12-01"]},
                  13: {"dates": ["2026-11-05"], "plugin": ["2026-11-05"]}}), 0,
         expect_in_out=["2 expired row(s) pruned", "0 current/future occurrence(s) missing"])

# 2. The two confirmed future removals, modeled on the evidence:
#    Sevier 10/22 via plugin full save (post modified; mec_not_in_days) -> explained
#    RIO 12/1 via cancel/manual path (meta only, no post modified bump) -> unexplained (1 <= 2)
confirmed = ev_with({
    12: {"dates": ["2026-11-19", "2026-12-17"], "plugin": ["2026-11-19", "2026-12-17"], "not_in": ["2026-10-22"]},
    10: {"dates": ["2026-11-03"], "plugin": ["2026-10-06", "2026-11-03", "2026-12-01"]},
})
scenario("confirmed removals pass: Sevier 10/22 explained, RIO 12/1 unexplained within budget", confirmed, 0,
         expect_in_out=["missing (explained): 10/22/2026", "excluded in MEC mec_not_in_days",
                        "missing (unexplained): 12/1/2026", "1 explained, 1 unexplained"])
scenario("RIO 12/1 explained if plugin copy also lacks it",
         ev_with({10: {"dates": ["2026-11-03"], "plugin": ["2026-11-03"]}}), 0,
         expect_in_out=["removed from mec_in_days and the RAM plugin schedule copy"])

# 2b. Real MEC state seen by diagnostic run 39481555398 (Oct 20, 2026):
#     RIO mec_in_days = {10/6} only; plugin copy + MEC events table still {10/6, 11/3, 12/1}.
#     Sevier mec_in_days/events table = {11/19, 12/17}; plugin copy still has 10/22.
real_state = ev_with({
    10: {"dates": ["2026-10-06"], "plugin": ["2026-10-06", "2026-11-03", "2026-12-01"],
         "table": ["2026-10-06", "2026-11-03", "2026-12-01"]},
    12: {"dates": ["2026-11-19", "2026-12-17"], "plugin": ["2026-10-22", "2026-11-19", "2026-12-17"]},
})
scenario("real Oct 20 MEC state (RIO lost 11/3 + 12/1, Sevier 10/22) blocks: 3 unexplained > 2", real_state, 1,
         expect_in_out=["missing (unexplained): 11/3/2026", "missing (unexplained): 12/1/2026",
                        "missing (unexplained): 10/22/2026", '"in_mec_events_table": true',
                        '"event_mec_in_days": ["2026-10-06"]'])
corrected = ev_with({
    10: {"dates": ["2026-10-06", "2026-11-03"], "plugin": ["2026-10-06", "2026-11-03", "2026-12-01"]},
    12: {"dates": ["2026-11-19", "2026-12-17"], "plugin": ["2026-10-22", "2026-11-19", "2026-12-17"]},
})
scenario("after RIO 11/3 is restored in MEC: only the 2 confirmed removals remain and pass", corrected, 0,
         expect_in_out=["2 unexplained (max 2)", "missing (unexplained): 12/1/2026", "missing (unexplained): 10/22/2026"])

# 2c. Diagnostic candidate copy: written even when validation fails; CSV_PATH untouched.
tmp = tempfile.mkdtemp()
p = os.path.join(tmp, "schedule.csv")
run(sync, Backend(BASE_EVENTS), p, {"MAX_UNEXPLAINED_FUTURE_ROW_LOSS": "", "MAX_FUTURE_ROW_LOSS": ""})
before = open(p, "rb").read()
rp = os.path.join(tmp, "reconciliation.json")
code, out = run(sync, Backend(real_state), p, {"RECONCILIATION_REPORT_PATH": rp})
cand_path = os.path.join(tmp, "reconciliation_candidate.csv")
cand_rows = read_rows(cand_path) if os.path.exists(cand_path) else []
check("diagnostic report + candidate copy written on failure, CSV_PATH unchanged",
      code == 1 and open(p, "rb").read() == before and os.path.exists(rp) and len(cand_rows) > 0
      and list(cand_rows[0].keys()) == sync.DEFAULT_HEADER
      and [r["start_date"] for r in cand_rows if "rio" in r["url"]] == ["10/6/2026"])
os.remove(cand_path)
code, out = run(sync, Backend(BASE_EVENTS), p)
check("no diagnostic copy without RECONCILIATION_REPORT_PATH", code == 0 and not os.path.exists(cand_path))
shutil.rmtree(tmp)

# 3. Unexpected missing future dates.
scenario("3 unexplained future removals (> 2) block",
         ev_with({11: {"dates": TH_B[:1] + TH_B[4:], "plugin": TH_B}}), 1,
         expect_in_out=["3 unexplained missing occurrence(s) exceed MAX_UNEXPLAINED_FUTURE_ROW_LOSS=2"])
scenario("mec_in_days wiped (plugin copy intact) blocks",
         ev_with({11: {"dates": [], "plugin": TH_B}}), 1, expect_in_out=["unexplained missing occurrence(s) exceed"])
def drop_one_row(cfg, d, occ_hints=None):
    rows = orig_build(cfg, d, occ_hints=occ_hints)
    return [r for r in rows if r["_start_ymd"] != "2026-11-09"] if d["data"]["ID"] == 11 else rows


scenario("sync regression (date still in mec_in_days, row missing) always blocks", BASE_EVENTS, 1, build=drop_one_row,
         expect_in_out=["missing (regression): 11/9/2026", "still scheduled in MEC but missing from the CSV"])
big = [tele(30 + i, f"series-{i}", [f"2026-11-{d:02d}" for d in range(10, 16)]) for i in range(3)]  # 18 future rows
tmp = tempfile.mkdtemp()
p = os.path.join(tmp, "schedule.csv")
run(sync, Backend(BASE_EVENTS + big), p, {"MAX_UNEXPLAINED_FUTURE_ROW_LOSS": "", "MAX_FUTURE_ROW_LOSS": ""})
code, out = run(sync, Backend(BASE_EVENTS), p)  # all three series unpublished at once: 18 explained misses
check("18 explained misses in one sync exceed total cap 15 and block",
      code == 1 and "18 missing current/future occurrence(s) exceed MAX_FUTURE_ROW_LOSS=15" in out, out[-600:])
code, out = run(sync, Backend(BASE_EVENTS + big[:2]), p)  # one series (6) unpublished: within cap
check("6 explained misses (one whole series unpublished) pass", code == 0 and "6 explained, 0 unexplained" in out, out[-600:])
shutil.rmtree(tmp)

# 4. Missing events and detail 404 responses.
scenario("unpublished Telehealth event (final cancellation -> draft) explained", ev_with({12: None}), 0,
         expect_in_out=["event no longer published", "3 explained, 0 unexplained"])
scenario("detail 404 for a published event blocks (unexplained > 2)", BASE_EVENTS, 1,
         backend_kw={"detail_404": {11}}, expect_in_out=["a discovered event returned 404"])
scenario("detail 404 small loss stays within budget but is reported", BASE_EVENTS, 0,
         backend_kw={"detail_404": {2}}, expect_in_out=["missing (unexplained): 12/11/2026"])

# 5. Required WordPress discovery failures (unchanged + new completeness check).
for label, kw, text in (("WP REST 500", {"wp_status": 500}, "unexpected HTTP 500"),
                        ("WP REST 404", {"wp_status": 404}, "endpoint returned HTTP 404"),
                        ("WP REST partial list vs X-WP-Total", {"wp_total_delta": 3}, "X-WP-Total reported")):
    scenario(f"{label} blocks (REQUIRE_WP_REST_DISCOVERY)", BASE_EVENTS, 1, backend_kw=kw, expect_in_out=[text])
scenario("WP REST recovers event missing from MEC pagination", BASE_EVENTS, 0, backend_kw={"mec_list_hide": {3}},
         extra=lambda out, rows: check("  recovered event has its row", any("pop-future-2" in r["url"] for r in rows)))

# 6. Missing or malformed baseline -> fail closed, CSV untouched.
GOOD_HEADER = ",".join(sync.DEFAULT_HEADER)
bad_baselines = {
    "missing baseline": None,
    "empty file": b"",
    "header only": (GOOD_HEADER + "\n").encode(),
    "wrong header": b"a,b,c\n1,2,3\n",
    "wrong field count": (GOOD_HEADER + "\nNo,1,2\n").encode(),
    "unparseable date": (GOOD_HEADER + "\n" + ",".join(["No"] + [""] * 12 + ["13/45/2026"] + [""] * 8) + "\n").encode(),
    "undecodable bytes": (GOOD_HEADER + "\n").encode() + b"\xff\xfe\xfa,\n",
}
for label, content in bad_baselines.items():
    tmp = tempfile.mkdtemp()
    p = os.path.join(tmp, "schedule.csv")
    if content is not None:
        open(p, "wb").write(content)
    code, out = run(sync, Backend(BASE_EVENTS), p)
    unchanged = (not os.path.exists(p)) if content is None else open(p, "rb").read() == content
    check(f"baseline {label} -> fails closed, CSV not written", code == 1 and "[FAIL] Baseline CSV" in out and unchanged,
          out[-500:])
    shutil.rmtree(tmp)

# 7. Date changes and Popup ranges.
scenario("Telehealth date moved via plugin save is explained, new date added",
         ev_with({12: {"dates": ["2026-10-29", "2026-11-19", "2026-12-17"], "plugin": ["2026-10-29", "2026-11-19", "2026-12-17"]}}), 0,
         expect_in_out=["missing (explained): 10/22/2026"],
         extra=lambda out, rows: check("  10/29 row present", any("sevier" in r["url"] and r["start_date"] == "10/29/2026" for r in rows)))
scenario("Telehealth time change on same date is not a loss",
         ev_with({12: {"times": ("10-00-AM", "02-00-PM")}}), 0,
         expect_in_out=["0 current/future occurrence(s) missing"],
         extra=lambda out, rows: check("  new times written", any("sevier" in r["url"] and r["start_time"] == "10:00 AM" for r in rows)))
scenario("Popup range change is not a loss and range is preserved",
         ev_with({2: {"start": "2026-12-18", "end": "2026-12-20"}}), 0,
         expect_in_out=["0 current/future occurrence(s) missing"],
         extra=lambda out, rows: check("  popup range 12/18-12/20",
                                       any("pop-future/" in r["url"] and (r["start_date"], r["end_date"]) == ("12/18/2026", "12/20/2026") for r in rows)))

# 8. Existing cancellation behavior.
scenario("Popup cancellation keeps row with canceled=Yes", ev_with({2: {"canceled": True}}), 0,
         expect_in_out=["0 current/future occurrence(s) missing"],
         extra=lambda out, rows: check("  canceled=Yes", any("pop-future/" in r["url"] and r["canceled"] == "Yes" for r in rows)))

# 9. Other validations unchanged.
scenario("MIN_ROW_COUNT still blocks", BASE_EVENTS, 1, env={"MIN_ROW_COUNT": "999"}, expect_in_out=["below MIN_ROW_COUNT=999"])
scenario("checks off when both env vars unset (staging/back-compat)", ev_with({11: None}), 0,
         env={"MAX_UNEXPLAINED_FUTURE_ROW_LOSS": "", "MAX_FUTURE_ROW_LOSS": ""})
scenario("map-eligible zero-row guard still blocks", BASE_EVENTS, 1, expect_in_out=["Map-eligible zero-row validation failed"],
         build=lambda cfg, d, occ_hints=None: [] if d["data"]["ID"] == 2 else orig_build(cfg, d, occ_hints=occ_hints))

# 10. Row generation unchanged: patched vs origin/main output byte-identical.
for label, evs in (("base data", BASE_EVENTS), ("confirmed-removal data", confirmed)):
    tmp = tempfile.mkdtemp()
    a, b = os.path.join(tmp, "a.csv"), os.path.join(tmp, "b.csv")
    ca, _ = run(sync, Backend(evs), a, {"MAX_UNEXPLAINED_FUTURE_ROW_LOSS": "", "MAX_FUTURE_ROW_LOSS": ""})
    cb, _ = run(orig, Backend(evs), b, {})
    same = ca == cb == 0 and open(a, "rb").read() == open(b, "rb").read()
    check(f"CSV byte-identical to origin/main script ({label})", same)
    if label == "base data":
        rows = read_rows(a)
        check("  header unchanged", list(rows[0].keys()) == sync.DEFAULT_HEADER)
        check("  Telehealth times kept (9:30 AM-2:30 PM)",
              any("rio" in r["url"] and (r["start_time"], r["end_time"]) == ("9:30 AM", "2:30 PM") for r in rows))
        check("  Popup range kept (12/11-12/12, blank times)",
              any("pop-future/" in r["url"] and (r["start_date"], r["end_date"], r["start_time"]) == ("12/11/2026", "12/12/2026", "") for r in rows))
        check("  display duplicate collapsed (care-cuts-alias 11/2 merged)",
              not any("care-cuts-alias" in r["url"] for r in rows))
        check("  links are plain permalinks", all(r["url"].startswith("https://www.ramusa.org/events/") and "?" not in r["url"] for r in rows))
    shutil.rmtree(tmp)

# 11. Real committed CSV as baseline (INFERRED candidate, labelled as such).
base = read_rows(COMMITTED)
remove = {("rio-townsend-telehealth", "10/6/2026"), ("rio-townsend-telehealth", "12/1/2026"),
          ("hamblen-county-central-services-telehealth", "10/8/2026"),
          ("knoxville-tn-mobile-th-carecuts", "10/9/2026"), ("knoxville-tn-mobile-th-carecuts", "10/16/2026"),
          ("appalachian-ministries-of-the-smokies-telehealth", "10/13/2026"),
          ("sustainable-future-center-telehealth", "10/14/2026"),
          ("church-street-united-methodist-church", "10/15/2026"),
          ("sevier-co-food-ministries-telehealth", "10/22/2026")}
cand = [r for r in base if (r["url"].rstrip("/").rsplit("/", 1)[-1], r["start_date"]) not in remove]
cand.append(dict(base[0], url="https://www.ramusa.org/events/brookfield-ohio/", start_date="9/25/2027",
                 end_date="9/26/2027", telehealth="No", clinic_type="Popup Clinic"))
findings = sync._reconcile_future_occurrences(base, cand, {}, TODAY, any_detail_404=False)
names = sorted((f["row"]["url"].rstrip("/").rsplit("/", 1)[-1], f["row"]["start_date"]) for f in findings)
check("real committed CSV: exact matching finds only RIO 12/1 + Sevier 10/22 as current/future misses",
      len(cand) == 123 and names == [("rio-townsend-telehealth", "12/1/2026"), ("sevier-co-food-ministries-telehealth", "10/22/2026")],
      str(names))

print()
failed = [n for n, ok in results if not ok]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
