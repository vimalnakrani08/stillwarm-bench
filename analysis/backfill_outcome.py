#!/usr/bin/env python3
"""Backfill `outcome` for warm rows that predate outcome stamping.

outcome is a PURE FUNCTION of recorded verdicts (no re-measurement):
  probe PASS + reuse PASS -> WORKS
  probe FAIL             -> SILENTLY-WRONG
  probe PASS + reuse FAIL -> SILENTLY-INEFFECTIVE
Only rows with mode in WARM_MODES, empty outcome, and BOTH verdicts present are
backfilled; each gets outcome_backfilled marker via labels? (schema has no labels col)
-> the backfill rule and count are documented in the dataset card."""
import csv, glob, os, sys
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "harness"))
from schema import CSV_FIELDS_V1
WARM = {"prefix_reuse", "ram_parked", "disk_restore", "d_restore"}
n = 0
touched = {}
for path in sorted(glob.glob(os.path.join(REPO, "results", "block*.csv"))):
    rows = list(csv.DictReader(open(path)))
    changed = 0
    for r in rows:
        r.pop(None, None)
        if (r["mode"] in WARM and not r.get("outcome")
                and r.get("probe_result") in ("PASS", "FAIL")
                and r.get("reuse_assert") in ("PASS", "FAIL")):
            if r["probe_result"] == "FAIL":
                r["outcome"] = "SILENTLY-WRONG"
            elif r["reuse_assert"] == "FAIL":
                r["outcome"] = "SILENTLY-INEFFECTIVE"
            else:
                r["outcome"] = "WORKS"
            changed += 1
    if changed:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS_V1)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in CSV_FIELDS_V1})
        touched[os.path.basename(path)] = changed
        n += changed
print({"backfilled_rows": n, "files": touched})
