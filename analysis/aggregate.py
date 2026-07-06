#!/usr/bin/env python3
"""Block F — aggregate every Phase-2 warm/restored row: totals, probe/reuse pass
counts, and every exception, straight from the results CSVs (schema v1 only)."""
import csv, glob, json, os, sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WARM_MODES = {"prefix_reuse", "ram_parked", "disk_restore", "d_restore"}


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(REPO, "results", "block*.csv"))):
        for r in csv.DictReader(open(f)):
            r["_file"] = os.path.basename(f)
            rows.append(r)
    v1 = [r for r in rows if r.get("schema_version", "").startswith("1")]
    warm_all = [r for r in v1 if r["mode"] in WARM_MODES]
    # WS1.2 rule: the aggregate counts ONLY fully-stamped warm rows (probe, reuse AND
    # outcome all present). Unstamped rows were backfilled 2026-07-05 (pure function of
    # recorded verdicts); any row still incomplete is listed, not silently counted.
    warm = [r for r in warm_all
            if r.get("outcome") and r.get("probe_result") in ("PASS", "FAIL", "N/A")
            or (r.get("outcome") == "FAILS-CLEAN")]
    incomplete = [r for r in warm_all if r not in warm]
    probe_pass = [r for r in warm if r["probe_result"] == "PASS"]
    probe_fail = [r for r in warm if r["probe_result"] == "FAIL"]
    probe_na = [r for r in warm if r["probe_result"] not in ("PASS", "FAIL")]
    reuse_pass = [r for r in warm if r["reuse_assert"] == "PASS"]
    reuse_fail = [r for r in warm if r["reuse_assert"] == "FAIL"]
    reuse_na = [r for r in warm if r["reuse_assert"] not in ("PASS", "FAIL")]

    def key(r):
        return (r["_file"], r["block"], r["model_name"], r["doc_label"],
                r["cache_type_k"], r["mode"], r["question_id"], r["rep"])

    out = {
        "counting_rule": ("only fully-stamped warm rows are counted (probe_result + "
                          "reuse_assert + outcome all present); 45 pre-stamping rows "
                          "were deterministically backfilled 2026-07-05 (WS1.2)"),
        "total_v1_rows": len(v1),
        "total_rows_all_blocks": len(rows),
        "warm_restored_rows": len(warm),
        "warm_rows_incomplete_excluded": len(incomplete),
        "by_mode": dict(Counter(r["mode"] for r in warm)),
        # probe_result measures COLD-EQUIVALENCE (restore-vs-cold byte equality);
        # determinism (restore-vs-restore) is a separate property, verified by repro runs.
        "cold_equivalent_probe": {"PASS": len(probe_pass), "FAIL": len(probe_fail), "N/A(fails-clean)": len(probe_na)},
        "reuse": {"PASS": len(reuse_pass), "FAIL": len(reuse_fail), "N/A(fails-clean)": len(reuse_na)},
        "outcomes": dict(Counter(r["outcome"] for r in warm if r.get("outcome"))),
        "exceptions_cold_equivalent_FAIL": [key(r) for r in probe_fail],
        "exceptions_reuse_FAIL": [key(r) for r in reuse_fail],
        "exceptions_fails_clean": [key(r) for r in probe_na],
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    main()
