#!/usr/bin/env python3
"""Schema v1.1 upgrade (additive only).

Rewrites every results/block*.csv with the v1.1 header (adds resume_total_ms) and
computes resume_total_ms = restore_ms + client_ttft_s*1000 for rows that have a
restore step (disk_restore / d_restore with restore_ms present). All rows are
bumped to schema_version=1.1 (a v1 row is a valid v1.1 row; the new column is
blank where not applicable). Prints a per-file report.
"""
import csv, glob, os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "harness"))
from schema import CSV_FIELDS_V1, SCHEMA_VERSION  # noqa: E402


def main():
    for path in sorted(glob.glob(os.path.join(REPO, "results", "block*.csv"))):
        rows = list(csv.DictReader(open(path)))
        computed = 0
        for r in rows:
            r.pop(None, None)
            try:
                restore_ms = float(r.get("restore_ms") or "")
                ttft_s = float(r.get("client_ttft_s") or "")
                r["resume_total_ms"] = round(restore_ms + ttft_s * 1000, 1)
                computed += 1
            except ValueError:
                r["resume_total_ms"] = ""
            r["schema_version"] = SCHEMA_VERSION
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS_V1)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in CSV_FIELDS_V1})
        print(f"{os.path.basename(path):<44} rows={len(rows):>3} resume_total computed={computed}")


if __name__ == "__main__":
    main()
