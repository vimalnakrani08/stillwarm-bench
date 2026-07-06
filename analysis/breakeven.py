#!/usr/bin/env python3
"""Block B — break-even analysis from Block A CSVs (+ dd reference).

Break-even question: above how many PROMPT TOKENS does disk-restore beat cold
recompute on TTFT? Both sides measured per rung; both are ~linear in tokens, so
the crossover comes from linear fits (least squares through the rung medians).
The automated column is page_cache_state=warm_read; the cold_read column is
produced in the supervised sudo-purge session and computed the same way.
"""
import csv, glob, json, os, statistics, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rows_for(pattern):
    out = []
    for f in glob.glob(os.path.join(REPO, "results", pattern)):
        out += [r for r in csv.DictReader(open(f))]
    return out


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def fit(pts):  # least-squares y = a + b*x through (x, y) points
    n = len(pts)
    sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
    sxx = sum(p[0] ** 2 for p in pts); sxy = sum(p[0] * p[1] for p in pts)
    b = (n * sxy - sx * sy) / (n * sxx - sx ** 2)
    a = (sy - b * sx) / n
    return a, b


def main(run_filter=None, page_state="warm_read"):
    rows = rows_for("blockA_*.csv")
    if run_filter:
        rows = [r for r in rows if run_filter(r)]
    rungs = sorted({r["doc_label"] for r in rows}, key=lambda l: int(l[:-1]))
    table, cold_pts, rest_pts = [], [], []
    for rung in rungs:
        rr = [r for r in rows if r["doc_label"] == rung]
        cold = [r for r in rr if r["mode"] == "cold"]
        rest = [r for r in rr if r["mode"] == "disk_restore" and r["page_cache_state"] == page_state]
        if not cold or not rest:
            continue
        toks = fnum(cold[0]["doc_tokens"])
        prefill_ms = med([fnum(r["server_prompt_ms"]) for r in cold])
        cold_ttft = med([fnum(r["client_ttft_s"]) for r in cold])
        rest_ttft = med([fnum(r["client_ttft_s"]) for r in rest])
        restore_ms = med([fnum(r["restore_ms"]) for r in rest])
        save_ms = med([fnum(r["save_ms"]) for r in rest])
        save_bytes = fnum(rest[0]["save_bytes"])
        table.append(dict(rung=rung, tokens=int(toks), prefill_ms=round(prefill_ms, 1),
                          cold_ttft_s=round(cold_ttft, 3), restore_ms=round(restore_ms, 1),
                          restore_ttft_s=round(rest_ttft, 4), save_ms=round(save_ms, 1),
                          save_bytes=int(save_bytes),
                          bytes_per_token=round(save_bytes / toks),
                          eff_read_MBps=round(save_bytes / 1e6 / (restore_ms / 1e3), 1)))
        cold_pts.append((toks, cold_ttft))
        # restore path total = restore_ms + question prefill (in restore_ttft already)
        rest_pts.append((toks, rest_ttft + restore_ms / 1e3))
    a_c, b_c = fit(cold_pts)
    a_r, b_r = fit(rest_pts)
    # Global linear crossover is INVALID here: cold TTFT is super-linear in tokens
    # (attention scaling), so a straight-line fit through 2K..64K misrepresents small N.
    # Honest crossover: restore already wins at the smallest rung, so extrapolate with
    # the SMALLEST rung's local cold slope (most favorable to cold) below 2K.
    n0, cold0 = cold_pts[0]
    local_cold_s_per_tok = cold0 / n0            # ~1/prefill-rate at the smallest rung
    n_star_local = a_r / (local_cold_s_per_tok - b_r) if local_cold_s_per_tok > b_r else float("inf")
    out = {"page_cache_state": page_state, "table": table,
           "restore_wins_at_every_measured_rung": all(
               t["restore_ttft_s"] + t["restore_ms"] / 1e3 < t["cold_ttft_s"] for t in table),
           "global_linear_fit_note": ("cold is super-linear in tokens; global fit kept for "
                                      "reference only, crossover from it is NOT meaningful"),
           "cold_fit_s": {"intercept": round(a_c, 4), "per_token": b_c},
           "restore_fit_s": {"intercept": round(a_r, 4), "per_token": b_r},
           "crossover_tokens_local_slope": round(n_star_local, 1),
           "crossover_method": (f"restore_intercept {round(a_r,4)}s / (local cold slope "
                                f"{round(local_cold_s_per_tok*1e3,3)} ms/tok at {int(n0)} tok "
                                f"- restore slope) — an upper bound; true crossover <= this"),
           "note": ("restore side includes restore_ms + new-question prefill; "
                    "crossover < smallest rung means restore wins everywhere measured")}
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    # default: only clean-SHA (non-dirty) headline rows
    main(run_filter=lambda r: r.get("harness_git_dirty") in ("0", 0, "") and r.get("block") == "A")
