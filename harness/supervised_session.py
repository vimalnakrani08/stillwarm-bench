#!/usr/bin/env python3
"""SUPERVISED SESSION (hardened) — run interactively in a terminal with Vimal present.

One sitting, four parts (Gate-2 WS2). Pauses at every manual sudo step; the harness
itself NEVER invokes sudo.

  A. Thermal-controlled headline re-run — f16, rungs 8K/32K/64K, 5x ALTERNATING
     cold/restore per rung (same thermal state both sides).            (~42 min)
  B. Cold-read restores — `sudo purge` before EVERY rep, x3 per rung.  (~12 min)
  C. dd of the 64K save file after a purge (+ warm contrast).          (~3 min)
  D. Guided powermetrics capture: one 32K cold + one 32K restore.      (~6 min)

HARDENING (2026-07-05, after the premature-completion incident):
  * every measured rep is APPENDED to its CSV immediately — a crash loses at most
    one rep, never a session;
  * every prompt survives EOF/Ctrl-C with an explicit "SESSION ABORTED at <step>";
  * --resume skips parts (and Part-A rungs) whose outputs already exist;
  * start-of-run self-check prints every output destination;
  * END-OF-RUN ARTIFACT VERIFICATION: checks every expected file exists and is
    non-empty, prints a manifest (paths, sizes, row counts) and ONLY THEN prints
    the exact completion line (with a manifest digest) to paste back. Anything
    missing -> "SESSION INCOMPLETE" + the missing list instead. The completion
    declaration cannot be produced without the artifacts.

Usage:
    .venv/bin/python harness/supervised_session.py                # all parts
    .venv/bin/python harness/supervised_session.py --resume       # skip done parts
    .venv/bin/python harness/supervised_session.py A B --resume   # subset
"""
import argparse, csv, datetime, hashlib, json, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase2 import (BIN, PORT, RUNG_TOKENS, model_entry, doc_text, questions,
                    base_row, fill_metrics, probe_and_reuse, write_csv, common_for,
                    REPO, now)
from serverproc import ServerProc
from client import complete
from provenance import Provenance

RUNGS_A = ["8k", "32k", "64k"]
REPS_A, REPS_B = 5, 3
FLAGS = ["-ctk", "f16", "-ctv", "f16", "-fa", "on", "--cache-ram", "0"]
SAVE_DIR = os.path.join(REPO, "cache_saves") + "/"
RAWDIR = os.path.join(REPO, "results/raw")
RESULTS = os.path.join(REPO, "results")
A2_CSV = os.path.join(RESULTS, "blockA2_supervised.csv")
DD_JSON = os.path.join(RESULTS, "blockB_dd_reference.json")
ENERGY_JSON = os.path.join(RESULTS, "energy_markers.json")
POWERMETRICS_RAW = os.path.join(RAWDIR, "powermetrics_32k.txt")

prov = Provenance(os.path.join(REPO, "results/evidence.jsonl"))
m, MODEL = model_entry("primary")
TMPL, QS = questions()


class SessionAborted(Exception):
    def __init__(self, step):
        self.step = step
        super().__init__(step)


def ask(msg, step):
    try:
        return input(msg)
    except (EOFError, KeyboardInterrupt):
        raise SessionAborted(step) from None


def pause(msg, step):
    ask(f"\n>>> {msg}\n>>> Press Enter when done... ", step)


def ts():
    return datetime.datetime.now().isoformat(timespec="seconds")


def csv_rows(path, **match):
    if not os.path.exists(path):
        return []
    out = []
    for r in csv.DictReader(open(path)):
        if all(r.get(k) == v for k, v in match.items()):
            out.append(r)
    return out


def start_server(ctx, log):
    s = ServerProc(BIN, MODEL, PORT, ctx, SAVE_DIR, extra_flags=FLAGS, log_path=log)
    s.start()
    return s


def ensure_save(rung, s=None):
    """Make sure supervised_<rung>.bin exists (prime+save is prep, not measurement).
    Uses the given running server or starts/stops one."""
    savefile = f"supervised_{rung}.bin"
    if os.path.exists(SAVE_DIR + savefile):
        return savefile
    own = s is None
    if own:
        s = start_server(RUNG_TOKENS[rung] + 1024,
                         os.path.join(RAWDIR, f"ensure_save_{rung}.log"))
    s.erase_slot(0)
    complete(PORT, doc_text(rung), 0, cache_prompt=True)
    s.save_slot(savefile)
    if own:
        s.stop()
    return savefile


# --------------------------------------------------------------------------- A
def part_A(resume):
    print(f"\n===== PART A: interleaved thermal-controlled re-run ({ts()}) =====")
    notes = ask("Ambient notes (room temp/fan/anything unusual; Enter to skip): ",
                "A/ambient-notes").strip()
    for rung in RUNGS_A:
        done = len(csv_rows(A2_CSV, doc_label=rung, mode="cold")) \
             + len(csv_rows(A2_CSV, doc_label=rung, mode="disk_restore"))
        if resume and done >= 2 * REPS_A:
            print(f"--- rung {rung}: already complete in CSV ({done} rows) — SKIP")
            continue
        doc = doc_text(rung)
        p1 = TMPL.format(doc=doc, question=QS[0]["text"])
        ctx = RUNG_TOKENS[rung] + 1024
        run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + f"_A2_{rung}"
        log = os.path.join(RAWDIR, f"{run_id}.log")
        print(f"\n--- rung {rung} (ctx {ctx}) started {ts()} ---")
        s = start_server(ctx, log)
        try:
            doc_tok = len(s.tokenize(doc)); full_tok = len(s.tokenize(p1))
            savefile = ensure_save(rung, s)
            save_bytes = os.path.getsize(SAVE_DIR + savefile)
            common = common_for("A2", run_id, m, rung, ctx, "f16", "f16", FLAGS)
            common.update(scenario="supervised_interleaved", question_id="q1",
                          full_prompt_tokens=full_tok, new_input_tokens=full_tok - doc_tok,
                          cache_ram_mib=0, save_file=savefile, save_bytes=save_bytes,
                          server_log=os.path.relpath(log, REPO),
                          verbosity=f"default; ambient={notes or 'n/a'}")
            # warmup pair (discarded)
            s.erase_slot(0); complete(PORT, p1, 64, cache_prompt=False)
            s.erase_slot(0); s.restore_slot(savefile); complete(PORT, p1, 64, cache_prompt=True)
            for rep in range(1, REPS_A + 1):
                s.erase_slot(0)
                res_c = complete(PORT, p1, 64, cache_prompt=False)          # COLD
                rc = base_row(common); rc.update(mode="cold", rep=rep, is_warmup=0)
                fill_metrics(rc, res_c)
                s.erase_slot(0)
                rest = s.restore_slot(savefile)                              # RESTORE
                res_r = complete(PORT, p1, 64, cache_prompt=True)
                rr = base_row(common); rr.update(
                    mode="disk_restore", rep=rep, is_warmup=0, page_cache_state="warm_read",
                    restore_ms=rest["timings"]["restore_ms"], restore_bytes=rest["n_read"],
                    n_restored=rest["n_restored"],
                    resume_total_ms=round(rest["timings"]["restore_ms"] + res_r["ttft_s"] * 1000, 1))
                fill_metrics(rr, res_r)
                probe_and_reuse(rr, res_r, res_c, full_tok - doc_tok)
                write_csv(A2_CSV, [rc, rr])          # <-- incremental: per-rep append
                print(f"  rep {rep}: cold {res_c['ttft_s']}s | restore {rest['timings']['restore_ms']}ms"
                      f" + ttft {res_r['ttft_s']}s | probe {rr['probe_result']} reuse {rr['reuse_assert']}")
        finally:
            s.stop()
    prov.verification("A2_supervised", {"check": "thermal_interleaved",
                                        "rows": len(csv_rows(A2_CSV)), "ambient": notes})
    print(f"===== PART A done ({ts()}); save files kept for parts B-D =====")


# --------------------------------------------------------------------------- B
def part_B(resume):
    print(f"\n===== PART B: cold-read restores ({ts()}) =====")
    for rung in RUNGS_A:
        out_csv = os.path.join(RESULTS, f"blockB_coldread_{rung}.csv")
        if resume and len(csv_rows(out_csv, mode="disk_restore")) >= REPS_B:
            print(f"--- {rung}: already complete — SKIP"); continue
        doc = doc_text(rung)
        p1 = TMPL.format(doc=doc, question=QS[0]["text"])
        ctx = RUNG_TOKENS[rung] + 1024
        savefile = ensure_save(rung)
        run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + f"_coldread_{rung}"
        log = os.path.join(RAWDIR, f"{run_id}.log")
        s = start_server(ctx, log)
        doc_tok = len(s.tokenize(doc)); full_tok = len(s.tokenize(p1))
        baseline = complete(PORT, p1, 64, cache_prompt=False)
        s.stop()
        common = common_for("B", run_id, m, rung, ctx, "f16", "f16", FLAGS)
        common.update(scenario="coldread", question_id="q1", full_prompt_tokens=full_tok,
                      new_input_tokens=full_tok - doc_tok, page_cache_state="cold_read",
                      cache_ram_mib=0, save_file=savefile,
                      save_bytes=os.path.getsize(SAVE_DIR + savefile),
                      server_log=os.path.relpath(log, REPO))
        for rep in range(1, REPS_B + 1):
            pause(f"[{rung} rep {rep}/{REPS_B}] Server STOPPED. In another terminal run:  sudo purge",
                  f"B/{rung}/rep{rep}/purge")
            s = start_server(ctx, log)
            try:
                rest = s.restore_slot(savefile)
                res = complete(PORT, p1, 64, cache_prompt=True)
            finally:
                s.stop()
            r = base_row(common); r.update(
                mode="disk_restore", rep=rep, is_warmup=0,
                restore_ms=rest["timings"]["restore_ms"], restore_bytes=rest["n_read"],
                n_restored=rest["n_restored"],
                resume_total_ms=round(rest["timings"]["restore_ms"] + res["ttft_s"] * 1000, 1))
            fill_metrics(r, res)
            probe_and_reuse(r, res, baseline, full_tok - doc_tok)
            write_csv(out_csv, [r])                  # <-- incremental
            print(f"  COLD-READ restore_ms={rest['timings']['restore_ms']} ttft={res['ttft_s']}s"
                  f" resume_total={r['resume_total_ms']}ms probe={r['probe_result']}")
    print(f"===== PART B done ({ts()}) =====")


# --------------------------------------------------------------------------- C
def _dd_supervised_done():
    if not os.path.exists(DD_JSON):
        return False
    for line in open(DD_JSON):
        try:
            if "supervised" in json.loads(line):
                return True
        except json.JSONDecodeError:
            pass
    return False


def part_C(resume):
    print(f"\n===== PART C: dd reference, 64K file ({ts()}) =====")
    if resume and _dd_supervised_done():
        print("--- supervised dd entry already present — SKIP"); return
    savefile = ensure_save("64k")
    f = SAVE_DIR + savefile
    out = []
    pause("Run:  sudo purge   (for the COLD dd read)", "C/purge")
    for label in ("cold_dd_64k", "warm_dd_64k"):
        t0 = time.perf_counter()
        subprocess.run(["dd", f"if={f}", "of=/dev/null", "bs=8m"], capture_output=True)
        dt = time.perf_counter() - t0
        sz = os.path.getsize(f)
        rec = {"ts": ts(), "label": label, "bytes": sz, "seconds": round(dt, 3),
               "MBps": round(sz / 1e6 / dt, 1),
               "page_cache_state": label.split("_")[0] + "_read"}
        out.append(rec); print(" ", json.dumps(rec))
    with open(DD_JSON, "a") as fh:
        fh.write(json.dumps({"supervised": out}) + "\n")
    print(f"===== PART C done ({ts()}) =====")


# --------------------------------------------------------------------------- D
def part_D(resume):
    print(f"\n===== PART D: powermetrics energy capture ({ts()}) =====")
    if resume and os.path.exists(ENERGY_JSON) and os.path.exists(POWERMETRICS_RAW) \
            and os.path.getsize(POWERMETRICS_RAW) > 0:
        print("--- energy markers + raw capture already present — SKIP"); return
    rung, ctx = "32k", RUNG_TOKENS["32k"] + 1024
    doc = doc_text(rung); p1 = TMPL.format(doc=doc, question=QS[0]["text"])
    savefile = ensure_save(rung)
    log = os.path.join(RAWDIR, "powermetrics_session.log")
    s = start_server(ctx, log)
    try:
        pause("In a SECOND terminal start:\n"
              "    sudo powermetrics --samplers gpu_power,cpu_power -i 1000 -o /tmp/powermetrics_32k.txt\n"
              "and leave it running", "D/start-powermetrics")
        markers = {"session": ts(), "interval_ms": 1000, "samplers": "gpu_power,cpu_power"}
        markers["cold_start"] = ts()
        s.erase_slot(0)
        res_c = complete(PORT, p1, 64, cache_prompt=False)
        markers["cold_end"] = ts(); markers["cold_ttft_s"] = res_c["ttft_s"]
        markers["cold_total_s"] = res_c["total_s"]
        time.sleep(5)   # separator gap in the power trace
        markers["restore_start"] = ts()
        s.erase_slot(0)
        rest = s.restore_slot(savefile)
        res_r = complete(PORT, p1, 64, cache_prompt=True)
        markers["restore_end"] = ts()
        markers["restore_ms"] = rest["timings"]["restore_ms"]
        markers["restore_ttft_s"] = res_r["ttft_s"]; markers["restore_total_s"] = res_r["total_s"]
    finally:
        s.stop()
    pause("Ctrl-C the powermetrics terminal, then run:\n"
          f"    cp /tmp/powermetrics_32k.txt {POWERMETRICS_RAW}", "D/copy-powermetrics")
    with open(ENERGY_JSON, "w") as fh:
        json.dump(markers, fh, indent=2)
    print("  markers:", json.dumps(markers))
    print(f"===== PART D done ({ts()}) =====")


# ------------------------------------------------------------------ artifacts
def expected_artifacts(parts):
    exp = []
    if "A" in parts:
        exp.append((A2_CSV, "csv", 2 * REPS_A * len(RUNGS_A)))
    if "B" in parts:
        exp += [(os.path.join(RESULTS, f"blockB_coldread_{r}.csv"), "csv", REPS_B)
                for r in RUNGS_A]
    if "C" in parts:
        exp.append((DD_JSON, "dd-json", 1))
    if "D" in parts:
        exp.append((ENERGY_JSON, "json", 1))
        exp.append((POWERMETRICS_RAW, "raw", 1))
    return exp


def verify_artifacts(parts):
    manifest, missing = [], []
    for path, kind, min_rows in expected_artifacts(parts):
        rel = os.path.relpath(path, REPO)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            missing.append(f"{rel}  (missing or empty)")
            continue
        rows = ""
        if kind == "csv":
            n = len(csv_rows(path))
            rows = n
            if n < min_rows:
                missing.append(f"{rel}  (only {n} rows, expected >= {min_rows})")
                continue
        if kind == "dd-json" and not _dd_supervised_done():
            missing.append(f"{rel}  (no supervised dd entry)")
            continue
        manifest.append({"path": rel, "bytes": os.path.getsize(path), "rows": rows})
    return manifest, missing


def print_verdict(parts):
    manifest, missing = verify_artifacts(parts)
    print("\n===================== ARTIFACT VERIFICATION =====================")
    for e in manifest:
        print(f"  OK  {e['path']:<44} {e['bytes']:>12,} B  rows={e['rows']}")
    if missing:
        print("\nSESSION INCOMPLETE — missing/short artifacts:")
        for x in missing:
            print(f"  !!  {x}")
        print("\nDo NOT declare the session complete. Re-run with --resume to fill the gaps.")
        return False
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
    total_rows = sum(e["rows"] or 0 for e in manifest if isinstance(e["rows"], int))
    print("\nAll expected artifacts present and non-empty.")
    print("Paste EXACTLY this line back as the session-completion report:")
    print(f"\nSUPERVISED SESSION COMPLETE — ARTIFACTS VERIFIED "
          f"(manifest {digest}, {len(manifest)} files, {total_rows} csv rows)\n")
    return True


def cleanup():
    for rung in RUNGS_A:
        f = SAVE_DIR + f"supervised_{rung}.bin"
        if os.path.exists(f):
            a = ask(f"Delete {f} ({os.path.getsize(f)/1e9:.2f} GB)? [Y/n] ", "cleanup").strip().lower()
            if a in ("", "y", "yes"):
                os.remove(f); print("  removed")
            else:
                print("  kept")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="*", default=[], help="subset of A B C D (default all)")
    ap.add_argument("--resume", action="store_true", help="skip parts/rungs whose outputs exist")
    ns = ap.parse_args()
    parts = [p.upper() for p in ns.parts] or ["A", "B", "C", "D"]

    print("SUPERVISED SESSION (hardened) — parts:", parts, "| resume:", ns.resume)
    print("Checklist: AC power connected · lid open · no other heavy apps.")
    print("\nOutput destinations (self-check):")
    for path, kind, min_rows in expected_artifacts(parts):
        print(f"  {os.path.relpath(path, REPO):<46} [{kind}, expect >= {min_rows} rows]" if kind == "csv"
              else f"  {os.path.relpath(path, REPO):<46} [{kind}]")
    print(f"  raw server logs -> results/raw/  |  save files -> cache_saves/supervised_*.bin")

    caf = subprocess.Popen(["caffeinate", "-dims"])
    print(f"\ncaffeinate held (pid {caf.pid}). Session start: {ts()}")
    aborted = None
    try:
        if "A" in parts: part_A(ns.resume)
        if "B" in parts: part_B(ns.resume)
        if "C" in parts: part_C(ns.resume)
        if "D" in parts: part_D(ns.resume)
    except SessionAborted as e:
        aborted = e.step
        print(f"\n*** SESSION ABORTED at step: {e.step} ***")
    finally:
        caf.terminate()
        print(f"caffeinate released. Session end: {ts()}")
    ok = print_verdict(parts)
    if ok and not aborted:
        try:
            cleanup()
        except SessionAborted:
            print("(cleanup skipped)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
