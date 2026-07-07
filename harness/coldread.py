#!/usr/bin/env python3
"""Supervised COLD-READ restore protocol (Block B, page_cache_state=cold_read).

RUN THIS INTERACTIVELY IN A TERMINAL WITH AN OPERATOR PRESENT — it pauses before every
restore rep so `sudo purge` can be run in another terminal (purging the page cache
needs sudo; the harness itself never invokes sudo).

Per rung: build the save file once (doc_prefill), then for each of N reps:
  1. server stopped -> PAUSE: operator runs `sudo purge`, presses Enter
  2. fresh server start -> restore (reads the file from a purged page cache)
  3. ask q1 -> row with page_cache_state=cold_read, probe + reuse as usual
Also records a `dd` sequential read of the save file right after a purge (raw
SSD cold-read reference) and one right after (warm reference).

Usage: .venv/bin/python harness/coldread.py [rungs...]   (default: 2k 8k 32k)
"""
import json, os, subprocess, sys, time, datetime
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase2 import (BIN, PORT, RUNG_TOKENS, model_entry, doc_text, questions,
                    base_row, fill_metrics, probe_and_reuse, write_csv, common_for,
                    free_gb, REPO, now)
from serverproc import ServerProc
from client import complete
from provenance import Provenance

RUNGS = sys.argv[1:] or ["2k", "8k", "32k"]
REPS = 5


def dd_read(path, label):
    t0 = time.perf_counter()
    subprocess.run(["dd", f"if={path}", "of=/dev/null", "bs=8m"],
                   capture_output=True, text=True)
    dt = time.perf_counter() - t0
    sz = os.path.getsize(path)
    r = {"label": label, "bytes": sz, "seconds": round(dt, 3),
         "MBps": round(sz / 1e6 / dt, 1)}
    print("dd:", json.dumps(r), flush=True)
    return r


def main():
    m, model_path = model_entry("primary")
    tmpl, qs = questions()
    save_dir = os.path.join(REPO, "cache_saves") + "/"
    rawdir = os.path.join(REPO, "results/raw")
    prov = Provenance(os.path.join(REPO, "results/evidence.jsonl"))
    dd_results = []
    for rung in RUNGS:
        doc = doc_text(rung)
        p1 = tmpl.format(doc=doc, question=qs[0]["text"])
        ctx = RUNG_TOKENS[rung] + 1024
        run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + f"_coldread_{rung}"
        flags = ["-ctk", "f16", "-ctv", "f16", "-fa", "on", "--cache-ram", "0"]
        savefile = f"coldread_{rung}.bin"
        log = os.path.join(rawdir, f"{run_id}.log")

        # save once + cold baseline for the probe
        s = ServerProc(BIN, model_path, PORT, ctx, save_dir, extra_flags=flags, log_path=log)
        s.start()
        doc_tok = len(s.tokenize(doc)); full_tok = len(s.tokenize(p1))
        baseline = complete(PORT, p1, 64, cache_prompt=False)
        s.erase_slot(0)
        complete(PORT, doc, 0, cache_prompt=True)
        save = s.save_slot(savefile)
        s.stop()

        common = common_for("B", run_id, m, rung, ctx, "f16", "f16", flags)
        common.update(scenario="coldread", full_prompt_tokens=full_tok,
                      new_input_tokens=full_tok - doc_tok, question_id="q1",
                      page_cache_state="cold_read", save_point="doc_prefill",
                      save_file=savefile, save_bytes=save["n_written"],
                      save_ms=save["timings"]["save_ms"], n_saved=save["n_saved"],
                      cache_ram_mib=0, server_log=os.path.relpath(log, REPO))
        rows = []
        for rep in range(1, REPS + 1):
            input(f"\n[{rung} rep {rep}/{REPS}] Server is STOPPED. Run `sudo purge` in "
                  f"another terminal, wait for it to finish, then press Enter here... ")
            if rep == 1:
                dd_results.append(dd_read(os.path.join(save_dir, savefile), f"{rung}_cold_dd"))
                input("dd cold-read done (this warmed the cache!). Run `sudo purge` AGAIN, "
                      "then press Enter... ")
            s = ServerProc(BIN, model_path, PORT, ctx, save_dir, extra_flags=flags, log_path=log)
            s.start()
            rest = s.restore_slot(savefile)
            res = complete(PORT, p1, 64, cache_prompt=True)
            s.stop()
            r = base_row(common)
            r.update(mode="disk_restore", rep=rep, is_warmup=0,
                     restore_ms=rest["timings"]["restore_ms"], restore_bytes=rest["n_read"],
                     n_restored=rest["n_restored"])
            fill_metrics(r, res)
            probe_and_reuse(r, res, baseline, full_tok - doc_tok)
            rows.append(r)
            print(f"  restore_ms={rest['timings']['restore_ms']} ttft={res['ttft_s']}s "
                  f"probe={r['probe_result']} reuse={r['reuse_assert']}", flush=True)
        dd_results.append(dd_read(os.path.join(save_dir, savefile), f"{rung}_warm_dd"))
        write_csv(os.path.join(REPO, "results", f"blockB_coldread_{rung}.csv"), rows)
        prov.run_started({**common, "scenario": f"coldread_{rung}"})
        prov.verification(run_id, {"check": "coldread", "rung": rung,
                                   "restore_ms": [r["restore_ms"] for r in rows]})
        prov.run_completed(run_id, {"rung": rung, "rows": len(rows)})
        os.remove(os.path.join(save_dir, savefile))
    with open(os.path.join(REPO, "results", "blockB_dd_reference.json"), "a") as f:
        f.write(json.dumps({"ts": now(), "dd": dd_results}) + "\n")
    print(json.dumps(dd_results, indent=2))


if __name__ == "__main__":
    main()
