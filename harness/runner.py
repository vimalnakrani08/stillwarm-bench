#!/usr/bin/env python3
"""stillwarm-bench measurement runner.

Launches llama-server, runs a scenario, writes one CSV row per measured rep with
the complete config + client TTFT + server timings (verbatim) + prompt_n + memory
+ probe/reuse verdicts + raw-log paths, emits AuditWeave evidence, then enforces the
disk budget. REFUSES any measured config without flash-attn 'on' (policy).

Scenarios:
  determinism   - BLOCKING pre-test: same cold recompute x2, byte-diff 64 tokens.
  docchat_restore - cold baseline + save + kill + restore + warm, with the
                    verified-restore probe and reuse assertion on the restored path.

Usage:
  runner.py --scenario determinism     --doc 8k [--reps 1]
  runner.py --scenario docchat_restore --doc 8k [--reps 3 --warmup 1]
"""
from __future__ import annotations
import argparse, csv, json, os, subprocess, sys, datetime, statistics
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from serverproc import ServerProc            # noqa: E402
from client import complete, GREEDY          # noqa: E402
from diskbudget import enforce, remove_files  # noqa: E402
from provenance import Provenance, ADAPTER_NOTE, HAVE_AUDITWEAVE  # noqa: E402

BIN = "/Users/vimal/llamacpp-stillwarm/build/bin/llama-server"
from schema import SCHEMA_VERSION, CSV_FIELDS_V1 as CSV_FIELDS  # frozen v1


def git_sha():
    try:
        sha = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", REPO, "status", "--porcelain"], text=True).strip())
        return sha, dirty
    except Exception:
        return None, None


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def load_cfg():
    with open(os.path.join(HERE, "config.yaml")) as f:
        return yaml.safe_load(f)


def build_prompt(doc_label):
    doc = open(os.path.join(REPO, f"workloads/docchat/doc_{doc_label}.txt"), encoding="utf-8").read()
    q = json.load(open(os.path.join(REPO, "workloads/docchat/questions.json")))
    tmpl, qid = q["prompt_template"], q["primary_question_id"]
    qtext = next(x["text"] for x in q["questions"] if x["id"] == qid)
    return doc, tmpl.format(doc=doc, question=qtext), qid


def doc_tokens_of(doc_label):
    cc = json.load(open(os.path.join(REPO, "workloads/docchat/cuts_tokencount.json")))
    return next(c["actual_tokens"] for c in cc["cuts"] if c["label"] == doc_label)


def validate_fa(flash_attn):
    if flash_attn != "on":
        raise SystemExit(f"REFUSED: measured runs require flash_attn 'on' (got {flash_attn!r}). "
                         "Policy: 'auto' is banned in measurements.")


def base_row(cfg_common):
    r = {k: "" for k in CSV_FIELDS}
    r.update(cfg_common)
    return r


def fill_metrics(row, res):
    row.update(
        client_ttft_s=res["ttft_s"], client_total_s=res["total_s"],
        server_prompt_n=res["prompt_n"], server_prompt_ms=res["prompt_ms"],
        server_predicted_n=res["predicted_n"], server_predicted_ms=res["predicted_ms"],
        server_prompt_tok_s=res["prompt_per_second"], server_predicted_tok_s=res["predicted_per_second"],
        tokens_cached=res["tokens_cached"], tokens_evaluated=res["tokens_evaluated"],
        server_timings_json=json.dumps(res["server_timings"], separators=(",", ":")),
        gen_content_sha256=res["content_sha256"], gen_tokens_sha256=res["gen_tokens_sha256"],
    )
    return row


# ---------------------------------------------------------------------------
def run_determinism(cfg_common, model, ctx, extra_flags, save_dir, full_prompt, log_path, n_predict=64):
    """BLOCKING pre-test: two cold recomputes; byte-diff generated tokens."""
    sp = ServerProc(BIN, model, 8080, ctx, save_dir, extra_flags=extra_flags, log_path=log_path)
    sp.start(); sp.start_rss()
    a = complete(8080, full_prompt, n_predict, cache_prompt=False)
    b = complete(8080, full_prompt, n_predict, cache_prompt=False)
    sp.stop_rss(); sp.stop()
    # content_sha256 is always populated (streaming may or may not carry token ids);
    # tokens_match is corroboration when both token lists are present.
    content_match = a["content_sha256"] == b["content_sha256"]
    both_toks = bool(a["gen_tokens"]) and bool(b["gen_tokens"])
    tokens_match = (a["gen_tokens"] == b["gen_tokens"]) if both_toks else None
    identical = content_match and (tokens_match is not False)
    return {
        "identical": identical, "content_match": content_match, "tokens_match": tokens_match,
        "run_a_tokens_sha": a["gen_tokens_sha256"], "run_b_tokens_sha": b["gen_tokens_sha256"],
        "run_a_content_sha": a["content_sha256"], "run_b_content_sha": b["content_sha256"],
        "n_predict": n_predict, "a_predicted_n": a["predicted_n"], "b_predicted_n": b["predicted_n"],
    }


def run_docchat_restore(cfg_common, model, ctx, extra_flags, save_dir, doc, full_prompt,
                        full_prompt_tokens, doc_tokens, reps, warmup, log_a, log_b, prov, n_predict=64):
    rows, savefile = [], f"gate1_docchat_{cfg_common['doc_label']}.bin"
    new_input = full_prompt_tokens - doc_tokens

    # --- server S1: cold baseline + prime + save ---
    s1 = ServerProc(BIN, model, 8080, ctx, save_dir, extra_flags=extra_flags, log_path=log_a)
    s1.start(); s1.start_rss()
    for i in range(warmup):
        complete(8080, full_prompt, n_predict, cache_prompt=False)   # discard
    cold_baseline = None
    for rep in range(1, reps + 1):
        res = complete(8080, full_prompt, n_predict, cache_prompt=False)
        cold_baseline = res  # keep last as probe baseline
        row = base_row(cfg_common); row.update(mode="cold", rep=rep, is_warmup=0,
                                                new_input_tokens=new_input, ts_end=now())
        rows.append(fill_metrics(row, res))
    # prime the doc cache, then save
    complete(8080, doc, 1, cache_prompt=True)          # slot 0 now holds the doc prefix
    save = s1.save_slot(savefile)
    s1.stop_rss(); s1_mem = s1.rss_stats_mb(); s1.stop()

    # --- server S2: restore + warm reps (each rep re-restores for an independent measurement) ---
    s2 = ServerProc(BIN, model, 8080, ctx, save_dir, extra_flags=extra_flags, log_path=log_b)
    s2.start(); s2.start_rss()
    restore_ms_first = None
    for rep in range(1, reps + 1):
        s2.erase_slot(0)
        rest = s2.restore_slot(savefile)
        if restore_ms_first is None:
            restore_ms_first = rest["timings"]["restore_ms"]
        res = complete(8080, full_prompt, n_predict, cache_prompt=True)
        reuse_ok = (res["prompt_n"] is not None) and (res["prompt_n"] <= new_input + 8)
        # byte-exact probe on generated content (always populated); token ids corroborate if present.
        probe_ok = (res["content_sha256"] == cold_baseline["content_sha256"])
        if res["gen_tokens"] and cold_baseline["gen_tokens"]:
            probe_ok = probe_ok and (res["gen_tokens"] == cold_baseline["gen_tokens"])
        row = base_row(cfg_common); row.update(
            mode="restore", rep=rep, is_warmup=0, new_input_tokens=new_input, ts_end=now(),
            reuse_assert="PASS" if reuse_ok else "FAIL", reuse_bool=reuse_ok,
            probe_result="PASS" if probe_ok else "FAIL",
            probe_baseline_sha=cold_baseline["gen_tokens_sha256"], probe_restored_sha=res["gen_tokens_sha256"],
            save_file=savefile, save_bytes=save["n_written"], save_ms=save["timings"]["save_ms"],
            n_saved=save["n_saved"], restore_ms=rest["timings"]["restore_ms"],
            restore_bytes=rest["n_read"], n_restored=rest["n_restored"])
        rows.append(fill_metrics(row, res))
    s2.stop_rss(); s2_mem = s2.rss_stats_mb(); s2.stop()

    # attach memory (S1 for cold rows, S2 for restore rows)
    for r in rows:
        mem = s1_mem if r["mode"] == "cold" else s2_mem
        r.update(rss_peak_mb=mem["rss_peak_mb"], rss_mean_mb=mem["rss_mean_mb"], rss_n=mem["rss_n"])
    return rows, savefile, save


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True, choices=["determinism", "docchat_restore"])
    ap.add_argument("--doc", default="8k")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=16384)
    ap.add_argument("--n-predict", type=int, default=64)
    ap.add_argument("--cache-type-k", default="f16")
    ap.add_argument("--cache-type-v", default="f16")
    ap.add_argument("--flash-attn", default="on")
    ap.add_argument("--swa-full", action="store_true")
    args = ap.parse_args()

    validate_fa(args.flash_attn)   # REFUSAL point
    cfg = load_cfg()
    m = cfg["models"]["primary"]
    model_path = os.path.expanduser(os.path.join(cfg["paths"]["models_dir"], m["file"]))
    save_dir = os.path.join(REPO, "cache_saves") + "/"
    resdir = os.path.join(REPO, "results"); rawdir = os.path.join(resdir, "raw")
    os.makedirs(rawdir, exist_ok=True)
    sha, dirty = git_sha()
    run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + "_" + args.scenario

    extra_flags = ["-ctk", args.cache_type_k, "-ctv", args.cache_type_v, "-fa", args.flash_attn]
    if args.swa_full:
        extra_flags.append("--swa-full")

    doc, full_prompt, qid = build_prompt(args.doc)
    doc_tok = doc_tokens_of(args.doc)

    free_gb = round(__import__("shutil").disk_usage(REPO).free / 1e9, 1)
    cfg_common = dict(
        schema_version=SCHEMA_VERSION, block="pre", side_experiment=0,
        save_point="doc_prefill", page_cache_state="n/a", free_disk_gb_start=free_gb,
        run_id=run_id, ts_start=now(), scenario=args.scenario,
        harness_git_sha=sha, harness_git_dirty=int(bool(dirty)),
        llamacpp_tag=cfg["llamacpp"]["release_tag"], llamacpp_sha=cfg["llamacpp"]["commit_sha"],
        model_name=m["name"], model_sha256=m["sha256"],
        doc_label=args.doc, doc_tokens=doc_tok, question_id=qid,
        ctx_size=args.ctx, n_slots=1, ngl=99,
        cache_type_k=args.cache_type_k, cache_type_v=args.cache_type_v,
        flash_attn=args.flash_attn, swa_full=int(args.swa_full),
        seed=GREEDY["seed"], temperature=GREEDY["temperature"], top_k=GREEDY["top_k"],
        n_predict=args.n_predict, verbosity="default(timed)/5(profile)",
    )
    # Profile launch (-lv 5, no timed requests): capture KV/Metal buffer sizes AND
    # tokenize the full prompt with this build's tokenizer. Timing runs stay quiet.
    prof_log = os.path.join(rawdir, f"{run_id}_profile.log")
    sp = ServerProc(BIN, model_path, 8080, args.ctx, save_dir, extra_flags=extra_flags,
                    verbosity=5, log_path=prof_log)
    sp.start()
    full_prompt_tokens = len(sp.tokenize(full_prompt))
    bufs = sp.parse_buffer_sizes()
    sp.stop()
    cfg_common.update(full_prompt_tokens=full_prompt_tokens,
                      kv_total_mib=bufs.get("kv_total_mib"), kv_k_type=bufs.get("kv_k_type"),
                      kv_v_type=bufs.get("kv_v_type"),
                      compute_buffer_mtl_mib=bufs.get("compute_buffer_mtl_mib"),
                      model_buffer_mtl_mib=bufs.get("model_buffer_mtl_mib"),
                      projected_device_mib=bufs.get("projected_device_mib"),
                      server_log=os.path.relpath(prof_log, REPO))

    prov = Provenance(os.path.join(resdir, "evidence.jsonl"), enabled=cfg.get("provenance", True))
    prov.run_started(cfg_common)

    csv_path = os.path.join(resdir, f"{run_id}.csv")

    if args.scenario == "determinism":
        log = os.path.join(rawdir, f"{run_id}_determinism.log")
        det = run_determinism(cfg_common, model_path, args.ctx, extra_flags, save_dir, full_prompt, log, args.n_predict)
        prov.verification(run_id, {"check": "determinism_pretest", **det})
        prov.run_completed(run_id, {"scenario": "determinism", "pass": det["identical"]})
        # write a minimal 2-row record
        rows = []
        for tag, sha_t, sha_c in (("A", det["run_a_tokens_sha"], det["run_a_content_sha"]),
                                  ("B", det["run_b_tokens_sha"], det["run_b_content_sha"])):
            r = base_row(cfg_common); r.update(mode="cold_recompute", rep=tag, is_warmup=0,
                                               gen_tokens_sha256=sha_t, gen_content_sha256=sha_c,
                                               server_log=os.path.relpath(log, REPO), ts_end=now(),
                                               probe_result="IDENTICAL" if det["identical"] else "DIFFERENT")
            rows.append(r)
        write_csv(csv_path, rows)
        print(json.dumps({"scenario": "determinism", "identical": det["identical"], **det}, indent=2))
        if not det["identical"]:
            print("\n*** DETERMINISM PRE-TEST FAILED — STOP. Probe methodology invalid on Metal. ***")
            sys.exit(3)
        clean = enforce(save_dir, cfg["DISK_BUDGET_GB"])
        print("csv:", os.path.relpath(csv_path, REPO), "| cleanup:", clean)
        return

    # docchat_restore
    log_a = os.path.join(rawdir, f"{run_id}_cold.log")
    log_b = os.path.join(rawdir, f"{run_id}_restore.log")
    rows, savefile, save = run_docchat_restore(
        cfg_common, model_path, args.ctx, extra_flags, save_dir, doc, full_prompt,
        full_prompt_tokens, doc_tok, args.reps, args.warmup, log_a, log_b, prov, args.n_predict)
    for r in rows:
        r.setdefault("server_log", os.path.relpath(log_a if r["mode"] == "cold" else log_b, REPO))
    write_csv(csv_path, rows)

    # provenance: one MEASUREMENT per row + a VERIFICATION for the restore probe/reuse
    restore_rows = [r for r in rows if r["mode"] == "restore"]
    for r in rows:
        prov.measurement(run_id, r, {"mode": r["mode"], "rep": r["rep"],
                                     "ttft_s": r["client_ttft_s"], "prompt_n": r["server_prompt_n"]})
    probe_pass = all(r["probe_result"] == "PASS" for r in restore_rows)
    reuse_pass = all(r["reuse_bool"] for r in restore_rows)
    prov.verification(run_id, {"check": "verified_restore_probe", "all_pass": probe_pass,
                               "reuse_assertion_all_pass": reuse_pass,
                               "restore_reps": len(restore_rows)})
    prov.run_completed(run_id, {"scenario": "docchat_restore", "probe_pass": probe_pass,
                                "reuse_pass": reuse_pass})
    # cleanup: this scenario's save file is not flagged keep -> remove, then enforce budget
    rm = remove_files([os.path.join(save_dir, savefile)])
    clean = enforce(save_dir, cfg["DISK_BUDGET_GB"])

    summarize(rows, csv_path, save, probe_pass, reuse_pass, prov, rm, clean)


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})


def summarize(rows, csv_path, save, probe_pass, reuse_pass, prov, rm, clean):
    cold = [r["client_ttft_s"] for r in rows if r["mode"] == "cold"]
    warm = [r["client_ttft_s"] for r in rows if r["mode"] == "restore"]
    def med(xs): return round(statistics.median(xs), 4) if xs else None
    out = {
        "csv": os.path.relpath(csv_path, REPO),
        "cold_ttft_s_median": med(cold), "cold_ttft_s_minmax": [min(cold), max(cold)] if cold else None,
        "restore_ttft_s_median": med(warm), "restore_ttft_s_minmax": [min(warm), max(warm)] if warm else None,
        "save_bytes": save["n_written"], "save_ms": save["timings"]["save_ms"],
        "probe_all_pass": probe_pass, "reuse_all_pass": reuse_pass,
        "provenance_enabled": prov.enabled, "auditweave_present": HAVE_AUDITWEAVE,
        "evidence_verify": (lambda v: {"ok": v.ok, "checked": v.checked} if v else None)(prov.verify()),
        "cleanup_removed": rm, "budget": clean,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
