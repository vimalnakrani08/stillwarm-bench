#!/usr/bin/env python3
"""

Block A modes (mechanisms isolated — only the mode's own cache path is active):
  cold          --cache-ram 0, cache_prompt=false, q1 x (warmup+N)
  cold_baseline --cache-ram 0, one cold run per q2..q5 (probe baselines + spread data)
  prefix_reuse  same process, --cache-reuse 256, --cache-ram 0; doc primed once
                (n_predict=0); rep i asks question qi (cache_prompt=true)
  ram_parked    --cache-ram <sized>, no --cache-reuse; per rep: flush prompt parks
                doc state to host RAM -> doc+qi triggers RAM load; timed request is
                the doc+qi one. Verbose evidence cycle runs separately (8K rung).
  disk_restore  --cache-ram 0; S1 primes doc (n_predict=0) + saves + killed;
                S2: per rep erase -> restore -> doc+qi.
"""
from __future__ import annotations
import argparse, csv, json, os, shutil, subprocess, sys, datetime, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from serverproc import ServerProc                    # noqa: E402
from client import complete, GREEDY                  # noqa: E402
from diskbudget import enforce, remove_files         # noqa: E402
from provenance import Provenance                    # noqa: E402
from schema import SCHEMA_VERSION, CSV_FIELDS_V1     # noqa: E402
import yaml                                          # noqa: E402

BIN = "/Users/vimal/llamacpp-stillwarm/build/bin/llama-server"
PORT = 8080
RUNGS = ["2k", "4k", "8k", "16k", "32k", "64k"]
RUNG_TOKENS = {"2k": 2048, "4k": 4096, "8k": 8192, "16k": 16384, "32k": 32768, "64k": 65536}
CACHE_RAM_MIB = {"2k": 1024, "4k": 1024, "8k": 2048, "16k": 4096, "32k": 8192, "64k": 12288}
FLUSH_PROMPT = ("Unrelated flush request: list the numbers one through ten as words, "
                "separated by commas.")

cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def git_sha():
    sha = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", REPO, "status", "--porcelain"], text=True).strip())
    return sha, int(dirty)


def free_gb():
    return round(shutil.disk_usage(REPO).free / 1e9, 1)


def questions():
    q = json.load(open(os.path.join(REPO, "workloads/docchat/questions.json")))
    return q["prompt_template"], q["questions"]


def doc_text(rung):
    return open(os.path.join(REPO, f"workloads/docchat/doc_{rung}.txt"), encoding="utf-8").read()


def model_entry(key="primary"):
    m = cfg["models"][key]
    path = os.path.expanduser(os.path.join(cfg["paths"]["models_dir"], m["file"]))
    return m, path


def base_row(common):
    r = {k: "" for k in CSV_FIELDS_V1}
    r.update(common)
    return r


def fill_metrics(row, res):
    row.update(
        client_ttft_s=res["ttft_s"], client_total_s=res["total_s"],
        server_prompt_n=res["prompt_n"], server_prompt_ms=res["prompt_ms"],
        server_predicted_n=res["predicted_n"], server_predicted_ms=res["predicted_ms"],
        server_prompt_tok_s=res["prompt_per_second"], server_predicted_tok_s=res["predicted_per_second"],
        tokens_cached=res["tokens_cached"], tokens_evaluated=res["tokens_evaluated"],
        server_timings_json=json.dumps(res["server_timings"], separators=(",", ":")),
        gen_content_sha256=res["content_sha256"], gen_tokens_sha256=res["gen_tokens_sha256"], ts_end=now())
    return row


def probe_and_reuse(row, res, baseline, new_input):
    reuse_ok = (res["prompt_n"] is not None) and (res["prompt_n"] <= new_input + 8)
    probe_ok = res["content_sha256"] == baseline["content_sha256"]
    if res["gen_tokens"] and baseline["gen_tokens"]:
        probe_ok = probe_ok and (res["gen_tokens"] == baseline["gen_tokens"])
    outcome = ("WORKS" if (probe_ok and reuse_ok) else
               "SILENTLY-WRONG" if not probe_ok else
               "SILENTLY-INEFFECTIVE")
    row.update(reuse_assert="PASS" if reuse_ok else "FAIL", reuse_bool=reuse_ok,
               probe_result="PASS" if probe_ok else "FAIL",
               probe_baseline_sha=baseline["gen_tokens_sha256"],
               probe_restored_sha=res["gen_tokens_sha256"], outcome=outcome)
    return row


def write_csv(path, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS_V1)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS_V1})


def common_for(block, run_id, m, rung, ctx, ctk, ctv, extra):
    sha, dirty = git_sha()
    return dict(
        schema_version=SCHEMA_VERSION, block=block, run_id=run_id, ts_start=now(),
        scenario=f"block{block}", side_experiment=0,
        harness_git_sha=sha, harness_git_dirty=dirty,
        llamacpp_tag=cfg["llamacpp"]["release_tag"], llamacpp_sha=cfg["llamacpp"]["commit_sha"],
        model_name=m["name"], model_sha256=m["sha256"],
        doc_label=rung, doc_tokens=RUNG_TOKENS[rung],
        ctx_size=ctx, n_slots=1, ngl=99,
        cache_type_k=ctk, cache_type_v=ctv, flash_attn="on",
        swa_full=0, seed=GREEDY["seed"], temperature=GREEDY["temperature"],
        top_k=GREEDY["top_k"], n_predict=64,
        save_point="doc_prefill", page_cache_state="n/a",
        free_disk_gb_start=free_gb(),
        save_build_tag=cfg["llamacpp"]["release_tag"], restore_build_tag=cfg["llamacpp"]["release_tag"],
        verbosity="default(timed)/5(profile)",
    )


def profile_and_tokenize(model_path, ctx, extra_flags, save_dir, log_path, prompts):
    """-lv 5 launch: buffer sizes + token counts for a dict of prompts."""
    sp = ServerProc(BIN, model_path, PORT, ctx, save_dir, extra_flags=extra_flags,
                    verbosity=5, log_path=log_path)
    sp.start()
    counts = {k: len(sp.tokenize(v)) for k, v in prompts.items()}
    bufs = sp.parse_buffer_sizes()
    sp.stop()
    return counts, bufs


# ---------------------------------------------------------------------------
def block_a(rung, reps=5, warmup=1, model_key="primary", ctk="f16", ctv="f16",
            block="A", extra_server_flags=None, swa_full=False, csv_name=None,
            modes=("cold", "prefix_reuse", "ram_parked", "disk_restore"), keep_save=False,
            fa="on", side_experiment=0):
    # fa != "on" is allowed ONLY for the labeled side-cell (side_experiment=1) and
    # Block-D compatibility cells — never for headline measurements (Gate-0 amendment).
    if fa != "on" and not side_experiment:
        raise SystemExit("REFUSED: fa != 'on' outside a labeled side-experiment")
    m, model_path = model_entry(model_key)
    doc = doc_text(rung)
    tmpl, qs = questions()
    prompts = {q["id"]: tmpl.format(doc=doc, question=q["text"]) for q in qs}
    ctx = RUNG_TOKENS[rung] + 1024
    save_dir = os.path.join(REPO, "cache_saves") + "/"
    rawdir = os.path.join(REPO, "results/raw"); os.makedirs(rawdir, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + f"_block{block}_{rung}"
    csv_path = os.path.join(REPO, "results", csv_name or f"block{block}_{rung}.csv")
    prov = Provenance(os.path.join(REPO, "results/evidence.jsonl"))

    base_flags = ["-ctk", ctk, "-ctv", ctv, "-fa", fa] + list(extra_server_flags or [])
    if swa_full:
        base_flags.append("--swa-full")

    # profile: buffer sizes + all prompt token counts (incl. doc alone).
    # The doc cuts are LLAMA-token-exact; other tokenizers may yield MORE tokens for the
    # same text (Gemma ~+11%, Qwen +?%), so profile with generous ctx, then size the
    # measured servers from the MODEL'S OWN count: ctx = max(prompt counts) + 1024.
    ctx_profile = int(RUNG_TOKENS[rung] * 1.4) + 1024
    counts, bufs = profile_and_tokenize(
        model_path, ctx_profile, base_flags + ["--cache-ram", "0"], save_dir,
        os.path.join(rawdir, f"{run_id}_profile.log"), {**prompts, "_doc": doc})
    doc_tok_actual = counts["_doc"]
    ctx = max(counts.values()) + 1024
    if bufs.get("kv_cells") and bufs["kv_cells"] != ctx:
        # buffer sizes were profiled at ctx_profile; re-profile at the measured ctx so
        # kv_total_mib etc. describe the servers that actually produce the rows
        counts2, bufs = profile_and_tokenize(
            model_path, ctx, base_flags + ["--cache-ram", "0"], save_dir,
            os.path.join(rawdir, f"{run_id}_profile2.log"), {"_probe": "x"})

    def mk_common():
        c = common_for(block, run_id, m, rung, ctx, ctk, ctv, base_flags)
        c.update(kv_total_mib=bufs.get("kv_total_mib"), kv_k_type=bufs.get("kv_k_type"),
                 kv_v_type=bufs.get("kv_v_type"), compute_buffer_mtl_mib=bufs.get("compute_buffer_mtl_mib"),
                 model_buffer_mtl_mib=bufs.get("model_buffer_mtl_mib"),
                 projected_device_mib=bufs.get("projected_device_mib"), swa_full=int(swa_full),
                 flash_attn=fa, side_experiment=side_experiment)
        return c

    prov.run_started({**mk_common(), "scenario": f"block{block}_{rung}"})
    rows = []
    baselines = {}   # qid -> cold result (probe baselines)

    # ---- mode: cold (q1 x warmup+reps) + cold_baseline (q2..q5 x1) ----
    log = os.path.join(rawdir, f"{run_id}_cold.log")
    s = ServerProc(BIN, model_path, PORT, ctx, save_dir,
                   extra_flags=base_flags + ["--cache-ram", "0"], log_path=log)
    s.start(); s.start_rss()
    for i in range(warmup):
        complete(PORT, prompts["q1"], 64, cache_prompt=False)
    for rep in range(1, reps + 1):
        res = complete(PORT, prompts["q1"], 64, cache_prompt=False)
        baselines["q1"] = res
        r = base_row(mk_common()); r.update(mode="cold", rep=rep, is_warmup=0, question_id="q1",
                                            cache_ram_mib=0, full_prompt_tokens=counts["q1"],
                                            new_input_tokens=counts["q1"] - doc_tok_actual,
                                            server_log=os.path.relpath(log, REPO))
        rows.append(fill_metrics(r, res))
    for q in ("q2", "q3", "q4", "q5"):
        res = complete(PORT, prompts[q], 64, cache_prompt=False)
        baselines[q] = res
        r = base_row(mk_common()); r.update(mode="cold_baseline", rep=1, is_warmup=0, question_id=q,
                                            cache_ram_mib=0, full_prompt_tokens=counts[q],
                                            new_input_tokens=counts[q] - doc_tok_actual,
                                            server_log=os.path.relpath(log, REPO))
        rows.append(fill_metrics(r, res))
    s.stop_rss(); mem_cold = s.rss_stats_mb(); s.stop()
    for r in rows:
        r.update(**{k: v for k, v in mem_cold.items()})

    qorder = ["q1", "q2", "q3", "q4", "q5"]

    def warm_reps(s, mode, log, cache_ram_mib, prep=None):
        out = []
        # warmup (discarded row, q1)
        if prep:
            prep()
        complete(PORT, prompts["q1"], 64, cache_prompt=True)
        for rep in range(1, reps + 1):
            q = qorder[(rep - 1) % len(qorder)]
            if prep:
                prep()
            res = complete(PORT, prompts[q], 64, cache_prompt=True)
            new_input = counts[q] - doc_tok_actual
            r = base_row(mk_common()); r.update(mode=mode, rep=rep, is_warmup=0, question_id=q,
                                                cache_ram_mib=cache_ram_mib,
                                                full_prompt_tokens=counts[q], new_input_tokens=new_input,
                                                server_log=os.path.relpath(log, REPO))
            fill_metrics(r, res)
            probe_and_reuse(r, res, baselines[q], new_input)
            out.append(r)
        return out

    # ---- mode: prefix_reuse ----
    if "prefix_reuse" in modes:
      log = os.path.join(rawdir, f"{run_id}_prefix.log")
      s = ServerProc(BIN, model_path, PORT, ctx, save_dir,
                     extra_flags=base_flags + ["--cache-ram", "0", "--cache-reuse", "256"], log_path=log)
      s.start(); s.start_rss()
      complete(PORT, doc, 0, cache_prompt=True)          # prime doc only (n_predict=0)
      rws = warm_reps(s, "prefix_reuse", log, 0)
      for r in rws:
          r.update(cache_reuse_n=256)
      s.stop_rss(); mem = s.rss_stats_mb(); s.stop()
      for r in rws:
          r.update(**mem)
      rows += rws

    # ---- mode: ram_parked ----
    if "ram_parked" in modes:
      cram = CACHE_RAM_MIB[rung]
      log = os.path.join(rawdir, f"{run_id}_ram.log")
      s = ServerProc(BIN, model_path, PORT, ctx, save_dir,
                     extra_flags=base_flags + ["--cache-ram", str(cram)], log_path=log)
      s.start(); s.start_rss()
      complete(PORT, doc, 0, cache_prompt=True)          # prime doc
      def flush():
          complete(PORT, FLUSH_PROMPT, 1, cache_prompt=True)   # parks doc state to host RAM
      rws = warm_reps(s, "ram_parked", log, cram, prep=flush)
      s.stop_rss(); mem = s.rss_stats_mb(); s.stop()
      for r in rws:
          r.update(**mem)
      rows += rws

    # ---- mode: disk_restore (S1 prime+save, kill; S2 erase/restore per rep) ----
    save, restores = None, []
    savefile = f"blk{block}_{rung}_{ctk}.bin"
    if "disk_restore" in modes:
        log1 = os.path.join(rawdir, f"{run_id}_save.log")
        s1 = ServerProc(BIN, model_path, PORT, ctx, save_dir,
                        extra_flags=base_flags + ["--cache-ram", "0"], log_path=log1)
        s1.start()
        complete(PORT, doc, 0, cache_prompt=True)
        save = s1.save_slot(savefile)
        s1.stop()
        log2 = os.path.join(rawdir, f"{run_id}_restore.log")
        s2 = ServerProc(BIN, model_path, PORT, ctx, save_dir,
                        extra_flags=base_flags + ["--cache-ram", "0"], log_path=log2)
        s2.start(); s2.start_rss()
        def rr():
            s2.erase_slot(0)
            restores.append(s2.restore_slot(savefile))
        rws = warm_reps(s2, "disk_restore", log2, 0, prep=rr)
        s2.stop_rss(); mem = s2.rss_stats_mb(); s2.stop()
        for i, r in enumerate(rws):
            rest = restores[i + 1] if len(restores) > i + 1 else restores[-1]  # index 0 = warmup's restore
            r.update(**mem, page_cache_state="warm_read", save_file=savefile,
                     save_bytes=save["n_written"], save_ms=save["timings"]["save_ms"],
                     n_saved=save["n_saved"], restore_ms=rest["timings"]["restore_ms"],
                     restore_bytes=rest["n_read"], n_restored=rest["n_restored"])
        rows += rws

    write_csv(csv_path, rows)

    # provenance + verdicts
    warm_rows = [r for r in rows if r["mode"] in ("prefix_reuse", "ram_parked", "disk_restore")]
    for r in rows:
        prov.measurement(run_id, r, {"mode": r["mode"], "rep": r["rep"], "q": r["question_id"],
                                     "ttft_s": r["client_ttft_s"], "prompt_n": r["server_prompt_n"]})
    probe_pass = all(r["probe_result"] == "PASS" for r in warm_rows)
    reuse_pass = all(r["reuse_bool"] for r in warm_rows)
    prov.verification(run_id, {"check": "probe+reuse", "probe_all_pass": probe_pass,
                               "reuse_all_pass": reuse_pass, "warm_rows": len(warm_rows)})
    prov.run_completed(run_id, {"probe_pass": probe_pass, "reuse_pass": reuse_pass,
                                "rows": len(rows)})
    if savefile and not keep_save:
        remove_files([os.path.join(save_dir, savefile)])
    clean = enforce(os.path.join(REPO, "cache_saves"), cfg["DISK_BUDGET_GB"])

    med = lambda xs: round(statistics.median(xs), 4) if xs else None
    summary = {"rung": rung, "block": block, "model": m["name"], "ctk": ctk, "rows": len(rows),
               "cold_ttft_median": med([r["client_ttft_s"] for r in rows if r["mode"] == "cold"]),
               "cold_prefill_tok_s_median": med([r["server_prompt_tok_s"] for r in rows if r["mode"] == "cold"]),
               "decode_tok_s_median": med([r["server_predicted_tok_s"] for r in rows if r["mode"] in ("cold", "disk_restore")]),
               "prefix_ttft_median": med([r["client_ttft_s"] for r in rows if r["mode"] == "prefix_reuse"]),
               "ram_ttft_median": med([r["client_ttft_s"] for r in rows if r["mode"] == "ram_parked"]),
               "disk_ttft_median": med([r["client_ttft_s"] for r in rows if r["mode"] == "disk_restore"]),
               "save_bytes": save["n_written"] if save else None,
               "save_ms": save["timings"]["save_ms"] if save else None,
               "restore_ms_all": [rt["timings"]["restore_ms"] for rt in restores],
               "probe_all_pass": probe_pass, "reuse_all_pass": reuse_pass,
               "ram_prompt_n": [r["server_prompt_n"] for r in rows if r["mode"] == "ram_parked"],
               "free_gb": free_gb(), "cleanup": clean}
    print(json.dumps(summary))
    return summary


def ram_evidence(rung="8k", model_key="primary"):
    """One verbose (-lv 10) prime->flush->reload cycle; grep the RAM-hit log line."""
    m, model_path = model_entry(model_key)
    doc = doc_text(rung)
    tmpl, qs = questions()
    p1 = tmpl.format(doc=doc, question=qs[0]["text"])
    ctx = RUNG_TOKENS[rung] + 1024
    rawdir = os.path.join(REPO, "results/raw")
    log = os.path.join(rawdir, f"ram_evidence_{rung}.log")
    s = ServerProc(BIN, model_path, PORT, ctx, os.path.join(REPO, "cache_saves") + "/",
                   extra_flags=["-ctk", "f16", "-ctv", "f16", "-fa", "on",
                                "--cache-ram", str(CACHE_RAM_MIB[rung])],
                   verbosity=10, log_path=log)
    s.start()
    complete(PORT, doc, 0, cache_prompt=True)
    complete(PORT, FLUSH_PROMPT, 1, cache_prompt=True)
    res = complete(PORT, p1, 64, cache_prompt=True)
    s.stop()
    hits = [ln.strip() for ln in open(log, encoding="utf-8", errors="replace")
            if ("found better prompt" in ln or "saving prompt" in ln or
                "cache state:" in ln)]
    print(json.dumps({"rung": rung, "prompt_n_after_reload": res["prompt_n"],
                      "ttft_s": res["ttft_s"], "log": os.path.relpath(log, REPO),
                      "evidence_lines": hits[:12]}, indent=2))


# ---------------------------------------------------------------------------
def block_c(rungs=("2k", "8k", "32k"), types=("f16", "q8_0", "q4_0"), reps=5, warmup=1):
    """Quantized-cache grid: cold + disk_restore per (rung x type). fa on everywhere
    (required for quantized V anyway) -> settles the Gate-0 decode question unconfounded."""
    out = []
    for rung in rungs:
        for t in types:
            print(f"### C cell rung={rung} type={t}", flush=True)
            out.append(block_a(rung, reps=reps, warmup=warmup, ctk=t, ctv=t, block="C",
                               modes=("cold", "disk_restore"),
                               csv_name=f"blockC_{rung}_{t}.csv"))
    print(json.dumps({"blockC_cells": len(out)}))
    return out


def d_cell(cell_id, question_text, doc_source, *, save_cfg=None, restore_cfg,
           existing_savefile=None, reps=5, csv_name="blockD.csv", note=""):
    """One Block-D portability cell.

    save_cfg/restore_cfg: dict(bin, flags(list), ngl, ctx, tag)  — tag = build tag label.
    doc_source: path to the document text file (prompt prefix that was/will be saved).
    Classification per rep: FAILS-CLEAN (restore HTTP error / server death) /
    SILENTLY-WRONG (probe fail) / SILENTLY-INEFFECTIVE (reuse fail) / WORKS.
    Baseline = cold generation on the RESTORE-SIDE server config (same full prompt).
    """
    import httpx
    m, _ = model_entry("primary")
    save_dir = os.path.join(REPO, "cache_saves") + "/"
    rawdir = os.path.join(REPO, "results/raw"); os.makedirs(rawdir, exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + f"_D_{cell_id}"
    csv_path = os.path.join(REPO, "results", csv_name)
    prov = Provenance(os.path.join(REPO, "results/evidence.jsonl"))
    doc = open(doc_source, encoding="utf-8").read()
    full_prompt = doc + question_text

    savefile = existing_savefile or f"blkD_{cell_id}.bin"
    link_to_cleanup = None
    if existing_savefile and "/" in existing_savefile:
        # The slots API accepts plain filenames only (fs_validate_filename rejects '/').
        # Hardlink the keep/ file into the save dir root — zero-copy, original untouched.
        src = os.path.join(save_dir, existing_savefile)
        savefile = os.path.basename(existing_savefile)
        dst = os.path.join(save_dir, savefile)
        if not os.path.exists(dst):
            os.link(src, dst)
            link_to_cleanup = dst
    # --- save side (skipped when reusing an existing file, e.g. keep/ or shared) ---
    if existing_savefile is None:
        sc = save_cfg
        s1 = ServerProc(sc["bin"], sc["model"], PORT, sc["ctx"], save_dir,
                        extra_flags=sc["flags"], ngl=sc["ngl"],
                        log_path=os.path.join(rawdir, f"{run_id}_save.log"))
        s1.start()
        complete(PORT, doc, 0, cache_prompt=True)
        save = s1.save_slot(savefile)
        s1.stop()
    else:
        save = None

    rc = restore_cfg
    log2 = os.path.join(rawdir, f"{run_id}_restore.log")
    s2 = ServerProc(rc["bin"], rc["model"], PORT, rc["ctx"], save_dir,
                    extra_flags=rc["flags"], ngl=rc["ngl"], log_path=log2)
    s2.start()
    # counts + baseline on restore side
    full_tok = len(s2.tokenize(full_prompt))
    doc_tok = len(s2.tokenize(doc))
    new_input = full_tok - doc_tok
    baseline = complete(PORT, full_prompt, 64, cache_prompt=False)

    sha, dirty = git_sha()
    common = dict(schema_version=SCHEMA_VERSION, block="D", run_id=run_id, ts_start=now(),
                  scenario=f"D_{cell_id}", side_experiment=0, harness_git_sha=sha,
                  harness_git_dirty=dirty, llamacpp_tag=rc["tag"],
                  llamacpp_sha=rc.get("sha", ""), model_name=m["name"], model_sha256=m["sha256"],
                  doc_label=os.path.basename(doc_source), doc_tokens=doc_tok, question_id="d",
                  full_prompt_tokens=full_tok, new_input_tokens=new_input,
                  ctx_size=rc["ctx"], n_slots=1, ngl=rc["ngl"],
                  cache_type_k=rc.get("ctk", "f16"), cache_type_v=rc.get("ctv", "f16"),
                  flash_attn=rc.get("fa", "on"), swa_full=0, cache_ram_mib=0,
                  seed=GREEDY["seed"], temperature=GREEDY["temperature"], top_k=GREEDY["top_k"],
                  n_predict=64, save_point=("after_generation" if existing_savefile and "phase0" in savefile
                                            else "doc_prefill"),
                  page_cache_state="warm_read", free_disk_gb_start=free_gb(),
                  save_build_tag=(save_cfg or {}).get("tag", "gate0-b9871" if existing_savefile else ""),
                  restore_build_tag=rc["tag"],
                  save_flags_json=json.dumps({k: v for k, v in (save_cfg or {}).items()
                                              if k in ("flags", "ngl", "ctx", "tag")}),
                  save_file=savefile, server_log=os.path.relpath(log2, REPO),
                  verbosity="default")
    rows = []
    b = base_row(common); b.update(mode="cold_baseline", rep=1, is_warmup=0, page_cache_state="n/a")
    rows.append(fill_metrics(b, baseline))

    outcome_final = None
    for rep in range(1, reps + 1):
        r = base_row(common); r.update(mode="d_restore", rep=rep, is_warmup=0)
        try:
            s2.erase_slot(0)
            rest = s2.restore_slot(savefile)
            r.update(restore_ms=rest["timings"]["restore_ms"], restore_bytes=rest["n_read"],
                     n_restored=rest["n_restored"])
        except httpx.HTTPStatusError as e:
            err = e.response.text[:300]
            r.update(outcome="FAILS-CLEAN", reuse_assert="N/A", probe_result="N/A",
                     server_timings_json=json.dumps({"restore_error": err}), ts_end=now())
            rows.append(r)
            outcome_final = ("FAILS-CLEAN", err)
            break
        except (httpx.TransportError, RuntimeError) as e:
            r.update(outcome="FAILS-CLEAN", reuse_assert="N/A", probe_result="N/A",
                     server_timings_json=json.dumps({"restore_error": f"transport/crash: {e}"[:300]}),
                     ts_end=now())
            rows.append(r)
            outcome_final = ("FAILS-CLEAN", str(e)[:200])
            break
        res = complete(PORT, full_prompt, 64, cache_prompt=True)
        fill_metrics(r, res)
        probe_and_reuse(r, res, baseline, new_input)
        if save:
            r.update(save_bytes=save["n_written"], save_ms=save["timings"]["save_ms"],
                     n_saved=save["n_saved"])
        rows.append(r)
        outcome_final = (r["outcome"], "")
    s2.stop()
    write_csv(csv_path, rows)
    prov.run_started({**common, "scenario": f"D_{cell_id}"})
    for r in rows:
        prov.measurement(run_id, r, {"mode": r["mode"], "rep": r["rep"], "outcome": r.get("outcome", "")})
    prov.verification(run_id, {"check": "d_taxonomy", "cell": cell_id, "outcome": outcome_final[0],
                               "detail": outcome_final[1], "note": note})
    prov.run_completed(run_id, {"cell": cell_id, "outcome": outcome_final[0]})
    if existing_savefile is None:
        remove_files([os.path.join(save_dir, savefile)])
    if link_to_cleanup:
        os.remove(link_to_cleanup)   # remove the hardlink only; keep/ original untouched
    print(json.dumps({"cell": cell_id, "outcome": outcome_final[0], "detail": outcome_final[1][:150],
                      "rows": len(rows), "note": note}), flush=True)
    return {"cell": cell_id, "outcome": outcome_final[0], "detail": outcome_final[1], "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("blockA"); a.add_argument("--rung", required=True); a.add_argument("--reps", type=int, default=5); a.add_argument("--warmup", type=int, default=1)
    c = sub.add_parser("blockC")
    e = sub.add_parser("ramEvidence"); e.add_argument("--rung", default="8k")
    args = ap.parse_args()
    if args.cmd == "blockA":
        block_a(args.rung, reps=args.reps, warmup=args.warmup)
    elif args.cmd == "blockC":
        block_c()
    elif args.cmd == "ramEvidence":
        ram_evidence(args.rung)
