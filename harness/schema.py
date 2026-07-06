#!/usr/bin/env python3
"""CSV schema v1 — FROZEN 2026-07-04 (Gate-1 housekeeping). All Phase-2 rows use v1.

Changes vs the unversioned Gate-1 dry-run schema:
  + schema_version        "1"
  + block                 experiment block (A|B|C|D|E|side|pre)
  + side_experiment       1 = labeled side-cell, excluded from main charts
  + save_point            doc_prefill | after_generation | ""   (Gate-0 keep/ files
                          are after_generation; Phase-2 saves are doc_prefill)
  + page_cache_state      warm_read | cold_read | n/a  (restore rows; n/a elsewhere)
  + free_disk_gb_start    free disk (GB, df) at run start
  + cache_reuse_n         value of --cache-reuse if set, else ""
  + cache_ram_mib         value of --cache-ram if set, else ""
  + save_build_tag        llama.cpp tag that WROTE the save file (D3 cross-build)
  + restore_build_tag     llama.cpp tag that RESTORED it (== serving process)
  + save_flags_json       exact flags of the SAVING server when != serving row (D4-D6)
  + outcome               D/E taxonomy: WORKS | FAILS-CLEAN | SILENTLY-WRONG |
                          SILENTLY-INEFFECTIVE | ""
  ~ ctx -> ctx_size       (rename; same meaning: -c passed to the serving process)

Rules of record: one row per measured rep; warm/restored rows always carry
probe_result + reuse_assert; server_timings_json is the server's object verbatim.
"""

SCHEMA_VERSION = "1.1"

# v1.1 (Gate-2 correction, additive only): + resume_total_ms — the user-felt cost of
# resuming from disk = restore_ms + client TTFT of the follow-up request, computed
# per row for disk-restore rows (blank elsewhere). The headline table is built on
# resume_total; TTFT-only is the secondary "post-restore TTFT" view.

CSV_FIELDS_V1 = [
    "schema_version", "run_id", "ts_start", "ts_end", "block", "scenario", "mode",
    "rep", "is_warmup", "side_experiment",
    "harness_git_sha", "harness_git_dirty", "llamacpp_tag", "llamacpp_sha",
    "model_name", "model_sha256",
    "doc_label", "doc_tokens", "question_id", "full_prompt_tokens", "new_input_tokens",
    "ctx_size", "n_slots", "ngl", "cache_type_k", "cache_type_v", "flash_attn", "swa_full",
    "cache_reuse_n", "cache_ram_mib", "seed", "temperature", "top_k", "n_predict",
    "client_ttft_s", "client_total_s",
    "server_prompt_n", "server_prompt_ms", "server_predicted_n", "server_predicted_ms",
    "server_prompt_tok_s", "server_predicted_tok_s", "tokens_cached", "tokens_evaluated",
    "server_timings_json",
    "reuse_assert", "reuse_bool", "probe_result", "probe_baseline_sha", "probe_restored_sha",
    "save_point", "page_cache_state", "save_file", "save_bytes", "save_ms", "n_saved",
    "restore_ms", "restore_bytes", "n_restored",
    "save_build_tag", "restore_build_tag", "save_flags_json", "outcome",
    "kv_total_mib", "kv_k_type", "kv_v_type", "compute_buffer_mtl_mib", "model_buffer_mtl_mib",
    "projected_device_mib", "rss_peak_mb", "rss_mean_mb", "rss_n",
    "free_disk_gb_start", "gen_content_sha256", "gen_tokens_sha256", "server_log", "verbosity",
    "resume_total_ms",   # v1.1: restore_ms + client_ttft_s*1000 (disk-restore rows only)
]
