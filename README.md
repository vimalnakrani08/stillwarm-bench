# stillwarm-bench

**Does saving a llama.cpp conversation's KV cache to disk actually work — and
what is it worth?** A frozen-harness benchmark on consumer Apple Silicon, plus
every measured row, behind the [stillwarm](<GITHUB_TOOL_URL>) tool.

**Setup (pinned):** MacBook Pro M3 Max (36 GB unified memory), macOS 26.5.1,
llama.cpp release **b9871** (`ef2d770117db45b05aa7ecd1b0acca36370c5470`, Metal,
Release build). Models pinned by SHA-256 (recorded in every CSV row's
`model_sha256` column): Llama-3.1-8B-Instruct Q4_K_M, Qwen2.5-7B-Instruct
Q4_K_M, Gemma-3-4B-it Q4_K_M. Greedy decoding, flash-attention explicitly on,
warmup 1 + N=5 reps per cell.

## Headline (same-session, thermally controlled — `results/blockA2_supervised.csv`)

| context | cold prefill | resume (restore + first token) | speedup |
|---|---|---|---|
| 8,192 tok | 18.995 s | 0.266 s | **71×** |
| 32,768 tok | 157.42 s | 0.989 s | **159×** |
| 65,536 tok | 441.58 s | 2.197 s | **201×** |

Save files run ~128 KiB/token f16 (q8_0 exactly 0.531×, q4_0 0.281× — 
`results/blockC_*.csv`). Even with a **purged page cache**, restore wins above
~132–295 tokens (`results/blockB_coldread_*.csv`; purged reads at 5.3–6.1 GB/s
≈ 84–95% of raw `dd` on the same file).

## What's here

```
harness/     the runner, schema, orchestrators, supervised-session protocol
workloads/   frozen inputs (public-domain text, token-exact cuts, MANIFEST.sha256)
analysis/    break-even / aggregate / figure scripts + the final figures
results/     every measured row (schema v1/v1.1 CSVs) + hash-chained evidence.jsonl
```

The complete dataset (with a data card, raw powermetrics capture, and integrity
manifest) is also published on Hugging Face: `<HF_DATASET_URL>`.

## Figures (each computed from the named files only)

- **fig1_headline.png** — cold vs resume-total, medians of 5 interleaved
  same-session reps. Source: `results/blockA2_supervised.csv`.
- **fig2_breakeven.png** — cold prefill vs resume in both page-cache states,
  with the measured 132–295-token break-even zone. Sources:
  `results/blockA2_supervised.csv`, `results/blockB_coldread_{8k,32k,64k}.csv`,
  `results/blockB_dd_reference.json`.
- **fig3_energy.png** — Wh per 32K turn, cold vs restore (powermetrics, 1 s
  sampling, CPU+GPU+ANE). Sources: `results/energy_sidebar.json`, markers in
  `results/energy_markers.json`, raw capture in the HF dataset (`raw/powermetrics_32k.txt`).
- **fig4_thermal_inversion.png** — 64K restores: purged-but-cool beat
  cached-but-hot; `dd` contrast (21.2 vs 6.4 GB/s) rules out page cache.
  Sources: `results/blockA2_supervised.csv`, `results/blockB_coldread_64k.csv`,
  `results/blockB_dd_reference.json`.

## Findings in one paragraph each

**Portability (`results/blockD.csv`):** the only hard invalidator found is
flash-attention on↔off (clean 400, both directions). Cross-build restore works
in both directions across ≥5 weeks of releases (save/restore build tags are
columns in the CSV); `-ngl` changes and fitting-ctx changes are safe;
state-larger-than-ctx is refused cleanly.

**Silently-ineffective restores (`results/blockE_swa_8k_f16_NOSWAFULL.csv`):**
on an SWA model (Gemma-3) without `--swa-full`, every restore "succeeds" and
then re-prefills anyway (`prompt_n` ≈ 520 vs the ~29 expected) — you pay ~8.5×
the proper resume cost without any error. The stillwarm tool ships a runtime
guard for exactly this.

**Is a restore bit-identical? (`results/blockC_*.csv`)** Two different
properties: restores are *deterministic* (same state → same bytes, every case
examined), but *cold-equivalence* (restored continuation == full recompute)
fails without corruption: 0/5 at f16, 1/5 at q8_0, **5/5 at q4_0** (8K rung) —
different prefill batch splits flip marginal-logit tokens and cache
quantization amplifies it; both outputs stay coherent. The CSV columns
`probe_result` / `reuse_assert` / `outcome` carry both checks for all 256
warm rows (`results/blockF_aggregate.json` totals them).

**Thermal honesty:** absolute throughput is thermal-state-dependent (same-rung
cold prefill measured 473→310 tok/s across a session; see the dataset card's
notes). Headline numbers therefore come from interleaved same-session
measurement only; the full 2K–64K ladder (`results/blockA_*.csv`) is kept as a
thermally-mixed appendix.

## Reproduce

`harness/` contains everything: `runner.py`/`phase2.py` (scenarios, schema v1.1,
refusing any measured run without `-fa on`), `build_workloads.py` (token-exact
cuts; workloads ship frozen with SHA-256 manifest), `supervised_session.py`
(the thermally-controlled + purged-page-cache protocol, with artifact
verification), `coldread.py`. Python 3.12, deps in `harness/` docstrings;
llama.cpp pinned at b9871.

License: MIT (code) — the results CSVs are also published under CC-BY-4.0 in
the HF dataset. Author: Vimal Nakrani.
