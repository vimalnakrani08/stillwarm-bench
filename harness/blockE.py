#!/usr/bin/env python3
"""Block E — architectures (reduced grid): Qwen2.5-7B (low-KV-head) and
Gemma-3-4B (SWA, --swa-full) x {8k, 32k} x {f16, q8_0}, cold vs disk_restore.
PLUS one deliberate Gemma no---swa-full cell: the documented SILENTLY-INEFFECTIVE
exhibit (Gate-0 finding, now measured under the frozen harness)."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase2 import block_a

GRID_MODELS = ["low_kv_head", "swa"]     # config.yaml registry keys
# Per-model rungs. Qwen2.5's n_ctx_train is exactly 32768 and llama-server CAPS -c to it
# ("the slot context ... exceeds the training context of the model (32768) - capping");
# the frozen 32K doc = 32,803 Qwen tokens (+35 vs Llama) -> 32K rung cannot fit. The long
# rung for Qwen is therefore 16k (documented architecture finding, not a harness choice).
RUNGS_BY_MODEL = {"low_kv_head": ["8k", "16k"], "swa": ["8k", "32k"]}
TYPES = ["f16", "q8_0"]

if __name__ == "__main__":
    out = []
    for mk in GRID_MODELS:
        for rung in RUNGS_BY_MODEL[mk]:
            for t in TYPES:
                csv_name = f"blockE_{mk}_{rung}_{t}.csv"
                if os.path.exists(os.path.join(os.path.dirname(HERE), "results", csv_name)):
                    print(f"### E cell model={mk} rung={rung} type={t} SKIP (csv exists)", flush=True)
                    continue
                print(f"### E cell model={mk} rung={rung} type={t}", flush=True)
                out.append(block_a(rung, model_key=mk, ctk=t, ctv=t, block="E",
                                   modes=("cold", "disk_restore"),
                                   swa_full=(mk == "swa"),
                                   csv_name=csv_name))
    # Exhibit: Gemma WITHOUT --swa-full (expected SILENTLY-INEFFECTIVE; that is the point)
    print("### E exhibit: gemma 8k f16 WITHOUT --swa-full", flush=True)
    out.append(block_a("8k", model_key="swa", ctk="f16", ctv="f16", block="E",
                       modes=("cold", "disk_restore"), swa_full=False,
                       csv_name="blockE_swa_8k_f16_NOSWAFULL.csv"))
    print("### BLOCK E SUMMARY ###")
    print(json.dumps([{k: s.get(k) for k in ("model", "rung", "ctk", "rows", "cold_ttft_median",
                                             "disk_ttft_median", "save_bytes", "probe_all_pass",
                                             "reuse_all_pass")} for s in out], indent=2))
