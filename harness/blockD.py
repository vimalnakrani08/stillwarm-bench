#!/usr/bin/env python3
"""Block D — portability cells d2..d6 (d1 = Block A 8K disk_restore baseline).

Outcome taxonomy per cell: WORKS / FAILS-CLEAN / SILENTLY-WRONG (probe fail) /
SILENTLY-INEFFECTIVE (reuse fail). Baseline = cold gen on the RESTORE-side config.

fa=off appears ONLY as a compatibility axis (d5), per the Gate-2 instructions —
it is not a headline measurement. Mac->Linux is deferred to the Space (Phase 4).
"""
import json, os, sys, traceback
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import phase2
from phase2 import d_cell, model_entry, REPO

NEW_BIN = "/Users/vimal/llamacpp-stillwarm/build/bin/llama-server"
OLD_BIN = "/Users/vimal/llamacpp-old-b9386/build/bin/llama-server"
NEW_TAG, OLD_TAG = "b9871", "b9386"

m, MODEL = model_entry("primary")
DOC_8K = os.path.join(REPO, "workloads/docchat/doc_8k.txt")
KEEP_PROMPT = os.path.join(REPO, "cache_saves/keep/prompt_8k_phase0.txt")

Q_STD = ("\n\n---\nBased only on the text above, answer concisely.\n"
         "Question: Summarize the text above in three sentences.\nAnswer:")
Q_PHASE0 = ("\n\nBased only on the briefing above, answer concisely.\n"
            "Q: What is the project codename, and where is the primary compute cluster installed?\nA:")

F16 = ["-ctk", "f16", "-ctv", "f16"]
def cfgd(bin, tag, *, fa="on", ngl=99, ctx=16384, ctk="f16", ctv="f16"):
    return {"bin": bin, "tag": tag, "model": MODEL, "ngl": ngl, "ctx": ctx,
            "ctk": ctk, "ctv": ctv, "fa": fa,
            "flags": ["-ctk", ctk, "-ctv", ctv, "-fa", fa, "--cache-ram", "0"]}

BASE_NEW = cfgd(NEW_BIN, NEW_TAG)

CELLS = [
    # d2 — Gate-0 keep/ files (same build, old flags incl. fa=auto at save, save AFTER generation)
    ("d2_keep_f16", dict(question_text=Q_PHASE0, doc_source=KEEP_PROMPT,
                         existing_savefile="keep/phase0_llama_8k_f16.bin",
                         restore_cfg=BASE_NEW,
                         note="Gate-0 f16 keep file (fa=auto at save, after_generation) -> b9871 fa=on")),
    ("d2_keep_q8", dict(question_text=Q_PHASE0, doc_source=KEEP_PROMPT,
                        existing_savefile="keep/phase0_llama_8k_q8.bin",
                        restore_cfg=cfgd(NEW_BIN, NEW_TAG, ctk="q8_0", ctv="q8_0"),
                        note="Gate-0 q8_0 keep file -> b9871 q8_0 fa=on")),
    # d3 — cross-build, both directions (f16, fa on, 8K)
    ("d3_old_to_new", dict(question_text=Q_STD, doc_source=DOC_8K,
                           save_cfg=cfgd(OLD_BIN, OLD_TAG), restore_cfg=BASE_NEW,
                           note=f"save@{OLD_TAG} -> restore@{NEW_TAG}")),
    ("d3_new_to_old", dict(question_text=Q_STD, doc_source=DOC_8K,
                           save_cfg=BASE_NEW, restore_cfg=cfgd(OLD_BIN, OLD_TAG),
                           note=f"save@{NEW_TAG} -> restore@{OLD_TAG}")),
    # d4 — ngl mismatch (b9871)
    ("d4_ngl999_to_24", dict(question_text=Q_STD, doc_source=DOC_8K,
                             save_cfg=cfgd(NEW_BIN, NEW_TAG, ngl=999),
                             restore_cfg=cfgd(NEW_BIN, NEW_TAG, ngl=24),
                             note="save -ngl 999 -> restore -ngl 24")),
    ("d4_ngl24_to_999", dict(question_text=Q_STD, doc_source=DOC_8K,
                             save_cfg=cfgd(NEW_BIN, NEW_TAG, ngl=24),
                             restore_cfg=cfgd(NEW_BIN, NEW_TAG, ngl=999),
                             note="save -ngl 24 -> restore -ngl 999")),
    # d5 — fa mismatch (compat axis; fa off allowed here only)
    ("d5_faon_to_faoff", dict(question_text=Q_STD, doc_source=DOC_8K,
                              save_cfg=cfgd(NEW_BIN, NEW_TAG, fa="on"),
                              restore_cfg=cfgd(NEW_BIN, NEW_TAG, fa="off"),
                              note="save fa=on -> restore fa=off")),
    ("d5_faoff_to_faon", dict(question_text=Q_STD, doc_source=DOC_8K,
                              save_cfg=cfgd(NEW_BIN, NEW_TAG, fa="off"),
                              restore_cfg=cfgd(NEW_BIN, NEW_TAG, fa="on"),
                              note="save fa=off -> restore fa=on")),
    # d6 — ctx mismatch (state is ~8193 tok)
    ("d6_ctx16k_to_8k", dict(question_text=Q_STD, doc_source=DOC_8K,
                             save_cfg=cfgd(NEW_BIN, NEW_TAG, ctx=16384),
                             restore_cfg=cfgd(NEW_BIN, NEW_TAG, ctx=8192),
                             note="save ctx 16384 -> restore ctx 8192 (state ~8193 tok)")),
    ("d6_ctx16k_to_32k", dict(question_text=Q_STD, doc_source=DOC_8K,
                              save_cfg=cfgd(NEW_BIN, NEW_TAG, ctx=16384),
                              restore_cfg=cfgd(NEW_BIN, NEW_TAG, ctx=32768),
                              note="save ctx 16384 -> restore ctx 32768")),
]

if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    results = []
    for cell_id, kw in CELLS:
        if only and only != cell_id:
            continue
        try:
            r = d_cell(cell_id, **kw)
            results.append({"cell": cell_id, "outcome": r["outcome"], "detail": r["detail"][:120]})
        except Exception as e:
            # a cell whose restore-side SERVER cannot even start with the config
            traceback.print_exc()
            results.append({"cell": cell_id, "outcome": "FAILS-CLEAN",
                            "detail": f"cell-level exception: {e}"[:200]})
            os.system("lsof -ti tcp:8080 2>/dev/null | xargs kill 2>/dev/null")
    print("### BLOCK D TAXONOMY ###")
    print(json.dumps(results, indent=2))
