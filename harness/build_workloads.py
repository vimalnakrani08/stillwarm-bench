#!/usr/bin/env python3
"""Build token-exact doc-chat cuts from the frozen Gutenberg source, using THIS
build's Llama-3.1-8B tokenizer via a running llama-server /tokenize + /detokenize.

Requires a llama-server (Llama-3.1-8B) already listening on --port. Deterministic:
tokenize(full) -> first N ids -> detokenize -> write -> re-tokenize to record the
ACTUAL count (boundary merges can shift it by a token or two; recorded honestly).

Usage: build_workloads.py [--port 8080]
"""
import argparse, json, sys, os
import httpx

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "workloads/source/frankenstein_pg84.txt")
OUTDIR = os.path.join(REPO, "workloads/docchat")
LADDER = [("2k", 2048), ("4k", 4096), ("8k", 8192), ("16k", 16384), ("32k", 32768),
          ("64k", 65536)]   # source has 97,954 tokens, so 64K is the top rung

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8080)
args = ap.parse_args()
BASE = f"http://127.0.0.1:{args.port}"


def tokenize(text):
    r = httpx.post(f"{BASE}/tokenize", json={"content": text}, timeout=180)
    r.raise_for_status()
    return r.json()["tokens"]


def detokenize(toks):
    r = httpx.post(f"{BASE}/detokenize", json={"tokens": toks}, timeout=180)
    r.raise_for_status()
    return r.json()["content"]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    text = open(SRC, encoding="utf-8").read()
    full = tokenize(text)
    print(json.dumps({"source": os.path.relpath(SRC, REPO),
                      "source_chars": len(text), "full_tokens": len(full)}))
    summary = []
    for label, N in LADDER:
        if len(full) < N:
            print(f"SKIP {label}: source only has {len(full)} tokens (< {N})")
            continue
        cut_toks = full[:N]
        cut_text = detokenize(cut_toks)
        path = os.path.join(OUTDIR, f"doc_{label}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(cut_text)
        actual = len(tokenize(cut_text))
        row = {"label": label, "target_tokens": N, "actual_tokens": actual,
               "chars": len(cut_text), "file": os.path.relpath(path, REPO)}
        summary.append(row)
        print(json.dumps(row))
    with open(os.path.join(OUTDIR, "cuts_tokencount.json"), "w") as f:
        json.dump({"full_tokens": len(full), "cuts": summary}, f, indent=2)
    print("wrote", os.path.join(OUTDIR, "cuts_tokencount.json"))


if __name__ == "__main__":
    main()
