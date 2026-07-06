#!/usr/bin/env python3
"""Streaming /completion client. TTFT = request-sent -> first non-empty CONTENT
SSE chunk (role-only/empty chunks ignored, per PLAN §5.3). Captures the server's
own `timings` object verbatim, the generated token ids (for byte-exact probes),
and prompt_n (tokens actually prefilled -> reuse evidence)."""
from __future__ import annotations
import json, time, hashlib
import httpx

GREEDY = {"temperature": 0, "top_k": 1, "seed": 42}


def complete(port, prompt, n_predict, *, cache_prompt=True, id_slot=0,
             sampling=None, timeout=900):
    body = {"prompt": prompt, "n_predict": n_predict, "cache_prompt": cache_prompt,
            "id_slot": id_slot, "stream": True, "return_tokens": True}
    body.update(sampling or GREEDY)

    t0 = time.perf_counter()
    ttft = None
    content, gen_tokens, final = [], [], {}
    with httpx.stream("POST", f"http://127.0.0.1:{port}/completion", json=body, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            chunk = obj.get("content", "")
            if chunk and ttft is None:
                ttft = time.perf_counter() - t0
            if chunk:
                content.append(chunk)
            toks = obj.get("tokens") or []
            if toks:
                gen_tokens.extend(toks)
            if obj.get("stop") or obj.get("stopped_eos") or obj.get("stopped_limit"):
                final = obj
    total = time.perf_counter() - t0

    tim = final.get("timings", {}) if isinstance(final, dict) else {}
    text = "".join(content)
    return {
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "total_s": round(total, 4),
        "content": text,
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "gen_tokens": gen_tokens,
        "gen_tokens_sha256": hashlib.sha256(
            json.dumps(gen_tokens, separators=(",", ":")).encode()).hexdigest(),
        "server_timings": tim,                       # verbatim
        "prompt_n": tim.get("prompt_n"),             # tokens prefilled THIS request
        "prompt_ms": tim.get("prompt_ms"),
        "predicted_n": tim.get("predicted_n"),
        "predicted_ms": tim.get("predicted_ms"),
        "prompt_per_second": tim.get("prompt_per_second"),
        "predicted_per_second": tim.get("predicted_per_second"),
        "tokens_cached": final.get("tokens_cached"),
        "tokens_evaluated": final.get("tokens_evaluated"),
    }
