#!/usr/bin/env python3
"""AuditWeave provenance for the harness (PLAN §10). Optional import; if auditweave
is absent this degrades to a no-op and the harness runs unchanged.

CANDIDATE-ADAPTER FINDING (dogfooding, recorded in LABLOG): AuditWeave's event
vocabulary is RAG/data-pipeline shaped — SOURCE / RETRIEVAL / TRANSFORMATION /
INFERENCE / DECISION / ATTESTATION — with NO benchmark lifecycle types. The four
benchmark events the plan wants are mapped onto the closest generics + a
`benchmark_event` label:

    RUN_STARTED   -> SOURCE       (frozen inputs + config/SHA hashes = the run's source)
    MEASUREMENT   -> INFERENCE    (a model produced tokens + timings)
    VERIFICATION  -> ATTESTATION  (probe/reuse verdict)  *semantic mismatch: ATTESTATION
                                   implies a HUMAN signer; ours is automated -> actor kind=system*
    RUN_COMPLETED -> DECISION     (the run reached a conclusion)

Recommendation to AuditWeave: add a benchmark/experiment event family (or a generic
MEASUREMENT + VERIFICATION pair whose actor may be `system`) so automated eval
harnesses don't have to overload ATTESTATION.
"""
from __future__ import annotations
import json, hashlib

try:
    from auditweave import Event, Actor, EventType
    from auditweave.core.trail import Trail
    from auditweave.store.jsonl import JsonlStore
    HAVE_AUDITWEAVE = True
except Exception:
    HAVE_AUDITWEAVE = False

BENCH_MAP = {
    "RUN_STARTED": "source",
    "MEASUREMENT": "inference",
    "VERIFICATION": "attestation",
    "RUN_COMPLETED": "decision",
}

ADAPTER_NOTE = ("AuditWeave lacks benchmark lifecycle event types; mapped "
                "RUN_STARTED->source, MEASUREMENT->inference, VERIFICATION->attestation "
                "(attestation implies a human signer; ours is automated system actor), "
                "RUN_COMPLETED->decision, each carrying a benchmark_event label.")


def sha256_json(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     default=str).encode("utf-8")).hexdigest()


class Provenance:
    """Append hash-chained evidence to results/evidence.jsonl via AuditWeave."""

    def __init__(self, path, enabled=True):
        self.enabled = bool(enabled and HAVE_AUDITWEAVE)
        self.path = str(path)
        self.reason = None
        if not HAVE_AUDITWEAVE:
            self.reason = "auditweave not importable (no-op)"
        elif not enabled:
            self.reason = "disabled by config (no-op)"
        if self.enabled:
            self.store = JsonlStore(self.path)
            self.trail = self.store.load()

    def _emit(self, bench_event, payload, links=None, actor_kind="system", actor_name="stillwarm-bench"):
        if not self.enabled:
            return None
        ev = Event(
            type=EventType(BENCH_MAP[bench_event]),
            actor=Actor(name=actor_name, kind=actor_kind),
            payload=payload,
            links=list(links or []),
            labels={"benchmark_event": bench_event, "harness": "stillwarm-bench"},
        )
        self.trail.record(ev)
        self.store.append(ev)
        return ev.id

    # -- benchmark lifecycle --------------------------------------------------
    def run_started(self, config: dict):
        return self._emit("RUN_STARTED", {
            "run_id": config.get("run_id"),
            "config_sha256": sha256_json(config),
            "llamacpp_tag": config.get("llamacpp_tag"),
            "llamacpp_sha": config.get("llamacpp_sha"),
            "model": config.get("model_name"),
            "model_sha256": config.get("model_sha256"),
            "harness_git_sha": config.get("harness_git_sha"),
            "scenario": config.get("scenario"),
            "flags": {k: config.get(k) for k in
                      ("ctx", "flash_attn", "cache_type_k", "cache_type_v", "swa_full", "seed")},
        })

    def measurement(self, run_id, csv_row: dict, metrics: dict, links=None):
        return self._emit("MEASUREMENT", {
            "run_id": run_id,
            "csv_row_sha256": sha256_json(csv_row),   # bind the row by hash, don't copy it all
            "metrics": metrics,
        }, links=links, actor_kind="model", actor_name="llama-server")

    def verification(self, run_id, probe: dict, links=None):
        return self._emit("VERIFICATION", {"run_id": run_id, **probe}, links=links)

    def run_completed(self, run_id, summary: dict, links=None):
        return self._emit("RUN_COMPLETED", {"run_id": run_id, **summary}, links=links)

    # -- integrity ------------------------------------------------------------
    def verify(self):
        if not self.enabled:
            return None
        return self.trail.verify()
