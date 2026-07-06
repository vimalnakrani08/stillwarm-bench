#!/usr/bin/env python3
"""llama-server subprocess lifecycle + memory sampling + startup-log buffer parsing.

Timing rule (PLAN §5.3): timed runs use DEFAULT verbosity (quiet) so per-request
logging can't skew the stopwatch. KV/Metal buffer sizes (config-deterministic) are
captured from a SEPARATE `-lv 5` profile launch that serves no timed requests.
"""
from __future__ import annotations
import subprocess, time, threading, os, re, signal
import httpx


class ServerProc:
    def __init__(self, binary, model, port, ctx, slot_save_path, *,
                 extra_flags=None, verbosity=None, log_path=None, ngl=99, n_slots=1, seed=42):
        self.binary = binary
        self.model = model
        self.port = int(port)
        self.ctx = int(ctx)
        self.slot_save_path = slot_save_path
        self.extra_flags = list(extra_flags or [])
        self.verbosity = verbosity
        self.log_path = log_path or f"/tmp/llama-server-{port}.log"
        self.ngl = ngl
        self.n_slots = n_slots
        self.seed = seed
        self.proc = None
        self._rss_thread = None
        self._rss_stop = None
        self.rss_samples = []   # list of (t_rel, rss_bytes)
        self.base = f"http://127.0.0.1:{self.port}"

    def cmd(self):
        c = [self.binary, "-m", self.model, "-c", str(self.ctx), "-np", str(self.n_slots),
             "-ngl", str(self.ngl), "--host", "127.0.0.1", "--port", str(self.port),
             "--slot-save-path", self.slot_save_path, "--seed", str(self.seed)]
        c += self.extra_flags
        if self.verbosity is not None:
            c += ["-lv", str(self.verbosity)]
        return c

    def start(self, timeout=180):
        self._logf = open(self.log_path, "w")
        self.proc = subprocess.Popen(self.cmd(), stdout=self._logf, stderr=subprocess.STDOUT)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early (rc={self.proc.returncode}); see {self.log_path}")
            try:
                if httpx.get(f"{self.base}/health", timeout=2).status_code == 200:
                    return time.perf_counter() - t0
            except Exception:
                pass
            time.sleep(0.4)
        self.stop()
        raise TimeoutError(f"server not ready in {timeout}s; see {self.log_path}")

    def stop(self):
        self.stop_rss()
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)
        try:
            self._logf.close()
        except Exception:
            pass

    # -- RSS sampling at 1 Hz -------------------------------------------------
    def start_rss(self, hz=1.0):
        import psutil
        self.rss_samples = []
        self._rss_stop = threading.Event()
        p = psutil.Process(self.proc.pid)
        t0 = time.perf_counter()

        def loop():
            while not self._rss_stop.is_set():
                try:
                    self.rss_samples.append((round(time.perf_counter() - t0, 3), p.memory_info().rss))
                except Exception:
                    break
                self._rss_stop.wait(1.0 / hz)
        self._rss_thread = threading.Thread(target=loop, daemon=True)
        self._rss_thread.start()

    def stop_rss(self):
        if self._rss_stop:
            self._rss_stop.set()
        if self._rss_thread:
            self._rss_thread.join(timeout=2)
            self._rss_thread = None
        self._rss_stop = None

    def rss_stats_mb(self):
        if not self.rss_samples:
            return {"rss_peak_mb": None, "rss_mean_mb": None, "rss_n": 0}
        vals = [r for _, r in self.rss_samples]
        return {"rss_peak_mb": round(max(vals) / 2**20, 1),
                "rss_mean_mb": round(sum(vals) / len(vals) / 2**20, 1),
                "rss_n": len(vals)}

    # -- HTTP helpers ---------------------------------------------------------
    def _post(self, path, payload, timeout=600):
        r = httpx.post(f"{self.base}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def tokenize(self, text):
        return self._post("/tokenize", {"content": text})["tokens"]

    def detokenize(self, tokens):
        return self._post("/detokenize", {"tokens": tokens})["content"]

    def save_slot(self, filename, id_slot=0):
        return self._post(f"/slots/{id_slot}?action=save", {"filename": filename})

    def restore_slot(self, filename, id_slot=0):
        return self._post(f"/slots/{id_slot}?action=restore", {"filename": filename})

    def erase_slot(self, id_slot=0):
        return self._post(f"/slots/{id_slot}?action=erase", {})

    # -- startup-log parsing (memory breakdown) -------------------------------
    def parse_buffer_sizes(self):
        """Parse KV/compute/model buffer sizes + projected device memory from the
        (verbose) startup log. Returns MiB values; None if a line is absent."""
        try:
            text = open(self.log_path, encoding="utf-8", errors="replace").read()
        except FileNotFoundError:
            return {}
        out = {}
        m = re.search(r"llama_kv_cache: size =\s+([\d.]+) MiB \(\s*(\d+) cells,\s*(\d+) layers.*?"
                      r"K \((\w+)\):\s+([\d.]+) MiB, V \((\w+)\):\s+([\d.]+) MiB", text)
        if m:
            out.update(kv_total_mib=float(m.group(1)), kv_cells=int(m.group(2)), kv_layers=int(m.group(3)),
                       kv_k_type=m.group(4), kv_k_mib=float(m.group(5)), kv_v_type=m.group(6), kv_v_mib=float(m.group(7)))
        mtl = re.findall(r"sched_reserve:\s+MTL0 compute buffer size =\s+([\d.]+) MiB", text)
        if mtl:
            out["compute_buffer_mtl_mib"] = float(mtl[-1])
        mm = re.findall(r"load_tensors:\s+MTL0 model buffer size =\s+([\d.]+) MiB", text)
        if mm:
            out["model_buffer_mtl_mib"] = float(mm[-1])
        pj = re.findall(r"projected to use (\d+) MiB of device memory", text)
        if pj:
            out["projected_device_mib"] = int(pj[-1])
        return out
