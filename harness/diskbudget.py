#!/usr/bin/env python3
"""Disk-budget enforcement for cache_saves/.

- remove_files(): delete specific scenario save files after measurement (unless kept).
- enforce(): if cache_saves total exceeds DISK_BUDGET_GB, LRU-prune non-protected
  save files (never touches cache_saves/keep/). Returns a report dict."""
from __future__ import annotations
import os


def _iter_files(root, protect_dirs):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if os.path.join(dirpath, d) not in
                       {os.path.join(root, p) for p in protect_dirs}]
        for fn in filenames:
            yield os.path.join(dirpath, fn)


def dir_size_bytes(root, protect_dirs=()):
    total = 0
    for f in _iter_files(root, protect_dirs):
        try:
            total += os.path.getsize(f)
        except OSError:
            pass
    return total


def remove_files(paths):
    removed = []
    for p in paths:
        try:
            if os.path.exists(p):
                sz = os.path.getsize(p)
                os.remove(p)
                removed.append({"file": os.path.basename(p), "bytes": sz})
        except OSError as e:
            removed.append({"file": os.path.basename(p), "error": str(e)})
    return removed


def enforce(save_dir, budget_gb, protect_dirs=("keep",)):
    budget = budget_gb * (10 ** 9)
    prunable = sorted(
        (f for f in _iter_files(save_dir, protect_dirs)),
        key=lambda f: os.path.getmtime(f))            # LRU by mtime
    # Total includes protected (keep/) files: they count toward disk usage but are
    # never pruned. If only protected files remain and we're still over, report it.
    total = dir_size_bytes(save_dir)
    pruned = []
    i = 0
    while total > budget and i < len(prunable):
        f = prunable[i]; i += 1
        try:
            sz = os.path.getsize(f); os.remove(f)
            pruned.append({"file": os.path.relpath(f, save_dir), "bytes": sz})
            total -= sz
        except OSError:
            pass
    return {"budget_gb": budget_gb, "used_gb_after": round(total / 1e9, 3),
            "over_budget": total > budget, "pruned": pruned}
