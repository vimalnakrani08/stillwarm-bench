#!/usr/bin/env python3
"""Final article figures — computed from the results data files ONLY.

Style: validated categorical palette (CVD ΔE 24.2, light surface), color follows
the ENTITY across all figures (cold=blue, warm-read resume=aqua, cold-read=green,
restore=green), one axis per chart, recessive grid, direct value labels
(relief rule for the sub-3:1 aqua/yellow slots), text in ink colors.
Sources per figure are written to analysis/figures/captions.md.
"""
import csv, json, statistics as st, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "analysis/figures")
INK, INK2, GRID, SURF = "#1a1a19", "#5f5e56", "#e5e4df", "#fcfcfb"
BLUE, AQUA, YELLOW, GREEN = "#2a78d6", "#1baf7a", "#eda100", "#008300"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
})


def rows(name):
    return list(csv.DictReader(open(os.path.join(REPO, "results", name))))


def med(xs):
    return st.median(xs)


A2 = rows("blockA2_supervised.csv")
RUNGS = ["8k", "32k", "64k"]
TOK = {"8k": 8192, "32k": 32768, "64k": 65536}
cold_s = {r: med([float(x["client_ttft_s"]) for x in A2 if x["doc_label"] == r and x["mode"] == "cold"]) for r in RUNGS}
resume_s = {r: med([float(x["resume_total_ms"]) for x in A2 if x["doc_label"] == r and x["mode"] == "disk_restore"]) / 1000 for r in RUNGS}
coldread = {r: [float(x["resume_total_ms"]) / 1000 for x in rows(f"blockB_coldread_{r}.csv")] for r in RUNGS}

# ---------------------------------------------------------------- fig 1: headline
fig, ax = plt.subplots(figsize=(7.2, 4.2))
x = range(len(RUNGS)); w = 0.38
b1 = ax.bar([i - w/2 for i in x], [cold_s[r] for r in RUNGS], w, color=BLUE,
            label="Cold: re-read everything", edgecolor=SURF, linewidth=2)
b2 = ax.bar([i + w/2 for i in x], [resume_s[r] for r in RUNGS], w, color=GREEN,
            label="stillwarm resume (restore + first token)", edgecolor=SURF, linewidth=2)
ax.set_yscale("log"); ax.set_ylim(0.05, 900)
for i, r in enumerate(RUNGS):
    ax.text(i - w/2, cold_s[r] * 1.15, f"{cold_s[r]:.0f}s", ha="center", color=INK, fontsize=9)
    ax.text(i + w/2, resume_s[r] * 1.15, f"{resume_s[r]:.2f}s", ha="center", color=INK, fontsize=9)
    gm = (cold_s[r] * resume_s[r]) ** 0.5          # log-scale midpoint between the pair
    ax.text(i, gm, f"{cold_s[r]/resume_s[r]:.0f}×\nfaster", ha="center", va="center",
            color=GREEN, fontsize=10, fontweight="bold")
ax.set_xticks(list(x)); ax.set_xticklabels([f"{TOK[r]:,} tokens" for r in RUNGS])
ax.set_ylabel("seconds until the answer starts (log)")
ax.set_title("Resuming a session from disk vs re-reading it\nLlama-3.1-8B Q4_K_M · M3 Max · same-session, thermally controlled", fontsize=11, loc="left")
ax.legend(frameon=False, loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_headline.png"), dpi=300); plt.close(fig)

# ------------------------------------------------------------- fig 2: break-even
fig, ax = plt.subplots(figsize=(7.2, 4.2))
toks = [TOK[r] for r in RUNGS]
ax.plot(toks, [cold_s[r] for r in RUNGS], "o-", color=BLUE, linewidth=2, markersize=7,
        label="cold prefill")
ax.plot(toks, [resume_s[r] for r in RUNGS], "o-", color=AQUA, linewidth=3.5, markersize=9,
        label="resume, file in page cache (warm_read)", zorder=2)
ax.plot(toks, [med(coldread[r]) for r in RUNGS], "s--", color=GREEN, linewidth=1.6, markersize=6,
        label="resume, purged page cache (cold_read)  — nearly identical", zorder=3)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("document length (tokens, log)"); ax.set_ylabel("seconds (log)")
for t, r in zip(toks, RUNGS):
    ax.text(t, cold_s[r] * 1.3, f"{cold_s[r]:.0f}s", ha="center", fontsize=8.5, color=INK)
    ax.text(t * 0.88, med(coldread[r]), f"{med(coldread[r]):.2f}s", ha="right", va="center", fontsize=8.5, color=INK)
ax.axvspan(132, 295, color=GRID, alpha=0.6, zorder=0)
ax.text(197, 2.2, "break-even zone\n~132–295 tokens\n(purged cache)", ha="center",
        fontsize=8.5, color=INK2)
ax.set_title("Break-even: above a few hundred tokens, restoring always wins\nsame-session medians; purged reads at 5.3–6.1 GB/s ≈ 84–95% of raw dd", fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=9, loc="center left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_breakeven.png"), dpi=300); plt.close(fig)

# ---------------------------------------------------------------- fig 3: energy
E = json.load(open(os.path.join(REPO, "results/energy_sidebar.json")))
fig, ax = plt.subplots(figsize=(6.4, 3.8))
labels = ["Cold prefill\n(116.5 s wall)", "Restore + answer\n(2.4 s wall)"]
vals = [E["cold_window"]["Wh"], E["restore_window"]["Wh"]]
bars = ax.bar(labels, vals, 0.5, color=[BLUE, GREEN], edgecolor=SURF, linewidth=2)
ax.set_ylim(0, 1.42)
for b, v, w in zip(bars, vals, [E["cold_window"]["avg_W"], E["restore_window"]["avg_W"]]):
    ax.text(b.get_x() + b.get_width()/2, v * 1.04 + 0.005, f"{v:.3f} Wh",
            ha="center", color=INK, fontsize=10, fontweight="bold")
    ax.text(b.get_x() + b.get_width()/2, v / 2 if v > 0.1 else v + 0.06,
            f"avg {w:.0f} W", ha="center", color=INK2, fontsize=9)
ax.set_ylabel("energy (Wh, CPU+GPU+ANE package)")
ax.set_title(f"Energy per 32K-token turn: ~{E['energy_ratio']:.0f}× less\n"
             f"(wall clock {E['wall_ratio']:.0f}×; 1 s powermetrics sampling; restore window ±1 sample)",
             fontsize=11, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_energy.png"), dpi=300); plt.close(fig)

# ------------------------------------------------ fig 4: thermal inversion exhibit
hot = [float(x["restore_ms"]) for x in A2 if x["doc_label"] == "64k" and x["mode"] == "disk_restore"]
cool = [float(x["restore_ms"]) for x in rows("blockB_coldread_64k.csv")]
fig, ax = plt.subplots(figsize=(6.8, 3.8))
ax.scatter(hot, [1] * len(hot), s=70, color=YELLOW, zorder=3,
           label="hot machine, file IN page cache", edgecolor=SURF, linewidth=1.5)
ax.scatter(cool, [0] * len(cool), s=70, color=GREEN, zorder=3, marker="s",
           label="cool machine, page cache PURGED", edgecolor=SURF, linewidth=1.5)
ax.set_yticks([0, 1]); ax.set_yticklabels(["cool + purged", "hot + cached"])
ax.set_ylim(-0.6, 1.6)
ax.set_xlabel("64K-state restore time (ms) — lower is better")
for v in hot: pass
ax.text(med(hot), 1.22, f"median {med(hot):.0f} ms", ha="center", fontsize=9, color=INK)
ax.text(med(cool), -0.38, f"median {med(cool):.0f} ms", ha="center", fontsize=9, color=INK)
ax.set_title("The inversion: purged-but-cool beats cached-but-hot\n"
             "page cache is not the bottleneck (dd: 21.2 GB/s warm vs 6.4 GB/s cold) — thermal-state hypothesis",
             fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=9, loc="center right")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_thermal_inversion.png"), dpi=300); plt.close(fig)

print("figures written:", sorted(os.listdir(OUT)))
