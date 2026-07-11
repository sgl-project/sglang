#!/usr/bin/env python3
"""Render the ISL-ablation grouped bar chart from run_isl_client.sh results.

Per ISL rung (80K -> 1M), one bar for GLM-5.2-NVFP4 on v0.5.15 (fused top-k +
deferred finalize on) and one for the day-0 tree (@ 22dce5720). Y =
interactivity (tok/s/user = 1000 / TPOT ms) at concurrency 1, TP4, 4xGB300.
The y-axis is truncated below FLOOR (zigzag break; '0' anchors the origin).

Expects: results/{v0515,day0}/<rung>/isl_<rung>/parallel_1_number_4/benchmark_summary.json
Series with no data are skipped (running only v0515 gives a single-series chart).

Usage:  python3 plot_isl_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
RUNGS = ["80k", "128k", "256k", "512k", "1m"]
LABELS = {"80k": "80K", "128k": "128K", "256k": "256K", "512k": "512K", "1m": "1M"}

SERIES = [
    ("GLM-5.2 NVFP4 v0.5.15", "v0515", "#A43223"),
    ("GLM-5.2 NVFP4 day-0", "day0", "#9C948A"),
]

FLOOR = 120  # broken-axis working floor (below the lowest expected bar)


def interactivity(series: str, tag: str):
    p = (BASE / "results" / series / tag / f"isl_{tag}"
         / "parallel_1_number_4" / "benchmark_summary.json")
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    return 1000.0 / s["TPOT (ms)"]


def _axis_break(ax) -> None:
    ax.spines["left"].set_bounds(FLOOR, ax.get_ylim()[1])
    y0, y1 = ax.get_ylim()
    yf = (FLOOR - y0) / (y1 - y0)
    seg_x = [0, 0, 0.020, -0.020, 0, 0]
    seg_y = [0.000, 0.35 * yf, 0.50 * yf, 0.70 * yf, 0.83 * yf, yf]
    ax.plot(seg_x, seg_y, transform=ax.transAxes, clip_on=False,
            color="black", linewidth=1.4, solid_capstyle="round", zorder=5)
    ax.text(-0.018, 0.0, "0", transform=ax.transAxes, ha="right",
            va="center", fontsize=10, color="#222222")


def main() -> None:
    xs = list(range(len(RUNGS)))
    present = [s for s in SERIES
               if any(interactivity(s[1], t) is not None for t in RUNGS)]
    if not present:
        raise SystemExit("no data under results/ — run run_isl_client.sh first")
    n = len(present)
    width = 0.62 if n == 1 else 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=200)
    for j, (label, series, color) in enumerate(present):
        vals = [interactivity(series, t) for t in RUNGS]
        offs = 0.0 if n == 1 else (j - (n - 1) / 2) * width
        pos = [x + offs for x in xs]
        for x, v in zip(pos, vals):
            if v is None:
                continue
            ax.bar(x, v, width=width, color=color, edgecolor="white",
                   linewidth=0.8, zorder=3,
                   label=label if x == pos[0] else None)
            ax.annotate(f"{v:.0f}", (x, v), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=9, color="#333333")

    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[t] for t in RUNGS])
    ax.set_ylim(FLOOR - 26, 480)
    ax.set_yticks([150, 200, 250, 300, 350, 400, 450])
    ax.set_facecolor("white")
    ax.grid(True, axis="y", linestyle="--", color="#9aa0a6", alpha=0.4,
            linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.4)
    ax.tick_params(colors="#222222", labelsize=10, width=1.2)
    ax.set_xlabel("Input Sequence Length", fontsize=11, labelpad=10)
    ax.set_ylabel("Interactivity (tok/s/user)", fontsize=11, labelpad=10)
    _axis_break(ax)
    fig.suptitle("ISL Ablation (c=1, TP=4)", fontsize=15, fontweight="bold")
    if n > 1:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=n,
                   frameon=False, fontsize=10)
        fig.subplots_adjust(left=0.11, right=0.96, top=0.86, bottom=0.20)
    else:
        fig.subplots_adjust(left=0.11, right=0.96, top=0.86, bottom=0.14)

    out = BASE / "isl_ablation.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
