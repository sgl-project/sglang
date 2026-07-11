#!/usr/bin/env python3
"""Render the blog's main figure from sweep results produced by run_client.sh.

Expected layout (only curves that exist are plotted):
  results/gb300/{day0,glm52_v0515,glm51_v0515}/{tp4,tep4}/parallel_*/benchmark_summary.json
  results/b300/{day0,glm52_v0515,glm51_v0515}/{tp8,tep8}/parallel_*/benchmark_summary.json

2x2 panels: GB300 TP=4 / TEP=4 (top), B300 TP=8 / TEP=8 (bottom). Per panel,
three curves at concurrency 1,2,4,8:
  X = interactivity = 1000 / TPOT(ms)
  Y = total throughput (prompt+completion tok/s, as evalscope reports) / n_gpus
The y-axis is truncated (break glyph, '0' anchors the origin).

Usage:  python3 plot_main_figure.py  [results-dir]   (default ./results)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "results"

CONCS = (1, 2, 4, 8)

# (curve label, results subdir, color, linestyle, annotate above/below)
CURVES = [
    ("GLM-5.2 day-0", "day0", "#9C948A", "--", False),
    ("GLM-5.2 v0.5.15", "glm52_v0515", "#A43223", "-", True),
    ("GLM-5.1 v0.5.15", "glm51_v0515", "#D45927", "-", False),
]

# (panel title, platform dir, run name, n_gpus)
PANELS = [
    ("TP=4", "gb300", "tp4", 4),
    ("TEP=4", "gb300", "tep4", 4),
    ("TP=8", "b300", "tp8", 8),
    ("TEP=8", "b300", "tep8", 8),
]


def read_points(platform: str, curve_dir: str, run_name: str, n_gpus: int):
    base = RESULTS / platform / curve_dir / run_name
    pts = []
    for f in sorted(base.glob("parallel_*/benchmark_summary.json")):
        s = json.loads(f.read_text())
        conc = int(s["Concurrency"])
        if conc not in CONCS:
            continue
        tpot = s["TPOT (ms)"]
        pts.append((1000.0 / tpot if tpot else 0.0,
                    s["Total Throughput (tok/s)"] / n_gpus, conc))
    pts.sort(key=lambda p: p[2])
    return pts


def _axis_break(ax, work_min):
    ax.spines["left"].set_bounds(work_min, ax.get_ylim()[1])
    y0, y1 = ax.get_ylim()
    wf = (work_min - y0) / (y1 - y0)
    seg_x = [0, 0, 0.020, -0.020, 0, 0]
    seg_y = [0.0, 0.30 * wf, 0.42 * wf, 0.58 * wf, 0.69 * wf, wf]
    ax.plot(seg_x, seg_y, transform=ax.transAxes, clip_on=False,
            color="black", linewidth=1.4, solid_capstyle="round", zorder=5)
    ax.text(-0.018, 0.0, "0", transform=ax.transAxes, ha="right",
            va="center", fontsize=10, color="#222222")


def plot_line(ax, pts, *, color, linestyle, above, label):
    if not pts:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, label=label, color=color, linestyle=linestyle,
            marker="o", linewidth=1.5, markersize=6,
            markeredgecolor="white", markeredgewidth=0.8)
    off = (5, 7) if above else (-3, -15)
    for x, y, c in zip(xs, ys, (p[2] for p in pts)):
        ax.annotate(str(c), (x, y), textcoords="offset points",
                    xytext=off, fontsize=8, color="#8a8a8a")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.4), dpi=200)
    flat = axes.flatten()
    legend_ax = None

    for row in (0, 1):
        idx = (row * 2, row * 2 + 1)
        allpts = [p for i in idx for _, cdir, *_ in CURVES
                  for p in read_points(PANELS[i][1], cdir, PANELS[i][2], PANELS[i][3])]
        if not allpts:
            for i in idx:
                flat[i].set_title(PANELS[i][0] + " (no data)", fontsize=12.5)
            continue
        xmax = max(p[0] for p in allpts)
        ymax = max(p[1] for p in allpts)
        work_min = 10000 if row == 0 else 5000
        ytop = ymax * 1.07
        wf = 0.12
        for i in idx:
            title, platform, run_name, ng = PANELS[i]
            ax = flat[i]
            for label, cdir, color, ls, above in CURVES:
                pts = read_points(platform, cdir, run_name, ng)
                plot_line(ax, pts, color=color, linestyle=ls, above=above, label=label)
                if pts:
                    legend_ax = legend_ax or ax
            ax.set_xlim(0, xmax * 1.07)
            ax.set_ylim((work_min - wf * ytop) / (1 - wf), ytop)
            ax.set_xticks(list(range(50, int(xmax * 1.07) + 1, 50)))
            ax.yaxis.set_major_locator(MultipleLocator(5000))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:g}k"))
            ax.set_title(title, fontsize=12.5, fontweight="bold", pad=8)
            ax.set_facecolor("white")
            ax.grid(True, linestyle="--", color="#9aa0a6", alpha=0.4, linewidth=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color("black")
                ax.spines[sp].set_linewidth(1.4)
            ax.tick_params(colors="#222222", labelsize=10, width=1.2)
            _axis_break(ax, work_min)

    fig.suptitle("GLM NVFP4 Perf on SGLang", fontsize=17, fontweight="bold")
    fig.supxlabel("Interactivity (tok/s/user)", fontsize=12, y=0.055)
    fig.supylabel("Token Throughput per GPU (tok/s/gpu)", fontsize=12, x=0.015)
    if legend_ax is not None:
        handles, labels = legend_ax.get_legend_handles_labels()
        # de-dup while preserving order
        seen, h2, l2 = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l); h2.append(h); l2.append(l)
        fig.legend(h2, l2, loc="lower center", bbox_to_anchor=(0.5, 0.005),
                   ncol=3, frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.89, bottom=0.12,
                        wspace=0.16, hspace=0.40)

    top_y = flat[0].get_position().y1
    bot_y = flat[2].get_position().y1
    for text, y in (("GB300", top_y), ("B300", bot_y)):
        fig.text(0.5, y + 0.038, text, ha="center", va="bottom",
                 fontsize=14, fontweight="bold", color="#333333")

    out = Path(__file__).resolve().parent / "main_figure.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
