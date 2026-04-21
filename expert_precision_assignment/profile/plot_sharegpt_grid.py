"""Heatmap of sharegpt sweep: mc × variant grid, each cell annotated with
total_throughput / median_ttft_ms / median_itl_ms / concurrency.

Usage:
    python plot_sharegpt_grid.py \
        --csv results/sharegpt/summary.csv \
        --out results/sharegpt/grid.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MC_ORDER = [8, 16, 32, 64, 128, 256]
HOT_VARIANTS = ["hot0", "hot20", "hot40", "hot60", "hot80", "hot100"]
THR_VARIANTS_REV = ["thr512", "thr256", "thr128", "thr64", "thr32"]


def _fmt(x: float, d: int = 3) -> str:
    if x is None:
        return "—"
    return f"{round(float(x), d):g}"


def _plot_subset(cells, variants, title_suffix, out_path):
    ncol, nrow = len(MC_ORDER), len(variants)
    tput = np.full((nrow, ncol), np.nan)
    labels = [[""] * ncol for _ in range(nrow)]
    for i, v in enumerate(variants):
        for j, mc in enumerate(MC_ORDER):
            r = cells.get((mc, v))
            if not r:
                continue
            t = float(r["total_throughput"])
            ttft = float(r["median_ttft_ms"])
            itl = float(r["median_itl_ms"])
            conc = float(r["concurrency"])
            tput[i, j] = t
            labels[i][j] = (
                f"{_fmt(t)}\n{_fmt(ttft)}\n{_fmt(itl)}\n{_fmt(conc)}"
            )

    fig, ax = plt.subplots(figsize=(1.5 * ncol + 1.3, 0.75 * nrow + 1.5))
    im = ax.imshow(tput, cmap="viridis", aspect="auto")

    ax.set_xticks(range(ncol))
    ax.set_xticklabels([f"mc={m}" for m in MC_ORDER])
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(variants)

    vmax = np.nanmax(tput)
    vmin = np.nanmin(tput)
    for i in range(nrow):
        for j in range(ncol):
            if np.isnan(tput[i, j]):
                continue
            frac = (tput[i, j] - vmin) / (vmax - vmin + 1e-9)
            color = "white" if frac < 0.55 else "black"
            ax.text(j, i, labels[i][j], ha="center", va="center",
                    fontsize=9, color=color, linespacing=1.15)

    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlabel("max_concurrency", fontsize=11)
    ax.set_ylabel("variant", fontsize=11)
    ax.set_title(
        f"sharegpt sweep — heter-MoE (Qwen3-30B-A3B, n=1024, ctx=4096) — {title_suffix}\n"
        "per cell: total_tput (tok/s) / median TTFT (ms) / median ITL (ms) / concurrency\n"
        "color: total_throughput (higher = brighter)",
        fontsize=11,
    )
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("total_throughput (tok/s)")
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_hot", required=True)
    ap.add_argument("--out_thr", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    cells: dict[tuple[int, str], dict] = {}
    for r in rows:
        cells[(int(r["mc"]), r["variant"])] = r

    _plot_subset(cells, HOT_VARIANTS, "hot% dispatch", args.out_hot)
    _plot_subset(cells, THR_VARIANTS_REV, "threshold dispatch (reversed)", args.out_thr)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
