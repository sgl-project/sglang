"""Heatmaps of gsm8k sweep over the mc × hot-variant grid.

Produces two figures:
  * accuracy heatmap — color by exact_match (flexible-extract), cell shows
    acc / ±stderr.
  * SLO heatmap — color by mean rank across median TTFT and median ITL
    (lower rank = better latency = brighter), cell shows
    total_tput / median TTFT / median ITL / concurrency.

Usage:
    python plot_gsm8k_grid.py \
        --csv results/gsm8k/summary.csv \
        --out_acc results/gsm8k/grid_acc.png \
        --out_slo results/gsm8k/grid_slo.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MC_ORDER = [8, 16, 32, 64, 128, 256]
HOT_VARIANTS = ["hot0", "hot20", "hot40", "hot60", "hot80", "hot100"]

ACC_KEY = "acc_exact_match__flexible-extract"
ACC_STDERR_KEY = "acc_exact_match_stderr__flexible-extract"


def _fmt(x, d: int = 3) -> str:
    if x is None or x == "":
        return "—"
    try:
        return f"{round(float(x), d):g}"
    except (TypeError, ValueError):
        return "—"


def _get(r: dict, key: str):
    v = r.get(key)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _render(values, labels, title, cbar_label, out_path,
            lower_is_better: bool = False):
    ncol, nrow = len(MC_ORDER), len(HOT_VARIANTS)
    fig, ax = plt.subplots(figsize=(1.5 * ncol + 1.3, 0.75 * nrow + 1.5))
    cmap = "viridis_r" if lower_is_better else "viridis"
    im = ax.imshow(values, cmap=cmap, aspect="auto")

    ax.set_xticks(range(ncol))
    ax.set_xticklabels([f"mc={m}" for m in MC_ORDER])
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(HOT_VARIANTS)

    vmax = np.nanmax(values)
    vmin = np.nanmin(values)
    for i in range(nrow):
        for j in range(ncol):
            if np.isnan(values[i, j]):
                continue
            frac = (values[i, j] - vmin) / (vmax - vmin + 1e-9)
            # viridis: bright at high frac; viridis_r: bright at low frac.
            brightness = 1.0 - frac if lower_is_better else frac
            color = "white" if brightness < 0.55 else "black"
            ax.text(j, i, labels[i][j], ha="center", va="center",
                    fontsize=9, color=color, linespacing=1.15)

    ax.tick_params(axis="both", labelsize=10)
    ax.set_xlabel("max_concurrency", fontsize=11)
    ax.set_ylabel("variant", fontsize=11)
    ax.set_title(title, fontsize=11)
    cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def _plot_accuracy(cells, out_path):
    ncol, nrow = len(MC_ORDER), len(HOT_VARIANTS)
    acc = np.full((nrow, ncol), np.nan)
    labels = [[""] * ncol for _ in range(nrow)]
    for i, v in enumerate(HOT_VARIANTS):
        for j, mc in enumerate(MC_ORDER):
            r = cells.get((mc, v))
            if not r:
                continue
            a = _get(r, ACC_KEY)
            if a is None:
                continue
            stderr = _get(r, ACC_STDERR_KEY) or 0.0
            acc[i, j] = a
            labels[i][j] = f"{_fmt(a, 4)}\n±{_fmt(stderr, 4)}"

    _render(
        acc, labels,
        title=(
            "gsm8k sweep — heter-MoE (Qwen3-30B-A3B, n=1319, 5-shot) — accuracy\n"
            "per cell: exact_match (flexible-extract) / ±stderr\n"
            "color: exact_match (flexible-extract), higher = brighter"
        ),
        cbar_label="exact_match (flexible-extract)",
        out_path=out_path,
    )


def _rank(values: np.ndarray) -> np.ndarray:
    """Rank a 2D array ascending (smallest = rank 1). NaNs stay NaN."""
    flat = values.flatten()
    mask = ~np.isnan(flat)
    ranks = np.full_like(flat, np.nan, dtype=float)
    order = np.argsort(flat[mask], kind="stable")
    idx_valid = np.where(mask)[0]
    ranks_valid = np.empty(mask.sum(), dtype=float)
    ranks_valid[order] = np.arange(1, mask.sum() + 1)
    ranks[idx_valid] = ranks_valid
    return ranks.reshape(values.shape)


def _plot_slo(cells, out_path):
    ncol, nrow = len(MC_ORDER), len(HOT_VARIANTS)
    tput = np.full((nrow, ncol), np.nan)
    ttft_m = np.full((nrow, ncol), np.nan)
    itl_m = np.full((nrow, ncol), np.nan)
    conc_m = np.full((nrow, ncol), np.nan)
    for i, v in enumerate(HOT_VARIANTS):
        for j, mc in enumerate(MC_ORDER):
            r = cells.get((mc, v))
            if not r:
                continue
            ttft = _get(r, "median_ttft_ms")
            itl = _get(r, "median_itl_ms")
            if ttft is None or itl is None:
                continue
            tput[i, j] = _get(r, "total_throughput") or np.nan
            ttft_m[i, j] = ttft
            itl_m[i, j] = itl
            conc_m[i, j] = _get(r, "concurrency") or np.nan

    rank_ttft = _rank(ttft_m)
    rank_itl = _rank(itl_m)
    composite = (rank_ttft + rank_itl) / 2.0  # lower = better latency

    labels = [[""] * ncol for _ in range(nrow)]
    for i in range(nrow):
        for j in range(ncol):
            if np.isnan(composite[i, j]):
                continue
            labels[i][j] = (
                f"#{_fmt(composite[i, j], 1)}\n"
                f"ttft={_fmt(ttft_m[i, j])}\n"
                f"itl={_fmt(itl_m[i, j])}\n"
                f"tput={_fmt(tput[i, j])}"
            )

    _render(
        composite, labels,
        title=(
            "gsm8k sweep — heter-MoE (Qwen3-30B-A3B, n=1319, 5-shot) — SLO\n"
            "per cell: mean rank / median TTFT (ms) / median ITL (ms) / total_tput (tok/s)\n"
            "color: mean rank of (TTFT, ITL), lower rank = better latency = brighter"
        ),
        cbar_label="mean rank across (TTFT, ITL) — 1 = best",
        out_path=out_path,
        lower_is_better=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_acc", required=True)
    ap.add_argument("--out_slo", required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    cells: dict[tuple[int, str], dict] = {}
    for r in rows:
        cells[(int(r["mc"]), r["variant"])] = r

    _plot_accuracy(cells, args.out_acc)
    _plot_slo(cells, args.out_slo)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
