#!/usr/bin/env python3
"""Generate denoising speedup chart from Ideogram 4 delta sweep results.

Usage:
    python scratch/spectral-progressive-ideogram/gen_speedup_plot.py --json <results.json>
    python scratch/spectral-progressive-ideogram/gen_speedup_plot.py --json <results.json> --out /tmp/speedup.png
"""

import argparse
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# X-brand purple used across the multi-model speedup charts in final_PR_smoke
IDEOGRAM_COLOR = "#7856FF"
BG = "#FFFFFF"
GRID_COL = "#E7E7E7"
TEXT_COL = "#0F1419"
SUB_COL = "#536471"


SERIES_COLORS = [IDEOGRAM_COLOR, "#00B4D8", "#FF6B6B", "#51CF66"]
SERIES_ALPHA = [0.85, 0.75, 0.75, 0.75]


def _load_series(json_paths: list[str]) -> list[tuple[str, dict]]:
    """Load one or more results JSON files.  Each JSON may have one model key."""
    series = []
    for path in json_paths:
        with open(path) as f:
            data = json.load(f)
        for key, md in data.items():
            steps = md.get("steps", "?")
            steps_suffix = f"({steps}-step)"
            label = key if steps_suffix in key else f"{key} {steps_suffix}"
            series.append((label, md))
    return series


def plot(json_paths: list[str], out_path: str) -> None:
    series = _load_series(json_paths)

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BG)
    ax.set_facecolor(BG)

    for spine in ax.spines.values():
        spine.set_color(GRID_COL)
        spine.set_linewidth(0.8)

    ax.axhline(y=1.0, color=GRID_COL, linewidth=1.2, zorder=1)

    all_speedups = []
    for i, (label, md) in enumerate(series):
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        pts = [
            (p["delta"], p["speedup"])
            for p in md.get("points", [])
            if p.get("speedup") is not None
        ]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        all_speedups.extend(ys)

        # End-point annotation (only for first series to avoid clutter)
        if i == 0:
            ax.annotate(
                f"{ys[-1]:.2f}×",
                xy=(xs[-1], ys[-1]),
                xytext=(8, 0),
                textcoords="offset points",
                fontsize=16,
                fontweight="bold",
                color=color,
                va="center",
            )

        # Glow + line
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=8,
            alpha=0.12,
            solid_capstyle="round",
            zorder=2,
        )
        ax.plot(
            xs,
            ys,
            color=color,
            marker="D",
            markersize=11,
            markeredgewidth=2.0,
            markeredgecolor="white",
            linewidth=3.0,
            solid_capstyle="round",
            label=label,
            zorder=3,
        )

        # Per-point labels (only for first series on combined chart)
        if len(series) == 1:
            for x, y in zip(xs, ys):
                ax.annotate(
                    f"{y:.2f}×",
                    xy=(x, y),
                    xytext=(0, 11),
                    textcoords="offset points",
                    fontsize=12,
                    color=color,
                    ha="center",
                )

    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xlim(0.008, 0.18)

    ymax = max(all_speedups) * 1.20 if all_speedups else 2.5
    ax.set_ylim(0.85, ymax)
    yticks = np.arange(1.0, ymax, 0.25)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y:.2f}×"))

    ax.grid(axis="y", color=GRID_COL, linewidth=0.8, zorder=0)
    ax.grid(axis="x", color=GRID_COL, linewidth=0.5, linestyle=":", zorder=0)
    ax.tick_params(colors=SUB_COL, labelsize=16)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(SUB_COL)

    if len(series) > 1:
        ax.legend(fontsize=12, framealpha=0.9, loc="upper left")

    ax.set_xlabel("Progressive delta (δ)", fontsize=18, color=SUB_COL, labelpad=10)
    ax.set_ylabel("Denoising speedup (×)", fontsize=18, color=SUB_COL, labelpad=10)

    # Subtitle: list all series
    sub_parts = []
    for _, md in series:
        fullres = md.get("fullres_denoise_s")
        steps = md.get("steps", "?")
        if fullres:
            sub_parts.append(f"{steps}-step (baseline={fullres:.1f}s)")
        else:
            sub_parts.append(f"{steps}-step")
    sub = "Ideogram 4 fp8 · " + "  /  ".join(sub_parts) + " · 1024×1024"
    fig.suptitle(
        "Spectral Progressive Diffusion — Ideogram 4\nDenoising Speedup vs. δ",
        fontsize=20,
        fontweight="bold",
        color=TEXT_COL,
        y=1.02,
    )
    ax.set_title(sub, fontsize=13, color=SUB_COL, pad=4)

    ax.text(0.0083, 1.006, "1× baseline", fontsize=13, color=SUB_COL, va="bottom")

    fig.tight_layout(pad=1.2)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=BG)
    print(f"Saved: {out_path}")

    # Summary table
    for label, md in series:
        fullres = md.get("fullres_denoise_s")
        print(f"\n{label}  (fullres={fullres:.2f}s)")
        print(f"{'delta':>8}  {'denoise_s':>10}  {'speedup':>9}")
        print("-" * 32)
        for p in md.get("points", []):
            sp = p.get("speedup")
            ds = p.get("denoise_s")
            sp_str = f"{sp:.4f}×" if sp else "N/A"
            ds_str = f"{ds:.2f}s" if ds else "N/A"
            print(f"  {p['delta']:>6}  {ds_str:>10}  {sp_str:>9}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        required=True,
        action="append",
        dest="jsons",
        help="Results JSON (repeat for multiple series)",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "speedup_ideogram4.png"
        ),
    )
    args = parser.parse_args()
    plot(args.jsons, args.out)


if __name__ == "__main__":
    main()
