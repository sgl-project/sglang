"""Render the historical integrated-build comparison, not a kernel microbenchmark.

Run with Python and matplotlib from this directory or any working directory.
The data and the comparison limits are recorded in measurements.json.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "measurements.json").read_text())
arms = [data["arms"][key] for key in ("off", "on")]

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.axisbelow": True,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    }
)

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.30, wspace=0.38)
panels = (
    ("output_tokens_per_second", "Output throughput (tokens/s)", 800, 200),
    ("p95_ttft_seconds", "p95 time to first token (s)", 30, 5),
)
colors = ("#777777", "#0072B2")
for ax, (key, label, upper, tick) in zip(axes, panels):
    values = [arm[key] for arm in arms]
    bars = ax.bar([0, 1], values, width=0.57, color=colors, zorder=3)
    bars[0].set_hatch("//")
    bars[0].set_edgecolor("white")
    ax.set_xticks([0, 1], ["Feature OFF", "Feature ON\n+ runtime profile"])
    ax.set_ylabel(label)
    ax.set_ylim(0, upper)
    ax.yaxis.set_major_locator(MultipleLocator(tick))
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.tick_params(axis="both", length=0)
    for index, value in enumerate(values):
        ax.text(index, value + upper * 0.025, f"{value:.3f}", ha="center", va="bottom")
    change = 100 * (values[1] / values[0] - 1)
    ax.text(
        0.98,
        0.98,
        f"{change:+.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=colors[1],
        fontweight="bold",
    )

fig.text(
    0.5,
    0.10,
    "B300 · TP8/DCP8 · no speculation · 32K input / 300 output incl. reasoning · C64\n"
    "Historical pre-fix build; one run per arm, 128 requests each. Whole package + profile comparison.",
    ha="center",
    va="center",
    fontsize=9,
    linespacing=1.7,
)
for suffix in ("pdf", "png"):
    fig.savefig(HERE / f"integrated-comparison.{suffix}", facecolor="white")
plt.close(fig)
