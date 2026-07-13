#!/usr/bin/env python3
"""Plot TTFT and TPOT comparison charts from the live PD flip CSV exports.

The script reads:
  - csv_export/baseline_results.csv
  - csv_export/state_machine_results.csv

and writes PNG/SVG charts under plots_python/ by default.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import seaborn as sns
except ModuleNotFoundError:  # The cluster Docker image has matplotlib but not seaborn.
    sns = None


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

COLORS = {
    "baseline": "#5477C4",
    "state_machine": "#CC6F47",
    "slo": "#464C55",
    "warmup": "#7A828F",
}


def parse_request_index(request_id: str) -> int:
    return int("".join(ch for ch in request_id if ch.isdigit()))


def parse_float(value: str | None) -> float:
    if value is None or str(value).strip() == "":
        return math.nan
    return float(value)


def read_results(path: Path, series_name: str) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "series": series_name,
                    "request_id": row["request_id"],
                    "request_index": parse_request_index(row["request_id"]),
                    "category": row["category"],
                    "prompt_chars": int(row["prompt_chars"]),
                    "start_epoch": parse_float(row.get("start_epoch")),
                    "ttft_s": parse_float(row.get("ttft_s")),
                    "avg_tpot_s": parse_float(row.get("avg_tpot_s")),
                    "chunks": int(row["chunks"] or 0),
                    "finish_reason": row.get("finish_reason", ""),
                }
            )
    return sorted(rows, key=lambda r: r["request_index"])


def configure_theme() -> None:
    if sns is not None:
        sns.set_theme(
            style="whitegrid",
            rc={
                "figure.facecolor": TOKENS["surface"],
                "axes.facecolor": TOKENS["panel"],
                "axes.edgecolor": TOKENS["axis"],
                "axes.labelcolor": TOKENS["ink"],
                "grid.color": TOKENS["grid"],
                "grid.linewidth": 0.8,
                "font.family": "sans-serif",
                "font.sans-serif": [
                    "Aptos",
                    "Inter",
                    "Segoe UI",
                    "DejaVu Sans",
                    "Arial",
                    "sans-serif",
                ],
            },
        )
    else:
        plt.rcParams.update(
            {
                "figure.facecolor": TOKENS["surface"],
                "axes.facecolor": TOKENS["panel"],
                "axes.edgecolor": TOKENS["axis"],
                "axes.labelcolor": TOKENS["ink"],
                "axes.grid": True,
                "grid.color": TOKENS["grid"],
                "grid.linewidth": 0.8,
                "font.family": "sans-serif",
                "font.sans-serif": [
                    "Aptos",
                    "Inter",
                    "Segoe UI",
                    "DejaVu Sans",
                    "Arial",
                    "sans-serif",
                ],
            }
        )


def values(rows: Iterable[dict], field: str) -> tuple[list[int], list[float]]:
    ordered = sorted(rows, key=lambda r: r["request_index"])
    return [r["request_index"] for r in ordered], [r[field] for r in ordered]


def estimate_event_x(rows: list[dict], event_epoch: float) -> float:
    """Map a wall-clock event epoch onto the request-index x-axis."""
    ordered = sorted(rows, key=lambda r: r["request_index"])
    starts = [
        (r["request_index"], r.get("start_epoch", math.nan))
        for r in ordered
        if not math.isnan(r.get("start_epoch", math.nan))
    ]
    if not starts:
        return math.nan
    if event_epoch <= starts[0][1]:
        return float(starts[0][0])
    for (prev_idx, prev_start), (next_idx, next_start) in zip(starts, starts[1:]):
        if prev_start <= event_epoch <= next_start:
            span = next_start - prev_start
            if span <= 0:
                return float(prev_idx)
            return prev_idx + ((event_epoch - prev_start) / span) * (
                next_idx - prev_idx
            )
    return float(starts[-1][0])


def add_event_marker(ax, x: float, label: str, *, y_fraction: float = 0.88) -> None:
    if math.isnan(x):
        return
    ax.axvline(
        x,
        color=COLORS["state_machine"],
        linewidth=1.25,
        linestyle=(0, (2, 2)),
        alpha=0.95,
    )
    ymin, ymax = ax.get_ylim()
    ax.text(
        x + 0.7,
        ymin + (ymax - ymin) * y_fraction,
        label,
        color=COLORS["state_machine"],
        fontsize=8.5,
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": TOKENS["panel"],
            "edgecolor": "#FFBDA1",
            "linewidth": 0.8,
            "alpha": 0.92,
        },
    )


def add_header(fig, title: str, subtitle: str) -> None:
    fig.text(
        0.085,
        0.965,
        title,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        0.085,
        0.92,
        subtitle,
        ha="left",
        va="top",
        fontsize=10,
        color=TOKENS["muted"],
    )


def plot_line_comparison(
    *,
    baseline: list[dict],
    state_machine: list[dict],
    field: str,
    ylabel: str,
    title: str,
    subtitle: str,
    slo: float,
    output_base: Path,
    y_as_ms: bool = False,
    event_x: float = math.nan,
    event_label: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 6.2), dpi=160)
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.14, top=0.80)

    bx, by = values(baseline, field)
    sx, sy = values(state_machine, field)
    if y_as_ms:
        by = [v * 1000 if not math.isnan(v) else math.nan for v in by]
        sy = [v * 1000 if not math.isnan(v) else math.nan for v in sy]
        slo_y = slo * 1000
    else:
        slo_y = slo

    ax.plot(
        bx,
        by,
        color=COLORS["baseline"],
        linewidth=1.7,
        marker="o",
        markersize=3.0,
        label="No state machine",
    )
    ax.plot(
        sx,
        sy,
        color=COLORS["state_machine"],
        linewidth=1.7,
        marker="s",
        markersize=3.0,
        linestyle="--",
        label="State machine",
    )

    ax.axhline(
        slo_y,
        color=COLORS["slo"],
        linewidth=1.1,
        linestyle=(0, (4, 3)),
        label=f"SLO = {slo:g}s",
    )
    ax.axvline(
        16,
        color=COLORS["warmup"],
        linewidth=1.0,
        linestyle=":",
    )
    ax.text(
        16.6,
        ax.get_ylim()[1] * 0.94,
        "first 16 concurrent burst",
        color=COLORS["warmup"],
        fontsize=8.5,
        va="top",
    )
    if event_label:
        add_event_marker(ax, event_x, event_label)

    missing = sum(1 for v in sy if math.isnan(v))
    if missing:
        ax.text(
            0.995,
            0.04,
            f"State-machine missing points: {missing} zero-chunk requests",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.5,
            color=TOKENS["muted"],
        )

    ax.set_xlim(1, 100)
    ax.set_xlabel("Request index (trace-001 to trace-100)")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))
    ax.tick_params(colors=TOKENS["muted"], labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TOKENS["axis"])
    ax.spines["bottom"].set_color(TOKENS["axis"])
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        ncol=3,
        borderaxespad=0,
    )
    add_header(fig, title, subtitle)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_two_panel(
    baseline: list[dict],
    state_machine: list[dict],
    output_base: Path,
    event_x: float = math.nan,
    event_label: str = "",
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 9.2), dpi=160, sharex=True)
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.10, top=0.84, hspace=0.28)

    specs = [
        (axes[0], "ttft_s", "TTFT seconds", 8.0, False),
        (axes[1], "avg_tpot_s", "Avg TPOT milliseconds", 0.02, True),
    ]
    for ax, field, ylabel, slo, y_as_ms in specs:
        bx, by = values(baseline, field)
        sx, sy = values(state_machine, field)
        if y_as_ms:
            by = [v * 1000 if not math.isnan(v) else math.nan for v in by]
            sy = [v * 1000 if not math.isnan(v) else math.nan for v in sy]
            slo_y = slo * 1000
            slo_label = f"SLO = {slo * 1000:g}ms"
        else:
            slo_y = slo
            slo_label = f"SLO = {slo:g}s"

        ax.plot(bx, by, color=COLORS["baseline"], linewidth=1.6, marker="o", markersize=2.8, label="No state machine")
        ax.plot(sx, sy, color=COLORS["state_machine"], linewidth=1.6, marker="s", markersize=2.8, linestyle="--", label="State machine")
        ax.axhline(slo_y, color=COLORS["slo"], linewidth=1.1, linestyle=(0, (4, 3)), label=slo_label)
        ax.axvline(16, color=COLORS["warmup"], linewidth=1.0, linestyle=":")
        if event_label:
            add_event_marker(ax, event_x, event_label, y_fraction=0.84)
        ax.set_ylabel(ylabel)
        ax.tick_params(colors=TOKENS["muted"], labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(TOKENS["axis"])
        ax.spines["bottom"].set_color(TOKENS["axis"])

    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), frameon=False, ncol=3, borderaxespad=0)
    axes[1].set_xlim(1, 100)
    axes[1].set_xlabel("Request index (trace-001 to trace-100)")
    axes[1].xaxis.set_major_locator(mticker.MultipleLocator(10))
    add_header(
        fig,
        "TTFT and TPOT comparison by request",
        "Baseline versus state-machine run on the same 100-request trace. Gaps in the state-machine line are zero-chunk requests.",
    )

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def write_chart_data(path: Path, baseline: list[dict], state_machine: list[dict]) -> None:
    fields = [
        "series",
        "request_id",
        "request_index",
        "category",
        "prompt_chars",
        "start_epoch",
        "ttft_s",
        "avg_tpot_s",
        "chunks",
        "finish_reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(baseline)
        writer.writerows(state_machine)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv-dir",
        default="csv_export",
        help="Directory containing baseline_results.csv and state_machine_results.csv",
    )
    parser.add_argument("--out-dir", default="plots_python")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    baseline = read_results(csv_dir / "baseline_results.csv", "No state machine")
    state_machine = read_results(csv_dir / "state_machine_results.csv", "State machine")
    first_state_start = min(r["start_epoch"] for r in state_machine)
    d_to_p_epoch = first_state_start + 5.0
    d_to_p_x = estimate_event_x(state_machine, d_to_p_epoch)
    d_to_p_label = "D->P triggered\nnode3 decode -> prefill"

    configure_theme()
    plot_line_comparison(
        baseline=baseline,
        state_machine=state_machine,
        field="ttft_s",
        ylabel="TTFT seconds",
        title="TTFT comparison by request",
        subtitle="Same 100-request trace; request index is ordered by trace id, not wall-clock spacing. Vertical guide marks the first 16 concurrent requests.",
        slo=8.0,
        output_base=out_dir / "ttft_comparison_python",
        y_as_ms=False,
        event_x=d_to_p_x,
        event_label=d_to_p_label,
    )
    plot_line_comparison(
        baseline=baseline,
        state_machine=state_machine,
        field="avg_tpot_s",
        ylabel="Avg TPOT milliseconds",
        title="Average TPOT comparison by request",
        subtitle="Per-request average inter-token latency; state-machine gaps are requests with no first token/zero chunks.",
        slo=0.02,
        output_base=out_dir / "tpot_comparison_python",
        y_as_ms=True,
        event_x=d_to_p_x,
        event_label=d_to_p_label,
    )
    plot_two_panel(
        baseline,
        state_machine,
        out_dir / "ttft_tpot_comparison_python",
        event_x=d_to_p_x,
        event_label=d_to_p_label,
    )
    write_chart_data(out_dir / "chart_data_long.csv", baseline, state_machine)
    print(f"Estimated D->P marker x={d_to_p_x:.2f} from epoch={d_to_p_epoch:.3f}")
    print(f"Wrote charts to {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
