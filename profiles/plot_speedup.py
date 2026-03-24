#!/usr/bin/env python3
"""Plot inductor compilation speedup bar charts from bench_one_batch JSONL profiles.

Usage:
    python profiles/plot_speedup.py profiles/openai/gpt-oss-20b-bf16
    python profiles/plot_speedup.py profiles/Qwen/Qwen3-30B-A3B
    python profiles/plot_speedup.py profiles/Qwen/Qwen3-30B-A3B --baseline "inductor[rmsnorm]"

Expects the directory to contain JSONL files named like:
    b1b-moe[auto]-inductor[None].jsonl
    b1b-moe[auto]-inductor[moe].jsonl
    ...

The baseline defaults to inductor[None]. Override with --baseline.
Two charts are produced: median decode throughput speedup and overall throughput speedup.
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


def load_profiles(directory):
    """Load all JSONL profile files from directory, keyed by inductor config name."""
    pattern = os.path.join(directory, "*.jsonl")
    data = {}
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)
        m = re.search(r"inductor\[(.+?)\]\.jsonl$", fname)
        if not m:
            continue
        config_name = f"inductor[{m.group(1)}]"
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        if entries:
            data[config_name] = entries
    return data


def plot_speedup(data, baseline_name, model_name, output_path):
    baseline = data[baseline_name]
    batch_sizes = [e["batch_size"] for e in baseline]
    baseline_decode = {e["batch_size"]: e["median_decode_throughput"] for e in baseline}
    baseline_overall = {e["batch_size"]: e["overall_throughput"] for e in baseline}

    configs = [k for k in data if k != baseline_name]
    n = len(configs)
    if n == 0:
        print("No non-baseline configs found, nothing to plot.")
        sys.exit(1)

    palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
    colors = palette[:n]
    width = min(0.8 / n, 0.25)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    x = np.arange(len(batch_sizes))

    def add_labels(ax, bars, speedups):
        for bar, s in zip(bars, speedups):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{s:.2f}x",
                ha="left",
                va="bottom",
                fontsize=max(6, 8 - n * 0.5),
                fontweight="bold",
                rotation=45,
                rotation_mode="anchor",
            )

    for chart_idx, (metric, title) in enumerate(
        [
            ("median_decode_throughput", "Median Decode Throughput Speedup"),
            ("overall_throughput", "Overall Throughput Speedup"),
        ]
    ):
        ax = axes[chart_idx]
        bl = baseline_decode if metric == "median_decode_throughput" else baseline_overall

        for i, cfg in enumerate(configs):
            speedups = [e[metric] / bl[e["batch_size"]] for e in data[cfg]]
            bars = ax.bar(x + i * width, speedups, width, label=cfg, color=colors[i])
            add_labels(ax, bars, speedups)

        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, label=f"baseline ({baseline_name})")
        ax.set_xlabel("Batch Size", fontsize=12)
        ax.set_ylabel(f"Speedup over {baseline_name}", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x + width * (n - 1) / 2)
        ax.set_xticklabels([str(bs) for bs in batch_sizes])
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)

    plt.suptitle(
        f"{model_name} — Inductor Compilation Speedups vs Baseline",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", help="Directory containing JSONL profile files")
    parser.add_argument("--baseline", default="inductor[None]", help='Baseline config name (default: "inductor[None]")')
    parser.add_argument("--model-name", default=None, help="Model name for the chart title (default: derived from directory path)")
    parser.add_argument("-o", "--output", default=None, help="Output PNG path (default: <directory>/speedup_charts.png)")
    args = parser.parse_args()

    data = load_profiles(args.directory)
    if not data:
        print(f"No JSONL profile files found in {args.directory}")
        sys.exit(1)

    if args.baseline not in data:
        print(f"Baseline '{args.baseline}' not found. Available: {list(data.keys())}")
        sys.exit(1)

    model_name = args.model_name or "/".join(args.directory.rstrip("/").split("/")[-2:])
    output_path = args.output or os.path.join(args.directory, "speedup_charts.png")

    plot_speedup(data, args.baseline, model_name, output_path)


if __name__ == "__main__":
    main()
