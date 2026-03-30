"""Plot and compare DCP benchmark results across concurrency levels.

Parses bench_serving output files (cc*.txt) from benchmark result folders,
generates comparison charts and markdown tables.

Usage:
    python3 benchmark/dcp/plot_dcp_bench.py --results-dir benchmark/dcp/results/<run>
    python3 benchmark/dcp/plot_dcp_bench.py  # auto-discover from script directory
    python3 benchmark/dcp/plot_dcp_bench.py --ignore-folder q_rep
"""

import argparse
import os
import re

import matplotlib.pyplot as plt
from tabulate import tabulate

METRICS = {
    "Output token throughput (tok/s)": r"Output token throughput \(tok/s\):\s+([\d.]+)",
    "Mean TTFT (ms)": r"Mean TTFT \(ms\):\s+([\d.]+)",
    "Mean TPOT (ms)": r"Mean TPOT \(ms\):\s+([\d.]+)",
    "Mean ITL (ms)": r"Mean ITL \(ms\):\s+([\d.]+)",
}


def parse_bench_folder(folder_path):
    """Parse all cc*.txt files in a folder, return {concurrency: {metric: value}}."""
    results = {}
    for fname in os.listdir(folder_path):
        m = re.match(r"cc(\d+)\.txt", fname)
        if not m:
            continue
        cc = int(m.group(1))
        with open(os.path.join(folder_path, fname)) as f:
            text = f.read()
        metrics = {}
        for name, pattern in METRICS.items():
            match = re.search(pattern, text)
            if match:
                metrics[name] = float(match.group(1))
        if metrics:
            results[cc] = metrics
    return results


def discover_folders(base_dir, ignore_folders=None):
    """Walk base_dir to find all folders containing cc*.txt files."""
    ignore_set = set(ignore_folders or [])
    folder_configs = []
    for root, _, files in os.walk(base_dir):
        if any(re.match(r"cc\d+\.txt", f) for f in files):
            rel = os.path.relpath(root, base_dir)
            label = rel.replace(os.sep, "_")
            if any(ign in label or ign in rel for ign in ignore_set):
                continue
            folder_configs.append((root, label))
    return sorted(folder_configs, key=lambda x: x[1])


def plot_comparison(all_data, output_file):
    """Plot benchmark comparison across folders."""
    _, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, metric_name in enumerate(METRICS):
        ax = axes[idx]
        for label, data in all_data.items():
            ccs = sorted(data.keys())
            values = [data[cc].get(metric_name, float("nan")) for cc in ccs]
            ax.plot(ccs, values, marker="o", label=label)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("DCP Benchmark: Throughput & Latency vs Concurrency", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Chart saved to {output_file}")


def save_markdown_tables(all_data, output_file):
    """Write markdown tables for all metrics."""
    all_ccs = sorted(set(cc for data in all_data.values() for cc in data))
    labels = list(all_data.keys())
    cc_headers = [f"cc{cc}" for cc in all_ccs]

    md_lines = ["# DCP Benchmark Results\n"]
    for metric_name in METRICS:
        rows = []
        for label in labels:
            data = all_data[label]
            vals = [
                (
                    f"{data.get(cc, {}).get(metric_name, 0):.2f}"
                    if data.get(cc, {}).get(metric_name) is not None
                    else "-"
                )
                for cc in all_ccs
            ]
            rows.append([label] + vals)

        print(f"\n  {metric_name}")
        print(tabulate(rows, headers=["Config"] + cc_headers, tablefmt="simple_grid"))

        md_lines.append(f"## {metric_name}\n")
        md_lines.append(
            tabulate(rows, headers=["Config"] + cc_headers, tablefmt="github")
        )
        md_lines.append("")

    with open(output_file, "w") as f:
        f.write("\n".join(md_lines))
    print(f"\nMarkdown saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot and compare DCP benchmark results."
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Directory containing benchmark result folders. "
        "Defaults to auto-discover from script directory.",
    )
    parser.add_argument(
        "--output-chart",
        default=None,
        help="Output chart filename (default: bench_comparison.png in results dir).",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Output markdown filename (default: bench_results.md in results dir).",
    )
    parser.add_argument(
        "--ignore-folder",
        action="append",
        default=[],
        help="Substring to ignore in folder labels (repeatable).",
    )
    args = parser.parse_args()

    base = args.results_dir or os.path.dirname(os.path.abspath(__file__))
    folder_configs = discover_folders(base, args.ignore_folder)

    if not folder_configs:
        print(f"No benchmark folders found in {base}")
        return

    print(f"Found {len(folder_configs)} benchmark folders:")
    for _, label in folder_configs:
        print(f"  {label}")

    all_data = {}
    for folder_path, label in folder_configs:
        all_data[label] = parse_bench_folder(folder_path)

    chart_file = args.output_chart or os.path.join(base, "bench_comparison.png")
    md_file = args.output_md or os.path.join(base, "bench_results.md")

    plot_comparison(all_data, chart_file)
    save_markdown_tables(all_data, md_file)


if __name__ == "__main__":
    main()
