"""Pool bench_topk.py runs into per-shape medians and markdown tables.

    python3 analyze_bench.py <result dir> [--rerun-threshold 2.0]

Samples of both opposite-order runs (and of re-runs, which replace a cell's
samples) are pooled; the median of the pool is the cell value. Prints the raw
per-cell table, geometric-mean summaries, and the labels whose sgl-kernel
before/after delta exceeds the threshold.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
from collections import defaultdict

VARIANTS = ["sgl-kernel-base", "sgl-kernel", "flashinfer", "flashinfer-small"]


def load(result_dir):
    cells = defaultdict(list)
    reruns = defaultdict(list)
    meta = {}
    order = []
    for path in sorted(glob.glob(os.path.join(result_dir, "*.jsonl"))):
        is_rerun = os.path.basename(path).startswith("rerun")
        for line in open(path):
            record = json.loads(line)
            if "environment" in record:
                meta[record["variant"]] = record["environment"]
                continue
            key = (record["suite"], record["label"], record["variant"])
            (reruns if is_rerun else cells)[key].extend(record["samples_us"])
            if (record["suite"], record["label"]) not in order:
                order.append((record["suite"], record["label"]))
    for key, samples in reruns.items():
        cells[key] = samples
    return cells, meta, order, set(k[:2] for k in reruns)


def geomean(values):
    values = [v for v in values if v > 0]
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir")
    parser.add_argument("--rerun-threshold", type=float, default=2.0)
    args = parser.parse_args()
    cells, meta, order, rerun_labels = load(args.result_dir)

    print("| suite | shape | sgl-kernel base (us) | sgl-kernel branch (us) | delta | FlashInfer fused (us) | FI / base | FlashInfer fused small (us) | FI small / base |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    ratios = defaultdict(lambda: defaultdict(list))
    over = []
    for suite, label in order:
        med = {v: statistics.median(cells[(suite, label, v)]) for v in VARIANTS if cells.get((suite, label, v))}
        if "sgl-kernel-base" not in med:
            continue
        base = med["sgl-kernel-base"]
        row = [suite, label + (" (re-run)" if (suite, label) in rerun_labels else ""), f"{base:.3f}"]
        if "sgl-kernel" in med:
            delta = (med["sgl-kernel"] / base - 1) * 100
            row += [f"{med['sgl-kernel']:.3f}", f"{delta:+.2f}%"]
            ratios[suite]["sgl-kernel"].append(med["sgl-kernel"] / base)
            if abs(delta) > args.rerun_threshold and (suite, label) not in rerun_labels:
                over.append((suite, label))
        else:
            row += ["", ""]
        for variant in ("flashinfer", "flashinfer-small"):
            if variant in med:
                row += [f"{med[variant]:.3f}", f"{med[variant] / base:.2f}x"]
                ratios[suite][variant].append(med[variant] / base)
            else:
                row += ["", ""]
        print("| " + " | ".join(row) + " |")

    print()
    print("| suite | shapes | sgl-kernel branch / base | FlashInfer fused / base | FlashInfer fused small / base |")
    print("|---|---:|---:|---:|---:|")
    for suite, by_variant in ratios.items():
        n = len(by_variant["sgl-kernel"])
        print(
            f"| {suite} | {n} | "
            + " | ".join(f"{geomean(by_variant[v]):.4f}x" for v in ("sgl-kernel", "flashinfer", "flashinfer-small"))
            + " |"
        )
    print()
    print("environment:", json.dumps(meta, indent=1, sort_keys=True))
    print()
    print(f"cells over {args.rerun_threshold}% (sgl-kernel branch vs base):", json.dumps(over))


if __name__ == "__main__":
    main()
