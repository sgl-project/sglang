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

VARIANTS = [
    "sgl-kernel-base",
    "sgl-kernel",
    "sgl-kernel-mode2",
    "flashinfer",
    "flashinfer-small",
    "flashinfer-small-det",
]
# per-shape microseconds after base / branch / delta, and suite ratios to base
OTHERS = ["sgl-kernel-mode2", "flashinfer", "flashinfer-small", "flashinfer-small-det"]
HEADERS = {
    "sgl-kernel-mode2": "sgl-kernel mode 2",
    "flashinfer": "FlashInfer fused",
    "flashinfer-small": "FlashInfer fused small",
    "flashinfer-small-det": "FlashInfer fused small det",
}


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

    print(
        "| suite | shape | sgl-kernel base (us) | sgl-kernel branch (us) | delta | "
        + " | ".join(f"{HEADERS[v]} (us)" for v in OTHERS)
        + " |"
    )
    print("|---|---|" + "---:|" * (3 + len(OTHERS)))
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
        for variant in OTHERS:
            if variant in med:
                row.append(f"{med[variant]:.3f}")
                ratios[suite][variant].append(med[variant] / base)
                if variant == "flashinfer-small-det" and "sgl-kernel-mode2" in med:
                    ratios[suite]["det/mode2"].append(med[variant] / med["sgl-kernel-mode2"])
            else:
                row.append("")
        print("| " + " | ".join(row) + " |")

    print()
    print(
        "| suite | shapes | sgl-kernel branch / base | "
        + " | ".join(f"{HEADERS[v]} / base" for v in OTHERS)
        + " | FlashInfer fused small det / sgl-kernel mode 2 |"
    )
    print("|---|---:|" + "---:|" * (2 + len(OTHERS)))
    for suite, by_variant in ratios.items():
        n = len(by_variant["sgl-kernel"])
        print(
            f"| {suite} | {n} | "
            + " | ".join(f"{geomean(by_variant[v]):.4f}x" for v in ["sgl-kernel", *OTHERS, "det/mode2"])
            + " |"
        )
    print()
    print("environment:", json.dumps(meta, indent=1, sort_keys=True))
    print()
    print(f"cells over {args.rerun_threshold}% (sgl-kernel branch vs base):", json.dumps(over))


if __name__ == "__main__":
    main()
