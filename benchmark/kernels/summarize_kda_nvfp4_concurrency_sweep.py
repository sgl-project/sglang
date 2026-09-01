#!/usr/bin/env python3
"""Summarize a KDA NVFP4 E2E concurrency sweep."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

METRICS = (
    ("output_throughput", True),
    ("mean_tpot_ms", False),
    ("mean_ttft_ms", False),
    ("mean_e2e_latency_ms", False),
)


def load_jsonl(path: Path):
    with path.open() as file:
        return [json.loads(line) for line in file if line.strip()]


def mean(rows, key):
    return statistics.mean(float(row[key]) for row in rows)


def summarize(results_dir: Path):
    model = (results_dir / "model_name.txt").read_text().strip()
    concurrencies = [
        int(value) for value in (results_dir / "concurrencies.txt").read_text().split()
    ]
    rows = []
    for concurrency in concurrencies:
        baseline = load_jsonl(results_dir / f"baseline_c{concurrency}.jsonl")
        candidate = load_jsonl(results_dir / f"candidate_c{concurrency}.jsonl")
        adjacent = load_jsonl(results_dir / f"baseline_adjacent_c{concurrency}.jsonl")
        metrics = {}
        for key, higher_is_better in METRICS:
            base = mean(baseline, key)
            cand = mean(candidate, key)
            adjacent_value = mean(adjacent, key)
            raw_change = (cand / base - 1.0) * 100.0
            improvement = raw_change if higher_is_better else -raw_change
            metrics[key] = {
                "baseline": base,
                "candidate": cand,
                "baseline_adjacent": adjacent_value,
                "improvement_pct": improvement,
            }
        rows.append({"model": model, "concurrency": concurrency, "metrics": metrics})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dirs", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    rows = [row for path in args.results_dirs for row in summarize(path)]
    if args.json:
        print(json.dumps(rows, indent=2))
        return

    print(
        "| Model | Concurrency | Baseline tok/s | KDA tok/s | "
        "Adjacent baseline | Throughput improvement | TPOT improvement |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        throughput = row["metrics"]["output_throughput"]
        tpot = row["metrics"]["mean_tpot_ms"]
        print(
            f"| {row['model']} | {row['concurrency']} | "
            f"{throughput['baseline']:.3f} | {throughput['candidate']:.3f} | "
            f"{throughput['baseline_adjacent']:.3f} | "
            f"{throughput['improvement_pct']:+.2f}% | "
            f"{tpot['improvement_pct']:+.2f}% |"
        )


if __name__ == "__main__":
    main()
