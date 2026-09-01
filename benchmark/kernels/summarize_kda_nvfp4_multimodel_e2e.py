#!/usr/bin/env python3
"""Summarize baseline/candidate JSONL produced by the multi-model E2E run."""

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
    baseline = load_jsonl(results_dir / "baseline.jsonl")
    candidate = load_jsonl(results_dir / "candidate.jsonl")
    adjacent = load_jsonl(results_dir / "baseline_adjacent.jsonl")
    model_name = (results_dir / "model_name.txt").read_text().strip()
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
            "candidate_change_pct": raw_change,
            "improvement_pct": improvement,
        }
    return {"model": model_name, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dirs", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    summaries = [summarize(path) for path in args.results_dirs]
    if args.json:
        print(json.dumps(summaries, indent=2))
        return

    print(
        "| Model | Metric | Baseline mean | KDA mean | Adjacent baseline | Improvement |"
    )
    print("|---|---|---:|---:|---:|---:|")
    for summary in summaries:
        for key, values in summary["metrics"].items():
            print(
                f"| {summary['model']} | {key} | {values['baseline']:.3f} | "
                f"{values['candidate']:.3f} | {values['baseline_adjacent']:.3f} | "
                f"{values['improvement_pct']:+.2f}% |"
            )


if __name__ == "__main__":
    main()
