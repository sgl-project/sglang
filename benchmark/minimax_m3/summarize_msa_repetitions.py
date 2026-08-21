#!/usr/bin/env python3
"""Fail-closed aggregation for three alternating MiniMax-M3 MSA A/B runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

CONCURRENCIES = (1, 8, 32, 128)
SERVING_METRICS = {
    "output_throughput": True,
    "request_throughput": True,
    "median_ttft_ms": False,
    "p99_ttft_ms": False,
    "median_itl_ms": False,
    "p99_itl_ms": False,
}
ACCURACY_METRICS = ("gpqa", "longbench_v2")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def number(value, description: str, *, positive: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"invalid {description}: {value!r}")
    return result


def gain(baseline: float, candidate: float, higher_is_better: bool) -> float:
    if higher_is_better:
        return candidate / baseline - 1.0
    return baseline / candidate - 1.0


def expected_order(repetition: int) -> list[str]:
    if repetition % 2:
        return ["baseline", "candidate"]
    return ["candidate", "baseline"]


def build_summary(root: Path, repetitions: int = 3) -> dict:
    if repetitions != 3:
        raise ValueError(
            "the publishable MiniMax-M3 gate requires exactly 3 repetitions"
        )

    comparisons = []
    orders = []
    provider_dirs: list[Path] = []
    for repetition in range(1, repetitions + 1):
        repetition_dir = root / f"rep{repetition:02d}"
        recorded_order = load(repetition_dir / "order.json")["order"]
        wanted_order = expected_order(repetition)
        if recorded_order != wanted_order:
            raise ValueError(
                f"rep{repetition:02d} order is {recorded_order}, expected {wanted_order}"
            )
        orders.append(recorded_order)
        comparisons.append(load(repetition_dir / "comparison.json"))
        provider_dirs.extend(
            [repetition_dir / "baseline", repetition_dir / "candidate"]
        )

    gpqa_hashes = {
        (provider_dir / "gpqa_dataset.sha256").read_text().strip()
        for provider_dir in provider_dirs
    }
    if len(gpqa_hashes) != 1:
        raise ValueError("GPQA-Diamond SHA-256 is not identical across all six runs")
    longbench_hashes = {
        load(provider_dir / "longbench_v2_subset_manifest.json")["subset_sha256"]
        for provider_dir in provider_dirs
    }
    if len(longbench_hashes) != 1:
        raise ValueError("LongBench-v2 SHA-256 is not identical across all six runs")

    fixed_answers: dict[str, set[str]] = {}
    for provider_dir in provider_dirs:
        for record in load(provider_dir / "fixed_parity.json")["records"]:
            if not record.get("exact_expected", False):
                raise ValueError(
                    f"fixed probe failed its expected answer: {record['name']}"
                )
            fixed_answers.setdefault(record["name"], set()).add(record["content"])
    expected_fixed = {"short", "long_32768", "long_65536"}
    if set(fixed_answers) != expected_fixed:
        raise ValueError(f"fixed probes are incomplete: {sorted(fixed_answers)}")
    unstable = sorted(
        name for name, answers in fixed_answers.items() if len(answers) != 1
    )
    if unstable:
        raise ValueError(
            "temperature-zero fixed answers changed across repetitions: "
            + ", ".join(unstable)
        )

    result = {
        "schema_version": 1,
        "repetitions": repetitions,
        "orders": orders,
        "dataset_sha256": {
            "gpqa_diamond": next(iter(gpqa_hashes)),
            "longbench_v2": next(iter(longbench_hashes)),
        },
        "accuracy": {},
        "serving": {},
    }
    for eval_name in ACCURACY_METRICS:
        baseline = [
            number(row["accuracy"][eval_name]["baseline"], f"{eval_name} baseline")
            for row in comparisons
        ]
        candidate = [
            number(row["accuracy"][eval_name]["candidate"], f"{eval_name} candidate")
            for row in comparisons
        ]
        baseline_median = statistics.median(baseline)
        candidate_median = statistics.median(candidate)
        result["accuracy"][eval_name] = {
            "baseline_runs": baseline,
            "candidate_runs": candidate,
            "baseline_median": baseline_median,
            "candidate_median": candidate_median,
            "median_delta": candidate_median - baseline_median,
        }

    for concurrency in CONCURRENCIES:
        concurrency_result = {}
        for metric, higher_is_better in SERVING_METRICS.items():
            baseline = [
                number(
                    row["serving"][str(concurrency)][metric]["baseline"],
                    f"c{concurrency} {metric} baseline",
                    positive=True,
                )
                for row in comparisons
            ]
            candidate = [
                number(
                    row["serving"][str(concurrency)][metric]["candidate"],
                    f"c{concurrency} {metric} candidate",
                    positive=True,
                )
                for row in comparisons
            ]
            baseline_median = statistics.median(baseline)
            candidate_median = statistics.median(candidate)
            paired_gains = [
                gain(base, cand, higher_is_better)
                for base, cand in zip(baseline, candidate, strict=True)
            ]
            concurrency_result[metric] = {
                "baseline_runs": baseline,
                "candidate_runs": candidate,
                "baseline_median": baseline_median,
                "candidate_median": candidate_median,
                "gain_from_backend_medians": gain(
                    baseline_median, candidate_median, higher_is_better
                ),
                "median_paired_gain": statistics.median(paired_gains),
            }
        result["serving"][str(concurrency)] = concurrency_result
    return result


def accuracy_noninferiority_failures(
    summary: dict, score_tolerances: dict[str, float]
) -> list[str]:
    if set(score_tolerances) != set(ACCURACY_METRICS):
        raise ValueError("score tolerances must cover every accuracy metric")
    if any(value < 0 for value in score_tolerances.values()):
        raise ValueError("score tolerances must be non-negative")

    failures = []
    for eval_name, tolerance in score_tolerances.items():
        accuracy = summary["accuracy"][eval_name]
        accuracy["noninferiority_tolerance"] = tolerance
        if accuracy["candidate_median"] + tolerance < accuracy["baseline_median"]:
            failures.append(
                f"{eval_name} median exceeded its noninferiority margin: "
                f"{accuracy['candidate_median']:.6f} + {tolerance:.6f} < "
                f"{accuracy['baseline_median']:.6f}"
            )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--min-median-output-throughput-gain",
        type=float,
        default=0.0,
        help="Minimum candidate gain computed from the two backend medians",
    )
    parser.add_argument("--gpqa-score-tolerance", type=float, required=True)
    parser.add_argument("--longbench-score-tolerance", type=float, required=True)
    args = parser.parse_args()

    summary = build_summary(args.root)
    score_tolerances = {
        "gpqa": args.gpqa_score_tolerance,
        "longbench_v2": args.longbench_score_tolerance,
    }
    failures = accuracy_noninferiority_failures(summary, score_tolerances)
    for concurrency in CONCURRENCIES:
        observed = summary["serving"][str(concurrency)]["output_throughput"]
        observed = observed["gain_from_backend_medians"]
        if observed < args.min_median_output_throughput_gain:
            failures.append(
                f"c{concurrency} median output throughput gain {observed:.2%} < "
                f"{args.min_median_output_throughput_gain:.2%}"
            )

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    output = args.output or args.root / "summary.json"
    output.write_text(rendered + "\n")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
