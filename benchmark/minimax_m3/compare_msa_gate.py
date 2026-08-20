#!/usr/bin/env python3
"""Compare baseline/candidate MiniMax-M3 MSA gate artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def load_last_jsonl(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not records:
        raise ValueError(f"no benchmark records in {path}")
    return records[-1]


def relative_gain(baseline: float, candidate: float) -> float:
    return candidate / baseline - 1.0


def latency_gain(baseline: float, candidate: float) -> float:
    return baseline / candidate - 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--score-tolerance", type=float, default=0.0)
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument(
        "--min-output-throughput-gain",
        type=float,
        help="Optional minimum fractional output-throughput gain at every concurrency",
    )
    args = parser.parse_args()

    failures = []
    baseline_parity = load(args.baseline_dir / "fixed_parity.json")
    candidate_parity = load(args.candidate_dir / "fixed_parity.json")
    baseline_records = {row["name"]: row for row in baseline_parity["records"]}
    candidate_records = {row["name"]: row for row in candidate_parity["records"]}
    for name, baseline in baseline_records.items():
        candidate = candidate_records.get(name)
        if (
            candidate is None
            or candidate["response_sha256"] != baseline["response_sha256"]
        ):
            failures.append(f"fixed output mismatch: {name}")

    comparisons = {"accuracy": {}, "serving": {}}
    for eval_name in ("gpqa", "longbench_v2"):
        baseline = load(args.baseline_dir / f"{eval_name}.json")
        candidate = load(args.candidate_dir / f"{eval_name}.json")
        baseline_score = float(baseline["score"])
        candidate_score = float(candidate["score"])
        comparisons["accuracy"][eval_name] = {
            "baseline": baseline_score,
            "candidate": candidate_score,
            "delta": candidate_score - baseline_score,
        }
        if candidate_score + args.score_tolerance < baseline_score:
            failures.append(
                f"{eval_name} regressed: {candidate_score:.6f} < {baseline_score:.6f}"
            )

    for concurrency in (1, 8, 32, 128):
        baseline = load_last_jsonl(
            args.baseline_dir / f"serving_c{concurrency}.jsonl"
        )
        candidate = load_last_jsonl(
            args.candidate_dir / f"serving_c{concurrency}.jsonl"
        )
        for label, record in (("baseline", baseline), ("candidate", candidate)):
            if int(record["completed"]) != args.num_prompts:
                failures.append(
                    f"{label} c{concurrency} completed {record['completed']} / "
                    f"{args.num_prompts} requests"
                )
        throughput_gain = relative_gain(
            float(baseline["output_throughput"]),
            float(candidate["output_throughput"]),
        )
        comparisons["serving"][str(concurrency)] = {
            "output_throughput": {
                "baseline": baseline["output_throughput"],
                "candidate": candidate["output_throughput"],
                "gain": throughput_gain,
            },
            "request_throughput": {
                "baseline": baseline["request_throughput"],
                "candidate": candidate["request_throughput"],
                "gain": relative_gain(
                    float(baseline["request_throughput"]),
                    float(candidate["request_throughput"]),
                ),
            },
            "median_ttft_ms": {
                "baseline": baseline["median_ttft_ms"],
                "candidate": candidate["median_ttft_ms"],
                "gain": latency_gain(
                    float(baseline["median_ttft_ms"]),
                    float(candidate["median_ttft_ms"]),
                ),
            },
            "p99_ttft_ms": {
                "baseline": baseline["p99_ttft_ms"],
                "candidate": candidate["p99_ttft_ms"],
                "gain": latency_gain(
                    float(baseline["p99_ttft_ms"]),
                    float(candidate["p99_ttft_ms"]),
                ),
            },
            "median_itl_ms": {
                "baseline": baseline["median_itl_ms"],
                "candidate": candidate["median_itl_ms"],
                "gain": latency_gain(
                    float(baseline["median_itl_ms"]),
                    float(candidate["median_itl_ms"]),
                ),
            },
            "p99_itl_ms": {
                "baseline": baseline["p99_itl_ms"],
                "candidate": candidate["p99_itl_ms"],
                "gain": latency_gain(
                    float(baseline["p99_itl_ms"]),
                    float(candidate["p99_itl_ms"]),
                ),
            },
        }
        if (
            args.min_output_throughput_gain is not None
            and throughput_gain < args.min_output_throughput_gain
        ):
            failures.append(
                f"c{concurrency} output throughput gain {throughput_gain:.2%} < "
                f"{args.min_output_throughput_gain:.2%}"
            )

    print(json.dumps(comparisons, indent=2, sort_keys=True))
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
