#!/usr/bin/env python3
"""Compare baseline/candidate MiniMax-M3 MSA gate artifacts."""

from __future__ import annotations

import argparse
import json
import re
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


def load_sha256(path: Path) -> str:
    value = path.read_text().strip()
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"invalid SHA-256 artifact: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--score-tolerance", type=float, default=0.0)
    parser.add_argument("--num-prompts", type=int, default=256)
    parser.add_argument(
        "--output",
        type=Path,
        help="Also write the machine-readable comparison JSON to this path",
    )
    parser.add_argument(
        "--min-output-throughput-gain",
        type=float,
        help="Optional minimum fractional output-throughput gain at every concurrency",
    )
    args = parser.parse_args()

    failures = []
    baseline_gpqa_sha = load_sha256(args.baseline_dir / "gpqa_dataset.sha256")
    candidate_gpqa_sha = load_sha256(args.candidate_dir / "gpqa_dataset.sha256")
    if baseline_gpqa_sha != candidate_gpqa_sha:
        failures.append("GPQA-Diamond dataset hashes differ")
    baseline_parity = load(args.baseline_dir / "fixed_parity.json")
    candidate_parity = load(args.candidate_dir / "fixed_parity.json")
    baseline_records = {row["name"]: row for row in baseline_parity["records"]}
    candidate_records = {row["name"]: row for row in candidate_parity["records"]}
    expected_fixed_names = {"short", "long_32768", "long_65536"}
    if set(baseline_records) != expected_fixed_names:
        failures.append(
            f"baseline fixed probes are incomplete: {sorted(baseline_records)}"
        )
    if set(candidate_records) != expected_fixed_names:
        failures.append(
            f"candidate fixed probes are incomplete: {sorted(candidate_records)}"
        )
    for name, baseline in baseline_records.items():
        candidate = candidate_records.get(name)
        if not baseline.get("exact_expected", False):
            failures.append(f"baseline fixed probe failed its expected answer: {name}")
        if candidate is not None and not candidate.get("exact_expected", False):
            failures.append(f"candidate fixed probe failed its expected answer: {name}")
        if (
            candidate is None
            or candidate["response_sha256"] != baseline["response_sha256"]
        ):
            failures.append(f"fixed output mismatch: {name}")
        if name.startswith("long_"):
            minimum_tokens = int(name.removeprefix("long_"))
            if int(baseline.get("prompt_tokens", 0)) < minimum_tokens:
                failures.append(f"baseline probe {name} is shorter than {minimum_tokens}")
            if (
                candidate is not None
                and int(candidate.get("prompt_tokens", 0)) < minimum_tokens
            ):
                failures.append(
                    f"candidate probe {name} is shorter than {minimum_tokens}"
                )

    baseline_subset = load(
        args.baseline_dir / "longbench_v2_subset_manifest.json"
    )
    candidate_subset = load(
        args.candidate_dir / "longbench_v2_subset_manifest.json"
    )
    if baseline_subset.get("subset_sha256") != candidate_subset.get("subset_sha256"):
        failures.append("LongBench-v2 subset hashes differ")
    for label, manifest in (
        ("baseline", baseline_subset),
        ("candidate", candidate_subset),
    ):
        if int(manifest.get("num_examples", 0)) != 100:
            failures.append(
                f"{label} LongBench-v2 subset does not contain 100 examples"
            )
        if int(manifest.get("minimum_observed_tokens", 0)) < 32768:
            failures.append(f"{label} LongBench-v2 subset contains a sub-32K prompt")

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

    rendered = json.dumps(comparisons, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
