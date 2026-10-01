"""Summarize a complete pilot run; reject missing/mismatched measurements."""

import argparse
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.results / "manifest.json").read_text())
    modes = ("baseline", "static", "planned")
    metrics = (
        "median_tpot_ms",
        "p90_tpot_ms",
        "median_ttft_ms",
        "p90_ttft_ms",
        "median_e2e_latency_ms",
        "output_throughput",
    )
    report = {"scope": manifest["scope"], "accuracy": {}, "comparisons": []}
    for mode in modes:
        report["accuracy"][mode] = json.loads(
            (args.results / mode / "accuracy.json").read_text()
        )
        accuracy = report["accuracy"][mode]
        if accuracy["invalid"] or accuracy["accuracy"] < 0.9:
            raise ValueError(f"Accuracy smoke gate failed for {mode}")
    for workload in ("uniform8k", "mixed32k"):
        for concurrency in manifest["concurrency"]:
            measurements = {}
            for mode in modes:
                measurements[mode] = []
                for repeat in range(manifest["repetitions"]):
                    name = f"{workload}-c{concurrency}-r{repeat}.jsonl"
                    path = args.results / mode / name
                    lines = path.read_text().splitlines()
                    if len(lines) != 1:
                        raise ValueError(f"Expected one measurement in {path}")
                    data = json.loads(lines[0])
                    if data["completed"] != max(8, concurrency * 2):
                        raise ValueError(f"Incomplete requests in {path}")
                    if any(data["errors"]):
                        raise ValueError(f"Request errors in {path}")
                    if any(
                        not math.isfinite(data[metric]) or data[metric] <= 0
                        for metric in metrics
                    ):
                        raise ValueError(f"Invalid timing metrics in {path}")
                    measurements[mode].append(data)
            for repeat in range(manifest["repetitions"]):
                for field in (
                    "total_input_tokens",
                    "total_output_tokens",
                    "input_lens",
                    "output_lens",
                ):
                    values = [measurements[mode][repeat][field] for mode in modes]
                    if any(value != values[0] for value in values[1:]):
                        raise ValueError(
                            f"Unmatched {field}: {workload}, c={concurrency}, r={repeat}"
                        )
            summary = {}
            for mode, samples in measurements.items():
                summary[mode] = {}
                for metric in metrics:
                    values = [sample[metric] for sample in samples]
                    summary[mode][metric] = {
                        "median": statistics.median(values),
                        "min": min(values),
                        "max": max(values),
                    }
            changes = {}
            for mode in ("static", "planned"):
                changes[mode] = {}
                for metric in metrics:
                    base = summary["baseline"][metric]["median"]
                    candidate = summary[mode][metric]["median"]
                    # Positive values mean improvement for all listed metrics.
                    changes[mode][metric] = 100 * (
                        candidate / base - 1
                        if metric == "output_throughput"
                        else 1 - candidate / base
                    )
            report["comparisons"].append(
                {
                    "workload": workload,
                    "concurrency": concurrency,
                    "measurements": summary,
                    "improvement_pct": changes,
                    "throughput_repeat_spread_pct": {
                        mode: 100
                        * (
                            values["output_throughput"]["max"]
                            - values["output_throughput"]["min"]
                        )
                        / values["output_throughput"]["median"]
                        for mode, values in summary.items()
                    },
                }
            )
    output = args.results / "summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "| Workload | Concurrency | Static TPOT reduction | Planned TPOT reduction | Planned throughput gain |"
    )
    print("| --- | --- | --- | --- | --- |")
    for row in report["comparisons"]:
        change = row["improvement_pct"]
        print(
            f"| {row['workload']} | {row['concurrency']} | {change['static']['median_tpot_ms']:.1f}% | "
            f"{change['planned']['median_tpot_ms']:.1f}% | {change['planned']['output_throughput']:.1f}% |"
        )


if __name__ == "__main__":
    main()
