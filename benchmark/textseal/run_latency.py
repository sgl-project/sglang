#!/usr/bin/env python3
import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
METRICS = (
    "median_tpot_ms",
    "mean_tpot_ms",
    "median_ttft_ms",
    "mean_ttft_ms",
    "output_throughput",
    "request_throughput",
)


def _run_trial(base_url: str, output: Path, watermarked: bool):
    command = [
        sys.executable,
        "-m",
        "sglang.benchmark.serving",
        "--backend",
        "sglang-oai-chat",
        "--base-url",
        base_url,
        "--model",
        MODEL,
        "--dataset-name",
        "random",
        "--random-input-len",
        "128",
        "--random-output-len",
        "256",
        "--random-range-ratio",
        "1.0",
        "--num-prompts",
        "100",
        "--max-concurrency",
        "8",
        "--warmup-requests",
        "16",
        "--temperature",
        "0.8",
        "--top-p",
        "0.95",
        "--extra-request-body",
        json.dumps({"temperature": 0.8, "top_p": 0.95}),
        "--seed",
        "20260901",
        "--output-file",
        str(output),
    ]
    if watermarked:
        command.extend(["--header", "X-SGLang-Watermark=textseal"])
    subprocess.run(command, check=True)
    return json.loads(output.read_text().splitlines()[-1])


def _percent_change(baseline: float, textseal: float) -> float:
    return (textseal - baseline) / baseline * 100


def _summarize_metric(pairs, metric: str):
    baseline = [pair[f"baseline_{metric}"] for pair in pairs]
    textseal = [pair[f"textseal_{metric}"] for pair in pairs]
    changes = [pair[f"{metric}_change_percent"] for pair in pairs]
    return {
        "baseline_mean": statistics.mean(baseline),
        "baseline_median": statistics.median(baseline),
        "baseline_stdev": statistics.stdev(baseline),
        "textseal_mean": statistics.mean(textseal),
        "textseal_median": statistics.median(textseal),
        "textseal_stdev": statistics.stdev(textseal),
        "paired_change_mean_percent": statistics.mean(changes),
        "paired_change_median_percent": statistics.median(changes),
        "paired_change_stdev_percent": statistics.stdev(changes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    for pair_index in range(5):
        results = {}
        order = (False, True) if pair_index % 2 == 0 else (True, False)
        for watermarked in order:
            name = "textseal" if watermarked else "baseline"
            output = args.output_dir / f"pair-{pair_index + 1}-{name}.jsonl"
            results[name] = _run_trial(args.base_url, output, watermarked)
        pair = {"pair": pair_index + 1}
        for metric in METRICS:
            baseline = results["baseline"][metric]
            textseal = results["textseal"][metric]
            pair[f"baseline_{metric}"] = baseline
            pair[f"textseal_{metric}"] = textseal
            pair[f"{metric}_change_percent"] = _percent_change(baseline, textseal)
        pairs.append(pair)

    regressions = [pair["median_tpot_ms_change_percent"] for pair in pairs]
    report = {
        "model": MODEL,
        "pairs": pairs,
        "metric_summaries": {
            metric: _summarize_metric(pairs, metric) for metric in METRICS
        },
        "median_paired_tpot_regression_percent": statistics.median(regressions),
        "mean_paired_tpot_regression_percent": statistics.mean(regressions),
        "stdev_paired_tpot_regression_percent": statistics.stdev(regressions),
        "accepted": statistics.median(regressions) < 5.0,
    }
    report_path = args.output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
