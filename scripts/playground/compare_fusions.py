#!/usr/bin/env python3
"""Compare one-batch performance with SGLANG_DISABLE_FUSIONS on and off.

Usage:
    scripts/playground/compare_fusions.py MODEL BATCH_SIZES INPUT_LEN OUTPUT_LEN \
        [additional sglang.benchmark.one_batch arguments]

Example:
    scripts/playground/compare_fusions.py \
        RedHatAI/Qwen2-7B-Instruct-FP8 1,2,4,8,16 1024 1024

Set FUSION_BENCH_RESULT_DIR to retain results in a specific directory.
Set PYTHON to select the interpreter used to launch each benchmark. By default,
the script uses its own Python interpreter.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Metric:
    label: str
    key: str
    unit: str
    scale: float
    higher_is_better: bool


METRICS = (
    Metric("Prefill latency", "prefill_latency", "ms", 1_000.0, False),
    Metric("Prefill throughput", "prefill_throughput", "token/s", 1.0, True),
    Metric("Median decode latency", "median_decode_latency", "ms", 1_000.0, False),
    Metric(
        "Median decode throughput",
        "median_decode_throughput",
        "token/s",
        1.0,
        True,
    ),
    Metric("Total latency", "total_latency", "s", 1.0, False),
    Metric("Overall throughput", "overall_throughput", "token/s", 1.0, True),
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Compare one-batch performance with fusions disabled and enabled.",
        usage=("%(prog)s MODEL BATCH_SIZES INPUT_LEN OUTPUT_LEN [ONE_BATCH_ARGS ...]"),
        epilog=(
            "Additional arguments are forwarded to sglang.benchmark.one_batch.\n"
            "Example: %(prog)s RedHatAI/Qwen2-7B-Instruct-FP8 "
            "1,2,4,8,16 1024 1024"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("model", help="Hugging Face model ID or local model path")
    parser.add_argument(
        "batch_sizes",
        type=parse_batch_sizes,
        help="one batch size or a comma-separated sweep, such as 1,2,4,8,16",
    )
    parser.add_argument("input_len", type=positive_int)
    parser.add_argument("output_len", type=positive_int)
    return parser.parse_known_args()


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def parse_batch_sizes(value: str) -> tuple[int, ...]:
    try:
        batch_sizes = tuple(positive_int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated positive integers, got {value}"
        ) from exc
    if not batch_sizes or any(not part.strip() for part in value.split(",")):
        raise argparse.ArgumentTypeError(
            f"expected comma-separated positive integers, got {value}"
        )
    if len(set(batch_sizes)) != len(batch_sizes):
        raise argparse.ArgumentTypeError(f"batch sizes must be unique, got {value}")
    return batch_sizes


def result_directory() -> Path:
    configured = os.environ.get("FUSION_BENCH_RESULT_DIR")
    if configured:
        path = Path(configured).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
    return Path(tempfile.mkdtemp(prefix="sglang-fusion-bench."))


def run_benchmark(
    *,
    label: str,
    enable_fusions: bool,
    args: argparse.Namespace,
    extra_args: list[str],
    result_dir: Path,
) -> Path:
    result_file = result_dir / f"{label}.jsonl"
    log_file = result_dir / f"{label}.log"
    result_file.write_text("")

    python = os.environ.get("PYTHON", sys.executable)
    command = [
        python,
        "-m",
        "sglang.benchmark.one_batch",
        "--model",
        args.model,
        *extra_args,
        "--batch-size",
        *(str(batch_size) for batch_size in args.batch_sizes),
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        # Prefill CUDA graph "batch sizes" are aggregate-token buckets. Keep
        # startup focused on the one input length requested by this benchmark.
        "--cuda-graph-bs-prefill",
        str(args.input_len),
        "--run-name",
        label,
        "--result-filename",
        str(result_file),
    ]
    environment = os.environ.copy()
    environment.pop("SGLANG_ENABLE_FUSIONS", None)
    environment["SGLANG_DISABLE_FUSIONS"] = "0" if enable_fusions else "1"
    batch_sizes = ",".join(str(batch_size) for batch_size in args.batch_sizes)
    run_summary = (
        f"Running {label}: model={args.model}, "
        f"SGLANG_DISABLE_FUSIONS={environment['SGLANG_DISABLE_FUSIONS']}, "
        f"BS={batch_sizes}, ISL={args.input_len}, OSL={args.output_len}"
    )

    print(f"\n{run_summary}")
    print(f"Results: {result_file}")

    with log_file.open("w") as log:
        log.write(f"{run_summary}\nResults: {result_file}\n")
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()

    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return result_file


def load_results(
    path: Path, expected_batch_sizes: tuple[int, ...]
) -> dict[int, dict[str, Any]]:
    with path.open() as file:
        rows = [json.loads(line) for line in file if line.strip()]
    if not rows:
        raise RuntimeError(f"No benchmark results found in {path}")
    results = {int(row["batch_size"]): row for row in rows}
    missing = set(expected_batch_sizes) - results.keys()
    if missing:
        raise RuntimeError(f"Missing batch sizes in {path}: {sorted(missing)}")
    return results


def metric_gain(metric: Metric, fusion_off: float, fusion_on: float) -> float:
    relative_change = fusion_on / fusion_off - 1.0
    gain = relative_change * (100.0 if metric.higher_is_better else -100.0)
    return 0.0 if abs(gain) < 0.005 else gain


def color_gain(text: str, gain: float, width: int) -> str:
    cell = f"{text:>{width}}"
    if not sys.stdout.isatty() or "NO_COLOR" in os.environ or gain == 0.0:
        return cell
    color = "\033[32m" if gain > 0.0 else "\033[31m"
    return f"{color}{cell}\033[0m"


def format_comparison(
    fusion_off: dict[str, Any],
    fusion_on: dict[str, Any],
    *,
    model: str,
    batch_size: int,
    input_len: int,
    output_len: int,
) -> None:
    rows: list[tuple[str, str, str, str, float]] = []
    for metric in METRICS:
        off = fusion_off.get(metric.key)
        on = fusion_on.get(metric.key)
        if off is None or on is None:
            continue
        gain = metric_gain(metric, off, on)
        rows.append(
            (
                metric.label,
                f"{off * metric.scale:,.2f} {metric.unit}",
                f"{on * metric.scale:,.2f} {metric.unit}",
                f"{gain:+.2f}%",
                gain,
            )
        )

    headers = ("Metric", "Fusion off", "Fusion on", "Gain")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("\nFusion comparison")
    print(f"Model: {model}")
    print(f"BS: {batch_size}  ISL: {input_len}  OSL: {output_len}\n")
    print(
        f"{headers[0]:<{widths[0]}}  "
        f"{headers[1]:>{widths[1]}}  "
        f"{headers[2]:>{widths[2]}}  "
        f"{headers[3]:>{widths[3]}}"
    )
    print("-" * (sum(widths) + 6))
    for label, off, on, gain_text, gain in rows:
        gain_cell = color_gain(gain_text, gain, widths[3])
        print(
            f"{label:<{widths[0]}}  {off:>{widths[1]}}  {on:>{widths[2]}}  {gain_cell}"
        )


def format_gain_summary(
    fusion_off: dict[int, dict[str, Any]],
    fusion_on: dict[int, dict[str, Any]],
    batch_sizes: tuple[int, ...],
    *,
    model: str,
    input_len: int,
    output_len: int,
) -> None:
    headers = ("Metric", *(f"BS{batch_size}" for batch_size in batch_sizes), "Mean")
    rows: list[tuple[str, list[float]]] = []
    for metric in METRICS:
        if any(
            metric.key not in fusion_off[batch_size]
            or metric.key not in fusion_on[batch_size]
            for batch_size in batch_sizes
        ):
            continue
        gains = [
            metric_gain(
                metric,
                fusion_off[batch_size][metric.key],
                fusion_on[batch_size][metric.key],
            )
            for batch_size in batch_sizes
        ]
        rows.append((metric.label, [*gains, sum(gains) / len(gains)]))

    plain_rows = [
        (label, *(f"{gain:+.2f}%" for gain in gains)) for label, gains in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in plain_rows))
        for index in range(len(headers))
    ]

    batch_size_list = ",".join(str(batch_size) for batch_size in batch_sizes)
    print("\nGain summary by batch size (positive gain means fusion is faster)")
    print(f"Model: {model}")
    print(f"BS: {batch_size_list}  ISL: {input_len}  OSL: {output_len}\n")
    print(
        f"{headers[0]:<{widths[0]}}  "
        + "  ".join(
            f"{header:>{widths[index]}}"
            for index, header in enumerate(headers[1:], start=1)
        )
    )
    print("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for (label, gains), plain_row in zip(rows, plain_rows):
        gain_cells = [
            color_gain(plain_row[index], gain, widths[index])
            for index, gain in enumerate(gains, start=1)
        ]
        print(f"{label:<{widths[0]}}  " + "  ".join(gain_cells))


def main() -> None:
    args, extra_args = parse_args()
    result_dir = result_directory()
    fusion_off_path = run_benchmark(
        label="fusion_off",
        enable_fusions=False,
        args=args,
        extra_args=extra_args,
        result_dir=result_dir,
    )
    fusion_on_path = run_benchmark(
        label="fusion_on",
        enable_fusions=True,
        args=args,
        extra_args=extra_args,
        result_dir=result_dir,
    )
    fusion_off = load_results(fusion_off_path, args.batch_sizes)
    fusion_on = load_results(fusion_on_path, args.batch_sizes)
    for batch_size in args.batch_sizes:
        format_comparison(
            fusion_off[batch_size],
            fusion_on[batch_size],
            model=args.model,
            batch_size=batch_size,
            input_len=args.input_len,
            output_len=args.output_len,
        )
    if len(args.batch_sizes) > 1:
        format_gain_summary(
            fusion_off,
            fusion_on,
            args.batch_sizes,
            model=args.model,
            input_len=args.input_len,
            output_len=args.output_len,
        )
    print(f"\nLogs and JSONL results: {result_dir}")


if __name__ == "__main__":
    main()
