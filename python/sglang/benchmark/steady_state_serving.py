"""Dedicated serving benchmark entry point with steady-state reporting.

Usage example::

    python -m sglang.benchmark.steady_state_serving \
        --steady-state-concurrency-ratio 0.8 \
        --backend sglang --dataset-name random --num-prompts 1000

All arguments other than the two steady-state options are forwarded unchanged to
``sglang.benchmark.serving``.  The regular serving module is not modified and its
normal full-run result remains intact.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from sglang.benchmark import serving
from sglang.benchmark.steady_state import (
    SteadyStateMetrics,
    calculate_steady_state_metrics,
)


def _print_metrics(metrics: SteadyStateMetrics) -> None:
    print("\n{s:{c}^{n}}".format(s=" Steady-State Result ", n=50, c="="))
    print("{:<40} {:<10.2f}".format("Concurrency ratio:", metrics.concurrency_ratio))
    print("{:<40} {:<10.2f}".format("Measurement duration (s):", metrics.duration))
    print("{:<40} {:<10}".format("Minimum concurrency:", metrics.concurrency_threshold))
    print("{:<40} {:<10}".format("Completed requests:", metrics.completed))
    if metrics.input_throughput is None:
        print("{:<40} {:<10}".format("Input token throughput (tok/s):", "N/A"))
    else:
        print("{:<40} {:<10}".format("Input tokens:", metrics.total_input))
        print(
            "{:<40} {:<10.2f}".format(
                "Input token throughput (tok/s):", metrics.input_throughput
            )
        )
    print("{:<40} {:<10.2f}".format("Generated tokens:", metrics.total_output))
    print(
        "{:<40} {:<10.2f}".format(
            "Output token throughput (tok/s):", metrics.output_throughput
        )
    )
    print(
        "{:<40} {:<10.2f}".format(
            "Output throughput, retokenized (tok/s):",
            metrics.output_throughput_retokenized,
        )
    )
    print(
        "{:<40} {:<10.2f}".format("Average concurrency:", metrics.average_concurrency)
    )
    print("{:<40} {:<10}".format("Peak concurrency:", metrics.peak_concurrency))
    print(
        "{:<40} {:<10.2f}".format(
            "Peak output throughput (tok/s):", metrics.peak_output_throughput
        )
    )
    print("=" * 50)


def _append_metrics(path: str, metrics: SteadyStateMetrics) -> None:
    with Path(path).open("a") as file:
        file.write(json.dumps(asdict(metrics)) + "\n")


def _run_with_capture(
    args: argparse.Namespace,
    concurrency_ratio: float,
    output_file: Optional[str],
    run_serving_benchmark: Callable[[argparse.Namespace], Dict[str, Any]],
) -> Tuple[Dict[str, Any], SteadyStateMetrics]:
    captured: Dict[str, Any] = {}
    original_calculate_metrics = serving.calculate_metrics
    calculate_signature = inspect.signature(original_calculate_metrics)

    def capture_calculate_metrics(*call_args, **call_kwargs):
        bound = calculate_signature.bind(*call_args, **call_kwargs)
        input_requests = bound.arguments["input_requests"]
        captured["input_requests"] = (
            None if input_requests is None else list(input_requests)
        )
        captured["outputs"] = list(bound.arguments["outputs"])
        captured["tokenizer"] = bound.arguments["tokenizer"]
        return original_calculate_metrics(*call_args, **call_kwargs)

    serving.calculate_metrics = capture_calculate_metrics
    try:
        benchmark_result = run_serving_benchmark(args)
    finally:
        serving.calculate_metrics = original_calculate_metrics

    if "outputs" not in captured:
        raise RuntimeError(
            "serving benchmark finished without producing request results"
        )

    steady_state_metrics = calculate_steady_state_metrics(
        outputs=captured["outputs"],
        tokenizer=captured["tokenizer"],
        concurrency_ratio=concurrency_ratio,
        input_requests=captured["input_requests"],
    )
    _print_metrics(steady_state_metrics)
    if output_file:
        _append_metrics(output_file, steady_state_metrics)

    return benchmark_result, steady_state_metrics


def run_steady_state_benchmark(
    args: argparse.Namespace,
    concurrency_ratio: float,
    output_file: Optional[str] = None,
) -> Tuple[Dict[str, Any], SteadyStateMetrics]:
    """Run the normal serving benchmark and add an isolated steady-state report."""
    return _run_with_capture(
        args=args,
        concurrency_ratio=concurrency_ratio,
        output_file=output_file,
        run_serving_benchmark=serving.run_benchmark,
    )


def _custom_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add isolated steady-state metrics to a serving benchmark run.",
        epilog="All other options are forwarded to sglang.benchmark.serving.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--steady-state-concurrency-ratio",
        type=float,
        required=True,
        help=(
            "Measure the longest continuous interval whose concurrency is at "
            "least this fraction of peak concurrency. Must be in (0, 1]."
        ),
    )
    parser.add_argument(
        "--steady-state-output-file",
        type=str,
        default=None,
        help="Append the standalone steady-state result to this JSONL file.",
    )
    return parser


def cli_main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    custom_parser = _custom_parser()
    custom_args, serving_argv = custom_parser.parse_known_args(argv)

    # Validate before server setup so malformed values fail fast.
    if not 0 < custom_args.steady_state_concurrency_ratio <= 1:
        custom_parser.error("--steady-state-concurrency-ratio must be in (0, 1]")

    original_run_benchmark = serving.run_benchmark
    original_argv = sys.argv

    def run_and_report(args: argparse.Namespace):
        result, _ = _run_with_capture(
            args=args,
            concurrency_ratio=custom_args.steady_state_concurrency_ratio,
            output_file=custom_args.steady_state_output_file,
            run_serving_benchmark=original_run_benchmark,
        )
        return result

    serving.run_benchmark = run_and_report
    sys.argv = [original_argv[0], *serving_argv]
    try:
        serving.cli_main()
    finally:
        sys.argv = original_argv
        serving.run_benchmark = original_run_benchmark


if __name__ == "__main__":
    cli_main()
