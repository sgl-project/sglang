#!/usr/bin/env python3
"""Compare performance with SGLANG_DISABLE_FUSIONS on and off.

One-batch usage:
    scripts/playground/compare_fusions.py --model MODEL --bs BATCH_SIZES \
        --eval one-batch --isl INPUT_LEN --osl OUTPUT_LEN \
        [additional sglang.benchmark.one_batch arguments]

GSM8K usage:
    scripts/playground/compare_fusions.py --model MODEL --eval gsm8k \
        --concurrency CONCURRENCY [additional sglang serve arguments]

One-batch example:
    scripts/playground/compare_fusions.py \
        --model RedHatAI/Qwen2-7B-Instruct-FP8 --eval one-batch \
        --bs 1,2,4,8,16 --isl 1024 --osl 1024

GSM8K example:
    scripts/playground/compare_fusions.py \
        --model RedHatAI/Meta-Llama-3.1-8B-FP8 --eval gsm8k \
        --concurrency 16

Set FUSION_BENCH_RESULT_DIR to retain results in a specific directory.
Set PYTHON to select the interpreter used to launch each benchmark. By default,
the script uses its own Python interpreter.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

_GSM8K_HOST = "0.0.0.0"
_GSM8K_HEALTH_HOST = "127.0.0.1"
_GSM8K_PORT = 30000
_SERVER_STARTUP_TIMEOUT_SECONDS = 900


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

GSM8K_METRICS = (
    Metric("Score", "score", "", 1.0, True),
    Metric("Duration", "duration", "s", 1.0, False),
    Metric("Output throughput", "output_throughput", "token/s", 1.0, True),
)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Compare performance with fusions disabled and enabled.",
        usage=(
            "%(prog)s --model MODEL --eval {one-batch,gsm8k} "
            "[evaluation options] [forwarded arguments ...]"
        ),
        epilog=(
            "Unknown arguments are forwarded to sglang.benchmark.one_batch for "
            "one-batch or sglang serve for gsm8k.\n"
            "One-batch: %(prog)s --model MODEL --eval one-batch --bs 1,4,8,16 "
            "--isl 1024 --osl 1024\n"
            "GSM8K: %(prog)s --model MODEL --eval gsm8k --concurrency 16"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model ID or local model path",
    )
    parser.add_argument(
        "--eval",
        required=True,
        choices=("one-batch", "gsm8k"),
        help="evaluation workflow to run",
    )
    parser.add_argument(
        "--bs",
        dest="batch_sizes",
        type=parse_batch_sizes,
        help="one batch size or a comma-separated sweep, such as 1,2,4,8,16",
    )
    parser.add_argument(
        "--isl",
        dest="input_len",
        type=positive_int,
        help="one-batch input sequence length",
    )
    parser.add_argument(
        "--osl",
        dest="output_len",
        type=positive_int,
        help="one-batch output sequence length",
    )
    parser.add_argument(
        "--concurrency",
        type=positive_int,
        help="GSM8K server request limit and evaluator thread count",
    )
    args, extra_args = parser.parse_known_args()

    one_batch_args = (args.batch_sizes, args.input_len, args.output_len)
    if args.eval == "one-batch":
        if any(value is None for value in one_batch_args):
            parser.error("--eval one-batch requires --bs, --isl, and --osl")
        if args.concurrency is not None:
            parser.error("--concurrency is only valid with --eval gsm8k")
    else:
        if args.concurrency is None:
            parser.error("--eval gsm8k requires --concurrency")
        if any(value is not None for value in one_batch_args):
            parser.error("--bs, --isl, and --osl are only valid with --eval one-batch")

    return args, extra_args


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


def fusion_environment(enable_fusions: bool, *, local_server: bool = False) -> dict:
    environment = os.environ.copy()
    environment.pop("SGLANG_ENABLE_FUSIONS", None)
    environment["SGLANG_DISABLE_FUSIONS"] = "0" if enable_fusions else "1"
    if local_server:
        no_proxy_hosts = ("0.0.0.0", "127.0.0.1", "localhost")
        for variable in ("NO_PROXY", "no_proxy"):
            configured = [
                host for host in environment.get(variable, "").split(",") if host
            ]
            environment[variable] = ",".join(
                dict.fromkeys((*configured, *no_proxy_hosts))
            )
    return environment


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
    environment = fusion_environment(enable_fusions)
    batch_sizes = ",".join(str(batch_size) for batch_size in args.batch_sizes)
    run_summary = (
        f"Running {label}: model={args.model}, "
        f"SGLANG_DISABLE_FUSIONS={environment['SGLANG_DISABLE_FUSIONS']}, "
        f"eval=one-batch, BS={batch_sizes}, ISL={args.input_len}, "
        f"OSL={args.output_len}"
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


def stream_output(stream, log_file) -> None:
    try:
        for line in stream:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
    finally:
        stream.close()


def stop_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def ensure_gsm8k_port_available() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(1)
        if connection.connect_ex((_GSM8K_HEALTH_HOST, _GSM8K_PORT)) == 0:
            raise RuntimeError(
                f"Port {_GSM8K_PORT} is already in use; stop the existing server "
                "before running the GSM8K comparison"
            )


def wait_for_gsm8k_server(process: subprocess.Popen) -> None:
    url = f"http://{_GSM8K_HEALTH_HOST}:{_GSM8K_PORT}/health_generate"
    deadline = time.monotonic() + _SERVER_STARTUP_TIMEOUT_SECONDS
    session = requests.Session()
    session.trust_env = False
    try:
        while time.monotonic() < deadline:
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"SGLang server exited during startup with code {return_code}"
                )
            try:
                response = session.get(url, timeout=5)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
    finally:
        session.close()
    raise TimeoutError(
        f"SGLang server did not become ready within "
        f"{_SERVER_STARTUP_TIMEOUT_SECONDS} seconds"
    )


def run_logged_command(command: list[str], environment: dict, log_path: Path) -> str:
    output: list[str] = []
    with log_path.open("w") as log_file:
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
            log_file.write(line)
            output.append(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(
            return_code, command, output="".join(output)
        )
    return "".join(output)


def parse_gsm8k_metrics(output: str) -> dict[str, float]:
    number = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    patterns = {
        "duration": rf"Total latency:\s*{number}\s*s",
        "score": rf"Score:\s*{number}",
        "output_throughput": rf"Output throughput:\s*{number}\s*token/s",
    }
    metrics = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match is None:
            raise RuntimeError(f"GSM8K output is missing the {key} metric")
        metrics[key] = float(match.group(1))
    return metrics


def run_gsm8k_benchmark(
    *,
    label: str,
    enable_fusions: bool,
    args: argparse.Namespace,
    server_args: list[str],
    result_dir: Path,
) -> dict[str, float]:
    server_log_path = result_dir / f"{label}_server.log"
    eval_log_path = result_dir / f"{label}_gsm8k.log"
    environment = fusion_environment(enable_fusions, local_server=True)
    server_command = [
        "sglang",
        "serve",
        *server_args,
        "--model-path",
        args.model,
        "--host",
        _GSM8K_HOST,
        "--port",
        str(_GSM8K_PORT),
        "--max-running-requests",
        str(args.concurrency),
    ]
    python = os.environ.get("PYTHON", sys.executable)
    eval_command = [
        python,
        "-m",
        "sglang.test.run_eval",
        "--host",
        _GSM8K_HOST,
        "--port",
        str(_GSM8K_PORT),
        "--eval-name",
        "gsm8k",
        "--api",
        "generate",
        "--num-threads",
        str(args.concurrency),
    ]
    print(
        f"\nRunning {label}: model={args.model}, "
        f"SGLANG_DISABLE_FUSIONS={environment['SGLANG_DISABLE_FUSIONS']}, "
        f"eval=gsm8k, concurrency={args.concurrency}"
    )
    print(f"Server log: {server_log_path}")
    print(f"Evaluation log: {eval_log_path}")

    ensure_gsm8k_port_available()
    with server_log_path.open("w") as server_log:
        server = subprocess.Popen(
            server_command,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert server.stdout is not None
        output_thread = threading.Thread(
            target=stream_output,
            args=(server.stdout, server_log),
            daemon=True,
        )
        output_thread.start()
        try:
            wait_for_gsm8k_server(server)
            output = run_logged_command(eval_command, environment, eval_log_path)
        finally:
            stop_process_group(server)
            output_thread.join(timeout=30)

    return parse_gsm8k_metrics(output)


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
    if fusion_off == 0.0:
        if fusion_on == 0.0:
            return 0.0
        improved = (fusion_on > fusion_off) == metric.higher_is_better
        return float("inf") if improved else float("-inf")
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


def format_gsm8k_comparison(
    fusion_off: dict[str, float],
    fusion_on: dict[str, float],
    *,
    model: str,
    concurrency: int,
) -> None:
    rows: list[tuple[str, str, str, str, float]] = []
    for metric in GSM8K_METRICS:
        off = fusion_off[metric.key]
        on = fusion_on[metric.key]
        gain = metric_gain(metric, off, on)
        unit = f" {metric.unit}" if metric.unit else ""
        rows.append(
            (
                metric.label,
                f"{off * metric.scale:,.3f}{unit}",
                f"{on * metric.scale:,.3f}{unit}",
                f"{gain:+.2f}%",
                gain,
            )
        )

    headers = ("Metric", "Fusion off", "Fusion on", "Gain")
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    print("\nGSM8K fusion comparison")
    print(f"Model: {model}")
    print(f"Concurrency: {concurrency}\n")
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


def run_one_batch_comparison(args: argparse.Namespace, extra_args: list[str]) -> None:
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


def run_gsm8k_comparison(args: argparse.Namespace, server_args: list[str]) -> None:
    result_dir = result_directory()
    fusion_on = run_gsm8k_benchmark(
        label="fusion_on",
        enable_fusions=True,
        args=args,
        server_args=server_args,
        result_dir=result_dir,
    )
    fusion_off = run_gsm8k_benchmark(
        label="fusion_off",
        enable_fusions=False,
        args=args,
        server_args=server_args,
        result_dir=result_dir,
    )
    format_gsm8k_comparison(
        fusion_off,
        fusion_on,
        model=args.model,
        concurrency=args.concurrency,
    )
    print(f"\nLogs: {result_dir}")


def main() -> None:
    args, extra_args = parse_args()
    if args.eval == "one-batch":
        run_one_batch_comparison(args, extra_args)
    else:
        run_gsm8k_comparison(args, extra_args)


if __name__ == "__main__":
    main()
