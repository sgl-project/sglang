# SPDX-License-Identifier: Apache-2.0

"""
Small usage wrapper for benchmarking embedding serving with bench_serving.

By default this script only validates arguments and prints the launch/benchmark
commands. Add --run-benchmark to execute the bench_serving command, and add
--launch-server if this script should start and stop an embedding server around
the benchmark.
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_MODEL_PATH = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_PORT = 30000


@dataclass(frozen=True)
class EmbeddingServingCommands:
    launch_server: List[str]
    benchmark: List[str]


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to 0")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _range_ratio(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _json_object(value: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must be a JSON object")
    return parsed


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Print or run a minimal OpenAI embeddings serving benchmark through "
            "sglang.bench_serving."
        )
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Embedding model path used by launch_server and bench_serving tokenizer.",
    )
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Optional model name exposed by the server and sent in requests.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer path for synthetic dataset sizing. Defaults to --model-path.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host used by bench_serving when --base-url is not set.",
    )
    parser.add_argument(
        "--server-host",
        default="0.0.0.0",
        help="Host passed to launch_server when --launch-server is used.",
    )
    parser.add_argument("--port", type=_positive_int, default=DEFAULT_PORT)
    parser.add_argument(
        "--base-url",
        default=None,
        help="Existing server base URL for bench_serving, for example http://127.0.0.1:30000.",
    )
    parser.add_argument(
        "--dataset-name",
        choices=["random-ids", "random", "sharegpt", "custom"],
        default="random-ids",
        help=(
            "Dataset passed to bench_serving. random-ids is synthetic and avoids "
            "downloading a dataset."
        ),
    )
    parser.add_argument(
        "--dataset-path",
        default="",
        help="Optional dataset path for random/sharegpt/custom datasets.",
    )
    parser.add_argument("--num-prompts", type=_positive_int, default=100)
    parser.add_argument("--random-input-len", type=_positive_int, default=128)
    parser.add_argument(
        "--random-range-ratio",
        type=_range_ratio,
        default=0.0,
        help="Range ratio for random input lengths.",
    )
    parser.add_argument(
        "--request-rate",
        type=_positive_float,
        default=float("inf"),
        help="Requests per second. Use inf to send all requests immediately.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=_positive_int,
        default=None,
        help="Optional cap on concurrent in-flight requests.",
    )
    parser.add_argument(
        "--ready-check-timeout-sec",
        type=_non_negative_int,
        default=60,
        help="bench_serving readiness wait. Use 0 to skip readiness polling.",
    )
    parser.add_argument("--warmup-requests", type=_non_negative_int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        help="Custom HTTP header for bench_serving in Key=Value format. Repeatable.",
    )
    parser.add_argument(
        "--dimensions",
        type=_positive_int,
        default=None,
        help="Optional Matryoshka dimensions value added to the embedding request body.",
    )
    parser.add_argument(
        "--extra-request-body",
        type=_json_object,
        default=None,
        help='Extra JSON object merged into each /v1/embeddings request.',
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass --trust-remote-code to launch_server.",
    )
    parser.add_argument(
        "--launch-server",
        action="store_true",
        help="Start launch_server before running the benchmark.",
    )
    parser.add_argument(
        "--server-start-grace-sec",
        type=_non_negative_float,
        default=2.0,
        help="Seconds to wait after starting launch_server before invoking bench_serving.",
    )
    parser.add_argument(
        "--run-benchmark",
        action="store_true",
        help="Execute the benchmark command. Without this flag the script is a dry run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print commands without running anything.",
    )
    parser.add_argument(
        "server_args",
        nargs=argparse.REMAINDER,
        help="Extra launch_server args after '--', used only with --launch-server.",
    )
    return parser


def _normalized_server_args(server_args: Sequence[str]) -> List[str]:
    if server_args and server_args[0] == "--":
        server_args = server_args[1:]
    if "..." in server_args:
        raise ValueError(
            "literal '...' is not a valid launch_server argument; "
            "replace it with real arguments or omit it"
        )
    return list(server_args)


def _build_extra_request_body(args: argparse.Namespace) -> Optional[str]:
    body = {}
    if args.extra_request_body:
        body.update(args.extra_request_body)
    if args.dimensions is not None:
        existing = body.get("dimensions")
        if existing is not None and existing != args.dimensions:
            raise ValueError(
                "--dimensions conflicts with dimensions in --extra-request-body"
            )
        body["dimensions"] = args.dimensions
    if not body:
        return None
    return json.dumps(body, separators=(",", ":"), sort_keys=True)


def build_commands(args: argparse.Namespace) -> EmbeddingServingCommands:
    launch_cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model_path,
        "--is-embedding",
        "--host",
        args.server_host,
        "--port",
        str(args.port),
    ]
    if args.served_model_name:
        launch_cmd.extend(["--served-model-name", args.served_model_name])
    if args.trust_remote_code:
        launch_cmd.append("--trust-remote-code")
    launch_cmd.extend(_normalized_server_args(args.server_args))

    bench_cmd = [
        sys.executable,
        "-m",
        "sglang.bench_serving",
        "--backend",
        "sglang-embedding",
        "--dataset-name",
        args.dataset_name,
        "--model",
        args.model_path,
        "--num-prompts",
        str(args.num_prompts),
        "--random-input-len",
        str(args.random_input_len),
        "--random-output-len",
        "0",
        "--random-range-ratio",
        str(args.random_range_ratio),
        "--request-rate",
        str(args.request_rate),
        "--ready-check-timeout-sec",
        str(args.ready_check_timeout_sec),
        "--warmup-requests",
        str(args.warmup_requests),
        "--seed",
        str(args.seed),
    ]
    if args.base_url:
        bench_cmd.extend(["--base-url", args.base_url])
    else:
        bench_cmd.extend(["--host", args.host, "--port", str(args.port)])
    if args.served_model_name:
        bench_cmd.extend(["--served-model-name", args.served_model_name])
    if args.tokenizer:
        bench_cmd.extend(["--tokenizer", args.tokenizer])
    if args.dataset_path:
        bench_cmd.extend(["--dataset-path", args.dataset_path])
    if args.max_concurrency is not None:
        bench_cmd.extend(["--max-concurrency", str(args.max_concurrency)])
    if args.output_file:
        bench_cmd.extend(["--output-file", args.output_file])
    if args.disable_tqdm:
        bench_cmd.append("--disable-tqdm")
    if args.header:
        bench_cmd.append("--header")
        bench_cmd.extend(args.header)

    extra_request_body = _build_extra_request_body(args)
    if extra_request_body:
        bench_cmd.extend(["--extra-request-body", extra_request_body])

    return EmbeddingServingCommands(launch_server=launch_cmd, benchmark=bench_cmd)


def _print_commands(commands: EmbeddingServingCommands) -> None:
    print("Launch embedding server:")
    print(shlex.join(commands.launch_server))
    print()
    print("Run smoke benchmark:")
    print(shlex.join(commands.benchmark))
    print()
    print(
        "bench_serving reports request throughput, input token throughput, "
        "concurrency, and end-to-end latency percentiles for embeddings."
    )


def _terminate_process(process: subprocess.Popen) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            if hasattr(os, "killpg"):
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait(timeout=30)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_argument_parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.run_benchmark:
        parser.error("--dry-run cannot be used with --run-benchmark")

    try:
        commands = build_commands(args)
    except ValueError as exc:
        parser.error(str(exc))

    _print_commands(commands)
    if not args.run_benchmark:
        return 0

    server_process = None
    try:
        if args.launch_server:
            server_process = subprocess.Popen(
                commands.launch_server, start_new_session=True
            )
            time.sleep(args.server_start_grace_sec)

        completed = subprocess.run(commands.benchmark, check=False)
        return completed.returncode
    finally:
        if server_process is not None:
            _terminate_process(server_process)


if __name__ == "__main__":
    raise SystemExit(main())
