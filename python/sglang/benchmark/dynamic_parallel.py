"""Benchmark TP/CP/DCP execution modes on an identical request grid.

The driver launches one SGLang server per mode, runs deterministic batched
``/generate`` requests, and writes JSON records that can be compared across
deployments.  It deliberately uses token IDs instead of text so input lengths
and shared-prefix ratios are exact.

Example:

.. code-block:: bash

    python -m sglang.benchmark.dynamic_parallel \
      --model-path deepseek-ai/DeepSeek-V3.1 \
      --modes tp,prefill_cp,dcp,dynamic \
      --batch-sizes 1,8,32 \
      --input-lengths 1024,4096,16384,32768 \
      --prefix-hit-ratios 0,0.5,0.9 \
      --output-length 32 \
      --result-file dynamic_parallel.jsonl

This is an experiment harness, not a CI benchmark.  Large DeepSeek checkpoints
need a machine with enough accelerators for the requested TP size.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import requests
from typing_extensions import Self


class DeploymentMode(str, Enum):
    TP = "tp"
    PREFILL_CP = "prefill_cp"
    DCP = "dcp"
    DYNAMIC = "dynamic"


@dataclass(frozen=True)
class GridCase:
    batch_size: int
    input_length: int
    output_length: int
    prefix_hit_ratio: float
    repeat: int

    @property
    def key(self) -> str:
        return (
            f"bs={self.batch_size}/in={self.input_length}/out={self.output_length}/"
            f"prefix={self.prefix_hit_ratio:.6f}/repeat={self.repeat}"
        )


@dataclass
class RunRecord:
    mode: str
    case: dict[str, Any]
    case_key: str
    wall_latency_s: float
    cached_tokens: list[int]
    prompt_tokens: list[int]
    completion_tokens: list[int]
    output_ids: list[list[int]]
    output_logprobs: list[list[float]]
    mode_metrics_before: dict[str, float]
    mode_metrics_after: dict[str, float]
    server_info: dict[str, Any]
    parity: dict[str, Any] | None = None


def _csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def _csv_floats(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item < 0.0 or item >= 1.0 for item in values):
        raise argparse.ArgumentTypeError(
            "expected comma-separated ratios in the range [0, 1)"
        )
    return values


def _csv_modes(value: str) -> tuple[DeploymentMode, ...]:
    try:
        modes = tuple(
            DeploymentMode(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in DeploymentMode)
        raise argparse.ArgumentTypeError(
            f"unknown mode; expected one of: {valid}"
        ) from exc
    if not modes:
        raise argparse.ArgumentTypeError("at least one mode is required")
    if len(set(modes)) != len(modes):
        raise argparse.ArgumentTypeError("modes must not contain duplicates")
    return modes


def build_mode_server_args(
    mode: DeploymentMode,
    *,
    cp_size: int,
    dcp_size: int,
    dynamic_include_dcp: bool,
    dynamic_striped_min_context: int | None = None,
) -> list[str]:
    """Return only the mode-specific server arguments."""

    cp_args = [
        "--enable-prefill-cp",
        "--cp-strategy",
        "zigzag",
        "--attention-context-parallel-size",
        str(cp_size),
        "--enable-cp-decode-attn-tp",
    ]
    if mode is DeploymentMode.TP:
        return []
    if mode is DeploymentMode.PREFILL_CP:
        return cp_args
    if mode is DeploymentMode.DCP:
        return ["--dcp-size", str(dcp_size)]
    if mode is DeploymentMode.DYNAMIC:
        args = [*cp_args, "--enable-dynamic-attn-parallel"]
        if dynamic_include_dcp:
            args += [
                "--dcp-size",
                str(dcp_size),
                "--dynamic-attn-parallel-enable-dcp",
                "--page-size",
                "1",
                "--disable-overlap-schedule",
                "--no-dcp-replicate-q-proj",
            ]
            if dynamic_striped_min_context is not None:
                args += [
                    "--dynamic-attn-parallel-striped-min-context",
                    str(dynamic_striped_min_context),
                    "--disable-radix-cache",
                ]
        return args
    raise AssertionError(f"unhandled deployment mode: {mode}")


def build_inputs(
    *,
    batch_size: int,
    input_length: int,
    prefix_hit_ratio: float,
    seed: int,
    token_id_low: int,
    token_id_high: int,
) -> tuple[list[int], list[list[int]]]:
    """Build exact-length inputs with a deterministic shared prefix."""

    if batch_size <= 0 or input_length <= 0:
        raise ValueError("batch_size and input_length must be positive")
    if not 0.0 <= prefix_hit_ratio < 1.0:
        raise ValueError("prefix_hit_ratio must be in [0, 1)")
    if token_id_low < 0 or token_id_high <= token_id_low:
        raise ValueError("invalid token ID range")

    prefix_length = math.floor(input_length * prefix_hit_ratio)
    rng = np.random.default_rng(seed)
    shared_prefix = rng.integers(
        token_id_low, token_id_high, size=prefix_length, dtype=np.int64
    ).tolist()
    suffix_length = input_length - prefix_length
    inputs = []
    for _ in range(batch_size):
        suffix = rng.integers(
            token_id_low, token_id_high, size=suffix_length, dtype=np.int64
        ).tolist()
        inputs.append([*shared_prefix, *suffix])
    return shared_prefix, inputs


def _normalize_batch_response(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return payload
    raise RuntimeError(f"unexpected /generate response shape: {type(payload).__name__}")


def _extract_logprobs(meta_info: dict[str, Any]) -> list[float]:
    values = meta_info.get("output_token_logprobs") or []
    result = []
    for value in values:
        if isinstance(value, (list, tuple)):
            result.append(float(value[0]))
        elif isinstance(value, dict):
            result.append(float(value.get("logprob", float("nan"))))
        else:
            result.append(float(value))
    return result


def _extract_cache_tokens(meta_info: dict[str, Any]) -> int:
    details = meta_info.get("cached_tokens_details") or {}
    if details:
        return int(
            sum(int(details.get(tier) or 0) for tier in ("device", "host", "storage"))
        )
    return int(meta_info.get("cached_tokens") or 0)


def _parse_prometheus_metrics(text: str) -> dict[str, float]:
    """Extract only dynamic-parallel metrics from Prometheus text."""

    result: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, separator, raw_value = line.rpartition(" ")
        if not separator or "dynamic_attn_parallel" not in name:
            continue
        try:
            result[name] = float(raw_value)
        except ValueError:
            continue
    return result


class ServerProcess:
    def __init__(
        self,
        *,
        command: Sequence[str],
        base_url: str,
        log_file: Path,
        launch_timeout_s: int,
        env: dict[str, str],
    ):
        self.command = list(command)
        self.base_url = base_url
        self.log_file = log_file
        self.launch_timeout_s = launch_timeout_s
        self.env = env
        self.process: subprocess.Popen | None = None
        self._log_handle = None

    def __enter__(self) -> Self:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = self.log_file.open("w")
        self.process = subprocess.Popen(
            self.command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=self.env,
        )
        deadline = time.monotonic() + self.launch_timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"server exited with code {self.process.returncode}; "
                    f"see {self.log_file}"
                )
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    return self
            except requests.RequestException:
                pass
            time.sleep(2)
        raise TimeoutError(
            f"server was not healthy within {self.launch_timeout_s}s; "
            f"see {self.log_file}"
        )

    def __exit__(self, exc_type, exc, traceback):
        if self.process is not None:
            from sglang.srt.utils import kill_process_tree

            kill_process_tree(self.process.pid)
            self.process = None
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def flush_cache(self):
        response = requests.post(
            f"{self.base_url}/flush_cache",
            params={"timeout": 60},
            timeout=90,
        )
        response.raise_for_status()

    def server_info(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/server_info", timeout=30)
        response.raise_for_status()
        info = response.json()
        keep = (
            "max_total_num_tokens",
            "max_req_len",
            "tp_size",
            "dp_size",
            "dcp_size",
            "attn_cp_size",
        )
        return {key: info[key] for key in keep if key in info}

    def mode_metrics(self) -> dict[str, float]:
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=30)
            response.raise_for_status()
        except requests.RequestException:
            return {}
        return _parse_prometheus_metrics(response.text)

    def generate(
        self,
        *,
        input_ids: list[int] | list[list[int]],
        output_length: int,
        return_logprob: bool,
        timeout_s: int,
    ) -> tuple[float, list[dict[str, Any]]]:
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_length,
                "ignore_eos": True,
            },
            "return_logprob": return_logprob,
            "top_logprobs_num": 0,
            "logprob_start_len": -1,
            "return_prompt_token_ids": False,
        }
        start = time.perf_counter()
        response = requests.post(
            f"{self.base_url}/generate", json=payload, timeout=timeout_s
        )
        latency = time.perf_counter() - start
        response.raise_for_status()
        return latency, _normalize_batch_response(response.json())


def _build_server_command(args, mode: DeploymentMode) -> list[str]:
    command = [
        sys.executable,
        "-m",
        args.launch_module,
        "--model-path",
        args.model_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tp-size",
        str(args.tp_size),
        "--attention-backend",
        args.attention_backend,
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--random-seed",
        str(args.seed),
        "--enable-cache-report",
        "--enable-metrics",
        "--trust-remote-code",
    ]
    command += build_mode_server_args(
        mode,
        cp_size=args.cp_size,
        dcp_size=args.dcp_size,
        dynamic_include_dcp=args.dynamic_include_dcp,
        dynamic_striped_min_context=args.dynamic_striped_min_context,
    )
    command += shlex.split(args.extra_server_args)
    mode_args = args.mode_server_args.get(mode.value, "")
    command += shlex.split(mode_args)
    return command


def _run_case(
    server: ServerProcess,
    *,
    mode: DeploymentMode,
    case: GridCase,
    args,
) -> RunRecord:
    shared_prefix, inputs = build_inputs(
        batch_size=case.batch_size,
        input_length=case.input_length,
        prefix_hit_ratio=case.prefix_hit_ratio,
        seed=args.seed + case.repeat,
        token_id_low=args.token_id_low,
        token_id_high=args.token_id_high,
    )
    server.flush_cache()
    if shared_prefix:
        server.generate(
            input_ids=shared_prefix,
            output_length=1,
            return_logprob=False,
            timeout_s=args.request_timeout_s,
        )

    metrics_before = server.mode_metrics()
    latency, outputs = server.generate(
        input_ids=inputs,
        output_length=case.output_length,
        return_logprob=True,
        timeout_s=args.request_timeout_s,
    )
    metrics_after = server.mode_metrics()

    output_ids: list[list[int]] = []
    output_logprobs: list[list[float]] = []
    cached_tokens: list[int] = []
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    for output in outputs:
        meta = output.get("meta_info") or {}
        output_ids.append([int(token_id) for token_id in output.get("output_ids", [])])
        output_logprobs.append(_extract_logprobs(meta))
        cached_tokens.append(_extract_cache_tokens(meta))
        prompt_tokens.append(int(meta.get("prompt_tokens") or 0))
        completion_tokens.append(int(meta.get("completion_tokens") or 0))

    return RunRecord(
        mode=mode.value,
        case=asdict(case),
        case_key=case.key,
        wall_latency_s=latency,
        cached_tokens=cached_tokens,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        output_ids=output_ids,
        output_logprobs=output_logprobs,
        mode_metrics_before=metrics_before,
        mode_metrics_after=metrics_after,
        server_info=server.server_info(),
    )


def compare_with_tp(
    baseline: RunRecord, candidate: RunRecord, *, logprob_tolerance: float
) -> dict[str, Any]:
    if baseline.case_key != candidate.case_key:
        raise ValueError("cannot compare records from different grid cases")
    exact_output_match = baseline.output_ids == candidate.output_ids
    max_logprob_delta = 0.0
    comparable_values = 0
    shape_match = len(baseline.output_logprobs) == len(candidate.output_logprobs)
    if shape_match:
        for baseline_row, candidate_row in zip(
            baseline.output_logprobs, candidate.output_logprobs
        ):
            if len(baseline_row) != len(candidate_row):
                shape_match = False
                continue
            for baseline_value, candidate_value in zip(baseline_row, candidate_row):
                if math.isnan(baseline_value) or math.isnan(candidate_value):
                    continue
                comparable_values += 1
                max_logprob_delta = max(
                    max_logprob_delta, abs(baseline_value - candidate_value)
                )
    return {
        "baseline_mode": DeploymentMode.TP.value,
        "exact_output_match": exact_output_match,
        "logprob_shape_match": shape_match,
        "comparable_logprobs": comparable_values,
        "max_logprob_delta": max_logprob_delta,
        "logprob_tolerance": logprob_tolerance,
        "passed": (
            exact_output_match
            and shape_match
            and comparable_values > 0
            and max_logprob_delta <= logprob_tolerance
        ),
    }


def _iter_cases(args) -> Iterable[GridCase]:
    for repeat in range(args.repeats):
        for batch_size in args.batch_sizes:
            for input_length in args.input_lengths:
                for prefix_hit_ratio in args.prefix_hit_ratios:
                    yield GridCase(
                        batch_size=batch_size,
                        input_length=input_length,
                        output_length=args.output_length,
                        prefix_hit_ratio=prefix_hit_ratio,
                        repeat=repeat,
                    )


def run(args) -> int:
    result_path = Path(args.result_file)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    baseline_records: dict[str, RunRecord] = {}
    failed_parity = False

    for mode in args.modes:
        command = _build_server_command(args, mode)
        print(f"[{mode.value}] launching: {shlex.join(command)}", flush=True)
        env = os.environ.copy()
        env.update(args.server_env)
        with ServerProcess(
            command=command,
            base_url=f"http://127.0.0.1:{args.port}",
            log_file=log_dir / f"{mode.value}.log",
            launch_timeout_s=args.launch_timeout_s,
            env=env,
        ) as server:
            for case in _iter_cases(args):
                record = _run_case(server, mode=mode, case=case, args=args)
                if mode is DeploymentMode.TP:
                    baseline_records[case.key] = record
                elif case.key in baseline_records:
                    record.parity = compare_with_tp(
                        baseline_records[case.key],
                        record,
                        logprob_tolerance=args.logprob_tolerance,
                    )
                    failed_parity = failed_parity or not record.parity["passed"]
                with result_path.open("a") as output_file:
                    output_file.write(json.dumps(asdict(record), sort_keys=True) + "\n")
                print(
                    f"[{mode.value}] {case.key}: {record.wall_latency_s:.6f}s"
                    + (
                        f", parity={record.parity['passed']}"
                        if record.parity is not None
                        else ""
                    ),
                    flush=True,
                )

    if args.strict_parity and failed_parity:
        return 2
    return 0


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--modes",
        type=_csv_modes,
        default=(DeploymentMode.TP, DeploymentMode.PREFILL_CP),
    )
    parser.add_argument("--batch-sizes", type=_csv_ints, default=(1, 8, 32))
    parser.add_argument("--input-lengths", type=_csv_ints, default=(1024, 4096, 16384))
    parser.add_argument(
        "--prefix-hit-ratios", type=_csv_floats, default=(0.0, 0.5, 0.9)
    )
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--cp-size", type=int, default=8)
    parser.add_argument("--dcp-size", type=int, default=8)
    parser.add_argument("--attention-backend", default="aiter")
    parser.add_argument("--mem-fraction-static", type=float, default=0.88)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--token-id-low", type=int, default=100)
    parser.add_argument("--token-id-high", type=int, default=10000)
    parser.add_argument("--launch-timeout-s", type=int, default=2400)
    parser.add_argument("--request-timeout-s", type=int, default=7200)
    parser.add_argument("--logprob-tolerance", type=float, default=0.1)
    parser.add_argument("--strict-parity", action="store_true")
    parser.add_argument("--dynamic-include-dcp", action="store_true")
    parser.add_argument(
        "--dynamic-striped-min-context",
        type=int,
        default=None,
        help=(
            "Opt in to compact striped KV residency at this prompt length. "
            "The default keeps replicated KV so prefill can use CP before decode DCP."
        ),
    )
    parser.add_argument("--launch-module", default="sglang.launch_server")
    parser.add_argument("--extra-server-args", default="")
    parser.add_argument(
        "--mode-server-args-json",
        default="{}",
        help='Per-mode extra args, e.g. \'{"dcp": "--dcp-comm-backend a2a"}\'.',
    )
    parser.add_argument(
        "--server-env-json",
        default="{}",
        help="Environment variables merged into the server process environment.",
    )
    parser.add_argument("--result-file", default="dynamic_parallel.jsonl")
    parser.add_argument("--log-dir", default="dynamic_parallel_logs")
    args = parser.parse_args(argv)
    args.mode_server_args = json.loads(args.mode_server_args_json)
    args.server_env = {
        str(key): str(value) for key, value in json.loads(args.server_env_json).items()
    }
    if args.output_length <= 0 or args.repeats <= 0:
        parser.error("--output-length and --repeats must be positive")
    if (
        args.dynamic_striped_min_context is not None
        and args.dynamic_striped_min_context <= 0
    ):
        parser.error("--dynamic-striped-min-context must be positive")
    if args.tp_size % args.cp_size != 0 or args.tp_size % args.dcp_size != 0:
        parser.error("--cp-size and --dcp-size must divide --tp-size")
    if args.modes[0] is not DeploymentMode.TP and len(args.modes) > 1:
        parser.error("put 'tp' first when running multiple modes for parity comparison")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
