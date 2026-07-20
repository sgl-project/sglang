"""Compare cold checkpoint startup with two remote weight-reuse paths.

The benchmark keeps a source SGLang service online while target services start
in one of three modes:

* ``cold``: load the target from checkpoint files after dropping page cache.
* ``legacy``: use the homogeneous remote-instance TransferEngine path.
* ``manifest``: use the runtime-manifest path, including heterogeneous TP.

Every deterministic source and target response is appended to
``responses.jsonl``. The benchmark requires exact equality of generated text,
input/output token IDs, token counts, and finish reason. Dynamic metadata such
as request IDs, cache counters, and latency is retained in the raw response but
is intentionally excluded from the deterministic comparison.

Homogeneous example (runs all three modes by default):

PYTHONPATH=python python test/manual/benchmark_remote_instance_service_startup.py \
  --model /models/Qwen3.5-0.8B --source-gpus 0,1 --target-gpus 2,3 \
  --source-tp-size 2 --target-tp-size 2 --drop-page-cache --iterations 3

Large-model legacy example (runtime-manifest semantics may be model-specific):

PYTHONPATH=python python test/manual/benchmark_remote_instance_service_startup.py \
  --model /models/Qwen2-72B --source-gpus 0,1 --target-gpus 2,3 \
  --source-tp-size 2 --target-tp-size 2 --modes cold,legacy \
  --drop-page-cache --iterations 3

Heterogeneous example (legacy is reported as ineligible when TPs differ):

PYTHONPATH=python python test/manual/benchmark_remote_instance_service_startup.py \
  --model /models/Qwen3.5-0.8B --source-gpus 0,1 --target-gpus 2,3,4,5 \
  --source-tp-size 2 --target-tp-size 4 --drop-page-cache --iterations 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

ALL_MODES = ("cold", "legacy", "manifest")
REUSE_MODES = ("legacy", "manifest")


@dataclass
class ServerProcess:
    process: subprocess.Popen
    log_file: Any
    log_path: Path
    started_at: float


class ResponseRecorder:
    """Persist raw inference responses as they arrive, including probe traffic."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("w", encoding="utf-8")
        self._lock = threading.Lock()
        self._sequence = 0

    def record_response(
        self,
        *,
        mode: str,
        iteration: int,
        endpoint: str,
        phase: str,
        latency_s: float,
        response: Any,
        deterministic_response: dict[str, Any],
        expected: dict[str, Any] | None,
    ) -> dict[str, Any]:
        consistent = expected is None or deterministic_response == expected
        entry = {
            "kind": "response",
            "mode": mode,
            "iteration": iteration,
            "endpoint": endpoint,
            "phase": phase,
            "captured_at_unix_s": time.time(),
            "latency_s": latency_s,
            "consistent_with_source_baseline": consistent,
            "deterministic_response": deterministic_response,
            "raw_response": response,
        }
        return self._write(entry)

    def record_error(
        self,
        *,
        mode: str,
        iteration: int,
        endpoint: str,
        phase: str,
        error: BaseException,
    ) -> dict[str, Any]:
        return self._write(
            {
                "kind": "error",
                "mode": mode,
                "iteration": iteration,
                "endpoint": endpoint,
                "phase": phase,
                "captured_at_unix_s": time.time(),
                "error": repr(error),
            }
        )

    def close(self) -> None:
        self._file.close()

    def _write(self, entry: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            entry["sequence"] = self._sequence
            self._sequence += 1
            self._file.write(json.dumps(entry, sort_keys=True) + "\n")
            self._file.flush()
        return entry


class SourceProbe:
    """Continuously verify source inference while a target service starts."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        mode: str,
        iteration: int,
        expected: dict[str, Any],
        recorder: ResponseRecorder,
    ) -> None:
        self.args = args
        self.mode = mode
        self.iteration = iteration
        self.expected = expected
        self.recorder = recorder
        self.latencies: list[float] = []
        self.errors: list[str] = []
        self.mismatches: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=self.args.request_timeout_s + 5)
        if self._thread.is_alive():
            self.errors.append("source probe thread did not stop before timeout")
        ordered = sorted(self.latencies)
        p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
        return {
            "success_count": len(ordered),
            "error_count": len(self.errors),
            "mismatch_count": len(self.mismatches),
            "errors": self.errors[:5],
            "mismatches": self.mismatches[:5],
            "latency_p50_s": statistics.median(ordered) if ordered else None,
            "latency_p95_s": ordered[p95_index] if ordered else None,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                measurement = _generate_and_record(
                    self.args,
                    port=self.args.source_port,
                    mode=self.mode,
                    iteration=self.iteration,
                    endpoint="source",
                    phase="during_target_start",
                    expected=self.expected,
                    recorder=self.recorder,
                )
                self.latencies.append(measurement["latency_s"])
                if not measurement["consistent_with_source_baseline"]:
                    self.mismatches.append(
                        {
                            "sequence": measurement["response_sequence"],
                            "deterministic_response": measurement[
                                "deterministic_response"
                            ],
                        }
                    )
            except Exception as error:
                self.errors.append(repr(error))
            self._stop.wait(self.args.probe_interval_s)


def _parse_gpus(value: str) -> str:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values or any(not item.isdigit() for item in values):
        raise argparse.ArgumentTypeError("GPU list must contain comma-separated IDs")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("GPU list must not contain duplicates")
    return ",".join(values)


def _parse_modes(value: str) -> tuple[str, ...]:
    modes = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    unknown = sorted(set(modes) - set(ALL_MODES))
    if not modes:
        raise argparse.ArgumentTypeError("at least one benchmark mode is required")
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown benchmark modes {unknown}; choose from {list(ALL_MODES)}"
        )
    if len(set(modes)) != len(modes):
        raise argparse.ArgumentTypeError("benchmark modes must not contain duplicates")
    return modes


def _drop_page_cache() -> None:
    os.sync()
    Path("/proc/sys/vm/drop_caches").write_text("3\n", encoding="ascii")


def _server_command(
    args: argparse.Namespace,
    *,
    port: int,
    load_mode: str,
) -> list[str]:
    is_source = load_mode == "source"
    tp_size = args.source_tp_size if is_source else args.target_tp_size
    pp_size = args.source_pp_size if is_source else args.target_pp_size
    command = [
        args.python,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--tokenizer-path",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--tp-size",
        str(tp_size),
        "--pp-size",
        str(pp_size),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--disable-cuda-graph",
    ]
    if args.attention_backend:
        command.extend(["--attention-backend", args.attention_backend])
    if args.mm_attention_backend:
        command.extend(["--mm-attention-backend", args.mm_attention_backend])
    if args.sampling_backend:
        command.extend(["--sampling-backend", args.sampling_backend])
    if args.disable_custom_all_reduce:
        command.append("--disable-custom-all-reduce")
    if is_source:
        command.extend(
            [
                "--remote-instance-weight-loader-start-seed-via-transfer-engine",
                "--engine-info-bootstrap-port",
                str(args.bootstrap_port),
            ]
        )
        if "manifest" in args.modes:
            command.append("--enable-weight-runtime-manifest")
    elif load_mode in REUSE_MODES:
        command.extend(
            [
                "--load-format",
                "remote_instance",
                "--remote-instance-weight-loader-backend",
                "transfer_engine",
                "--remote-instance-weight-loader-seed-instance-ip",
                "127.0.0.1",
                "--remote-instance-weight-loader-seed-instance-service-port",
                str(args.source_port),
            ]
        )
        if load_mode == "manifest":
            command.append("--enable-weight-runtime-manifest")
    elif load_mode != "cold":
        raise ValueError(f"unknown load mode: {load_mode}")
    return command


def _start_server(
    args: argparse.Namespace,
    *,
    gpus: str,
    port: int,
    load_mode: str,
    log_path: Path,
) -> ServerProcess:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpus
    environment["MOONCAKE_PROTOCOL"] = args.protocol
    if args.transport_device:
        environment["MOONCAKE_DEVICE"] = args.transport_device
    environment["PYTHONUNBUFFERED"] = "1"
    log_file = log_path.open("w", encoding="utf-8")
    started_at = time.perf_counter()
    try:
        process = subprocess.Popen(
            _server_command(args, port=port, load_mode=load_mode),
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except BaseException:
        log_file.close()
        raise
    return ServerProcess(process, log_file, log_path, started_at)


def _tail(path: Path, lines: int = 80) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError as error:
        return repr(error)


def _wait_ready(server: ServerProcess, port: int, timeout_s: float) -> float:
    deadline = time.monotonic() + timeout_s
    url = f"http://127.0.0.1:{port}/health_generate"
    while time.monotonic() < deadline:
        if server.process.poll() is not None:
            raise RuntimeError(
                f"server exited with {server.process.returncode}:\n{_tail(server.log_path)}"
            )
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                return time.perf_counter() - server.started_at
        except requests.RequestException:
            pass
        time.sleep(0.2)
    raise TimeoutError(
        f"server did not become ready in {timeout_s}s:\n{_tail(server.log_path)}"
    )


def _inference_request(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "text": args.prompt,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": args.max_new_tokens,
            "sampling_seed": args.sampling_seed,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }


def _token_ids(meta_info: dict[str, Any], field: str) -> list[int]:
    entries = meta_info.get(field)
    if not isinstance(entries, list):
        raise ValueError(f"response meta_info.{field} must be a list")
    token_ids = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise ValueError(f"invalid {field}[{index}] entry: {entry!r}")
        token_id = entry[1]
        if not isinstance(token_id, int):
            raise ValueError(f"invalid {field}[{index}] token ID: {token_id!r}")
        token_ids.append(token_id)
    return token_ids


def _deterministic_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict) or not isinstance(response.get("text"), str):
        raise ValueError("generate response must be an object with a string text field")
    meta_info = response.get("meta_info")
    if not isinstance(meta_info, dict):
        raise ValueError("generate response must contain a meta_info object")
    return {
        "text": response["text"],
        "input_token_ids": _token_ids(meta_info, "input_token_logprobs"),
        "output_token_ids": _token_ids(meta_info, "output_token_logprobs"),
        "prompt_tokens": meta_info.get("prompt_tokens"),
        "completion_tokens": meta_info.get("completion_tokens"),
        "finish_reason": meta_info.get("finish_reason"),
    }


def _generate(args: argparse.Namespace, port: int) -> tuple[float, Any]:
    started = time.perf_counter()
    response = requests.post(
        f"http://127.0.0.1:{port}/generate",
        json=_inference_request(args),
        timeout=args.request_timeout_s,
    )
    response.raise_for_status()
    return time.perf_counter() - started, response.json()


def _generate_and_record(
    args: argparse.Namespace,
    *,
    port: int,
    mode: str,
    iteration: int,
    endpoint: str,
    phase: str,
    expected: dict[str, Any] | None,
    recorder: ResponseRecorder,
) -> dict[str, Any]:
    try:
        latency_s, response = _generate(args, port)
        deterministic_response = _deterministic_response(response)
    except Exception as error:
        recorder.record_error(
            mode=mode,
            iteration=iteration,
            endpoint=endpoint,
            phase=phase,
            error=error,
        )
        raise
    record = recorder.record_response(
        mode=mode,
        iteration=iteration,
        endpoint=endpoint,
        phase=phase,
        latency_s=latency_s,
        response=response,
        deterministic_response=deterministic_response,
        expected=expected,
    )
    return {
        "latency_s": latency_s,
        "response_sequence": record["sequence"],
        "consistent_with_source_baseline": record["consistent_with_source_baseline"],
        "deterministic_response": deterministic_response,
    }


def _stop_server(server: ServerProcess) -> None:
    if server.process.poll() is None:
        os.killpg(server.process.pid, signal.SIGTERM)
        try:
            server.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(server.process.pid, signal.SIGKILL)
            server.process.wait(timeout=30)
    server.log_file.close()


def _assert_iteration_consistency(
    *,
    mode: str,
    iteration: int,
    measurements: list[tuple[str, dict[str, Any]]],
    source_probe: dict[str, Any],
    responses_path: Path,
) -> None:
    failures = [
        label
        for label, measurement in measurements
        if not measurement["consistent_with_source_baseline"]
    ]
    if source_probe["success_count"] == 0:
        failures.append("source probe produced no successful inference")
    if source_probe["error_count"]:
        failures.append(f"source probe errors={source_probe['error_count']}")
    if source_probe["mismatch_count"]:
        failures.append(f"source probe mismatches={source_probe['mismatch_count']}")
    if failures:
        raise RuntimeError(
            f"{mode} iteration {iteration} failed strict inference consistency: "
            f"{failures}; inspect {responses_path}"
        )


def _run_target(
    args: argparse.Namespace,
    *,
    mode: str,
    iteration: int,
    output_dir: Path,
    source_baseline: dict[str, Any],
    recorder: ResponseRecorder,
) -> dict[str, Any]:
    if args.drop_page_cache:
        _drop_page_cache()

    before = _generate_and_record(
        args,
        port=args.source_port,
        mode=mode,
        iteration=iteration,
        endpoint="source",
        phase="before_target_start",
        expected=source_baseline,
        recorder=recorder,
    )
    probe = SourceProbe(
        args,
        mode=mode,
        iteration=iteration,
        expected=source_baseline,
        recorder=recorder,
    )
    server: ServerProcess | None = None
    probe_summary: dict[str, Any] | None = None
    probe.start()
    try:
        server = _start_server(
            args,
            gpus=args.target_gpus,
            port=args.target_port,
            load_mode=mode,
            log_path=output_dir / f"{mode}-{iteration}.log",
        )
        ready_s = _wait_ready(server, args.target_port, args.timeout_s)
        target = _generate_and_record(
            args,
            port=args.target_port,
            mode=mode,
            iteration=iteration,
            endpoint="target",
            phase="after_target_ready",
            expected=source_baseline,
            recorder=recorder,
        )
        probe_summary = probe.stop()
        after = _generate_and_record(
            args,
            port=args.source_port,
            mode=mode,
            iteration=iteration,
            endpoint="source",
            phase="after_target_ready",
            expected=source_baseline,
            recorder=recorder,
        )
        _assert_iteration_consistency(
            mode=mode,
            iteration=iteration,
            measurements=[
                ("source before target start", before),
                ("target after ready", target),
                ("source after target ready", after),
            ],
            source_probe=probe_summary,
            responses_path=recorder.path,
        )
        return {
            "mode": mode,
            "iteration": iteration,
            "spawn_to_ready_s": ready_s,
            "first_generation_s": target["latency_s"],
            "response_sequences": {
                "source_before": before["response_sequence"],
                "target": target["response_sequence"],
                "source_after": after["response_sequence"],
            },
            "source_probe": probe_summary,
            "log": str(server.log_path),
        }
    finally:
        if probe.is_alive():
            probe.stop()
        if server is not None:
            _stop_server(server)
        time.sleep(args.inter_run_delay_s)


def _mode_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    ready = [record["spawn_to_ready_s"] for record in records]
    first_generation = [record["first_generation_s"] for record in records]
    return {
        "iterations": len(records),
        "spawn_to_ready_p50_s": statistics.median(ready),
        "spawn_to_ready_mean_s": statistics.mean(ready),
        "first_generation_p50_s": statistics.median(first_generation),
        "first_generation_mean_s": statistics.mean(first_generation),
        "source_probe_success_count": sum(
            record["source_probe"]["success_count"] for record in records
        ),
        "source_probe_error_count": sum(
            record["source_probe"]["error_count"] for record in records
        ),
        "source_probe_mismatch_count": sum(
            record["source_probe"]["mismatch_count"] for record in records
        ),
    }


def _reuse_comparison(
    cold: dict[str, Any],
    reuse: dict[str, Any],
    *,
    max_reuse_to_cold_ratio: float,
) -> dict[str, Any]:
    cold_p50 = cold["spawn_to_ready_p50_s"]
    cold_mean = cold["spawn_to_ready_mean_s"]
    reuse_p50 = reuse["spawn_to_ready_p50_s"]
    reuse_mean = reuse["spawn_to_ready_mean_s"]
    threshold_s = cold_p50 * max_reuse_to_cold_ratio
    return {
        "cold_spawn_to_ready_p50_s": cold_p50,
        "reuse_spawn_to_ready_p50_s": reuse_p50,
        "cold_spawn_to_ready_mean_s": cold_mean,
        "reuse_spawn_to_ready_mean_s": reuse_mean,
        "p50_speedup": cold_p50 / reuse_p50,
        "mean_speedup": cold_mean / reuse_mean,
        "reuse_to_cold_p50_ratio": reuse_p50 / cold_p50,
        "p50_improvement_ratio": (cold_p50 - reuse_p50) / cold_p50,
        "max_reuse_to_cold_ratio": max_reuse_to_cold_ratio,
        "reuse_p50_must_be_lte_s": threshold_s,
        "passes_significant_improvement_threshold": reuse_p50 <= threshold_s,
    }


def _eligible_modes(args: argparse.Namespace) -> tuple[list[str], list[dict[str, str]]]:
    executed = []
    skipped = []
    for mode in args.modes:
        if mode == "legacy" and (
            args.source_tp_size != args.target_tp_size
            or args.source_pp_size != args.target_pp_size
        ):
            skipped.append(
                {
                    "mode": "legacy",
                    "reason": (
                        "legacy remote-instance reuse requires homogeneous TP/PP; "
                        f"source_tp_size={args.source_tp_size}, "
                        f"target_tp_size={args.target_tp_size}, "
                        f"source_pp_size={args.source_pp_size}, "
                        f"target_pp_size={args.target_pp_size}"
                    ),
                }
            )
        else:
            executed.append(mode)
    return executed, skipped


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "responses.jsonl"
    result_path = output_dir / "benchmark-result.json"
    recorder = ResponseRecorder(responses_path)
    source: ServerProcess | None = None
    result: dict[str, Any] | None = None
    try:
        source = _start_server(
            args,
            gpus=args.source_gpus,
            port=args.source_port,
            load_mode="source",
            log_path=output_dir / "source.log",
        )
        source_ready_s = _wait_ready(source, args.source_port, args.timeout_s)
        baseline = _generate_and_record(
            args,
            port=args.source_port,
            mode="source",
            iteration=-1,
            endpoint="source",
            phase="baseline",
            expected=None,
            recorder=recorder,
        )
        source_baseline = baseline["deterministic_response"]
        executed_modes, skipped_modes = _eligible_modes(args)
        records: dict[str, list[dict[str, Any]]] = {mode: [] for mode in executed_modes}
        for iteration in range(args.iterations):
            for mode in executed_modes:
                records[mode].append(
                    _run_target(
                        args,
                        mode=mode,
                        iteration=iteration,
                        output_dir=output_dir,
                        source_baseline=source_baseline,
                        recorder=recorder,
                    )
                )

        by_mode = {
            mode: _mode_summary(mode_records) for mode, mode_records in records.items()
        }
        comparisons = {
            f"{mode}_vs_cold": _reuse_comparison(
                by_mode["cold"],
                by_mode[mode],
                max_reuse_to_cold_ratio=args.max_reuse_to_cold_ratio,
            )
            for mode in REUSE_MODES
            if mode in by_mode
        }
        threshold_results = [
            comparison["passes_significant_improvement_threshold"]
            for comparison in comparisons.values()
        ]
        result = {
            "schema_version": 1,
            "model": args.model,
            "topology": {
                "source_tp_size": args.source_tp_size,
                "target_tp_size": args.target_tp_size,
                "source_pp_size": args.source_pp_size,
                "target_pp_size": args.target_pp_size,
                "source_gpus": args.source_gpus,
                "target_gpus": args.target_gpus,
            },
            "modes_requested": list(args.modes),
            "modes_executed": executed_modes,
            "skipped_modes": skipped_modes,
            "iterations": args.iterations,
            "page_cache_dropped_before_each_target": args.drop_page_cache,
            "disable_custom_all_reduce": args.disable_custom_all_reduce,
            "measurement_boundary": "target process spawn to health_generate ready",
            "deterministic_inference": {
                "request": _inference_request(args),
                "comparison": (
                    "exact text, input/output token IDs, token counts, and "
                    "finish reason"
                ),
                "source_baseline_response_sequence": baseline["response_sequence"],
                "source_baseline": source_baseline,
            },
            "source": {
                "spawn_to_ready_s": source_ready_s,
                "log": str(source.log_path),
            },
            "records": records,
            "summary": {
                "by_mode": by_mode,
                "comparisons": comparisons,
                "significant_improvement_threshold": {
                    "metric": "spawn_to_ready_p50_s",
                    "condition": "reuse <= cold * max_reuse_to_cold_ratio",
                    "max_reuse_to_cold_ratio": args.max_reuse_to_cold_ratio,
                },
                "all_executed_reuse_modes_pass_threshold": (
                    all(threshold_results) if threshold_results else None
                ),
                "strict_response_consistency_passed": True,
                "source_serving_continuity_passed": True,
            },
            "artifacts": {
                "result_json": str(result_path),
                "responses_jsonl": str(responses_path),
            },
        }
    finally:
        if source is not None:
            _stop_server(source)
        recorder.close()

    if result is None:
        raise RuntimeError("benchmark exited without producing a result")
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare cold checkpoint startup, homogeneous remote reuse, and "
            "runtime-manifest heterogeneous reuse."
        )
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--python", default=os.environ.get("PYTHON", "python"))
    parser.add_argument("--source-gpus", type=_parse_gpus, required=True)
    parser.add_argument("--target-gpus", type=_parse_gpus, required=True)
    parser.add_argument("--source-tp-size", type=int, required=True)
    parser.add_argument("--target-tp-size", type=int, required=True)
    parser.add_argument("--source-pp-size", type=int, default=1)
    parser.add_argument("--target-pp-size", type=int, default=1)
    parser.add_argument("--modes", type=_parse_modes, default=ALL_MODES)
    parser.add_argument("--source-port", type=int, default=31000)
    parser.add_argument("--target-port", type=int, default=32000)
    parser.add_argument("--bootstrap-port", type=int, default=31999)
    parser.add_argument("--protocol", default="rdma")
    parser.add_argument("--transport-device", default="")
    parser.add_argument("--mem-fraction-static", type=float, default=0.88)
    parser.add_argument("--attention-backend", default="")
    parser.add_argument("--mm-attention-backend", default="")
    parser.add_argument("--sampling-backend", default="")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=1200)
    parser.add_argument("--request-timeout-s", type=float, default=120)
    parser.add_argument("--probe-interval-s", type=float, default=0.2)
    parser.add_argument("--inter-run-delay-s", type=float, default=2)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--sampling-seed", type=int, default=0)
    parser.add_argument(
        "--max-reuse-to-cold-ratio",
        type=float,
        default=0.8,
        help=(
            "Significant-improvement gate applied to spawn-to-ready p50 "
            "(default: reuse <= cold * 0.8)."
        ),
    )
    parser.add_argument("--drop-page-cache", action="store_true")
    parser.add_argument("--output-dir", default="remote-instance-service-benchmark")
    args = parser.parse_args()

    if args.iterations <= 0:
        parser.error("iterations must be positive")
    if (
        min(
            args.source_tp_size,
            args.target_tp_size,
            args.source_pp_size,
            args.target_pp_size,
        )
        <= 0
    ):
        parser.error("source/target TP and PP sizes must be positive")
    source_world_size = args.source_tp_size * args.source_pp_size
    target_world_size = args.target_tp_size * args.target_pp_size
    if len(args.source_gpus.split(",")) != source_world_size:
        parser.error("source-gpus count must equal source-tp-size * source-pp-size")
    if len(args.target_gpus.split(",")) != target_world_size:
        parser.error("target-gpus count must equal target-tp-size * target-pp-size")
    if set(args.source_gpus.split(",")) & set(args.target_gpus.split(",")):
        parser.error("source and target GPU sets must not overlap")
    if len({args.source_port, args.target_port, args.bootstrap_port}) != 3:
        parser.error("source-port, target-port, and bootstrap-port must be distinct")
    if any(mode in args.modes for mode in REUSE_MODES) and "cold" not in args.modes:
        parser.error("reuse modes require cold mode for speedup comparison")
    if "cold" in args.modes and not args.drop_page_cache:
        parser.error("cold mode requires --drop-page-cache")
    if not 0 < args.mem_fraction_static < 1:
        parser.error("mem-fraction-static must be between 0 and 1")
    if args.timeout_s <= 0 or args.request_timeout_s <= 0:
        parser.error("timeout-s and request-timeout-s must be positive")
    if args.probe_interval_s <= 0 or args.inter_run_delay_s < 0:
        parser.error("probe interval must be positive and inter-run delay nonnegative")
    if args.max_new_tokens <= 0:
        parser.error("max-new-tokens must be positive")
    if not 0 < args.max_reuse_to_cold_ratio <= 1:
        parser.error("max-reuse-to-cold-ratio must be in (0, 1]")
    return args


if __name__ == "__main__":
    print("REMOTE_INSTANCE_SERVICE_BENCHMARK_JSON=" + json.dumps(run(parse_args())))
