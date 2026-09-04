"""SGLang serving benchmark adapter for simulator traffic.

This script deliberately reuses SGLang's benchmark implementation and dataset
loaders.  It only owns the simulator-specific parts of the protocol:

* convert request-rate or trace timestamps into logical arrival timestamps;
* inject the internal ``sampling_params.custom_params.simulation`` metadata;
* avoid client-side pacing in OFFLINE mode; and
* display backend-produced simulator metrics when they are locally available.

User datasets must not contain simulator metadata.
"""

import argparse
import contextlib
import json
import os
import re
import sys
from dataclasses import fields
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import aiohttp
import numpy as np
from sglang_simulator.compat import validate_benchmark_runtime
from sglang_simulator.dataset.autobench import register_autobench_dataset

register_autobench_dataset()

from sglang.benchmark import serving
from sglang.benchmark.datasets.common import DatasetRow

_ORIGINAL_AIOHTTP_REQUEST = None
_ORIGINAL_CALCULATE_METRICS = serving.calculate_metrics
_ORIGINAL_GET_REQUEST = serving.get_request
_ORIGINAL_RUN_BENCHMARK = serving.run_benchmark
_SIMULATOR_MODE = "offline"
_USE_TRACE_TIMESTAMPS = False


def _metrics_path() -> Path:
    output_dir = Path(
        os.getenv("SGLANG_SIMULATOR_OUTPUT_DIR", "/tmp/sglang_simulator/output")
    )
    return output_dir / "metrics.json"


def _load_backend_metrics() -> Optional[dict]:
    metrics_path = _metrics_path()
    if not metrics_path.is_file():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


class _DurationReplacingStream:
    """Keep SGLang's output format but print the simulated duration."""

    def __init__(self, target):
        self.target = target

    def write(self, text):
        if "Benchmark duration (s):" in text:
            metrics = _load_backend_metrics()
            if metrics is not None and "duration" in metrics:
                text = "{:<40} {:<10.2f}".format(
                    "Benchmark duration (s):", metrics["duration"]
                )
        return self.target.write(text)

    def flush(self):
        return self.target.flush()


def _set_simulation_metadata(
    request: DatasetRow, *, created_time_ms: float, total_request: int
) -> None:
    """Attach transient metadata without replacing dataset-specific parameters."""
    extra_request_body = dict(request.extra_request_body or {})
    extra_request_body["simulation"] = {
        "created_time_ms": created_time_ms,
        "total_request": total_request,
    }
    request.extra_request_body = extra_request_body


async def simulator_get_request(
    input_requests: List[DatasetRow],
    request_rate: float,
    use_trace_timestamps: bool = False,
    slowdown_factor: float = 1.0,
) -> AsyncGenerator[DatasetRow, None]:
    """Generate simulator traffic while retaining official BLOCKING pacing."""
    # The benchmark may not forward --use-trace-timestamps to get_request(),
    # so preserve the parsed value in this adapter.
    use_trace_timestamps = use_trace_timestamps or _USE_TRACE_TIMESTAMPS
    if _SIMULATOR_MODE == "blocking":
        async for request in _ORIGINAL_GET_REQUEST(
            input_requests,
            request_rate,
            use_trace_timestamps=use_trace_timestamps,
            slowdown_factor=slowdown_factor,
        ):
            yield request
        return

    total_request = len(input_requests)
    if use_trace_timestamps:
        if any(request.timestamp is None for request in input_requests):
            raise ValueError(
                "--use-trace-timestamps requires every request to have timestamp"
            )
        input_requests.sort(key=lambda request: request.timestamp)
        trace_start_time_ms = input_requests[0].timestamp if input_requests else 0.0
        for request in input_requests:
            created_time_ms = (
                float(request.timestamp) - float(trace_start_time_ms)
            ) * slowdown_factor
            _set_simulation_metadata(
                request,
                created_time_ms=created_time_ms,
                total_request=total_request,
            )
            yield request
        return

    created_time_ms = 0.0
    for request in input_requests:
        _set_simulation_metadata(
            request,
            created_time_ms=created_time_ms,
            total_request=total_request,
        )
        yield request
        if request_rate != float("inf"):
            created_time_ms += np.random.exponential(1.0 / request_rate) * 1000.0


def install_aiohttp_json_hijack(
    *, hijack_url_regex: Optional[str] = r"/generate(?:\?.*)?$"
) -> None:
    """Move transient metadata into the already-built sampling parameters."""
    global _ORIGINAL_AIOHTTP_REQUEST
    if _ORIGINAL_AIOHTTP_REQUEST is not None:
        return

    pattern = re.compile(hijack_url_regex) if hijack_url_regex else None
    _ORIGINAL_AIOHTTP_REQUEST = aiohttp.ClientSession._request

    async def patched_request(self, method, url, **kwargs):
        if pattern is None or pattern.search(str(url)):
            payload = kwargs.get("json")
            if isinstance(payload, dict) and "simulation" in payload:
                simulation = payload.pop("simulation")
                sampling_params = payload.setdefault("sampling_params", {})
                custom_params = sampling_params.setdefault("custom_params", {})
                custom_params["simulation"] = simulation
                kwargs["json"] = payload
        return await _ORIGINAL_AIOHTTP_REQUEST(self, method, url, **kwargs)

    aiohttp.ClientSession._request = patched_request


def simulator_calculate_metrics(*args, **kwargs):
    """Use simulator metrics; mark unsupported client-only fields with -1."""
    client_metrics, output_lens = _ORIGINAL_CALCULATE_METRICS(*args, **kwargs)
    backend_metrics = _load_backend_metrics()
    if backend_metrics is None:
        print(
            f"Simulator metrics are not available at {_metrics_path()}; "
            "showing client-side benchmark metrics."
        )
        return client_metrics, output_lens

    metric_names = {field.name for field in fields(serving.BenchmarkMetrics)}
    values = {name: backend_metrics.get(name, -1) for name in metric_names}
    return serving.BenchmarkMetrics(**values), output_lens


def _replace_output_file_duration(
    args: argparse.Namespace, simulated_duration: float
) -> None:
    output_file = getattr(args, "output_file", None)
    if not output_file:
        return
    path = Path(output_file)
    if not path.is_file():
        return
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return
    last_result = json.loads(lines[-1])
    last_result["duration"] = simulated_duration
    lines[-1] = json.dumps(last_result)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def simulator_run_benchmark(args: argparse.Namespace):
    global _USE_TRACE_TIMESTAMPS
    if args.backend != "sglang":
        raise ValueError(
            "benchmark/simulator/bench_serving.py requires --backend sglang"
        )
    if args.dataset_name == "mooncake":
        raise ValueError(
            "Mooncake's multi-round scheduler is not supported by the simulator "
            "benchmark adapter"
        )
    _USE_TRACE_TIMESTAMPS = getattr(args, "use_trace_timestamps", False)
    args.profile = True
    with contextlib.redirect_stdout(_DurationReplacingStream(sys.stdout)):
        result = _ORIGINAL_RUN_BENCHMARK(args)

    backend_metrics = _load_backend_metrics()
    if backend_metrics is not None and "duration" in backend_metrics:
        simulated_duration = backend_metrics["duration"]
        if isinstance(result, dict):
            result["duration"] = simulated_duration
        _replace_output_file_duration(args, simulated_duration)
    return result


def _extract_simulator_args(argv: list[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--simulator-mode",
        choices=("offline", "blocking"),
        default="offline",
        help=argparse.SUPPRESS,
    )
    args, remaining = parser.parse_known_args(argv)
    return args.simulator_mode, remaining


def _simulator_argument_parser(base_parser):
    """Include simulator-owned datasets in SGLang's hard-coded CLI choices."""

    class SimulatorArgumentParser(base_parser):
        def add_argument(self, *name_or_flags, **kwargs):
            choices = kwargs.get("choices")
            if (
                "--dataset-name" in name_or_flags
                and choices is not None
                and "autobench" not in choices
            ):
                kwargs["choices"] = [*choices, "autobench"]
            if "--warmup-requests" in name_or_flags:
                kwargs["default"] = 0
            return super().add_argument(*name_or_flags, **kwargs)

    return SimulatorArgumentParser


def cli_main() -> None:
    global _SIMULATOR_MODE
    validate_benchmark_runtime()
    if any(argument in ("-h", "--help") for argument in sys.argv[1:]):
        print(
            "SGLang Simulator option: "
            "--simulator-mode {offline,blocking} (default: offline)\n"
        )
    _SIMULATOR_MODE, remaining = _extract_simulator_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]

    serving.get_request = simulator_get_request
    serving.calculate_metrics = simulator_calculate_metrics
    serving.run_benchmark = simulator_run_benchmark
    install_aiohttp_json_hijack()

    print(f"SGLang Simulator benchmark mode: {_SIMULATOR_MODE.upper()}")
    original_parser = serving.ArgumentParser
    serving.ArgumentParser = _simulator_argument_parser(original_parser)
    try:
        serving.cli_main()
    finally:
        serving.ArgumentParser = original_parser


if __name__ == "__main__":
    cli_main()
