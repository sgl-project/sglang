# SPDX-License-Identifier: Apache-2.0
"""Calibrate diffusion batching memory profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any

import zmq

from sglang.multimodal_gen.configs.sample.sampling_params import (
    SamplingParams,
    generate_request_id,
)
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request
from sglang.multimodal_gen.runtime.scheduler_client import SchedulerClient
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class CalibrationRecord:
    width: int
    height: int
    num_frames: int
    batch_size: int
    success_count: int
    oom_count: int
    error_count: int
    peak_memory_mb: float
    peak_allocated_memory_mb: float
    duration_s: float
    errors: list[str]


@dataclass
class CalibrationProfileRecord:
    key_hash: str
    observation_count: int
    batch_sizes: list[int]
    peak_memory_mb: list[float]


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    shapes = _parse_calibration_shapes(args.shapes)
    requested_batch_sizes = _parse_calibration_batch_sizes(args.batch_sizes)
    batch_sizes = _add_geometric_ramp(requested_batch_sizes)

    max_batch_size = max(batch_sizes)
    server_args = ServerArgs.from_kwargs(
        model_path=args.model_path,
        batching_max_size=max_batch_size,
        batching_delay_ms=args.batching_delay_ms,
        batching_memory_profile_cache=args.batching_memory_profile_cache,
        warmup=args.warmup,
    )

    records: list[CalibrationRecord] = []
    with DiffGenerator.from_server_args(server_args, local_mode=True) as _generator:
        for width, height, num_frames in shapes:
            for batch_size in batch_sizes:
                logger.info(
                    "Calibrating shape=%dx%dx%d batch_size=%d",
                    width,
                    height,
                    num_frames,
                    batch_size,
                )
                record = _run_calibration_batch(
                    server_args=server_args,
                    prompt=args.prompt,
                    width=width,
                    height=height,
                    num_frames=num_frames,
                    num_inference_steps=args.num_inference_steps,
                    batch_size=batch_size,
                    timeout_s=args.per_batch_timeout_s,
                )
                records.append(record)
                if args.stop_on_oom and record.oom_count:
                    break
                if args.stop_on_error and record.error_count and not record.oom_count:
                    break

    summary = {
        "schema_version": 1,
        "model_path": args.model_path,
        "profile_cache": args.batching_memory_profile_cache,
        "records": [asdict(record) for record in records],
        "profiles": [
            asdict(record)
            for record in _read_profile_cache_summary(
                args.batching_memory_profile_cache
            )
        ],
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")
    return summary


def _run_calibration_batch(
    *,
    server_args: ServerArgs,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    num_inference_steps: int | None,
    batch_size: int,
    timeout_s: float,
) -> CalibrationRecord:
    start = time.monotonic()
    barrier = threading.Barrier(batch_size)

    def send_one(idx: int):
        sampling_params = SamplingParams.from_user_sampling_params_args(
            server_args.model_path,
            server_args=server_args,
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            request_id=f"batch-memory-calibration-{generate_request_id()}-{idx}",
            save_output=False,
            suppress_logs=True,
        )
        req = prepare_request(server_args=server_args, sampling_params=sampling_params)
        client = SchedulerClient()
        client.initialize(server_args)
        client.scheduler_socket.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        try:
            barrier.wait(timeout=timeout_s)
            return client.forward([req])
        finally:
            client.close()

    outputs = []
    errors = []
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = [executor.submit(send_one, idx) for idx in range(batch_size)]
        for future in as_completed(futures, timeout=timeout_s * max(1, batch_size)):
            try:
                outputs.append(future.result(timeout=timeout_s))
            except Exception as exc:
                errors.append(str(exc))

    output_errors = [out.error for out in outputs if getattr(out, "error", None)]
    errors.extend(str(error) for error in output_errors)
    oom_count = sum(1 for out in outputs if getattr(out, "is_oom", False))
    success_count = sum(
        1
        for out in outputs
        if getattr(out, "error", None) is None and not getattr(out, "is_oom", False)
    )
    return CalibrationRecord(
        width=width,
        height=height,
        num_frames=num_frames,
        batch_size=batch_size,
        success_count=success_count,
        oom_count=oom_count,
        error_count=len(errors) - oom_count,
        peak_memory_mb=max(
            (getattr(out, "peak_memory_mb", 0.0) for out in outputs), default=0.0
        ),
        peak_allocated_memory_mb=max(
            (getattr(out, "peak_allocated_memory_mb", 0.0) for out in outputs),
            default=0.0,
        ),
        duration_s=time.monotonic() - start,
        errors=errors,
    )


def _parse_calibration_shapes(raw: str) -> list[tuple[int, int, int]]:
    shapes = []
    for item in raw.split(","):
        item = item.strip().lower()
        if not item:
            continue
        parts = item.replace(":", "x").split("x")
        if len(parts) == 2:
            width, height = (int(parts[0]), int(parts[1]))
            num_frames = 1
        elif len(parts) == 3:
            width, height, num_frames = (int(parts[0]), int(parts[1]), int(parts[2]))
        else:
            raise ValueError(
                f"invalid shape {item!r}; expected WIDTHxHEIGHT or WIDTHxHEIGHTxFRAMES"
            )
        shapes.append((width, height, num_frames))
    if not shapes:
        raise ValueError("at least one shape is required")
    return shapes


def _parse_calibration_batch_sizes(raw: str) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or values[0] < 1:
        raise ValueError("batch sizes must be positive integers")
    return values


def _add_geometric_ramp(batch_sizes: list[int]) -> list[int]:
    expanded = set(batch_sizes)
    for target in batch_sizes:
        current = 1
        expanded.add(current)
        while current < target:
            current = min(target, current * 2)
            expanded.add(current)
    return sorted(expanded)


def _read_profile_cache_summary(
    cache_path: str | None,
) -> list[CalibrationProfileRecord]:
    if not cache_path:
        return []

    path = cache_path
    if not path.endswith(".json"):
        try:
            files = [
                os.path.join(path, name)
                for name in os.listdir(path)
                if name.endswith(".json")
            ]
        except OSError:
            return []
        if not files:
            return []
        path = max(files, key=os.path.getmtime)

    try:
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    records = []
    for item in payload.get("profiles", []):
        profile = item.get("profile", {}) if isinstance(item, dict) else {}
        key = item.get("key") if isinstance(item, dict) else None
        successes = profile.get("successes", [])
        if not isinstance(successes, list):
            continue
        batch_sizes = sorted(
            {
                int(obs.get("batch_size", 1))
                for obs in successes
                if isinstance(obs, dict)
            }
        )
        peaks = [
            float(obs.get("peak_memory_mb", 0.0))
            for obs in successes
            if isinstance(obs, dict)
        ]
        records.append(
            CalibrationProfileRecord(
                key_hash=_short_json_hash(key),
                observation_count=len(successes),
                batch_sizes=batch_sizes,
                peak_memory_mb=peaks,
            )
        )
    return records


def _short_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate diffusion dynamic batching memory profiles."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prompt", default="a cinematic landscape")
    parser.add_argument(
        "--shapes",
        required=True,
        help="Comma-separated WIDTHxHEIGHT or WIDTHxHEIGHTxFRAMES entries.",
    )
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--batching-delay-ms", type=float, default=25.0)
    parser.add_argument("--batching-memory-profile-cache", default=None)
    parser.add_argument("--per-batch-timeout-s", type=float, default=600.0)
    parser.add_argument(
        "--output-json", default="batch_memory_calibration_summary.json"
    )
    parser.add_argument("--stop-on-oom", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    return parser


def main() -> None:
    run_calibration(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
