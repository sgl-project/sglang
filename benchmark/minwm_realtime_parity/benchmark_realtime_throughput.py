#!/usr/bin/env python3
"""Measure one persistent MinWM realtime session with LingBot's timing window.

Run this client against two separately launched servers to compare execution
profiles.  Keep the checkpoint, hardware, request, and action contract fixed;
only the server implementation/profile may change.  The client deliberately
does not retain frame payloads, so a 220-chunk run does not consume gigabytes of
host memory.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import msgspec.msgpack

from common import action_weights, load_cases, materialize_first_frame, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--ws-url", default="ws://127.0.0.1:30000/v1/realtime_video/generate"
    )
    parser.add_argument("--model", default="minwm")
    parser.add_argument("--case", default="00_forward_pottery")
    parser.add_argument("--profile-name", required=True)
    parser.add_argument("--warmup-chunks", type=int, default=20)
    parser.add_argument("--measured-chunks", type=int, default=200)
    parser.add_argument("--kv-cache-num-frames", type=int)
    parser.add_argument("--timeout", type=float, default=1800.0)
    return parser.parse_args()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)
    return ordered[max(index, 0)]


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("latency summary requires at least one value")
    return {
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def validate_contract(manifest: dict, args: argparse.Namespace) -> tuple[dict, dict]:
    contract = manifest["contract"]
    if contract.get("action_type") != "primitive_token_residual":
        raise ValueError("throughput comparison requires primitive_token_residual")
    if int(contract["latent_frames_per_chunk"]) != 4:
        raise ValueError("MinWM throughput contract requires four latent frames/chunk")
    if int(contract["generated_pixel_frames"]) % int(
        contract["generated_latent_frames"]
    ):
        raise ValueError("pixel/latent frame ratio must be integral")
    if args.warmup_chunks < 1 or args.measured_chunks < 1:
        raise ValueError("warmup-chunks and measured-chunks must be positive")
    if args.kv_cache_num_frames is not None and args.kv_cache_num_frames < 1:
        raise ValueError("kv-cache-num-frames must be positive")
    cases = {case["id"]: case for case in manifest["cases"]}
    if args.case not in cases:
        raise ValueError(f"unknown case {args.case!r}; choose from {sorted(cases)}")
    return contract, cases[args.case]


def validate_frame_batch(
    header: dict[str, Any], payload: bytes, *, chunk_index: int
) -> tuple[int, int, int]:
    batch_frames = int(header["num_frames"])
    expected_bytes = (
        batch_frames
        * int(header["height"])
        * int(header["width"])
        * int(header["channels"])
    )
    batch_index = int(header.get("frame_batch_index", 0))
    num_batches = int(header.get("num_frame_batches", 1))
    expected_final = batch_index == num_batches - 1
    checks = {
        "chunk_index": int(header["chunk_index"]) == chunk_index,
        "positive_num_frames": batch_frames > 0,
        "content_type": header["content_type"] == "application/x-raw-rgb",
        "payload_bytes": len(payload) == expected_bytes,
        "bytes_per_frame": int(header["bytes_per_frame"])
        == expected_bytes // batch_frames,
        "raw_size": int(header.get("raw_size", len(payload))) == len(payload),
        "total_size": int(header.get("total_size", len(payload))) == len(payload),
        "batch_index": 0 <= batch_index < num_batches,
        "is_final_frame_batch": bool(header.get("is_final_frame_batch", expected_final))
        == expected_final,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(
            f"chunk {chunk_index} raw frame batch contract failed: {failed}; "
            f"header={header}"
        )
    return batch_index, num_batches, batch_frames


async def receive_run(args: argparse.Namespace, contract: dict, case: dict) -> dict:
    import websockets

    total_chunks = args.warmup_chunks + args.measured_chunks
    latent_frames_per_chunk = int(contract["latent_frames_per_chunk"])
    pixel_frames_per_latent = int(contract["generated_pixel_frames"]) // int(
        contract["generated_latent_frames"]
    )
    first_frame = materialize_first_frame(case, Path(args.output).parent / "inputs")
    if contract.get("action_output_format") == "primitive_float":
        action_condition = {
            "action_weights": [action_weights(case)]
            * total_chunks
            * latent_frames_per_chunk
            * pixel_frames_per_latent
        }
    else:
        action_condition = {
            "action_labels": [int(case["action_label"])]
            * total_chunks
            * latent_frames_per_chunk
        }
    request = {
        "type": "init",
        "model": args.model,
        "prompt": case["prompt"],
        "first_frame": first_frame.read_bytes(),
        "size": f"{contract['width']}x{contract['height']}",
        "fps": int(contract["fps"]),
        "seed": int(contract["seed"]),
        "generator_device": "cuda",
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "max_chunks": total_chunks,
        "realtime_output_format": "raw",
        "condition_inputs": action_condition,
    }
    if args.kv_cache_num_frames is not None:
        request["realtime_causal_kv_cache_num_frames"] = args.kv_cache_num_frames
    stats_by_chunk: dict[int, dict[str, Any]] = {}
    payload_complete_ns: dict[int, int] = {}
    frame_batches_by_chunk: dict[int, dict[str, Any]] = {}
    init_started_ns = time.perf_counter_ns()
    async with websockets.connect(
        args.ws_url, max_size=None, ping_interval=None, open_timeout=args.timeout
    ) as websocket:
        await websocket.send(msgspec.msgpack.encode(request))
        init_completed_ns = time.perf_counter_ns()
        while (
            len(stats_by_chunk) < total_chunks
            or len(payload_complete_ns) < total_chunks
        ):
            packed = await asyncio.wait_for(websocket.recv(), timeout=args.timeout)
            if not isinstance(packed, bytes):
                raise TypeError(
                    f"expected binary MessagePack, got {type(packed).__name__}"
                )
            header = msgspec.msgpack.decode(packed)
            message_type = header.get("type")
            if message_type == "error":
                raise RuntimeError(header.get("content", "unknown realtime error"))
            if message_type == "chunk_stats":
                stats_by_chunk[int(header["chunk_index"])] = header
                continue
            if message_type == "frame_batch":
                payload = header.pop("payload")
            elif message_type == "frame_batch_header":
                payload = await asyncio.wait_for(websocket.recv(), timeout=args.timeout)
                if not isinstance(payload, bytes):
                    raise TypeError("raw frame payload must be bytes")
            else:
                raise ValueError(f"unexpected realtime message: {header}")
            chunk_index = int(header["chunk_index"])
            expected_frames = (
                int(contract["reference_pixel_frames"])
                + pixel_frames_per_latent * latent_frames_per_chunk
                if chunk_index == 0
                else pixel_frames_per_latent * latent_frames_per_chunk
            )
            batch_index, num_batches, batch_frames = validate_frame_batch(
                header, payload, chunk_index=chunk_index
            )
            state = frame_batches_by_chunk.setdefault(
                chunk_index,
                {"num_batches": num_batches, "seen": set(), "frames": 0},
            )
            if state["num_batches"] != num_batches:
                raise AssertionError(
                    f"chunk {chunk_index} changed num_frame_batches from "
                    f"{state['num_batches']} to {num_batches}"
                )
            if batch_index in state["seen"]:
                raise AssertionError(
                    f"chunk {chunk_index} repeated frame batch {batch_index}"
                )
            state["seen"].add(batch_index)
            state["frames"] += batch_frames
            if batch_index == num_batches - 1:
                expected_batch_indices = set(range(num_batches))
                if state["seen"] != expected_batch_indices:
                    raise AssertionError(
                        f"chunk {chunk_index} frame batches are incomplete: "
                        f"seen={sorted(state['seen'])} expected="
                        f"{sorted(expected_batch_indices)}"
                    )
                if state["frames"] != expected_frames:
                    raise AssertionError(
                        f"chunk {chunk_index} produced {state['frames']} frames, "
                        f"expected {expected_frames}"
                    )
                payload_complete_ns[chunk_index] = time.perf_counter_ns()

    expected_indices = list(range(total_chunks))
    if sorted(stats_by_chunk) != expected_indices:
        raise AssertionError("chunk_stats indices are not contiguous")
    if sorted(payload_complete_ns) != expected_indices:
        raise AssertionError("frame payload indices are not contiguous")

    measured_indices = list(range(args.warmup_chunks, total_chunks))
    measured_stats = [stats_by_chunk[index] for index in measured_indices]
    measured_frames = sum(int(stat["num_frames"]) for stat in measured_stats)
    expected_measured_frames = (
        args.measured_chunks * pixel_frames_per_latent * latent_frames_per_chunk
    )
    if measured_frames != expected_measured_frames:
        raise AssertionError(
            f"measured {measured_frames} frames, expected {expected_measured_frames}"
        )

    timing_fields = (
        "request_prepare_ms",
        "scheduler_forward_ms",
        "video_serialize_ms",
        "raw_payload_build_ms",
        "pace_wait_ms",
        "ws_write_ms",
        "chunk_total_ms",
    )
    server = {}
    for field in timing_fields:
        values = [float(stat[field]) for stat in measured_stats if field in stat]
        if values:
            server[field] = latency_summary(values)
            if field in {"scheduler_forward_ms", "chunk_total_ms"}:
                server[field.replace("_ms", "_fps_ratio_of_sums")] = measured_frames / (
                    sum(values) / 1000.0
                )

    previous_ns = payload_complete_ns[args.warmup_chunks - 1]
    interarrival_ms = []
    for index in measured_indices:
        completion_ns = payload_complete_ns[index]
        interarrival_ms.append((completion_ns - previous_ns) / 1e6)
        previous_ns = completion_ns
    client_window_s = sum(interarrival_ms) / 1000.0
    return {
        "schema_version": "minwm-realtime-throughput/v1",
        "profile_name": args.profile_name,
        "comparison_contract": {
            "case": case["id"],
            "action_type": contract["action_type"],
            "action_label": int(case["action_label"]),
            "seed": int(contract["seed"]),
            "size": f"{contract['width']}x{contract['height']}",
            "steps": 4,
            "guidance_scale": 0.0,
            "latent_frames_per_chunk": latent_frames_per_chunk,
            "generated_pixel_frames_per_steady_chunk": pixel_frames_per_latent
            * latent_frames_per_chunk,
            "kv_cache_num_frames": args.kv_cache_num_frames,
            "required_fixed_between_profiles": [
                "checkpoint bytes",
                "GPU model and count",
                "software image",
                "attention backend",
                "request payload",
            ],
        },
        "warmup_chunks": args.warmup_chunks,
        "measured_chunks": args.measured_chunks,
        "measured_frames": measured_frames,
        "server": server,
        "client": {
            "init_send_start_to_first_payload_complete_ms": (
                payload_complete_ns[0] - init_started_ns
            )
            / 1e6,
            "init_send_complete_to_first_payload_complete_ms": (
                payload_complete_ns[0] - init_completed_ns
            )
            / 1e6,
            "steady_payload_interarrival_ms": latency_summary(interarrival_ms),
            "steady_received_fps_ratio_of_sums": measured_frames / client_window_s,
            "steady_window_seconds": client_window_s,
        },
    }


async def async_main(args: argparse.Namespace) -> None:
    manifest = load_cases(args.cases)
    contract, case = validate_contract(manifest, args)
    result = await receive_run(args, contract, case)
    write_json(args.output, result)
    print(json.dumps(result["server"], indent=2, sort_keys=True))
    print(json.dumps(result["client"], indent=2, sort_keys=True))


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
