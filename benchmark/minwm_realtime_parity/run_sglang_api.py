#!/usr/bin/env python3
"""Run MinWM manifest cases through SGLang's realtime WebSocket API."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import zlib
from pathlib import Path

import msgspec.msgpack
import numpy as np
from common import (
    action_label_sequence,
    action_weights,
    is_realtime_trace_event,
    load_cases,
    materialize_first_frame,
    prompt_switch_event,
    resolve_case_contract,
    save_video,
    sha256_file,
    write_json,
)

RAW_RGB_CONTENT_TYPE = "application/x-raw-rgb"
RAW_RGB_DELTA_GZIP_CONTENT_TYPE = "application/x-raw-rgb-delta-gzip"
RAW_RGBA_DELTA_GZIP_CONTENT_TYPE = "application/x-raw-rgba-delta-gzip"


def restore_delta_gzip_raw_rgb_payload(
    payload: bytes,
    *,
    bytes_per_frame: int,
    num_frames: int,
    reference_frame: bytes | None = None,
) -> bytes:
    """Decode the wire format without importing the GPU-serving package."""
    if reference_frame is not None and len(reference_frame) != bytes_per_frame:
        raise ValueError("delta gzip reference frame size mismatch")
    restored = bytearray(zlib.decompress(payload, wbits=31))
    expected = bytes_per_frame * num_frames
    if len(restored) != expected:
        raise ValueError(
            f"delta gzip payload size mismatch: expected {expected}, got {len(restored)}"
        )
    previous = (
        np.frombuffer(reference_frame, dtype=np.uint8)
        if reference_frame is not None
        else None
    )
    for frame_idx in range(num_frames):
        current = np.frombuffer(
            restored,
            dtype=np.uint8,
            count=bytes_per_frame,
            offset=frame_idx * bytes_per_frame,
        )
        if previous is not None:
            current ^= previous
        previous = current
    return bytes(restored)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--results", required=True)
    parser.add_argument(
        "--ws-url", default="ws://127.0.0.1:30000/v1/realtime_video/generate"
    )
    parser.add_argument("--model", default="minwm")
    parser.add_argument("--case", action="append", dest="selected_cases")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--output-prefix",
        default="sglang",
        help="Artifact stem, for example sglang_bitwise or sglang_optimized.",
    )
    parser.add_argument(
        "--engine-name",
        default="sglang-minwm-realtime-api",
        help="Engine label written to the aggregate run record.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Discard this many complete runs of each case before measuring it.",
    )
    parser.add_argument("--sink-size", type=int)
    parser.add_argument("--kv-cache-num-frames", type=int)
    return parser.parse_args()


def decode_frames(header: dict, payload: bytes, previous: bytes | None):
    content_type = header["content_type"]
    bytes_per_frame = int(header["bytes_per_frame"])
    num_frames = int(header["num_frames"])
    if content_type in (
        RAW_RGB_DELTA_GZIP_CONTENT_TYPE,
        RAW_RGBA_DELTA_GZIP_CONTENT_TYPE,
    ):
        reference = (
            previous if header.get("delta_reference") == "previous-frame" else None
        )
        payload = restore_delta_gzip_raw_rgb_payload(
            payload,
            bytes_per_frame=bytes_per_frame,
            num_frames=num_frames,
            reference_frame=reference,
        )
    elif content_type != RAW_RGB_CONTENT_TYPE:
        raise ValueError(f"unsupported realtime content type: {content_type}")
    expected = bytes_per_frame * num_frames
    if len(payload) != expected:
        raise ValueError(f"frame payload has {len(payload)} bytes, expected {expected}")
    height = int(header["height"])
    width = int(header["width"])
    channels = int(header["channels"])
    frames = []
    for index in range(num_frames):
        start = index * bytes_per_frame
        frame = np.frombuffer(
            payload[start : start + bytes_per_frame], dtype=np.uint8
        ).reshape(height, width, channels)
        frames.append(frame[:, :, :3].copy())
    return frames


async def run_case(args, case, contract, first_frame: Path | None):
    import websockets

    if contract.get("action_output_format") == "primitive_float":
        action_condition = {
            "action_weights": [action_weights(case)]
            * int(contract["generated_pixel_frames"])
        }
    else:
        action_condition = {"action_labels": action_label_sequence(case, contract)}
    payload = {
        "type": "init",
        "generation_mode": "t2v" if first_frame is None else "i2v",
        "model": args.model,
        "prompt": case["prompt"],
        "size": f"{contract['width']}x{contract['height']}",
        "fps": int(contract["fps"]),
        "seed": int(contract["seed"]),
        "generator_device": "cuda",
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "max_chunks": int(contract["chunks"]),
        "realtime_output_format": "raw",
        "condition_inputs": action_condition,
    }
    if first_frame is None:
        payload["num_frames"] = int(contract["generated_pixel_frames"])
    else:
        payload["first_frame"] = first_frame.read_bytes()
    if args.sink_size is not None:
        payload["realtime_causal_sink_size"] = args.sink_size
    if args.kv_cache_num_frames is not None:
        payload["realtime_causal_kv_cache_num_frames"] = args.kv_cache_num_frames
    frames = []
    stats = []
    frame_event_ids = {}
    completed_chunks = set()
    payload_completed_ns = {}
    previous_frame = None
    queued_prompt_event = prompt_switch_event(case)
    async with websockets.connect(
        args.ws_url,
        max_size=None,
        ping_interval=None,
        open_timeout=args.timeout,
    ) as websocket:
        init_send_started_ns = time.perf_counter_ns()
        await websocket.send(msgspec.msgpack.encode(payload))
        prompt_event_sent_ns = None
        if queued_prompt_event is not None:
            # Queue while chunk 0 is executing. The MinWM adapter does not
            # sample prompt events for chunk 0, so this deterministically
            # targets chunk 1 without relying on a client-side sleep.
            await websocket.send(msgspec.msgpack.encode(queued_prompt_event))
            prompt_event_sent_ns = time.perf_counter_ns()
        while len(completed_chunks) < int(contract["chunks"]) or len(stats) < int(
            contract["chunks"]
        ):
            packed = await asyncio.wait_for(websocket.recv(), timeout=args.timeout)
            header = msgspec.msgpack.decode(packed)
            message_type = header.get("type")
            if is_realtime_trace_event(header):
                continue
            if message_type == "error":
                raise RuntimeError(header.get("content", "unknown realtime error"))
            if message_type == "chunk_stats":
                stats.append(header)
                continue
            if message_type == "frame_batch":
                frame_payload = header.pop("payload")
            elif message_type == "frame_batch_header":
                frame_payload = await asyncio.wait_for(
                    websocket.recv(), timeout=args.timeout
                )
            else:
                raise ValueError(f"unexpected realtime message: {header}")
            chunk_frames = decode_frames(header, frame_payload, previous_frame)
            frames.extend(chunk_frames)
            if chunk_frames:
                previous_frame = chunk_frames[-1].tobytes()
            if header.get("is_final_frame_batch", True):
                chunk_index = int(header["chunk_index"])
                completed_chunks.add(chunk_index)
                payload_completed_ns[chunk_index] = time.perf_counter_ns()
                frame_event_ids[chunk_index] = header.get("event_id")
    ordered_completions = [
        payload_completed_ns[index] for index in range(int(contract["chunks"]))
    ]
    timing = {
        "init_send_start_to_first_payload_complete_ms": (
            ordered_completions[0] - init_send_started_ns
        )
        / 1e6,
        "steady_payload_interarrival_ms": [
            (current - previous) / 1e6
            for previous, current in zip(
                ordered_completions[:-1], ordered_completions[1:]
            )
        ],
    }
    prompt_switch_observation = None
    if queued_prompt_event is not None:
        target_chunk = int(case["prompt_switch"]["target_chunk"])
        event_id = int(queued_prompt_event["event_id"])
        stats_by_chunk = {int(item["chunk_index"]): item for item in stats}
        observed_stats_event_id = stats_by_chunk[target_chunk].get("event_id")
        observed_frame_event_id = frame_event_ids[target_chunk]
        first_stats_chunk = next(
            (
                chunk_index
                for chunk_index in range(int(contract["chunks"]))
                if stats_by_chunk[chunk_index].get("event_id") == event_id
            ),
            None,
        )
        first_frame_chunk = next(
            (
                chunk_index
                for chunk_index in range(int(contract["chunks"]))
                if frame_event_ids[chunk_index] == event_id
            ),
            None,
        )
        prompt_switch_observation = {
            "event": queued_prompt_event,
            "target_chunk": target_chunk,
            "sent_after_init_ms": (
                (prompt_event_sent_ns - init_send_started_ns) / 1e6
                if prompt_event_sent_ns is not None
                else None
            ),
            "stats_event_id_at_target": observed_stats_event_id,
            "frame_event_id_at_target": observed_frame_event_id,
            "first_stats_chunk_with_event": first_stats_chunk,
            "first_frame_chunk_with_event": first_frame_chunk,
        }
        if first_stats_chunk != target_chunk or first_frame_chunk != target_chunk:
            raise AssertionError(
                f"{case['id']}: prompt event {event_id} first affected stats/frame "
                f"chunks {first_stats_chunk}/{first_frame_chunk}, expected "
                f"{target_chunk}"
            )
    return (
        np.stack(frames),
        stats,
        payload,
        timing,
        prompt_switch_observation,
    )


async def async_main(args) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.output_prefix):
        raise ValueError("--output-prefix must be a safe filename stem")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs must be non-negative")
    if args.sink_size is not None and args.sink_size < 0:
        raise ValueError("--sink-size must be non-negative")
    if args.kv_cache_num_frames is not None and args.kv_cache_num_frames <= 0:
        raise ValueError("--kv-cache-num-frames must be positive")
    if (
        args.sink_size is not None
        and args.kv_cache_num_frames is not None
        and args.sink_size >= args.kv_cache_num_frames
    ):
        raise ValueError("--sink-size must be smaller than --kv-cache-num-frames")
    manifest = load_cases(args.cases)
    contract = manifest["contract"]
    selected = set(args.selected_cases or [])
    cases = [
        case for case in manifest["cases"] if not selected or case["id"] in selected
    ]
    known_case_ids = {case["id"] for case in manifest["cases"]}
    if selected - known_case_ids:
        raise ValueError(f"unknown case ids: {sorted(selected - known_case_ids)}")
    results = Path(args.results).resolve()
    inputs = results / "inputs"
    records = []
    for case in cases:
        case_contract = resolve_case_contract(case, contract)
        case_dir = results / "cases" / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        first_frame = (
            materialize_first_frame(case, inputs)
            if case.get("first_frame") is not None
            else None
        )
        warmups = []
        for warmup_index in range(args.warmup_runs):
            (
                warmup_frames,
                warmup_stats,
                _,
                warmup_timing,
                warmup_prompt_switch,
            ) = await run_case(args, case, case_contract, first_frame)
            warmups.append(
                {
                    "frames": int(warmup_frames.shape[0]),
                    "chunk_stats": warmup_stats,
                    "client_timing": warmup_timing,
                    "prompt_switch": warmup_prompt_switch,
                }
            )
            print(
                json.dumps(
                    {
                        "id": case["id"],
                        "warmup": warmup_index + 1,
                        "warmup_runs": args.warmup_runs,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        frames, stats, request, timing, prompt_switch = await run_case(
            args, case, case_contract, first_frame
        )
        expected_frames = int(case_contract["reference_pixel_frames"]) + int(
            case_contract["generated_pixel_frames"]
        )
        expected_shape = (
            expected_frames,
            int(case_contract["height"]),
            int(case_contract["width"]),
            3,
        )
        if frames.shape != expected_shape:
            raise AssertionError(
                f"{case['id']}: video shape {frames.shape} != {expected_shape}"
            )
        np.save(case_dir / f"{args.output_prefix}.npy", frames, allow_pickle=False)
        save_video(case_dir / f"{args.output_prefix}.mp4", frames, int(contract["fps"]))
        record = {
            "id": case["id"],
            "frames": int(frames.shape[0]),
            "warmup_runs": args.warmup_runs,
            "warmups": warmups,
            "video_sha256": sha256_file(case_dir / f"{args.output_prefix}.mp4"),
            "frames_sha256": sha256_file(case_dir / f"{args.output_prefix}.npy"),
            "chunk_stats": stats,
            "client_timing": timing,
            "prompt_switch": prompt_switch,
            "contract": case_contract,
            "request": {
                key: value
                for key, value in request.items()
                if key not in {"first_frame"}
            },
        }
        write_json(case_dir / f"{args.output_prefix}.json", record)
        records.append(record)
        print(json.dumps({"id": case["id"], "frames": len(frames)}, sort_keys=True))
    write_json(
        results / f"{args.output_prefix}_run.json",
        {
            "engine": args.engine_name,
            "output_prefix": args.output_prefix,
            "warmup_runs": args.warmup_runs,
            "ws_url": args.ws_url,
            "cases": records,
        },
    )


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
