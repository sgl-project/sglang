#!/usr/bin/env python3
"""Run the ten MinWM cases through SGLang's realtime WebSocket API."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import msgspec.msgpack
import numpy as np

from common import (
    action_weights,
    load_cases,
    materialize_first_frame,
    save_video,
    sha256_file,
    write_json,
)
from sglang.multimodal_gen.runtime.utils.realtime_video import (
    RAW_RGB_CONTENT_TYPE,
    RAW_RGB_DELTA_GZIP_CONTENT_TYPE,
    RAW_RGBA_DELTA_GZIP_CONTENT_TYPE,
    restore_delta_gzip_raw_rgb_payload,
)


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


async def run_case(args, case, contract, first_frame: Path):
    import websockets

    if contract.get("action_output_format") == "primitive_float":
        action_condition = {
            "action_weights": [action_weights(case)]
            * int(contract["generated_pixel_frames"])
        }
    else:
        action_condition = {
            "action_labels": [int(case["action_label"])]
            * int(contract["generated_latent_frames"])
        }
    payload = {
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
        "max_chunks": int(contract["chunks"]),
        "realtime_output_format": "raw",
        "condition_inputs": action_condition,
    }
    frames = []
    stats = []
    completed_chunks = set()
    payload_completed_ns = {}
    previous_frame = None
    async with websockets.connect(
        args.ws_url,
        max_size=None,
        ping_interval=None,
        open_timeout=args.timeout,
    ) as websocket:
        init_send_started_ns = time.perf_counter_ns()
        await websocket.send(msgspec.msgpack.encode(payload))
        while len(completed_chunks) < int(contract["chunks"]) or len(stats) < int(
            contract["chunks"]
        ):
            packed = await asyncio.wait_for(websocket.recv(), timeout=args.timeout)
            header = msgspec.msgpack.decode(packed)
            message_type = header.get("type")
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
    return np.stack(frames), stats, payload, timing


async def async_main(args) -> None:
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
        case_dir = results / "cases" / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        first_frame = materialize_first_frame(case, inputs)
        frames, stats, request, timing = await run_case(
            args, case, contract, first_frame
        )
        expected_frames = int(contract["reference_pixel_frames"]) + int(
            contract["generated_pixel_frames"]
        )
        expected_shape = (
            expected_frames,
            int(contract["height"]),
            int(contract["width"]),
            3,
        )
        if frames.shape != expected_shape:
            raise AssertionError(
                f"{case['id']}: video shape {frames.shape} != {expected_shape}"
            )
        np.save(case_dir / "sglang.npy", frames, allow_pickle=False)
        save_video(case_dir / "sglang.mp4", frames, int(contract["fps"]))
        record = {
            "id": case["id"],
            "frames": int(frames.shape[0]),
            "video_sha256": sha256_file(case_dir / "sglang.mp4"),
            "frames_sha256": sha256_file(case_dir / "sglang.npy"),
            "chunk_stats": stats,
            "client_timing": timing,
            "request": {
                key: value
                for key, value in request.items()
                if key not in {"first_frame"}
            },
        }
        write_json(case_dir / "sglang.json", record)
        records.append(record)
        print(json.dumps({"id": case["id"], "frames": len(frames)}, sort_keys=True))
    write_json(
        results / "sglang_run.json",
        {
            "engine": "sglang-minwm-realtime-api",
            "ws_url": args.ws_url,
            "cases": records,
        },
    )


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
