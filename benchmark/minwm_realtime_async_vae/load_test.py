#!/usr/bin/env python3
"""Concurrent WebSocket load generator for MinWM realtime sync/async profiles."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import msgspec.msgpack

from summarize import latency_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ws-url", required=True)
    parser.add_argument("--profile", choices=("sync", "async"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--concurrency", default="1,2,4,8")
    parser.add_argument("--warmup-chunks", type=int, default=2)
    parser.add_argument("--measured-chunks", type=int, default=6)
    parser.add_argument("--model", default="/work/model")
    parser.add_argument("--prompt", default="A cinematic forward-moving landscape")
    parser.add_argument("--size", default="832x480")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--hardware-json", type=Path)
    return parser.parse_args()


def with_identity(url: str, *, user_id: str, trace_id: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(user_id=user_id, trace_id=trace_id)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def action_event(event_id: int, actions: list[str]) -> bytes:
    now_ms = time.time() * 1000.0
    return msgspec.msgpack.encode(
        {
            "type": "event",
            "kind": "camera_actions",
            "event_id": event_id,
            "client_sent_epoch_ms": now_ms,
            "payload": {
                "mode": "state",
                "transitions": [
                    {"actions": actions, "client_ts_ms": int(now_ms)}
                ],
            },
        }
    )


def init_request(args: argparse.Namespace, *, total_chunks: int, trace_id: str) -> dict:
    return {
        "type": "init",
        "generation_mode": "t2v",
        "model": args.model,
        "prompt": args.prompt,
        "size": args.size,
        "fps": args.fps,
        "num_frames": max(121, 1 + total_chunks * 16),
        "seed": 42,
        "generator_device": "cuda",
        "num_inference_steps": 4,
        "guidance_scale": 0.0,
        "max_chunks": total_chunks,
        "realtime_output_format": "webp",
        "realtime_preview_max_width": 560,
        "realtime_output_pacing": False,
        "output_compression": 55,
        "trace_id": trace_id,
    }


def stage_values(trace_events: list[dict]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for event in trace_events:
        name = event.get("event")
        if name == "server.model_denoise_complete":
            values["denoise_ms"].append(float(event.get("cuda_ms") or event.get("duration_ms") or 0))
        elif name == "server.remote_vae_complete":
            for field in (
                "vae_queue_wait_ms",
                "vae_decode_ms",
                "frame_encode_ms",
                "latent_serialize_ms",
                "latent_send_ms",
                "vae_credit_wait_ms",
                "first_frame_ms",
                "overlap_with_next_denoise_ms",
                "overlap_ratio",
            ):
                if event.get(field) is not None:
                    values[field].append(float(event[field]))
    return dict(values)


async def run_session(args: argparse.Namespace, concurrency: int, index: int) -> dict:
    import websockets

    total_chunks = args.warmup_chunks + args.measured_chunks
    trace_id = uuid4().hex
    url = with_identity(
        args.ws_url,
        user_id=f"load-{concurrency}-{index}-{uuid4().hex}",
        trace_id=trace_id,
    )
    stats: dict[int, dict] = {}
    first_frame_at: dict[int, float] = {}
    trace_events: list[dict] = []
    action_sent_at: dict[int, float] = {}
    action_latencies: list[float] = []
    next_event_id = 1
    pending_raw_header = None

    async with websockets.connect(
        url,
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
        open_timeout=args.timeout_s,
    ) as websocket:
        await websocket.send(
            msgspec.msgpack.encode(
                init_request(args, total_chunks=total_chunks, trace_id=trace_id)
            )
        )
        while len(stats) < total_chunks:
            packed = await asyncio.wait_for(websocket.recv(), timeout=args.timeout_s)
            if not isinstance(packed, bytes):
                continue
            if pending_raw_header is not None:
                pending_raw_header = None
                continue
            message = msgspec.msgpack.decode(packed)
            message_type = message.get("type")
            if message_type == "error":
                raise RuntimeError(message.get("content") or "realtime server error")
            if message_type == "trace_event":
                trace_events.append(dict(message.get("trace") or {}))
                continue
            if message_type == "frame_batch_header":
                pending_raw_header = message
            if message_type in {"frame_batch", "frame_batch_header"}:
                chunk = int(message.get("chunk_index") or 0)
                first_frame_at.setdefault(chunk, time.perf_counter())
                sampled_event = int(message.get("event_id") or 0)
                eligible = [event for event in action_sent_at if event <= sampled_event]
                if eligible:
                    latest = max(eligible)
                    action_latencies.append(
                        (first_frame_at[chunk] - action_sent_at[latest]) * 1000.0
                    )
                    for event_id in eligible:
                        action_sent_at.pop(event_id, None)
                continue
            if message_type != "chunk_stats":
                continue

            chunk = int(message["chunk_index"])
            stats[chunk] = dict(message)
            if chunk >= args.warmup_chunks - 1 and chunk < total_chunks - 1:
                actions = ["w"] if next_event_id % 2 else ["a", "w"]
                action_sent_at[next_event_id] = time.perf_counter()
                await websocket.send(action_event(next_event_id, actions))
                next_event_id += 1

    measured = [stats[index] for index in range(args.warmup_chunks, total_chunks)]
    chunk_total = [float(item["chunk_total_ms"]) for item in measured]
    frame_count = sum(int(item.get("num_frames") or 0) for item in measured)
    measured_seconds = sum(chunk_total) / 1000.0
    return {
        "session_index": index,
        "trace_id": trace_id,
        "chunk_total_ms": chunk_total,
        "action_to_first_frame_ms": action_latencies,
        "frames": frame_count,
        "measured_seconds": measured_seconds,
        "stage_values": stage_values(trace_events),
    }


async def run_level(args: argparse.Namespace, concurrency: int) -> dict:
    results = await asyncio.gather(
        *(run_session(args, concurrency, index) for index in range(concurrency)),
        return_exceptions=True,
    )
    sessions = [result for result in results if isinstance(result, dict)]
    errors = [str(result) for result in results if isinstance(result, BaseException)]
    chunks = [value for session in sessions for value in session["chunk_total_ms"]]
    action = [
        value for session in sessions for value in session["action_to_first_frame_ms"]
    ]
    stages: dict[str, list[float]] = defaultdict(list)
    for session in sessions:
        for name, values in session["stage_values"].items():
            stages[name].extend(values)
    total_frames = sum(session["frames"] for session in sessions)
    wall_seconds = max(
        (session["measured_seconds"] for session in sessions), default=0.0
    )
    session_fps = [
        session["frames"] / session["measured_seconds"]
        for session in sessions
        if session["measured_seconds"]
    ]
    return {
        "concurrency": concurrency,
        "successful_sessions": len(sessions),
        "errors": errors,
        "error_rate": len(errors) / concurrency,
        "chunk_total_ms": latency_summary(chunks),
        "action_to_first_frame_ms": latency_summary(action),
        "aggregate_fps": total_frames / wall_seconds if wall_seconds else 0.0,
        "per_session_fps": latency_summary(session_fps),
        "min_session_fps": min(session_fps, default=0.0),
        "stage_ms": {name: latency_summary(values) for name, values in stages.items()},
    }


async def async_main(args: argparse.Namespace) -> None:
    levels = [int(value) for value in args.concurrency.split(",") if value.strip()]
    if not levels or any(value < 1 for value in levels):
        raise ValueError("concurrency levels must be positive")
    runs = []
    for concurrency in levels:
        run = await run_level(args, concurrency)
        runs.append(run)
        print(json.dumps(run, indent=2, sort_keys=True))
    output = {
        "schema_version": "minwm-realtime-load/v1",
        "profile": args.profile,
        "request": {
            "model": args.model,
            "size": args.size,
            "fps": args.fps,
            "steps": 4,
            "warmup_chunks": args.warmup_chunks,
            "measured_chunks": args.measured_chunks,
        },
        "hardware": (
            json.loads(args.hardware_json.read_text()) if args.hardware_json else {}
        ),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")


def main() -> None:
    asyncio.run(async_main(parse_args()))


if __name__ == "__main__":
    main()
