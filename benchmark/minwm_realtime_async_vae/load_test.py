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
        "num_frames": 1 + (total_chunks - 1) * 16,
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


def stage_values(
    trace_events: list[dict], *, min_chunk_index: int = 0
) -> dict[str, list[float]]:
    values: dict[str, dict[int, float]] = defaultdict(dict)
    for event in trace_events:
        name = event.get("event")
        chunk_index_value = event.get("chunk_index")
        if chunk_index_value is None:
            continue
        chunk_index = int(chunk_index_value)
        if chunk_index < min_chunk_index:
            continue
        if name == "server.model_denoise_complete":
            values["denoise_ms"][chunk_index] = float(
                event.get("cuda_ms") or event.get("duration_ms") or 0
            )
        elif name == "server.vae_encode_complete":
            values["vae_encode_ms"][chunk_index] = float(
                event.get("cuda_ms") or event.get("duration_ms") or 0
            )
        elif name == "server.vae_decode_complete":
            values["vae_decode_ms"][chunk_index] = float(
                event.get("cuda_ms") or event.get("duration_ms") or 0
            )
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
                    values[field][chunk_index] = float(event[field])
        elif name == "server.vae_denoise_overlap_complete":
            for field in (
                "overlap_with_next_denoise_ms",
                "overlap_ratio",
            ):
                if event.get(field) is not None:
                    values[field][chunk_index] = float(event[field])
    return {
        name: [by_chunk[index] for index in sorted(by_chunk)]
        for name, by_chunk in values.items()
    }


def record_action_latency(
    message: dict,
    *,
    first_frame_at: dict[int, float],
    action_sent_at: dict[int, float],
    action_latencies: list[float],
    min_chunk_index: int,
) -> None:
    chunk = int(message.get("chunk_index") or 0)
    sampled_event = int(message.get("event_id") or 0)
    if sampled_event <= 0 or chunk not in first_frame_at:
        return
    eligible = [event for event in action_sent_at if event <= sampled_event]
    if not eligible:
        return
    latest = max(eligible)
    if chunk >= min_chunk_index:
        action_latencies.append(
            round((first_frame_at[chunk] - action_sent_at[latest]) * 1000.0, 3)
        )
    for event_id in eligible:
        action_sent_at.pop(event_id, None)


def server_action_latencies(
    trace_events: list[dict], *, min_chunk_index: int = 0
) -> dict[str, list[float]]:
    received: dict[int, tuple[float, float]] = {}
    markers: dict[int, dict] = {}
    for event in trace_events:
        name = event.get("event")
        event_id_value = event.get("event_id")
        if name == "server.event_received" and event_id_value is not None:
            client_epoch_ms = event.get("client_sent_epoch_ms")
            server_elapsed_ms = event.get("server_elapsed_ms")
            if client_epoch_ms is not None and server_elapsed_ms is not None:
                received[int(event_id_value)] = (
                    float(client_epoch_ms),
                    float(server_elapsed_ms),
                )
            continue
        if name not in {
            "server.remote_first_frame_received",
            "server.output_send_start",
        }:
            continue
        chunk_value = event.get("chunk_index")
        if chunk_value is None:
            continue
        chunk_index = int(chunk_value)
        if chunk_index >= min_chunk_index:
            markers.setdefault(chunk_index, event)

    client_to_first_frame: list[float] = []
    ingress_to_first_frame: list[float] = []
    for marker in (markers[index] for index in sorted(markers)):
        marker_event_id = marker.get("event_id")
        marker_epoch_ms = marker.get("server_epoch_ms")
        marker_elapsed_ms = marker.get("server_elapsed_ms")
        if (
            marker_event_id is None
            or marker_epoch_ms is None
            or marker_elapsed_ms is None
        ):
            continue
        eligible = [event_id for event_id in received if event_id <= int(marker_event_id)]
        if not eligible:
            continue
        client_epoch_ms, received_elapsed_ms = received[max(eligible)]
        client_delta = float(marker_epoch_ms) - client_epoch_ms
        ingress_delta = float(marker_elapsed_ms) - received_elapsed_ms
        if client_delta >= 0:
            client_to_first_frame.append(round(client_delta, 3))
        if ingress_delta >= 0:
            ingress_to_first_frame.append(round(ingress_delta, 3))

    return {
        "action_to_server_first_frame_ms": client_to_first_frame,
        "action_ingress_to_server_first_frame_ms": ingress_to_first_frame,
    }


def aggregate_measurement_seconds(sessions: list[dict]) -> float:
    starts = [
        float(session["measured_started_at"])
        for session in sessions
        if session.get("measured_started_at") is not None
    ]
    completions = [
        float(session["measured_completed_at"])
        for session in sessions
        if session.get("measured_completed_at") is not None
    ]
    if not starts or not completions:
        return 0.0
    return max(0.0, max(completions) - min(starts))


async def stream_actions(
    websocket,
    *,
    action_sent_at: dict[int, float],
    stop: asyncio.Event,
    interval_s: float = 0.1,
) -> None:
    event_id = 1
    while not stop.is_set():
        sent_at = time.perf_counter()
        actions = ["w"] if event_id % 2 else ["a", "w"]
        try:
            await websocket.send(action_event(event_id, actions))
        except Exception:
            return
        action_sent_at[event_id] = sent_at
        event_id += 1
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except TimeoutError:
            pass


def iter_trace_events(message: dict) -> list[dict]:
    message_type = message.get("type")
    if message_type == "trace_event":
        trace = message.get("trace")
        return [dict(trace)] if isinstance(trace, dict) else []
    if message_type == "trace_events":
        traces = message.get("traces")
        if not isinstance(traces, list):
            return []
        return [dict(trace) for trace in traces if isinstance(trace, dict)]
    return []


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
    pending_raw_header = None
    measured_started_at: float | None = None
    measured_completed_at: float | None = None

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
        action_stop = asyncio.Event()
        action_task = asyncio.create_task(
            stream_actions(
                websocket,
                action_sent_at=action_sent_at,
                stop=action_stop,
            )
        )
        try:
            while True:
                try:
                    packed = await asyncio.wait_for(
                        websocket.recv(), timeout=args.timeout_s
                    )
                except websockets.ConnectionClosed:
                    break
                if not isinstance(packed, bytes):
                    continue
                if pending_raw_header is not None:
                    pending_raw_header = None
                    continue
                message = msgspec.msgpack.decode(packed)
                message_type = message.get("type")
                if message_type == "error":
                    raise RuntimeError(
                        message.get("content") or "realtime server error"
                    )
                if message_type in {"trace_event", "trace_events"}:
                    for trace in iter_trace_events(message):
                        trace_events.append(trace)
                        trace_chunk = trace.get("chunk_index")
                        if (
                            measured_started_at is None
                            and trace.get("event")
                            == "server.scheduler_forward_start"
                            and trace_chunk is not None
                            and int(trace_chunk) >= args.warmup_chunks
                        ):
                            measured_started_at = time.perf_counter()
                    continue
                if message_type == "frame_batch_header":
                    pending_raw_header = message
                if message_type in {"frame_batch", "frame_batch_header"}:
                    chunk = int(message.get("chunk_index") or 0)
                    observed_at = time.perf_counter()
                    first_frame_at.setdefault(chunk, observed_at)
                    if chunk >= args.warmup_chunks and measured_started_at is None:
                        measured_started_at = observed_at
                    record_action_latency(
                        message,
                        first_frame_at=first_frame_at,
                        action_sent_at=action_sent_at,
                        action_latencies=action_latencies,
                        min_chunk_index=args.warmup_chunks,
                    )
                    continue
                if message_type != "chunk_stats":
                    continue

                chunk = int(message["chunk_index"])
                observed_at = time.perf_counter()
                stats[chunk] = dict(message)
                if chunk >= args.warmup_chunks:
                    if measured_started_at is None:
                        measured_started_at = first_frame_at.get(chunk, observed_at)
                    measured_completed_at = observed_at
                record_action_latency(
                    message,
                    first_frame_at=first_frame_at,
                    action_sent_at=action_sent_at,
                    action_latencies=action_latencies,
                    min_chunk_index=args.warmup_chunks,
                )
        finally:
            action_stop.set()
            await action_task

    if len(stats) != total_chunks:
        raise RuntimeError(
            f"session closed after {len(stats)} of {total_chunks} chunks"
        )

    measured = [stats[index] for index in range(args.warmup_chunks, total_chunks)]
    chunk_total = [float(item["chunk_total_ms"]) for item in measured]
    frame_count = sum(int(item.get("num_frames") or 0) for item in measured)
    measured_seconds = (
        max(0.0, measured_completed_at - measured_started_at)
        if measured_started_at is not None and measured_completed_at is not None
        else 0.0
    )
    server_action = server_action_latencies(
        trace_events, min_chunk_index=args.warmup_chunks
    )
    return {
        "session_index": index,
        "trace_id": trace_id,
        "chunk_total_ms": chunk_total,
        "action_to_first_frame_ms": server_action[
            "action_to_server_first_frame_ms"
        ],
        "action_ingress_to_first_frame_ms": server_action[
            "action_ingress_to_server_first_frame_ms"
        ],
        "client_observed_action_to_first_frame_ms": action_latencies,
        "frames": frame_count,
        "measured_seconds": measured_seconds,
        "measured_started_at": measured_started_at,
        "measured_completed_at": measured_completed_at,
        "stage_values": stage_values(
            trace_events, min_chunk_index=args.warmup_chunks
        ),
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
    action_ingress = [
        value
        for session in sessions
        for value in session["action_ingress_to_first_frame_ms"]
    ]
    client_observed_action = [
        value
        for session in sessions
        for value in session["client_observed_action_to_first_frame_ms"]
    ]
    stages: dict[str, list[float]] = defaultdict(list)
    for session in sessions:
        for name, values in session["stage_values"].items():
            stages[name].extend(values)
    total_frames = sum(session["frames"] for session in sessions)
    wall_seconds = aggregate_measurement_seconds(sessions)
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
        "action_ingress_to_first_frame_ms": latency_summary(action_ingress),
        "client_observed_action_to_first_frame_ms": latency_summary(
            client_observed_action
        ),
        "aggregate_fps": total_frames / wall_seconds if wall_seconds else 0.0,
        "measurement_wall_seconds": wall_seconds,
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
