#!/usr/bin/env python3
"""Exercise disconnect, reconnect, and slow-consumer behavior through Gateway."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

import msgspec.msgpack


def _url_with_identity(url: str, *, user_id: str, trace_id: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update(user_id=user_id, trace_id=trace_id)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def _init(model: str, *, chunks: int) -> bytes:
    return msgspec.msgpack.encode(
        {
            "type": "init",
            "generation_mode": "t2v",
            "model": model,
            "prompt": "A smooth forward camera move through a mountain valley",
            "size": "832x480",
            "fps": 24,
            "num_frames": 1 + (chunks - 1) * 16,
            "max_chunks": chunks,
            "seed": 42,
            "generator_device": "cuda",
            "num_inference_steps": 4,
            "guidance_scale": 0.0,
            "realtime_output_format": "webp",
            "realtime_preview_max_width": 560,
            "realtime_output_pacing": False,
            "output_compression": 55,
        }
    )


async def _consume(websocket, *, target_chunks: int, timeout_s: float) -> int:
    import websockets

    completed: set[int] = set()
    pending_header: dict | None = None
    deadline = time.monotonic() + timeout_s
    while len(completed) < target_chunks and time.monotonic() < deadline:
        try:
            packed = await asyncio.wait_for(
                websocket.recv(), timeout=max(0.1, deadline - time.monotonic())
            )
        except websockets.ConnectionClosed:
            break
        if not isinstance(packed, bytes):
            continue
        if pending_header is not None:
            if pending_header.get("is_final_frame_batch") is True:
                completed.add(int(pending_header["chunk_index"]))
            pending_header = None
            continue
        message = msgspec.msgpack.decode(packed)
        if message.get("type") == "error":
            raise RuntimeError(message.get("content") or "realtime server error")
        if message.get("type") == "media_chunk_complete":
            completed.add(int(message["chunk_index"]))
        elif message.get("type") == "frame_batch_header":
            pending_header = message
        elif (
            message.get("type") == "frame_batch"
            and message.get("is_final_frame_batch") is True
        ):
            completed.add(int(message["chunk_index"]))
    return len(completed)


async def _session(
    ws_url: str,
    model: str,
    *,
    user_id: str,
    chunks: int,
    stop_after_chunks: int | None = None,
    read_pause_s: float = 0.0,
) -> dict:
    import websockets

    trace_id = uuid4().hex
    started = time.perf_counter()
    completed = 0
    close_code = None
    outcome = "completed"
    try:
        async with websockets.connect(
            _url_with_identity(ws_url, user_id=user_id, trace_id=trace_id),
            max_size=None,
            max_queue=1,
            open_timeout=30,
            close_timeout=3,
        ) as websocket:
            await websocket.send(_init(model, chunks=chunks))
            if read_pause_s:
                await asyncio.sleep(read_pause_s)
            target = stop_after_chunks or chunks
            completed = await _consume(
                websocket, target_chunks=target, timeout_s=180
            )
            if stop_after_chunks is not None:
                outcome = "client_disconnected"
                await websocket.close(code=1000, reason="fault probe disconnect")
            elif completed != chunks:
                outcome = "server_closed_slow_consumer"
            close_code = websocket.close_code
    except websockets.ConnectionClosed as exc:
        close_code = exc.code
        outcome = "server_closed_slow_consumer"
    return {
        "trace_id": trace_id,
        "outcome": outcome,
        "completed_chunks": completed,
        "requested_chunks": chunks,
        "close_code": close_code,
        "elapsed_s": round(time.perf_counter() - started, 3),
    }


async def _reconnect_with_backoff(
    ws_url: str,
    model: str,
    *,
    user_id: str,
    chunks: int,
    timeout_s: float = 15.0,
) -> dict:
    started = time.perf_counter()
    attempts = 0
    while True:
        attempts += 1
        try:
            result = await _session(
                ws_url, model, user_id=user_id, chunks=chunks
            )
            result["admission_attempts"] = attempts
            result["admission_recovery_ms"] = round(
                (time.perf_counter() - started) * 1000.0, 3
            )
            return result
        except RuntimeError as exc:
            if "USER_SESSION_LIMIT" not in str(exc):
                raise
            if time.perf_counter() - started >= timeout_s:
                raise RuntimeError(
                    "same-user reconnect did not recover before deadline"
                ) from exc
            await asyncio.sleep(min(1.0, 0.1 * (2 ** min(attempts - 1, 4))))


async def _run(args: argparse.Namespace) -> dict:
    user_id = f"fault-{uuid4().hex}"
    disconnected = await _session(
        args.ws_url,
        args.model,
        user_id=user_id,
        chunks=8,
        stop_after_chunks=1,
    )
    reconnected = await _reconnect_with_backoff(
        args.ws_url, args.model, user_id=user_id, chunks=4
    )
    slow = await _session(
        args.ws_url,
        args.model,
        user_id=f"slow-{uuid4().hex}",
        chunks=12,
        read_pause_s=args.slow_pause_s,
    )
    post_slow = await _session(
        args.ws_url,
        args.model,
        user_id=f"post-slow-{uuid4().hex}",
        chunks=4,
    )
    if disconnected["completed_chunks"] != 1:
        raise RuntimeError("disconnect probe did not receive its first chunk")
    if reconnected["completed_chunks"] != 4:
        raise RuntimeError("same-user reconnect did not complete")
    if post_slow["completed_chunks"] != 4:
        raise RuntimeError("service did not recover after slow consumer")
    return {
        "schema_version": "minwm-realtime-fault-probe/v1",
        "disconnect": disconnected,
        "reconnect": reconnected,
        "slow_consumer": slow,
        "post_slow": post_slow,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ws-url", required=True)
    parser.add_argument("--model", default="/work/model")
    parser.add_argument("--slow-pause-s", type=float, default=8.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = asyncio.run(_run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
