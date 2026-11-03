# SPDX-License-Identifier: Apache-2.0
"""Client-side latency of a realtime video session (``/v1/realtime_video/generate``).

The server already reports per-chunk compute (``chunk_stats``). This benchmark
measures what a viewer sees on the client side of the WebSocket:

- time to first frame: from sending ``init`` to the first non-empty frame batch;
- event latency: from sending an event with ``event_id`` N to the first frame
  batch whose header echoes ``event_id`` N (this includes waiting for the chunk
  that was already running when the event arrived);
- delivered fps after the first frame, chunk inter-arrival gaps and stalls
  (gaps longer than ``--stall-factor`` times the median gap);
- the median server ``scheduler_forward_ms`` and ``chunk_total_ms``.

Sessions run one after another, since a realtime server admits one active
session. The first ``--warmup-sessions`` sessions are discarded (compilation,
cache allocation).

Usage::

    sglang serve --model-path robbyant/lingbot-world-fast-diffusers \\
        --pipeline-class-name LingBotWorldCausalDMDPipeline --port 30000
    python -m sglang.multimodal_gen.benchmarks.bench_realtime \\
        --init init.json --first-frame first_frame.jpg \\
        --event '{"kind": "camera_actions", "payload": [["d"], ["d"], ["d"]]}' \\
        --event-every 2 --max-chunks 12 --sessions 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import msgspec
import numpy as np
import websockets

FRAME_BATCH_TYPES = ("frame_batch", "frame_batch_header")


@dataclass
class SessionTimeline:
    """Client-side times (seconds since ``init`` was sent) for one session."""

    frame_batches: list[tuple[float, int, int | None, bool]] = field(
        default_factory=list
    )  # (t, num_frames, event_id, is_final_frame_batch)
    events_sent: dict[int, float] = field(default_factory=dict)
    chunk_stats: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    closed_at: float | None = None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values), q))


def summarize_session(timeline: SessionTimeline, stall_factor: float) -> dict:
    """Turn one session's client-side timeline into latency metrics (ms)."""
    batches = timeline.frame_batches
    first = next((t for t, n, _, _ in batches if n > 0), None)
    frames_after_first = sum(
        n for t, n, _, _ in batches if first is not None and t > first
    )
    last = batches[-1][0] if batches else None

    event_latency_ms = []
    seen = set()
    for t, _, event_id, _ in batches:
        if event_id is None or event_id in seen or event_id not in timeline.events_sent:
            continue
        seen.add(event_id)
        event_latency_ms.append((t - timeline.events_sent[event_id]) * 1000)

    chunk_ends = [t for t, _, _, final in batches if final]
    gaps_ms = [(b - a) * 1000 for a, b in zip(chunk_ends, chunk_ends[1:])]
    median_gap = statistics.median(gaps_ms) if gaps_ms else None
    stalls = [g for g in gaps_ms if median_gap and g > stall_factor * median_gap]

    def _median_stat(name: str) -> float | None:
        values = [
            float(s[name]) for s in timeline.chunk_stats if s.get(name) is not None
        ]
        return statistics.median(values) if values else None

    return {
        "time_to_first_frame_ms": first * 1000 if first is not None else None,
        "frames": sum(n for _, n, _, _ in batches),
        "chunks": len(chunk_ends),
        "delivered_fps": (
            frames_after_first / (last - first)
            if first is not None and last is not None and last > first
            else None
        ),
        "event_latency_ms": event_latency_ms,
        "events_sent": len(timeline.events_sent),
        "events_unanswered": len(timeline.events_sent) - len(event_latency_ms),
        "chunk_gap_ms_median": median_gap,
        "chunk_gap_ms_max": max(gaps_ms) if gaps_ms else None,
        "stalls": len(stalls),
        "server_scheduler_forward_ms_median": _median_stat("scheduler_forward_ms"),
        "server_chunk_total_ms_median": _median_stat("chunk_total_ms"),
        "errors": timeline.errors,
    }


async def run_session(
    url: str,
    init: dict,
    event_template: dict | None,
    event_every: int,
    recv_timeout_s: float,
) -> SessionTimeline:
    timeline = SessionTimeline()
    async with websockets.connect(
        f"{url}/v1/realtime_video/generate", max_size=None, ping_interval=None
    ) as ws:
        start = time.perf_counter()
        await ws.send(msgspec.msgpack.encode(init))
        chunks_done = 0
        next_event_id = 1
        while True:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=recv_timeout_s)
            except websockets.ConnectionClosed:
                break
            except asyncio.TimeoutError:
                timeline.errors.append(f"no message for {recv_timeout_s:.0f} s")
                break
            message = msgspec.msgpack.decode(raw)
            kind = message.get("type")
            if kind == "error":
                timeline.errors.append(str(message.get("content")))
                continue
            if kind == "chunk_stats":
                timeline.chunk_stats.append(message)
                continue
            if kind not in FRAME_BATCH_TYPES:
                continue
            if kind == "frame_batch_header":
                await ws.recv()  # the raw payload always follows its header
            now = time.perf_counter() - start
            final = bool(message.get("is_final_frame_batch", True))
            timeline.frame_batches.append(
                (
                    now,
                    int(message.get("num_frames") or 0),
                    message.get("event_id"),
                    final,
                )
            )
            if not final:
                continue
            chunks_done += 1
            if event_template is not None and chunks_done % event_every == 0:
                event = {"type": "event", **event_template, "event_id": next_event_id}
                sent_at = time.perf_counter() - start
                try:
                    await ws.send(msgspec.msgpack.encode(event))
                except websockets.ConnectionClosed:
                    break  # the server closed after its last chunk
                timeline.events_sent[next_event_id] = sent_at
                next_event_id += 1
        timeline.closed_at = time.perf_counter() - start
    return timeline


def _aggregate(sessions: list[dict]) -> dict:
    def _values(key: str) -> list[float]:
        return [s[key] for s in sessions if s.get(key) is not None]

    event_latency = [v for s in sessions for v in s["event_latency_ms"]]
    return {
        "sessions": len(sessions),
        "time_to_first_frame_ms_median": _percentile(
            _values("time_to_first_frame_ms"), 50
        ),
        "event_latency_ms_median": _percentile(event_latency, 50),
        "event_latency_ms_p90": _percentile(event_latency, 90),
        "events": len(event_latency),
        "events_unanswered": sum(s["events_unanswered"] for s in sessions),
        "delivered_fps_median": _percentile(_values("delivered_fps"), 50),
        "stalls": sum(s["stalls"] for s in sessions),
        "server_chunk_total_ms_median": _percentile(
            _values("server_chunk_total_ms_median"), 50
        ),
        "errors": [e for s in sessions for e in s["errors"]],
    }


async def _run(args: argparse.Namespace) -> dict:
    init = {"type": "init", "realtime_output_format": "raw"}
    if args.init:
        with open(args.init) as f:
            init.update(json.load(f))
    if args.first_frame:
        with open(args.first_frame, "rb") as f:
            init["first_frame"] = f.read()
    if args.max_chunks:
        init["max_chunks"] = args.max_chunks
    event_template = json.loads(args.event) if args.event else None
    measured = []
    for index in range(args.warmup_sessions + args.sessions):
        timeline = await run_session(
            args.url, init, event_template, args.event_every, args.recv_timeout
        )
        if index >= args.warmup_sessions:
            measured.append(summarize_session(timeline, args.stall_factor))
    return {"summary": _aggregate(measured), "per_session": measured}


def _fmt(value: float | None, unit: str = "") -> str:
    return "-" if value is None else f"{value:.0f}{unit}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="ws://127.0.0.1:30000")
    parser.add_argument("--init", help="JSON file with init fields (prompt, size, ...)")
    parser.add_argument("--first-frame", help="image file sent as init.first_frame")
    parser.add_argument("--max-chunks", type=int, default=12)
    parser.add_argument(
        "--event",
        help='event sent after every --event-every chunks, e.g. \'{"kind": "camera_actions", "payload": [["d"]]}\'',
    )
    parser.add_argument("--event-every", type=int, default=2)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--warmup-sessions", type=int, default=1)
    parser.add_argument("--stall-factor", type=float, default=2.0)
    parser.add_argument("--recv-timeout", type=float, default=600.0)
    parser.add_argument("--json", action="store_true", help="print the raw result")
    args = parser.parse_args()
    result = asyncio.run(_run(args))
    if args.json:
        print(json.dumps(result, indent=2))
        return
    s = result["summary"]
    print(
        f"{s['sessions']} sessions: first frame {_fmt(s['time_to_first_frame_ms_median'], ' ms')} median, "
        f"event to frame {_fmt(s['event_latency_ms_median'], ' ms')} median / "
        f"{_fmt(s['event_latency_ms_p90'], ' ms')} p90 ({s['events']} events, "
        f"{s['events_unanswered']} unanswered), "
        f"{s['delivered_fps_median'] or 0:.1f} fps delivered, {s['stalls']} stalls, "
        f"server chunk {_fmt(s['server_chunk_total_ms_median'], ' ms')} median"
    )
    for error in s["errors"]:
        print(f"  error: {error}")


if __name__ == "__main__":
    main()
