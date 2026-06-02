# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio
import msgspec.msgpack
import numpy as np
import pytest
from openai import Client

from sglang.multimodal_gen.runtime.utils.realtime_video import (
    RAW_RGB_CONTENT_TYPE,
    RAW_RGB_DELTA_GZIP_CONTENT_TYPE,
    RAW_RGBA_DELTA_GZIP_CONTENT_TYPE,
    restore_delta_gzip_raw_rgb_payload,
)
from sglang.multimodal_gen.test.server.testcase_configs import DiffusionSamplingParams
from sglang.multimodal_gen.test.test_utils import is_image_url

_REALTIME_WS_TIMEOUT_SECS = float(
    os.environ.get("SGLANG_TEST_REALTIME_WS_TIMEOUT_SECS", "1200")
)


@dataclass(frozen=True)
class RealtimeChunkStats:
    chunk_index: int
    request_id: str | None
    content_type: str
    num_frames: int
    raw_bytes: int
    ws_payload_bytes: int
    request_prepare_ms: float
    scheduler_forward_ms: float
    raw_payload_build_ms: float
    raw_write_ms: float
    ws_write_ms: float
    chunk_total_ms: float


@dataclass(frozen=True)
class RealtimeCollectionResult:
    frames: list[np.ndarray]
    chunk_stats: list[RealtimeChunkStats]


_REALTIME_CHUNK_STATS_BY_CASE: dict[str, list[RealtimeChunkStats]] = {}
_REALTIME_KEY_FRAMES_BY_CASE: dict[str, list[np.ndarray]] = {}


def realtime_ws_url(client: Client) -> str:
    base_url = str(client.base_url).rstrip("/")
    if base_url.startswith("https://"):
        return "wss://" + base_url[len("https://") :] + "/realtime_video/generate"
    if base_url.startswith("http://"):
        return "ws://" + base_url[len("http://") :] + "/realtime_video/generate"
    raise ValueError(f"Unsupported realtime client base_url: {base_url}")


def prepare_realtime_first_frame(
    image_path: Path | str | list[Path | str] | None,
) -> bytes | str | None:
    if image_path is None:
        return None
    if isinstance(image_path, list):
        image_path = image_path[0]
    if isinstance(image_path, str) and is_image_url(image_path):
        return image_path
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Realtime first frame file missing: {path}")
    return path.read_bytes()


def build_realtime_init_payload(
    *,
    model_path: str,
    sampling_params: DiffusionSamplingParams,
    output_size: str,
    first_frame: bytes | str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": "init",
        "model": model_path,
        "prompt": sampling_params.prompt,
        "size": output_size,
        "seconds": sampling_params.seconds,
        "first_frame": first_frame,
    }
    optional_fields = {
        "fps": sampling_params.fps,
        "num_frames": sampling_params.num_frames,
        "realtime_output_format": sampling_params.realtime_output_format,
    }
    payload.update({k: v for k, v in optional_fields.items() if v is not None})
    payload.update(dict(sampling_params.extras))
    return {k: v for k, v in payload.items() if v is not None}


def build_realtime_event_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = dict(event)
    payload.pop("after_chunk", None)
    payload.setdefault("type", "event")
    if "kind" not in payload:
        raise ValueError("realtime event config must include kind")
    return payload


def parse_realtime_chunk_stats(header: dict[str, Any]) -> RealtimeChunkStats:
    if header.get("type") != "chunk_stats":
        raise ValueError(f"Unexpected realtime chunk stats message: {header}")
    return RealtimeChunkStats(
        chunk_index=int(header["chunk_index"]),
        request_id=header.get("request_id"),
        content_type=str(header.get("content_type", "")),
        num_frames=int(header.get("num_frames", 0)),
        raw_bytes=int(header.get("raw_bytes", 0)),
        ws_payload_bytes=int(header.get("ws_payload_bytes", 0)),
        request_prepare_ms=float(header.get("request_prepare_ms", 0.0)),
        scheduler_forward_ms=float(header.get("scheduler_forward_ms", 0.0)),
        raw_payload_build_ms=float(header.get("raw_payload_build_ms", 0.0)),
        raw_write_ms=float(header.get("raw_write_ms", 0.0)),
        ws_write_ms=float(header.get("ws_write_ms", 0.0)),
        chunk_total_ms=float(header.get("chunk_total_ms", 0.0)),
    )


def summarize_realtime_perf_stats(
    chunk_stats: list[RealtimeChunkStats],
    *,
    ignore_initial_chunks: int = 0,
) -> dict[str, float]:
    if not chunk_stats:
        return {}
    if ignore_initial_chunks < 0:
        raise ValueError("ignore_initial_chunks must be non-negative")
    if ignore_initial_chunks >= len(chunk_stats):
        raise ValueError(
            "ignore_initial_chunks must leave at least one realtime chunk to guard"
        )

    ignored_stats = chunk_stats[:ignore_initial_chunks]
    guarded_stats = chunk_stats[ignore_initial_chunks:]

    metrics = {
        "request_prepare_ms": [s.request_prepare_ms for s in guarded_stats],
        "scheduler_forward_ms": [s.scheduler_forward_ms for s in guarded_stats],
        "raw_payload_build_ms": [s.raw_payload_build_ms for s in guarded_stats],
        "raw_write_ms": [s.raw_write_ms for s in guarded_stats],
        "ws_write_ms": [s.ws_write_ms for s in guarded_stats],
        "chunk_total_ms": [s.chunk_total_ms for s in guarded_stats],
        "ws_payload_mb": [s.ws_payload_bytes / (1024 * 1024) for s in guarded_stats],
    }
    summary: dict[str, float] = {
        "num_chunks": float(len(chunk_stats)),
        "total_frames": float(sum(s.num_frames for s in chunk_stats)),
        "ignored_initial_chunks": float(ignore_initial_chunks),
        "guarded_chunks": float(len(guarded_stats)),
    }
    if ignored_stats:
        summary["ignored_max_chunk_total_ms"] = max(
            s.chunk_total_ms for s in ignored_stats
        )
        summary["ignored_max_scheduler_forward_ms"] = max(
            s.scheduler_forward_ms for s in ignored_stats
        )
    for name, values in metrics.items():
        sorted_values = sorted(values)
        p95_idx = min(len(sorted_values) - 1, int(len(sorted_values) * 0.95))
        summary[f"avg_{name}"] = statistics.fmean(values)
        summary[f"p95_{name}"] = sorted_values[p95_idx]
        summary[f"max_{name}"] = max(values)
    return summary


def validate_realtime_perf_stats(
    case_id: str,
    chunk_stats: list[RealtimeChunkStats],
    thresholds: dict[str, float],
    *,
    ignore_initial_chunks: int = 0,
) -> None:
    if not thresholds:
        return
    summary = summarize_realtime_perf_stats(
        chunk_stats, ignore_initial_chunks=ignore_initial_chunks
    )
    if not summary:
        pytest.fail(f"{case_id}: no realtime chunk stats were received")

    failures = []
    for metric_name, threshold in thresholds.items():
        if metric_name not in summary:
            raise ValueError(
                f"{case_id}: unknown realtime perf metric {metric_name!r}; "
                f"available metrics: {sorted(summary)}"
            )
        actual = summary[metric_name]
        if actual > threshold:
            failures.append(
                f"{metric_name}: actual={actual:.2f}, limit={threshold:.2f}"
            )

    if failures:
        pytest.fail(
            f"Realtime performance guard failed for {case_id}:\n"
            + "\n".join(f"  - {failure}" for failure in failures)
        )


def record_realtime_perf_stats(
    case_id: str, chunk_stats: list[RealtimeChunkStats]
) -> None:
    _REALTIME_CHUNK_STATS_BY_CASE[case_id] = list(chunk_stats)


def pop_realtime_perf_stats(case_id: str) -> list[RealtimeChunkStats]:
    return _REALTIME_CHUNK_STATS_BY_CASE.pop(case_id, [])


def select_realtime_key_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    if not frames:
        return []
    key_indices = [0, len(frames) // 2, len(frames) - 1]
    return [frames[idx].copy() for idx in key_indices]


def record_realtime_key_frames(case_id: str, frames: list[np.ndarray]) -> None:
    _REALTIME_KEY_FRAMES_BY_CASE[case_id] = select_realtime_key_frames(frames)


def pop_realtime_key_frames(case_id: str) -> list[np.ndarray] | None:
    return _REALTIME_KEY_FRAMES_BY_CASE.pop(case_id, None)


def decode_realtime_raw_rgb_frames(
    header: dict[str, Any],
    payload: bytes,
    previous_frame: bytes | None = None,
) -> list[np.ndarray]:
    content_type = header.get("content_type")
    if content_type not in (
        RAW_RGB_CONTENT_TYPE,
        RAW_RGB_DELTA_GZIP_CONTENT_TYPE,
        RAW_RGBA_DELTA_GZIP_CONTENT_TYPE,
    ):
        raise ValueError(f"Unsupported realtime frame content type: {content_type}")

    width = int(header["width"])
    height = int(header["height"])
    channels = int(header["channels"])
    num_frames = int(header["num_frames"])
    bytes_per_frame = int(header["bytes_per_frame"])
    expected_size = num_frames * bytes_per_frame
    if content_type in (
        RAW_RGB_DELTA_GZIP_CONTENT_TYPE,
        RAW_RGBA_DELTA_GZIP_CONTENT_TYPE,
    ):
        if header.get("delta_reference") != "previous-frame":
            previous_frame = None
        payload = restore_delta_gzip_raw_rgb_payload(
            payload,
            bytes_per_frame=bytes_per_frame,
            num_frames=num_frames,
            reference_frame=previous_frame,
        )
    if len(payload) != expected_size:
        raise ValueError(
            f"Realtime payload size mismatch: expected {expected_size}, got {len(payload)}"
        )

    frames = []
    for frame_idx in range(num_frames):
        offset = frame_idx * bytes_per_frame
        frame = np.frombuffer(
            payload[offset : offset + bytes_per_frame], dtype=np.uint8
        )
        frame = frame.reshape(height, width, channels)
        if channels > 3:
            frame = frame[:, :, :3]
        frames.append(frame.copy())
    return frames


def encode_realtime_frames_to_mp4(frames: list[np.ndarray], fps: int) -> bytes:
    if not frames:
        raise ValueError("Cannot encode empty realtime frame list")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        output_path = tmp.name
    try:
        imageio.mimsave(
            output_path,
            frames,
            fps=fps,
            format="mp4",
            codec="libx264",
            quality=5,
        )
        return Path(output_path).read_bytes()
    finally:
        try:
            os.remove(output_path)
        except OSError:
            pass


async def collect_realtime_frames(
    *,
    ws_url: str,
    init_payload: dict[str, Any],
    events: list[dict[str, Any]],
    num_chunks: int,
) -> list[np.ndarray]:
    return (
        await collect_realtime_output(
            ws_url=ws_url,
            init_payload=init_payload,
            events=events,
            num_chunks=num_chunks,
        )
    ).frames


async def collect_realtime_output(
    *,
    ws_url: str,
    init_payload: dict[str, Any],
    events: list[dict[str, Any]],
    num_chunks: int,
    require_chunk_stats: bool = False,
) -> RealtimeCollectionResult:
    try:
        import websockets
    except ImportError:
        pytest.skip("websockets is required for realtime consistency checks")

    frames: list[np.ndarray] = []
    chunk_stats: list[RealtimeChunkStats] = []
    sent_event_indices: set[int] = set()

    async def send_events_for_boundary(ws, completed_chunk: int) -> None:
        for event_idx, event in enumerate(events):
            if event_idx in sent_event_indices:
                continue
            if int(event.get("after_chunk", 0)) != completed_chunk:
                continue
            await ws.send(msgspec.msgpack.encode(build_realtime_event_payload(event)))
            sent_event_indices.add(event_idx)

    async with websockets.connect(ws_url, max_size=None, ping_interval=None) as ws:
        await ws.send(msgspec.msgpack.encode(init_payload))
        await send_events_for_boundary(ws, -1)

        received_chunks: set[int] = set()
        previous_frame: bytes | None = None
        while len(received_chunks) < num_chunks or (
            require_chunk_stats and len(chunk_stats) < len(received_chunks)
        ):
            header_payload = await asyncio.wait_for(
                ws.recv(), timeout=_REALTIME_WS_TIMEOUT_SECS
            )
            header = msgspec.msgpack.decode(header_payload)
            message_type = header.get("type")
            if message_type == "error":
                pytest.fail(f"Realtime generation failed: {header.get('content')}")
            if message_type == "chunk_stats":
                chunk_stats.append(parse_realtime_chunk_stats(header))
                continue
            if message_type == "frame_batch":
                raw_payload = header.pop("payload", None)
                header["type"] = "frame_batch_header"
            elif message_type == "frame_batch_header":
                raw_payload = await asyncio.wait_for(
                    ws.recv(), timeout=_REALTIME_WS_TIMEOUT_SECS
                )
            else:
                raise ValueError(f"Unexpected realtime message: {header}")
            if not isinstance(raw_payload, bytes):
                raise ValueError("Realtime frame payload must be bytes")

            chunk_frames = decode_realtime_raw_rgb_frames(
                header,
                raw_payload,
                previous_frame,
            )
            frames.extend(chunk_frames)
            if chunk_frames:
                previous_frame = chunk_frames[-1].tobytes()
            chunk_index = int(header["chunk_index"])
            if header.get("is_final_frame_batch", True):
                received_chunks.add(chunk_index)
                await send_events_for_boundary(ws, chunk_index)

    return RealtimeCollectionResult(frames=frames, chunk_stats=chunk_stats)
