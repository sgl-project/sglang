# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING, TypedDict

from fastapi import WebSocket
from msgpack import packb

from sglang.multimodal_gen.runtime.utils.realtime_video import (
    RAW_RGB_CHANNELS,
    RAW_RGB_CONTENT_TYPE,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.generate_session import (
        GenerateSession,
    )
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
        OutputBatch,
        Req,
    )


class RealtimeFrameBatchHeader(TypedDict, total=False):
    type: str
    request_id: str
    chunk_index: int
    content_type: str
    num_frames: int
    total_size: int
    format: str
    width: int
    height: int
    channels: int
    bytes_per_frame: int


class RealtimeFrameSendStats(TypedDict):
    header_pack_ms: float
    header_write_ms: float
    raw_payload_build_ms: float
    raw_write_ms: float
    ws_write_ms: float
    raw_bytes: int
    ws_payload_bytes: int
    num_frames: int
    num_batches: int
    frame_shape: tuple[int, int, int] | None
    content_type: str


def empty_frame_send_stats(content_type: str = "") -> RealtimeFrameSendStats:
    return {
        "header_pack_ms": 0.0,
        "header_write_ms": 0.0,
        "raw_payload_build_ms": 0.0,
        "raw_write_ms": 0.0,
        "ws_write_ms": 0.0,
        "raw_bytes": 0,
        "ws_payload_bytes": 0,
        "num_frames": 0,
        "num_batches": 0,
        "frame_shape": None,
        "content_type": content_type,
    }


def _raw_rgb_frame_metadata(batch: Req) -> dict[str, int | str]:
    frame_width = batch.width
    frame_height = batch.height
    if frame_width is None or frame_height is None:
        return {}

    frame_width = int(frame_width)
    frame_height = int(frame_height)
    if batch.enable_upscaling:
        upscaling_scale = int(batch.upscaling_scale or 1)
        frame_width *= upscaling_scale
        frame_height *= upscaling_scale

    return {
        "format": "rgb24",
        "width": frame_width,
        "height": frame_height,
        "channels": RAW_RGB_CHANNELS,
        "bytes_per_frame": frame_width * frame_height * RAW_RGB_CHANNELS,
    }


def _frame_shape_from_metadata(
    metadata: dict[str, int | str] | None,
) -> tuple[int, int, int] | None:
    if not metadata:
        return None
    return (
        int(metadata["height"]),
        int(metadata["width"]),
        int(metadata["channels"]),
    )


class RawRGBRealtimeOutputAdapter:
    async def send(
        self,
        ws: WebSocket,
        session: GenerateSession,
        result: OutputBatch,
        batch: Req,
    ) -> RealtimeFrameSendStats:
        content_type = result.raw_frame_content_type
        if result.raw_frame_batches is None:
            return empty_frame_send_stats(content_type)

        frame_metadata = (
            result.raw_frame_metadata or _raw_rgb_frame_metadata(batch)
            if content_type == RAW_RGB_CONTENT_TYPE
            else {}
        )
        stats = await self._send_frame_batches(
            ws,
            result.raw_frame_batches,
            content_type=content_type,
            chunk_index_start=batch.block_idx,
            request_id=session.request_id,
            frame_metadata=frame_metadata,
        )
        stats["frame_shape"] = _frame_shape_from_metadata(frame_metadata)
        return stats

    async def _send_frame_batches(
        self,
        ws: WebSocket,
        frame_batches: list[list[bytes]],
        *,
        content_type: str,
        chunk_index_start: int,
        request_id: str,
        frame_metadata: dict[str, int | str] | None = None,
    ) -> RealtimeFrameSendStats:
        chunk_index = chunk_index_start
        metadata = frame_metadata or {}
        stats = empty_frame_send_stats(content_type)
        for frames in frame_batches:
            frame_bytes = sum(len(frame) for frame in frames)
            header: RealtimeFrameBatchHeader = {
                "type": "frame_batch_header",
                "request_id": request_id,
                "chunk_index": chunk_index,
                "content_type": content_type,
                "num_frames": len(frames),
                "total_size": frame_bytes,
            }
            header.update(metadata)

            stage_start = time.perf_counter()
            header_payload = packb(header, use_bin_type=True)
            stats["header_pack_ms"] += (time.perf_counter() - stage_start) * 1000.0

            stage_start = time.perf_counter()
            await ws.send_bytes(header_payload)
            stats["header_write_ms"] += (time.perf_counter() - stage_start) * 1000.0

            stage_start = time.perf_counter()
            raw_payload = b"".join(frames)
            stats["raw_payload_build_ms"] += (
                time.perf_counter() - stage_start
            ) * 1000.0

            stage_start = time.perf_counter()
            await ws.send_bytes(raw_payload)
            stats["raw_write_ms"] += (time.perf_counter() - stage_start) * 1000.0

            stats["raw_bytes"] += frame_bytes
            stats["ws_payload_bytes"] += len(header_payload) + len(raw_payload)
            stats["num_frames"] += len(frames)
            stats["num_batches"] += 1
            chunk_index += 1

        stats["ws_write_ms"] = stats["header_write_ms"] + stats["raw_write_ms"]
        return stats
