# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io
from typing import TYPE_CHECKING, TypedDict

from fastapi import WebSocket
from msgpack import packb
from PIL import Image

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.timing import (
    RealtimeStageTimer,
)
from sglang.multimodal_gen.runtime.utils.realtime_video import (
    JPEG_FRAME_CONTENT_TYPE,
    RAW_RGB_CHANNELS,
    RAW_RGB_CONTENT_TYPE,
    RAW_RGB_DELTA_GZIP_CONTENT_TYPE,
    WEBP_FRAME_CONTENT_TYPE,
    build_delta_gzip_raw_rgb_payload,
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
    raw_size: int
    encoding: str
    delta_reference: str
    event_id: int
    frame_batch_index: int
    num_frame_batches: int
    is_final_frame_batch: bool


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


RAW_RGB_FRAMES_PER_WS_MESSAGE = 1
WEBP_DEFAULT_QUALITY = 90
JPEG_DEFAULT_QUALITY = 95
JPEG_SUBSAMPLING = 0
RAW_LOSSLESS_OUTPUT_FORMAT = "raw"
ENCODED_PREVIEW_FORMATS = {"webp", "jpeg"}


def _split_frame_batch(frames: list[bytes]) -> list[list[bytes]]:
    if not frames:
        return [frames]
    return [
        frames[i : i + RAW_RGB_FRAMES_PER_WS_MESSAGE]
        for i in range(0, len(frames), RAW_RGB_FRAMES_PER_WS_MESSAGE)
    ]


def _encode_rgb_frame_to_webp(
    frame: bytes,
    *,
    width: int,
    height: int,
    quality: int,
) -> bytes:
    buffer = io.BytesIO()
    Image.frombytes("RGB", (width, height), frame).save(
        buffer,
        format="WEBP",
        quality=quality,
        method=0,
    )
    return buffer.getvalue()


def _encode_rgb_frame_to_jpeg(
    frame: bytes,
    *,
    width: int,
    height: int,
    quality: int,
) -> bytes:
    buffer = io.BytesIO()
    Image.frombytes("RGB", (width, height), frame).save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=JPEG_SUBSAMPLING,
    )
    return buffer.getvalue()


class RawRGBRealtimeOutputAdapter:
    """send raw RGB over WebSocket using lossless transport compression"""

    def __init__(self) -> None:
        self._last_raw_rgb_frame: bytes | None = None
        self._last_event_id: int | None = None

    def reset(self) -> None:
        self._last_raw_rgb_frame = None
        self._last_event_id = None

    async def send(
        self,
        ws: WebSocket,
        session: GenerateSession,
        result: OutputBatch,
        batch: Req,
    ) -> RealtimeFrameSendStats:
        """send frames through ws"""
        content_type = result.raw_frame_content_type
        if result.raw_frame_batches is None:
            return empty_frame_send_stats(content_type)
        if batch.block_idx == 0:
            self.reset()

        frame_metadata = (
            result.raw_frame_metadata or _raw_rgb_frame_metadata(batch)
            if content_type == RAW_RGB_CONTENT_TYPE
            else {}
        )
        output_format = getattr(batch, "realtime_output_format", None)
        stats = await self._send_frame_batches(
            ws,
            result.raw_frame_batches,
            content_type=content_type,
            chunk_index_start=batch.block_idx,
            request_id=batch.request_id,
            event_id=getattr(batch, "realtime_event_id", None),
            frame_metadata=frame_metadata,
            output_format=output_format,
            output_quality=getattr(batch, "output_compression", None),
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
        event_id: int | None = None,
        frame_metadata: dict[str, int | str] | None = None,
        output_format: str | None = None,
        output_quality: int | None = None,
    ) -> RealtimeFrameSendStats:
        chunk_index = chunk_index_start
        metadata = frame_metadata or {}
        stats = empty_frame_send_stats(content_type)
        for frames in frame_batches:
            split_batches = (
                [[frame] for frame in frames]
                if output_format in ENCODED_PREVIEW_FORMATS
                and content_type == RAW_RGB_CONTENT_TYPE
                else _split_frame_batch(frames)
                if content_type == RAW_RGB_CONTENT_TYPE
                else [frames]
            )
            num_frame_batches = len(split_batches)
            for frame_batch_index, transport_frames in enumerate(split_batches):
                timer = RealtimeStageTimer()
                transport_metadata = metadata
                payload_content_type = content_type
                payload_metadata: dict[str, int | str | bool] = {}
                raw_payload = b""
                if (
                    output_format in ENCODED_PREVIEW_FORMATS
                    and content_type == RAW_RGB_CONTENT_TYPE
                    and transport_frames
                ):
                    if output_format == "webp":
                        raw_payload = _encode_rgb_frame_to_webp(
                            transport_frames[0],
                            width=int(metadata["width"]),
                            height=int(metadata["height"]),
                            quality=int(output_quality or WEBP_DEFAULT_QUALITY),
                        )
                        payload_content_type = WEBP_FRAME_CONTENT_TYPE
                    else:
                        raw_payload = _encode_rgb_frame_to_jpeg(
                            transport_frames[0],
                            width=int(metadata["width"]),
                            height=int(metadata["height"]),
                            quality=int(output_quality or JPEG_DEFAULT_QUALITY),
                        )
                        payload_content_type = JPEG_FRAME_CONTENT_TYPE
                    payload_metadata = {
                        "format": output_format,
                        "encoding": output_format,
                    }
                elif (
                    output_format == RAW_LOSSLESS_OUTPUT_FORMAT
                    and content_type == RAW_RGB_CONTENT_TYPE
                    and transport_frames
                ):
                    raw_payload = b"".join(transport_frames)
                    payload_metadata = {
                        "raw_size": len(raw_payload),
                        "encoding": RAW_LOSSLESS_OUTPUT_FORMAT,
                    }
                elif content_type == RAW_RGB_CONTENT_TYPE and transport_frames:
                    reference_frame = self._last_raw_rgb_frame
                    if event_id != self._last_event_id:
                        reference_frame = None
                    raw_payload = build_delta_gzip_raw_rgb_payload(
                        transport_frames,
                        reference_frame=reference_frame,
                    )
                    payload_content_type = RAW_RGB_DELTA_GZIP_CONTENT_TYPE
                    payload_metadata = {
                        "raw_size": sum(len(frame) for frame in transport_frames),
                        "encoding": "delta-gzip",
                    }
                    if reference_frame is not None:
                        payload_metadata["delta_reference"] = "previous-frame"
                    self._last_raw_rgb_frame = transport_frames[-1]
                    self._last_event_id = event_id
                else:
                    raw_payload = b"".join(transport_frames)
                stats["raw_payload_build_ms"] += timer.mark_ms()

                header: RealtimeFrameBatchHeader = {
                    "type": "frame_batch_header",
                    "request_id": request_id,
                    "chunk_index": chunk_index,
                    "content_type": payload_content_type,
                    "num_frames": len(transport_frames),
                    "total_size": len(raw_payload),
                    "frame_batch_index": frame_batch_index,
                    "num_frame_batches": num_frame_batches,
                    "is_final_frame_batch": frame_batch_index
                    == num_frame_batches - 1,
                }
                if event_id is not None:
                    header["event_id"] = event_id
                header.update(transport_metadata)
                header.update(payload_metadata)

                header_payload = packb(header, use_bin_type=True)
                stats["header_pack_ms"] += timer.mark_ms()

                await ws.send_bytes(header_payload)
                stats["header_write_ms"] += timer.mark_ms()

                await ws.send_bytes(raw_payload)
                stats["raw_write_ms"] += timer.mark_ms()

                stats["raw_bytes"] += sum(len(frame) for frame in transport_frames)
                stats["ws_payload_bytes"] += len(header_payload) + len(raw_payload)
                stats["num_frames"] += len(transport_frames)
                stats["num_batches"] += 1
                stats["content_type"] = payload_content_type
            chunk_index += 1

        stats["ws_write_ms"] = stats["header_write_ms"] + stats["raw_write_ms"]
        return stats
