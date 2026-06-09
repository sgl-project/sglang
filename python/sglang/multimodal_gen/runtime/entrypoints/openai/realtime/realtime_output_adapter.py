# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

import msgspec.msgpack
from fastapi import WebSocket
from PIL import Image

from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.timing import (
    RealtimeStageTimer,
)
from sglang.multimodal_gen.runtime.utils.realtime_video import (
    JPEG_FRAME_CONTENT_TYPE,
    RAW_RGB_CHANNELS,
    RAW_RGB_CONTENT_TYPE,
    WEBP_FRAME_CONTENT_TYPE,
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
    payload_lengths: list[int]
    event_id: int
    frame_batch_index: int
    num_frame_batches: int
    is_final_frame_batch: bool


class RealtimeFrameBatchMessage(RealtimeFrameBatchHeader, total=False):
    payload: bytes


class RealtimeFrameSendStats(TypedDict):
    header_pack_ms: float
    header_write_ms: float
    raw_payload_build_ms: float
    raw_write_ms: float
    ws_write_ms: float
    pace_wait_ms: float
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
        "pace_wait_ms": 0.0,
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


RAW_RGB_FRAMES_PER_WS_MESSAGE = 16
ENCODED_PREVIEW_FRAMES_PER_WS_MESSAGE = 6
FRAME_BATCH_PACK_OFFLOAD_BYTES = 64 * 1024
WEBP_DEFAULT_QUALITY = 90
JPEG_DEFAULT_QUALITY = 95
JPEG_SUBSAMPLING = 0
RAW_LOSSLESS_OUTPUT_FORMAT = "raw"
ENCODED_PREVIEW_FORMATS = {"webp", "jpeg"}


@dataclass(frozen=True)
class _TransportPayload:
    content_type: str
    payload: bytes
    metadata: dict[str, int | str | bool | list[int]]


def _split_frame_batch(
    frames: list[bytes],
    frames_per_message: int = RAW_RGB_FRAMES_PER_WS_MESSAGE,
) -> list[list[bytes]]:
    if not frames:
        return [frames]
    return [
        frames[i : i + frames_per_message]
        for i in range(0, len(frames), frames_per_message)
    ]


def _encode_rgb_frame_to_webp(
    frame: bytes,
    *,
    width: int,
    height: int,
    quality: int,
    preview_max_width: int | None,
) -> bytes:
    buffer = io.BytesIO()
    image = _resize_preview_image(
        Image.frombytes("RGB", (width, height), frame),
        preview_max_width=preview_max_width,
    )
    image.save(
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
    preview_max_width: int | None,
) -> bytes:
    buffer = io.BytesIO()
    image = _resize_preview_image(
        Image.frombytes("RGB", (width, height), frame),
        preview_max_width=preview_max_width,
    )
    image.save(
        buffer,
        format="JPEG",
        quality=quality,
        subsampling=JPEG_SUBSAMPLING,
    )
    return buffer.getvalue()


def _preview_dimensions(
    *,
    width: int,
    height: int,
    preview_max_width: int | None,
) -> tuple[int, int]:
    if (
        preview_max_width is None
        or preview_max_width <= 0
        or width <= preview_max_width
    ):
        return width, height
    preview_width = int(preview_max_width)
    preview_height = max(1, round(height * preview_width / width))
    return preview_width, preview_height


def _resize_preview_image(
    image: Image.Image,
    *,
    preview_max_width: int | None,
) -> Image.Image:
    width, height = image.size
    preview_width, preview_height = _preview_dimensions(
        width=width,
        height=height,
        preview_max_width=preview_max_width,
    )
    if (preview_width, preview_height) == image.size:
        return image
    return image.resize((preview_width, preview_height), Image.Resampling.BICUBIC)


def _pack_frame_batch_message(
    header: RealtimeFrameBatchHeader,
    payload: bytes,
) -> bytes:
    message: RealtimeFrameBatchMessage = {
        **header,
        "type": "frame_batch",
        "payload": payload,
    }
    return msgspec.msgpack.encode(message)


def _pack_frame_batch_header(header: RealtimeFrameBatchHeader) -> bytes:
    return msgspec.msgpack.encode(header)


def _build_transport_payload(
    transport_frames: list[bytes],
    *,
    content_type: str,
    metadata: dict[str, int | str],
    output_format: str | None,
    transport_quality: int | None,
    preview_max_width: int | None,
) -> _TransportPayload:
    payload_content_type = content_type
    payload_metadata: dict[str, int | str | bool | list[int]] = {}
    raw_payload = b""

    if (
        output_format in ENCODED_PREVIEW_FORMATS
        and content_type == RAW_RGB_CONTENT_TYPE
        and transport_frames
    ):
        if output_format == "webp":
            encoded_frames = [
                _encode_rgb_frame_to_webp(
                    frame,
                    width=int(metadata["width"]),
                    height=int(metadata["height"]),
                    quality=int(transport_quality or WEBP_DEFAULT_QUALITY),
                    preview_max_width=preview_max_width,
                )
                for frame in transport_frames
            ]
            payload_content_type = WEBP_FRAME_CONTENT_TYPE
        else:
            encoded_frames = [
                _encode_rgb_frame_to_jpeg(
                    frame,
                    width=int(metadata["width"]),
                    height=int(metadata["height"]),
                    quality=int(transport_quality or JPEG_DEFAULT_QUALITY),
                    preview_max_width=preview_max_width,
                )
                for frame in transport_frames
            ]
            payload_content_type = JPEG_FRAME_CONTENT_TYPE
        raw_payload = b"".join(encoded_frames)
        preview_width, preview_height = _preview_dimensions(
            width=int(metadata["width"]),
            height=int(metadata["height"]),
            preview_max_width=preview_max_width,
        )
        payload_metadata = {
            "format": output_format,
            "encoding": output_format,
            "source_width": int(metadata["width"]),
            "source_height": int(metadata["height"]),
            "preview_width": preview_width,
            "preview_height": preview_height,
            "width": preview_width,
            "height": preview_height,
            "payload_lengths": [len(frame) for frame in encoded_frames],
        }
    elif content_type == RAW_RGB_CONTENT_TYPE and transport_frames:
        raw_payload = b"".join(transport_frames)
        payload_metadata = {
            "raw_size": len(raw_payload),
            "encoding": RAW_LOSSLESS_OUTPUT_FORMAT,
        }
    else:
        raw_payload = b"".join(transport_frames)

    return _TransportPayload(
        content_type=payload_content_type,
        payload=raw_payload,
        metadata=payload_metadata,
    )


def _should_build_payload_off_loop(
    *,
    content_type: str,
    output_format: str | None,
    transport_frames: list[bytes],
) -> bool:
    if content_type != RAW_RGB_CONTENT_TYPE or not transport_frames:
        return False
    return output_format in ENCODED_PREVIEW_FORMATS or output_format is None


def _is_encoded_preview_transport(
    *,
    content_type: str,
    output_format: str | None,
) -> bool:
    return (
        output_format in ENCODED_PREVIEW_FORMATS
        and content_type == RAW_RGB_CONTENT_TYPE
    )


async def _build_encoded_preview_payloads(
    split_batches: list[list[bytes]],
    *,
    content_type: str,
    metadata: dict[str, int | str],
    output_format: str,
    transport_quality: int | None,
    preview_max_width: int | None,
    event_id: int | None,
) -> list[_TransportPayload]:
    return list(
        await asyncio.gather(
            *(
                _build_encoded_preview_payload(
                    transport_frames,
                    metadata=metadata,
                    output_format=output_format,
                    transport_quality=transport_quality,
                    preview_max_width=preview_max_width,
                )
                for transport_frames in split_batches
            )
        )
    )


async def _build_encoded_preview_payload(
    transport_frames: list[bytes],
    *,
    metadata: dict[str, int | str],
    output_format: str,
    transport_quality: int | None,
    preview_max_width: int | None,
) -> _TransportPayload:
    width = int(metadata["width"])
    height = int(metadata["height"])
    if output_format == "webp":
        encoded_frames = list(
            await asyncio.gather(
                *(
                    asyncio.to_thread(
                        _encode_rgb_frame_to_webp,
                        frame,
                        width=width,
                        height=height,
                        quality=int(transport_quality or WEBP_DEFAULT_QUALITY),
                        preview_max_width=preview_max_width,
                    )
                    for frame in transport_frames
                )
            )
        )
        payload_content_type = WEBP_FRAME_CONTENT_TYPE
    else:
        encoded_frames = list(
            await asyncio.gather(
                *(
                    asyncio.to_thread(
                        _encode_rgb_frame_to_jpeg,
                        frame,
                        width=width,
                        height=height,
                        quality=int(transport_quality or JPEG_DEFAULT_QUALITY),
                        preview_max_width=preview_max_width,
                    )
                    for frame in transport_frames
                )
            )
        )
        payload_content_type = JPEG_FRAME_CONTENT_TYPE

    preview_width, preview_height = _preview_dimensions(
        width=width,
        height=height,
        preview_max_width=preview_max_width,
    )
    return _TransportPayload(
        content_type=payload_content_type,
        payload=b"".join(encoded_frames),
        metadata={
            "format": output_format,
            "encoding": output_format,
            "source_width": width,
            "source_height": height,
            "preview_width": preview_width,
            "preview_height": preview_height,
            "width": preview_width,
            "height": preview_height,
            "payload_lengths": [len(frame) for frame in encoded_frames],
        },
    )


class RawRGBRealtimeOutputAdapter:
    """send raw RGB over WebSocket using lossless transport"""

    def __init__(self) -> None:
        pass

    def reset(self) -> None:
        pass

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
        preview_max_width = getattr(batch, "realtime_preview_max_width", None)
        stats = await self._send_frame_batches(
            ws,
            result.raw_frame_batches,
            content_type=content_type,
            chunk_index_start=batch.block_idx,
            request_id=batch.request_id,
            event_id=getattr(batch, "realtime_event_id", None),
            frame_metadata=frame_metadata,
            output_format=output_format,
            transport_quality=getattr(batch, "output_compression", None),
            preview_max_width=preview_max_width,
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
        transport_quality: int | None = None,
        preview_max_width: int | None = None,
    ) -> RealtimeFrameSendStats:
        chunk_index = chunk_index_start
        metadata = frame_metadata or {}
        stats = empty_frame_send_stats(content_type)
        for frames in frame_batches:
            split_batches = (
                _split_frame_batch(frames, ENCODED_PREVIEW_FRAMES_PER_WS_MESSAGE)
                if _is_encoded_preview_transport(
                    content_type=content_type,
                    output_format=output_format,
                )
                else (
                    _split_frame_batch(frames)
                    if content_type == RAW_RGB_CONTENT_TYPE
                    else [frames]
                )
            )
            num_frame_batches = len(split_batches)
            encoded_preview_payloads: list[_TransportPayload] | None = None
            if _is_encoded_preview_transport(
                content_type=content_type,
                output_format=output_format,
            ):
                timer = RealtimeStageTimer()
                encoded_preview_payloads = await _build_encoded_preview_payloads(
                    split_batches,
                    content_type=content_type,
                    metadata=metadata,
                    output_format=output_format,
                    transport_quality=transport_quality,
                    preview_max_width=preview_max_width,
                    event_id=event_id,
                )
                stats["raw_payload_build_ms"] += timer.mark_ms()
            for frame_batch_index, transport_frames in enumerate(split_batches):
                timer = RealtimeStageTimer()
                transport_metadata = metadata
                if encoded_preview_payloads is not None:
                    transport_payload = encoded_preview_payloads[frame_batch_index]
                else:
                    if _should_build_payload_off_loop(
                        content_type=content_type,
                        output_format=output_format,
                        transport_frames=transport_frames,
                    ):
                        transport_payload = await asyncio.to_thread(
                            _build_transport_payload,
                            transport_frames,
                            content_type=content_type,
                            metadata=metadata,
                            output_format=output_format,
                            transport_quality=transport_quality,
                            preview_max_width=preview_max_width,
                        )
                    else:
                        transport_payload = _build_transport_payload(
                            transport_frames,
                            content_type=content_type,
                            metadata=metadata,
                            output_format=output_format,
                            transport_quality=transport_quality,
                            preview_max_width=preview_max_width,
                        )
                    stats["raw_payload_build_ms"] += timer.mark_ms()

                header: RealtimeFrameBatchHeader = {
                    "type": "frame_batch_header",
                    "request_id": request_id,
                    "chunk_index": chunk_index,
                    "content_type": transport_payload.content_type,
                    "num_frames": len(transport_frames),
                    "total_size": len(transport_payload.payload),
                    "frame_batch_index": frame_batch_index,
                    "num_frame_batches": num_frame_batches,
                    "is_final_frame_batch": frame_batch_index == num_frame_batches - 1,
                }
                if event_id is not None:
                    header["event_id"] = event_id
                header.update(transport_metadata)
                header.update(transport_payload.metadata)

                if len(transport_payload.payload) >= FRAME_BATCH_PACK_OFFLOAD_BYTES:
                    header_payload = _pack_frame_batch_header(header)
                    stats["header_pack_ms"] += timer.mark_ms()

                    await ws.send_bytes(header_payload)
                    stats["header_write_ms"] += timer.mark_ms()

                    await ws.send_bytes(transport_payload.payload)
                    stats["raw_write_ms"] += timer.mark_ms()

                    stats["ws_payload_bytes"] += len(header_payload) + len(
                        transport_payload.payload
                    )
                else:
                    message_payload = _pack_frame_batch_message(
                        header,
                        transport_payload.payload,
                    )
                    stats["header_pack_ms"] += timer.mark_ms()

                    stats["header_write_ms"] += timer.mark_ms()
                    await ws.send_bytes(message_payload)
                    stats["raw_write_ms"] += timer.mark_ms()

                    stats["ws_payload_bytes"] += len(message_payload)

                stats["raw_bytes"] += sum(len(frame) for frame in transport_frames)
                stats["num_frames"] += len(transport_frames)
                stats["num_batches"] += 1
                stats["content_type"] = transport_payload.content_type
            chunk_index += 1

        stats["ws_write_ms"] = stats["header_write_ms"] + stats["raw_write_ms"]
        return stats
