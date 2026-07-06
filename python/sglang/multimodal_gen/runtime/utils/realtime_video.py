# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
import zlib
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import (
        OutputBatch,
        Req,
    )

logger = init_logger(__name__)

RAW_RGB_CONTENT_TYPE = "application/x-raw-rgb"
RAW_RGB_DELTA_GZIP_CONTENT_TYPE = "application/x-raw-rgb-delta-gzip"
RAW_RGBA_DELTA_GZIP_CONTENT_TYPE = "application/x-raw-rgba-delta-gzip"
WEBP_FRAME_CONTENT_TYPE = "image/webp"
JPEG_FRAME_CONTENT_TYPE = "image/jpeg"
RAW_RGB_CHANNELS = 3
RAW_RGBA_CHANNELS = 4
_RAW_RGB_DELTA_GZIP_LEVEL = 0


def build_delta_gzip_raw_rgb_payload(
    frames: list[bytes],
    *,
    reference_frame: bytes | None = None,
) -> bytes:
    if not frames:
        return b""

    frame_size = len(frames[0])
    if reference_frame is not None and len(reference_frame) != frame_size:
        raise ValueError("raw RGB delta gzip reference frame size mismatch")

    previous = (
        np.frombuffer(reference_frame, dtype=np.uint8)
        if reference_frame is not None
        else None
    )
    # keep gzip framing for lossless transport without spending realtime budget on compression
    compressor = zlib.compressobj(
        level=_RAW_RGB_DELTA_GZIP_LEVEL, method=zlib.DEFLATED, wbits=31
    )
    compressed_chunks = []
    for frame in frames:
        if len(frame) != frame_size:
            raise ValueError("raw RGB delta gzip requires fixed-size frames")
        current = np.frombuffer(frame, dtype=np.uint8)
        if previous is None:
            delta_frame = frame
        else:
            delta_frame = np.bitwise_xor(current, previous).tobytes()
        compressed_chunks.append(compressor.compress(delta_frame))
        previous = current

    compressed_chunks.append(compressor.flush())
    return b"".join(compressed_chunks)


def restore_delta_gzip_raw_rgb_payload(
    payload: bytes,
    *,
    bytes_per_frame: int,
    num_frames: int,
    reference_frame: bytes | None = None,
) -> bytes:
    if reference_frame is not None and len(reference_frame) != bytes_per_frame:
        raise ValueError("delta gzip reference frame size mismatch")

    delta_payload = zlib.decompress(payload, wbits=31)
    expected_size = bytes_per_frame * num_frames
    if len(delta_payload) != expected_size:
        raise ValueError(
            "delta gzip payload size mismatch: "
            f"expected {expected_size}, got {len(delta_payload)}"
        )

    restored = bytearray(delta_payload)
    previous = (
        np.frombuffer(reference_frame, dtype=np.uint8)
        if reference_frame is not None
        else None
    )
    for frame_idx in range(num_frames):
        offset = frame_idx * bytes_per_frame
        current = np.frombuffer(
            restored, dtype=np.uint8, count=bytes_per_frame, offset=offset
        )
        if previous is not None:
            current ^= previous
        previous = current
    return bytes(restored)


def build_raw_rgb_frame_batches(
    output: Any,
    req: Req,
    output_batch: OutputBatch,
    post_process_sample_fn: Callable[..., Any],
) -> tuple[list[list[bytes]], dict[str, Any]]:
    """post-process for realtime responses, returns only the batched frames and metadata"""
    start = time.monotonic()
    sample_to_frames_ms = 0.0
    frames_to_bytes_ms = 0.0
    raw_bytes = 0
    num_frames = 0
    frame_shape = None
    frame_batches = []
    if isinstance(output, torch.Tensor):
        outputs = list(output)
    else:
        outputs = output if isinstance(output, Sequence) else [output]

    for sample in outputs:
        stage_start = time.monotonic()
        if (
            isinstance(sample, torch.Tensor)
            and not req.enable_frame_interpolation
            and not req.enable_upscaling
        ):
            frames = _tensor_sample_to_rgb24_array(sample)
        else:
            frames = post_process_sample_fn(
                sample,
                req.data_type,
                req.fps,
                False,
                None,
                audio_sample_rate=output_batch.audio_sample_rate,
                output_compression=req.output_compression,
                enable_frame_interpolation=req.enable_frame_interpolation,
                frame_interpolation_exp=req.frame_interpolation_exp,
                frame_interpolation_scale=req.frame_interpolation_scale,
                frame_interpolation_model_path=req.frame_interpolation_model_path,
                enable_upscaling=False,
                upscaling_model_path=req.upscaling_model_path,
                upscaling_scale=req.upscaling_scale,
            )
            if req.enable_upscaling and frames:
                from sglang.multimodal_gen.runtime.postprocess import (
                    batch_upscale_frames,
                )

                frames = batch_upscale_frames(
                    frames,
                    model_path=req.upscaling_model_path,
                    scale=req.upscaling_scale,
                )
        sample_to_frames_ms += (time.monotonic() - stage_start) * 1000.0

        stage_start = time.monotonic()

        # numpy frames to RGB24 bytes
        raw_frames = []
        for frame in frames:
            if frame.ndim == 2:
                frame = frame[:, :, None]
            if frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            elif frame.shape[-1] > RAW_RGB_CHANNELS:
                frame = frame[:, :, :RAW_RGB_CHANNELS]
            frame = np.ascontiguousarray(frame)
            frame_shape = tuple(int(dim) for dim in frame.shape)
            frame_bytes = frame.tobytes()
            raw_bytes += len(frame_bytes)
            num_frames += 1
            raw_frames.append(frame_bytes)
        frames_to_bytes_ms += (time.monotonic() - stage_start) * 1000.0
        frame_batches.append(raw_frames)

    total_ms = (time.monotonic() - start) * 1000.0
    logger.info(
        "realtime raw RGB frame batch timing: request_id=%s "
        "chunk_idx=%s sample_to_frames=%.2fms frames_to_bytes=%.2fms "
        "total=%.2fms batches=%d frames=%d frame_shape=%s "
        "raw_bytes=%d content_type=%s",
        req.request_id,
        req.block_idx,
        sample_to_frames_ms,
        frames_to_bytes_ms,
        total_ms,
        len(frame_batches),
        num_frames,
        frame_shape,
        raw_bytes,
        RAW_RGB_CONTENT_TYPE,
    )
    frame_metadata: dict[str, Any] = {}
    if frame_shape is not None and len(frame_shape) == 3:
        frame_height, frame_width, channels = frame_shape
        frame_metadata = {
            "format": "rgb24",
            "width": frame_width,
            "height": frame_height,
            "channels": channels,
            "bytes_per_frame": frame_width * frame_height * channels,
        }
    return frame_batches, frame_metadata


def _tensor_sample_to_rgb24_array(sample: torch.Tensor) -> np.ndarray:
    if sample.dim() == 3:
        sample = sample.unsqueeze(1)
    sample = (sample * 255).clamp(0, 255).to(torch.uint8)
    return sample.permute(1, 2, 3, 0).contiguous().cpu().numpy()
