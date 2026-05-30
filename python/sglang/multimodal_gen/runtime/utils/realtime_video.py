# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
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
RAW_RGB_CHANNELS = 3


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
