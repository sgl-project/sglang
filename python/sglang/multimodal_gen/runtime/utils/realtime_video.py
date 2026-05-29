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
    start = time.monotonic()
    postprocess_ms = 0.0
    pack_bytes_ms = 0.0
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
            enable_upscaling=req.enable_upscaling,
            upscaling_model_path=req.upscaling_model_path,
            upscaling_scale=req.upscaling_scale,
        )
        postprocess_ms += (time.monotonic() - stage_start) * 1000.0

        stage_start = time.monotonic()
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
        pack_bytes_ms += (time.monotonic() - stage_start) * 1000.0
        frame_batches.append(raw_frames)

    total_ms = (time.monotonic() - start) * 1000.0
    logger.info(
        "realtime raw RGB frame batches prepared: request_id=%s "
        "block_idx=%s postprocess=%.2fms pack_bytes=%.2fms "
        "total=%.2fms batches=%d frames=%d frame_shape=%s "
        "raw_bytes=%d content_type=%s",
        req.request_id,
        req.block_idx,
        postprocess_ms,
        pack_bytes_ms,
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
