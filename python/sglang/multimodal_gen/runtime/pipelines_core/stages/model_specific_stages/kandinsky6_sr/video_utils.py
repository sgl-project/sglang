# SPDX-License-Identifier: Apache-2.0
# Vendored from sr_core (parity-tested against the k6_video reference); adapted from
# the Kandinsky 6 SR inference reference (k6_video, Apache-2.0).
"""Frame selection and bounded-memory output resize for Kandinsky 6 SR."""

from __future__ import annotations

import logging

import torch
from torch.nn import functional as f  # noqa: N812

from sglang.multimodal_gen.configs.sample.kandinsky6_sr_resolution import (
    ASPECT_MISMATCH_TOLERANCE,
)

logger = logging.getLogger(__name__)

# --- Model / pipeline contract (from the reference ``constants.py``; NOT configurable) ---
# The SR model was trained on 5 s clips at 24 fps -> 121 pixel frames
# (= 1 + 8*15); feeding more frames or another rate is out of contract.
TARGET_FPS: int = 24
MAX_NUM_FRAMES: int = 121
# Treat ``|src_fps - TARGET_FPS| < tolerance`` as already-at-target (no
# resample): covers 23.98 (NTSC) and 25.00 sources.
RESAMPLE_FPS_TOLERANCE: float = 1.5
# Frames per interpolate call when downscaling the output (a 2K clip is
# ~2.4 GB as uint8, so the resize walks the clip in chunks).
RESIZE_FRAME_CHUNK: int = 8


def select_frame_indices(
    total_frames: int, src_fps: float, target_fps: float
) -> list[int]:
    """Select rounded frame indices to downsample src_fps to target_fps."""
    step = src_fps / target_fps
    indices = [round(i * step) for i in range(int(total_frames / step))]
    return [i for i in indices if i < total_frames]


def resample_to_target_fps(
    video: torch.Tensor,
    src_fps: float,
    target_fps: int = TARGET_FPS,
) -> tuple[torch.Tensor, int]:
    """Downsample high-fps inputs; retain near-target and lower-fps inputs.

    Return [T, C, H, W] video and its effective playback rate."""
    if abs(src_fps - target_fps) < RESAMPLE_FPS_TOLERANCE:
        return video, round(src_fps)
    if src_fps > target_fps:
        indices = select_frame_indices(video.shape[0], src_fps, target_fps)
        logger.warning(
            "Source fps %.2f > target %sfps: downsampling %s frames -> %s (fixed-stride); "
            "source temporal detail beyond %sfps is discarded.",
            src_fps,
            target_fps,
            video.shape[0],
            len(indices),
            target_fps,
        )
        return video[indices], target_fps
    logger.warning(
        "Source fps %.2f < target %sfps: keeping native frames (no minterpolate upsample); "
        "output is mildly out of distribution.",
        src_fps,
        target_fps,
    )
    return video, round(src_fps)


def clip_to_aligned_frames(
    video: torch.Tensor, max_num_frames: int = MAX_NUM_FRAMES
) -> torch.Tensor:
    """Cap the frame count, then floor-align to 1+8k; reject empty video."""
    if max_num_frames > 0:
        video = video[:max_num_frames]
    aligned = 1 + 8 * ((video.shape[0] - 1) // 8) if video.shape[0] > 0 else 0
    if aligned == 0:
        msg = "Video has no readable frames."
        raise ValueError(msg)
    return video[:aligned]


def resize_to_target(video: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Downscale [C, T, H, W] uint8 video in bounded-memory chunks, without cropping."""
    _c, _t, height, width = video.shape
    target_h, target_w = target_hw
    if (height, width) == (target_h, target_w):
        return video
    if target_h > height or target_w > width:
        msg = (
            f"target {target_w}x{target_h} exceeds the SR result {width}x{height} — that would "
            f"upscale. Pick a lower tier or a higher --resolution-scale for this source."
        )
        raise ValueError(msg)

    source_ratio, target_ratio = width / height, target_w / target_h
    if abs(source_ratio - target_ratio) / target_ratio > ASPECT_MISMATCH_TOLERANCE:
        logger.warning(
            "Aspect mismatch: SR %sx%s is %.3f, target %sx%s is %.3f — resizing anisotropically "
            "(no crop), the picture will be squeezed by %.1f%%.",
            width,
            height,
            source_ratio,
            target_w,
            target_h,
            target_ratio,
            abs(source_ratio / target_ratio - 1) * 100,
        )

    channels, frames = video.shape[0], video.shape[1]
    out = torch.empty((channels, frames, target_h, target_w), dtype=torch.uint8)
    for start in range(0, frames, RESIZE_FRAME_CHUNK):
        chunk = video[:, start : start + RESIZE_FRAME_CHUNK].permute(1, 0, 2, 3).float()
        resized = f.interpolate(
            chunk,
            size=(target_h, target_w),
            mode="bilinear",
            antialias=True,
            align_corners=False,
        )
        out[:, start : start + RESIZE_FRAME_CHUNK] = (
            resized.clamp(0, 255).round().to(torch.uint8).permute(1, 0, 2, 3)
        )

    logger.info(
        "Final output: %sx%s -> %sx%s (downscale x%.2f / x%.2f)",
        width,
        height,
        target_w,
        target_h,
        width / target_w,
        height / target_h,
    )
    return out
