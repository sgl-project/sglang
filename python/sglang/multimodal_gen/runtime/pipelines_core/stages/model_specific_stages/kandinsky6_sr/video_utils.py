# SPDX-License-Identifier: Apache-2.0
# Vendored from sr_core (parity-tested against the k6_video reference); adapted from
# the Kandinsky 6 SR inference reference (k6_video, Apache-2.0).
"""Pure-tensor frame-count / fps / output-size helpers for Kandinsky 6 video SR.

Ported from ``pipeline/video_io.py`` (frame alignment and fps resampling) and
``pipeline/output_resize.py`` (delivery-resolution resolution and resize). No
file or video-decoder I/O lives here; ``loguru`` is replaced by stdlib ``logging``.

The resolution-scale constant and the delivery-resize resolver (``SUPPORTED_RESOLUTION_SCALES``,
``resolve_target_hw`` and what they depend on) live in
``configs.sample.kandinsky6_sr_resolution`` instead of here: ``configs/sample/kandinsky6_sr.py``
needs them too, and importing anything under this runtime stage package from a ``configs/sample``
module closes an import cycle through ``registry`` -- see that module's docstring. They are
re-exported here so this module stays the one place the rest of the stage package imports them
from.
"""

from __future__ import annotations

import logging

import torch
from torch.nn import functional as f  # noqa: N812

from sglang.multimodal_gen.configs.sample.kandinsky6_sr_resolution import (
    ASPECT_MISMATCH_TOLERANCE,
    SUPPORTED_RESOLUTION_SCALES,
    TARGET_RESOLUTIONS,
    TargetResizeMode,
    fit_within,
    resolve_target_hw,
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


# --------------------------------------------------------------------------- #
# Frame alignment / fps resampling (pipeline/video_io.py)
# --------------------------------------------------------------------------- #


def align_to_vae_stride(t: int) -> int:
    """Round ``t`` to the nearest valid pixel-frame count ``1 + 8·k`` (ties up).

    SR data uses ``1 + 8·k`` frames so the temporal VAE stride round-trips
    cleanly. Rounding to nearest (not truncation) keeps a requested duration
    close to the target.

    Args:
        t: Raw frame count.

    Returns:
        Nearest valid frame count ``>= 1``.
    """
    if t <= 1:
        return 1
    k_floor = (t - 1) // 8
    t_floor = 1 + 8 * k_floor
    t_ceil = 1 + 8 * (k_floor + 1)
    return t_ceil if (t - t_floor) >= (t_ceil - t) else t_floor


def select_frame_indices(
    total_frames: int, src_fps: float, target_fps: float
) -> list[int]:
    """Fixed-stride frame indices that downsample ``src_fps`` to ``target_fps``.

    With ``step = src_fps / target_fps >= 1``, ``round(i * step)`` is strictly
    non-decreasing, so the selection never duplicates a source frame.

    Args:
        total_frames: Number of frames in the source.
        src_fps: Source frame rate (must be ``>= target_fps``).
        target_fps: Desired frame rate.

    Returns:
        Sorted list of integer source-frame indices.
    """
    step = src_fps / target_fps
    indices = [round(i * step) for i in range(int(total_frames / step))]
    return [i for i in indices if i < total_frames]


def resample_to_target_fps(
    video: torch.Tensor,
    src_fps: float,
    target_fps: int = TARGET_FPS,
) -> tuple[torch.Tensor, int]:
    """Resample a decoded video toward ``target_fps`` (downsample / no-op / keep).

    Three tiers:

    - ``|src_fps - target_fps| < RESAMPLE_FPS_TOLERANCE``: pass through unchanged.
    - ``src_fps > target_fps``: fixed-stride downsample via
      :func:`select_frame_indices`; the result is at ``target_fps``.
    - ``src_fps < target_fps``: kept at the native rate with a warning (no
      motion-compensated upsample); the clip stays mildly out of distribution.

    Args:
        video: ``[T, C, H, W]`` decoded source video.
        src_fps: Source frame rate.
        target_fps: Training frame rate to resample toward.

    Returns:
        ``(resampled_video, effective_fps)`` where ``effective_fps`` is the rate
        the returned frames play at (``target_fps`` when downsampled, otherwise
        ``round(src_fps)``) and is the correct rate to save the SR result at.
    """
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
    """Take the first ``max_num_frames`` and floor-align to ``1 + 8k`` frames.

    Args:
        video: ``[T, C, H, W]`` video.
        max_num_frames: Hard cap applied before alignment (``<= 0`` disables it).

    Returns:
        ``[T', C, H, W]`` with ``T' == 1 + 8k`` and ``T' <= max_num_frames``.

    Raises:
        ValueError: If the video has no frames.
    """
    if max_num_frames > 0:
        video = video[:max_num_frames]
    aligned = 1 + 8 * ((video.shape[0] - 1) // 8) if video.shape[0] > 0 else 0
    if aligned == 0:
        msg = "Video has no readable frames."
        raise ValueError(msg)
    return video[:aligned]


# --------------------------------------------------------------------------- #
# Output sizing (pipeline/output_resize.py)
# --------------------------------------------------------------------------- #
# ``fit_within`` and ``resolve_target_hw`` live in configs.sample.kandinsky6_sr_resolution
# (imported above and re-exported via this module's __all__); only the actual tensor resize
# stays here.


def resize_to_target(video: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Antialiased-downscale the SR result to exactly ``target_hw``. No cropping.

    Walks the clip in :data:`RESIZE_FRAME_CHUNK`-frame chunks to bound memory.

    Args:
        video: ``[C, T, H, W]`` uint8 SR result.
        target_hw: Exact output ``(H, W)``.

    Returns:
        ``[C, T, target_h, target_w]`` uint8.

    Raises:
        ValueError: If the target exceeds the SR result on either axis —
            that means the tier does not belong to this route, and silently
            upscaling would hide the mistake.
    """
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


__all__ = [
    "ASPECT_MISMATCH_TOLERANCE",
    "MAX_NUM_FRAMES",
    "RESAMPLE_FPS_TOLERANCE",
    "RESIZE_FRAME_CHUNK",
    "SUPPORTED_RESOLUTION_SCALES",
    "TARGET_FPS",
    "TARGET_RESOLUTIONS",
    "TargetResizeMode",
    "align_to_vae_stride",
    "clip_to_aligned_frames",
    "fit_within",
    "resample_to_target_fps",
    "resize_to_target",
    "resolve_target_hw",
    "select_frame_indices",
]
