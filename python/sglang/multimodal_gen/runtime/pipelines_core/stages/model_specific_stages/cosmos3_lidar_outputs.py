# SPDX-License-Identifier: Apache-2.0
"""Artifacts for a decoded Cosmos3 LiDAR clip: numeric range map plus preview videos.

The decoder hands over ``[3, T, H, W]`` metric sweeps (range in meters, unit
intensity, ``{0, 1}`` validity). Consumers that unproject points want the
numbers, so the safetensors file is the primary output; the range and
intensity videos are previews in the reference's conventions (viridis over the
sensor's range span, invalid rays black; grayscale intensity).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

LIDAR_RANGEMAP_SUFFIX = "_lidar.safetensors"
LIDAR_RANGE_VIDEO_SUFFIX = "_lidar_range.mp4"
LIDAR_INTENSITY_VIDEO_SUFFIX = "_lidar_intensity.mp4"
_ARRAY_KEYS = ("range_m", "intensity", "validity")


def _current_umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _colormap_lut(name: str = "viridis") -> np.ndarray:
    """``[256, 3]`` uint8 lookup table; grayscale when matplotlib is unavailable."""
    try:
        from matplotlib import colormaps

        lut = colormaps[name](np.linspace(0.0, 1.0, 256))[:, :3]
        return (lut * 255.0 + 0.5).astype(np.uint8)
    except (ImportError, KeyError):
        ramp = np.linspace(0, 255, 256).astype(np.uint8)
        return np.stack([ramp, ramp, ramp], axis=1)


def lidar_valid_mask(clip: torch.Tensor) -> torch.Tensor:
    """Kept rays of a metric clip: positive range and a set validity channel."""
    return (clip[0] > 0.0) & (clip[2] >= 0.5)


def render_lidar_range_frames(
    clip: torch.Tensor, *, min_range_m: float, max_range_m: float
) -> np.ndarray:
    """Colorize metric range ``[3, T, H, W]`` to uint8 ``[T, H, W, 3]``; dropped rays black."""
    range_m = clip[0].detach().float().cpu().numpy()
    valid = lidar_valid_mask(clip.detach().float().cpu()).numpy()
    unit = np.clip((range_m - min_range_m) / (max_range_m - min_range_m), 0.0, 1.0)
    frames = _colormap_lut()[(unit * 255.0 + 0.5).astype(np.int64)]
    frames[~valid] = 0
    return frames


def render_lidar_intensity_frames(clip: torch.Tensor) -> np.ndarray:
    """Grayscale unit intensity ``[3, T, H, W]`` to uint8 ``[T, H, W, 3]``; dropped rays black."""
    intensity = clip[1].detach().float().cpu()
    valid = lidar_valid_mask(clip.detach().float().cpu())
    gray = torch.where(valid, intensity.clamp(0.0, 1.0), torch.zeros_like(intensity))
    gray = (gray * 255.0).round().to(torch.uint8).numpy()
    return np.repeat(gray[..., None], 3, axis=-1)


def write_lidar_outputs(
    clip: torch.Tensor,
    *,
    directory: str,
    stem: str,
    fps: float,
    min_range_m: float,
    max_range_m: float,
) -> dict[str, str]:
    """Write ``<stem>_lidar.safetensors`` and the two preview videos; return their paths."""
    import imageio
    from safetensors.torch import save_file

    os.makedirs(directory, exist_ok=True)
    clip = clip.detach().float().cpu()
    valid = lidar_valid_mask(clip)
    zeros = torch.zeros_like(clip[0])
    tensors = {
        "range_m": torch.where(valid, clip[0], zeros).contiguous(),
        "intensity": torch.where(valid, clip[1].clamp(0.0, 1.0), zeros)
        .to(torch.float16)
        .contiguous(),
        "validity": valid.to(torch.uint8).contiguous(),
    }
    rangemap_path = os.path.join(directory, stem + LIDAR_RANGEMAP_SUFFIX)
    save_file(
        tensors,
        rangemap_path,
        metadata={
            "layout": "[sweeps, beams, azimuth]",
            "fps": str(fps),
            "min_range_m": str(min_range_m),
            "max_range_m": str(max_range_m),
            "invalid_range_m": "0.0",
        },
    )
    # safetensors writes through a private temp file (mode 0600); match the
    # umask-governed permissions the video next to it gets.
    os.chmod(rangemap_path, 0o666 & ~_current_umask())
    files = {"rangemap": rangemap_path}
    videos = {
        "range_video": (
            stem + LIDAR_RANGE_VIDEO_SUFFIX,
            render_lidar_range_frames(
                clip, min_range_m=min_range_m, max_range_m=max_range_m
            ),
        ),
        "intensity_video": (
            stem + LIDAR_INTENSITY_VIDEO_SUFFIX,
            render_lidar_intensity_frames(clip),
        ),
    }
    for key, (name, frames) in videos.items():
        path = os.path.join(directory, name)
        # macro_block_size 8 divides 128 x 1800 exactly, so ffmpeg does not resize.
        imageio.mimsave(
            path, list(frames), fps=fps, codec="libx264", quality=8, macro_block_size=8
        )
        files[key] = path
    return files


def lidar_output_payload(
    clip: torch.Tensor,
    *,
    fps: float,
    min_range_m: float,
    max_range_m: float,
    files: dict[str, str],
    include_arrays: bool,
) -> dict[str, Any]:
    """The ``lidar`` block of a generation result; arrays only when asked for."""
    clip = clip.detach().float().cpu()
    valid = lidar_valid_mask(clip)
    payload: dict[str, Any] = {
        "files": dict(files),
        "sweeps": int(clip.shape[1]),
        "fps": float(fps),
        "height": int(clip.shape[2]),
        "width": int(clip.shape[3]),
        "min_range_m": float(min_range_m),
        "max_range_m": float(max_range_m),
        "valid_fraction": float(valid.float().mean().item()),
    }
    if include_arrays:
        payload["range_m"] = torch.where(
            valid, clip[0], torch.zeros_like(clip[0])
        ).numpy()
        payload["intensity"] = (
            torch.where(valid, clip[1].clamp(0.0, 1.0), torch.zeros_like(clip[1]))
            .to(torch.float16)
            .numpy()
        )
        payload["validity"] = valid.numpy()
    return payload


def lidar_payload_for_response(payload: dict[str, Any]) -> dict[str, Any]:
    """The JSON-safe part of a ``lidar`` block: everything but the arrays."""
    return {key: value for key, value in payload.items() if key not in _ARRAY_KEYS}
