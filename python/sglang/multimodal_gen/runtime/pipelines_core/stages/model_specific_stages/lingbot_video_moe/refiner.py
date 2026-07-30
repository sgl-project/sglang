# Adapted from LingBot-Video (https://github.com/Robbyant/lingbot-video).
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch


def validate_refiner_sigmas(
    sigmas: np.ndarray, t_thresh: float | None = None
) -> np.ndarray:
    arr = np.asarray(list(sigmas), dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("refiner sigma schedule must be a non-empty 1D list")
    if not np.all(np.isfinite(arr)):
        raise ValueError("refiner sigma schedule contains non-finite values")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(
            f"refiner sigma schedule values must be in [0, 1], got {arr.tolist()}"
        )
    if arr.size > 1 and not np.all(np.diff(arr) < 0.0):
        raise ValueError(
            f"refiner sigma schedule must be strictly descending, got {arr.tolist()}"
        )
    if t_thresh is not None and abs(float(arr[0]) - float(t_thresh)) > 1e-6:
        raise ValueError(
            f"refiner sigma schedule must start at t_thresh={float(t_thresh)}, "
            f"got {float(arr[0])}"
        )
    return arr


def compute_refiner_sigmas(
    *,
    sigma_max: float,
    sigma_min: float,
    num_inference_steps: int,
    shift: float,
    t_thresh: float,
    tail_steps: int = 0,
) -> np.ndarray:
    """Sigma schedule starting at t_thresh, with the flow shift already applied."""

    if not 0.0 < float(t_thresh) <= 1.0:
        raise ValueError(f"refiner t_thresh must lie in (0, 1], got {t_thresh}")
    steps = int(num_inference_steps)
    if steps < 1:
        raise ValueError(f"num_inference_steps must be >= 1, got {steps}")
    tail = int(tail_steps)
    if tail < 0:
        raise ValueError(f"refiner_sigma_tail_steps must be >= 0, got {tail}")

    t_value = float(t_thresh)
    base = np.linspace(float(sigma_max), float(sigma_min), steps + 1).copy()[:-1]
    shifted = shift * base / (1.0 + (shift - 1.0) * base)
    eps = 1e-6
    sigmas = shifted[shifted <= t_value + eps]
    if sigmas.size == 0 or abs(float(sigmas[0]) - t_value) > eps:
        sigmas = np.concatenate([[t_value], sigmas])
    if tail > 0:
        start = float(sigmas[-1])
        stop = min(float(sigma_min), start)
        extra = np.linspace(start, stop, tail + 2, dtype=np.float64)[1:-1]
        sigmas = np.concatenate([sigmas, extra])
    return validate_refiner_sigmas(sigmas, t_value).astype(np.float32)


def prepare_refiner_latent(
    upscaled: torch.Tensor, noise: torch.Tensor, t_thresh: float
) -> torch.Tensor:
    return (1.0 - t_thresh) * upscaled + t_thresh * noise


def resize_video_pixels(pixels: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Bicubic resize of a B, C, T, H, W clip in 0..1."""

    batch, channels, frames = pixels.shape[:3]
    flat = pixels.transpose(1, 2).reshape(batch * frames, channels, *pixels.shape[3:])
    resized = torch.nn.functional.interpolate(
        flat.float(), size=(height, width), mode="bicubic", align_corners=False
    ).clamp(0.0, 1.0)
    return resized.reshape(batch, frames, channels, height, width).transpose(1, 2)
