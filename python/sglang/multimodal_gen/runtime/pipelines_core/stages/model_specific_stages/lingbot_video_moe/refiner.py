# Adapted from LingBot-Video (https://github.com/Robbyant/lingbot-video).
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch


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
    base = np.linspace(float(sigma_max), float(sigma_min), steps + 1)[:-1]
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
    # The threshold insertion guarantees a nonempty 1-D schedule starting at t_thresh.
    if not ((0 <= sigmas) & (sigmas <= 1)).all() or not (np.diff(sigmas) < 0).all():
        raise ValueError("refiner sigmas must be in [0, 1] and strictly descending")
    return sigmas.astype(np.float32)


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
