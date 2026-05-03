# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class UGDenoiseSchedule:
    """Shifted denoise timesteps and adjacent deltas for UG visual steps."""

    timesteps: torch.Tensor
    dts: torch.Tensor


def build_ug_denoise_schedule(
    *,
    num_inference_steps: int,
    timestep_shift: float,
    device: torch.device | str | None = None,
) -> UGDenoiseSchedule:
    """Build a shifted visual denoise schedule."""

    num_inference_steps = int(num_inference_steps)
    timestep_shift = float(timestep_shift)
    if num_inference_steps <= 0:
        raise ValueError(
            "num_inference_steps must be positive, got " f"{num_inference_steps}"
        )
    if not math.isfinite(timestep_shift) or timestep_shift <= 0.0:
        raise ValueError(f"timestep_shift must be positive, got {timestep_shift!r}")

    timesteps = torch.linspace(1, 0, num_inference_steps, device=device)
    timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
    dts = timesteps[:-1] - timesteps[1:]
    return UGDenoiseSchedule(timesteps=timesteps[:-1], dts=dts)


def get_ug_effective_cfg_scales(
    cfg_source: Any,
    timestep: torch.Tensor,
) -> tuple[float, float]:
    """Resolve timestep-gated text/image CFG scales."""

    t = float(timestep.flatten()[0].detach().cpu())
    start, end = _cfg_interval(cfg_source)
    if t > start and t <= end:
        return (
            float(getattr(cfg_source, "cfg_text_scale", 1.0)),
            float(getattr(cfg_source, "cfg_img_scale", 1.0)),
        )
    return 1.0, 1.0


def _cfg_interval(cfg_source: Any) -> tuple[float, float]:
    interval = getattr(cfg_source, "cfg_interval", (0.0, 1.0))
    if len(interval) != 2:
        raise ValueError("cfg_interval must contain [start, end]")
    start, end = (float(interval[0]), float(interval[1]))
    if not (0.0 <= start <= end <= 1.0):
        raise ValueError("cfg_interval must satisfy 0 <= start <= end <= 1")
    return start, end
