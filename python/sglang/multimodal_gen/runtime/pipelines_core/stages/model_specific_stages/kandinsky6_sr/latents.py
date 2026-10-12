# SPDX-License-Identifier: Apache-2.0
# Vendored from sr_core (parity-tested against the k6_video reference); adapted from
# the Kandinsky 6 SR inference reference (k6_video, Apache-2.0).
"""SR latent construction with reference RNG ordering.

Initial noise uses a per-request generator; conditioning noise uses the global RNG."""

from __future__ import annotations

from typing import Literal

import torch

InstructType = Literal["noise", "channel", "hybrid", "hybrid_anchor"]
LqNoiseType = Literal["linear", "ddpm"]


def degrade_lq_latent(
    lq_latent: torch.Tensor,
    noise_scale: float = 0.7,
    noise_type: LqNoiseType = "linear",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mix linear or variance-preserving (ddpm) noise; nonpositive scales are a no-op."""
    if noise_scale <= 0:
        return lq_latent
    eps = torch.randn(
        lq_latent.shape,
        device=lq_latent.device,
        dtype=lq_latent.dtype,
        generator=generator,
    )
    if noise_type == "ddpm":
        return (1 - noise_scale**2) ** 0.5 * lq_latent + noise_scale * eps
    return (1 - noise_scale) * lq_latent + noise_scale * eps


def build_initial_latent(
    *,
    instruct_type: InstructType,
    visual_cond: bool,
    in_visual_dim: int,
    lq_latent: torch.Tensor,
    device: str | int | torch.device,
    seed: int,
    lq_noise_scale: float,
    lq_noise_type: LqNoiseType,
    lq_channel_noise_scale: float,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build packed [B*T, H, W, C] noise and optional conditioning without reseeding global RNG."""
    if instruct_type not in ("noise", "channel", "hybrid", "hybrid_anchor"):
        raise ValueError(f"Unsupported SR instruct_type={instruct_type!r}")
    lq_latent = lq_latent.to(device=device, dtype=dtype)
    generator = torch.Generator(device=device).manual_seed(seed)
    if instruct_type == "channel":
        starting = torch.randn(
            (*lq_latent.shape[:-1], in_visual_dim), device=device, generator=generator
        )
    else:
        starting = degrade_lq_latent(
            lq_latent, lq_noise_scale, lq_noise_type, generator=generator
        )
    if instruct_type == "noise":
        if not visual_cond:
            return starting
        conditioning = torch.zeros_like(starting)
        mask = torch.zeros_like(starting[..., :1])
    elif instruct_type == "hybrid_anchor":
        # no HR anchor exists at inference; conditioning noise still uses the global RNG
        conditioning = degrade_lq_latent(
            torch.zeros_like(lq_latent), lq_channel_noise_scale, noise_type="linear"
        )
        mask = torch.zeros_like(lq_latent[..., :1])
    else:
        conditioning = degrade_lq_latent(
            lq_latent, lq_channel_noise_scale, noise_type="linear"
        )
        mask = torch.ones_like(lq_latent[..., :1])
    return torch.cat([starting, conditioning, mask], dim=-1)
