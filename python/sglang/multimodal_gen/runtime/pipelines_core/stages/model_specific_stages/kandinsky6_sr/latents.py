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


def _cast_floating(value: torch.Tensor, dtype: torch.dtype | None) -> torch.Tensor:
    """Cast floating-point ``value`` to ``dtype`` (no-op for ``None`` / non-float / same dtype)."""
    if dtype is not None and value.is_floating_point() and value.dtype != dtype:
        return value.to(dtype=dtype)
    return value


def build_initial_latent(  # noqa: PLR0913
    *,
    instruct_type: InstructType,
    visual_cond: bool,
    in_visual_dim: int,
    lq_latent: torch.Tensor,
    batch_size: int,
    duration: int,
    height: int,
    width: int,
    device: str | int | torch.device,
    seed: int,
    lq_noise_scale: float,
    lq_noise_type: LqNoiseType,
    lq_channel_noise_scale: float,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build [starting | conditioning | mask] for the selected instruction mode.

    Inputs and output use [B*T, H, W, C]; the output has 2C+1 channels except
    unconditioned noise mode. Cast LQ to the DiT dtype before drawing noise."""
    lq_latent = lq_latent.to(device)
    lq_latent = _cast_floating(lq_latent, dtype)

    if instruct_type == "noise":
        g_noise = torch.Generator(device=device)
        g_noise.manual_seed(seed)
        degraded_lq = degrade_lq_latent(
            lq_latent, lq_noise_scale, lq_noise_type, generator=g_noise
        )
        if not visual_cond:
            # No conditioning channels in the model input layer (e.g. the
            # t2v-initialized KVAE SR DiT) — feed the bare C-channel latent.
            return degraded_lq
        zero_cond = torch.zeros_like(degraded_lq)
        zero_mask = torch.zeros(
            [*degraded_lq.shape[:-1], 1], dtype=degraded_lq.dtype, device=device
        )
        return torch.cat([degraded_lq, zero_cond, zero_mask], dim=-1)

    if instruct_type in ("channel", "hybrid"):
        if instruct_type == "hybrid":
            g_noise = torch.Generator(device=device)
            g_noise.manual_seed(seed)
            starting = degrade_lq_latent(
                lq_latent.clone(), lq_noise_scale, lq_noise_type, generator=g_noise
            )
        else:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            starting = torch.randn(
                batch_size * duration,
                height,
                width,
                in_visual_dim,
                device=device,
                generator=g,
            )
        channel_lq = degrade_lq_latent(
            lq_latent, lq_channel_noise_scale, noise_type="linear"
        )
        mask = torch.ones_like(lq_latent[..., :1])
        return torch.cat([starting, channel_lq, mask], dim=-1)

    if instruct_type == "hybrid_anchor":
        # Anchor-free inference (tiled real-world LQ has no HR ground truth):
        # a zeroed anchor + zeroed mask runs the model with no anchor signal,
        # equivalent to anchor_indices=[] for the packed-latent datasets.
        anchor_latent = torch.zeros_like(lq_latent)
        anchor_mask = torch.zeros_like(lq_latent[..., :1])
        g_noise = torch.Generator(device=device)
        g_noise.manual_seed(seed)
        starting = degrade_lq_latent(
            lq_latent.clone(), lq_noise_scale, lq_noise_type, generator=g_noise
        )
        anchor = degrade_lq_latent(
            anchor_latent, lq_channel_noise_scale, noise_type="linear"
        )
        return torch.cat([starting, anchor, anchor_mask], dim=-1)

    msg = f"generate_sample_sr does not support instruct_type={instruct_type!r}"
    raise ValueError(msg)


__all__ = [
    "InstructType",
    "LqNoiseType",
    "build_initial_latent",
    "degrade_lq_latent",
]
