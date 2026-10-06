# SPDX-License-Identifier: Apache-2.0
# Vendored from sr_core (parity-tested against the k6_video reference); adapted from
# the Kandinsky 6 SR inference reference (k6_video, Apache-2.0).
"""Pure latent-construction functions for Kandinsky 6 video SR (no model object).

Ported from ``core/algo/train_utils.py`` (``degrade_lq_latent``) and ``core/algo/utils.py``
(``_build_initial_latent``). The timestep-schedule half of this module (the inline pi-Flow
segment schedule of ``piflow_generate`` and the warped Euler ``linspace`` of ``generate``) was
dropped: the denoising stage now drives the bundle's own diffusers scheduler
(``FlowMatchEulerDiscreteScheduler.set_timesteps`` / ``PiflowScheduler.set_timesteps``) instead
of re-deriving that schedule here -- see ``sampling.denoise_with_scheduler``.

RNG semantics are the reference's: ``torch.Generator(device=device).manual_seed(seed)``
seeds the initial noise, so results are bit-identical for the same
``device``/``seed``/dtype. Note (faithful to the reference) that the
channel-conditioning noise (``lq_channel_noise_scale``) is drawn from torch's
*global* RNG, not from the seeded generator.
"""

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
    """Mix random Gaussian noise into an LQ latent.

    Args:
        lq_latent: LQ latent tensor of arbitrary shape.
        noise_scale: Noise fraction ``s`` in ``(0, 1)``. When ``<= 0``, the tensor
            is returned unchanged (the very same object).
        noise_type: ``"linear"`` for ``(1-s)*lq + s*eps`` or ``"ddpm"`` for
            ``sqrt(1-s²)*lq + s*eps`` (variance-preserving).
        generator: Optional RNG generator (on ``lq_latent``'s device) for
            deterministic noise sampling; ``None`` uses torch's global RNG.

    Returns:
        Degraded LQ latent with the same shape and dtype as the input.
    """
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
    """Build the initial latent tensor for the SR denoising loop.

    Constructs ``[starting | cond | mask]`` depending on ``instruct_type``:

    - ``"noise"``: degraded LQ as starting point; with ``visual_cond`` also a zero
      LQ cond + zero mask, otherwise the bare degraded LQ latent.
    - ``"channel"``: random Gaussian noise as starting point, LQ cond + ones mask.
    - ``"hybrid"``: degraded LQ as starting point, LQ cond + ones mask.
    - ``"hybrid_anchor"``: anchor-free only — degraded LQ as starting point, a
      zeroed HR anchor (noised by ``lq_channel_noise_scale``) + a zeroed anchor
      mask, as the tiled path runs it.

    Args:
        instruct_type: One of ``"noise"``, ``"channel"``, ``"hybrid"``, ``"hybrid_anchor"``.
        visual_cond: ``dit.visual_cond`` — whether the DiT input layer has conditioning
            channels (only consulted for ``"noise"``).
        in_visual_dim: ``dit.in_visual_dim`` — latent channel count (only used by ``"channel"``).
        lq_latent: Scaled LQ latent ``[batch_size*duration, H, W, C]``.
        batch_size: Batch size (``bs``).
        duration: Number of temporal frames per sample.
        height: Latent height.
        width: Latent width.
        device: Target device (also the device of the noise generator).
        seed: RNG seed for the starting-noise generator.
        lq_noise_scale: Noise fraction mixed into the LQ starting point.
        lq_noise_type: ``"linear"`` or ``"ddpm"`` noising formula.
        lq_channel_noise_scale: Noise fraction mixed into the channel-cat
            conditioning (LQ for ``channel``/``hybrid``, anchor for ``hybrid_anchor``).
        dtype: Floating dtype to cast ``lq_latent`` to (the DiT's parameter dtype in
            the reference); ``None`` keeps ``lq_latent``'s dtype.

    Returns:
        Initial latent tensor ready for the denoising loop, ``[batch_size*duration, H, W, 2C+1]``
        (``[.., C]`` for ``"noise"`` without ``visual_cond``).

    Raises:
        ValueError: If ``instruct_type`` is not supported.
    """
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
