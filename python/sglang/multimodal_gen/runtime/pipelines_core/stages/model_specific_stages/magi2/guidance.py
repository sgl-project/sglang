# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def skimming_mask(
    *,
    latent: torch.Tensor,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Elements where all three sign tests agree; only those get the softened scale."""
    latent = latent.to(device=cond.device, dtype=cond.dtype)
    guided = (latent - uncond) + guidance_scale * ((latent - cond) - (latent - uncond))
    denoised = latent - guided

    matching_pred_signs = (cond - uncond).sign() == cond.sign()
    matching_diff_after = (
        cond.sign() == (cond * guidance_scale - uncond * (guidance_scale - 1)).sign()
    )
    deviation_influence = denoised.sign() == (denoised - latent).sign()
    return matching_pred_signs & matching_diff_after & deviation_influence


def skimmed_uncond(
    *,
    latent: torch.Tensor,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    guidance_scale: float,
    skimmed_scale: float,
) -> torch.Tensor:
    """Two rounds with swapped mask arguments (sampler.py:326-331)."""
    # An all-zero uncond has nothing to damp, and the sign tests would read zeros
    # as agreement.
    if guidance_scale <= 1.0 or not bool(uncond.any()):
        return uncond

    fallback_weight = (skimmed_scale - 1) / (guidance_scale - 1)
    mask = skimming_mask(
        latent=latent, cond=cond, uncond=uncond, guidance_scale=guidance_scale
    )
    uncond = torch.where(mask, torch.lerp(cond, uncond, fallback_weight), uncond)

    mask = skimming_mask(
        latent=latent, cond=uncond, uncond=cond, guidance_scale=guidance_scale
    )
    return torch.where(mask, torch.lerp(cond, uncond, fallback_weight), uncond)


def apply_guidance(
    *,
    latent: torch.Tensor,
    cond: torch.Tensor,
    uncond: torch.Tensor,
    guidance_scale: float,
    skimmed_scale: float | None = None,
) -> torch.Tensor:
    if skimmed_scale is not None:
        uncond = skimmed_uncond(
            latent=latent,
            cond=cond,
            uncond=uncond,
            guidance_scale=guidance_scale,
            skimmed_scale=skimmed_scale,
        )
    return uncond + guidance_scale * (cond - uncond)
