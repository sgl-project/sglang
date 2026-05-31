# SPDX-License-Identifier: Apache-2.0
"""
GPU-native latent upsample operations for progressive resolution growing.

All ops run entirely on GPU via torch.fft — no CPU↔GPU data movement.
Supported modes: "dct", "dct_rewind".

Each function takes a spatial latent tensor (..., H, W) and returns a 2× larger
tensor (..., 2H, 2W).  The rewind variant also returns t_eff.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.spectral_ops import (
    dct_2d,
    idct_2d,
)


def dct_upsample_2d(
    x: torch.Tensor,
    sigma_t: float,
    seed: int,
    rewind: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, float]:
    """DCT-II 2× upsample: embed low-res coefficients top-left, noise-pad, IDCT.

    x: (..., H, W) spatial latent tensor.
    sigma_t: current noise level (used to scale the high-freq padding noise).
    seed: deterministic RNG seed for the noise padding.
    rewind: if True, multiply by 2/(1+sigma_t) and return (result, t_eff).

    Matches the CPU reference in inference_progressive.py but runs fully on GPU.
    """
    *leading, H, W = x.shape
    H2, W2 = H * 2, W * 2

    # 2-D DCT-II of the source (ortho-normalized, Parseval identity preserved)
    X_low = dct_2d(x.float(), norm="ortho")  # (..., H, W)

    # Fill 2N×2N grid with white Gaussian noise of variance sigma_t^2 per bin
    gen = torch.Generator(device=x.device)
    gen.manual_seed(seed)
    big = torch.randn(*leading, H2, W2, generator=gen, dtype=x.dtype, device=x.device)
    big = big * sigma_t

    # Embed low-res DCT coefficients in the top-left corner
    big[..., :H, :W] = X_low.to(x.dtype)

    # 2-D IDCT-II → spatial domain
    result = idct_2d(big.float(), norm="ortho").to(x.dtype)

    if rewind:
        gamma = 1.0 + sigma_t
        result = result * (2.0 / gamma)
        t_eff = 2.0 * sigma_t / gamma
        return result, t_eff
    return result


def apply_upsample(
    x: torch.Tensor,
    sigma_t: float,
    seed: int,
    mode: str,
) -> torch.Tensor | tuple[torch.Tensor, float]:
    """Dispatch to the requested upsample function.

    Returns tensor for plain modes, (tensor, t_eff) for rewind modes.
    """
    if mode == "dct":
        return dct_upsample_2d(x, sigma_t, seed, rewind=False)
    if mode == "dct_rewind":
        return dct_upsample_2d(x, sigma_t, seed, rewind=True)
    raise ValueError(f"Unsupported progressive upsample mode: {mode!r}")
