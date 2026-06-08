# SPDX-License-Identifier: Apache-2.0
"""
GPU-native latent upsample operations for progressive resolution growing.

All ops run entirely on GPU via torch.fft — no CPU↔GPU data movement.
Supported modes: "dct", "dct_rewind".

Each function takes a spatial latent tensor (..., H, W) and returns a 2× larger
tensor (..., 2H, 2W).  The rewind variant also returns t_eff.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.progressive_resolution.spectral_ops import (
    dct_2d,
    idct_2d,
)


def dct_upsample_2d(
    x: torch.Tensor,
    sigma_t: float,
    seed: int | Sequence[int],
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

    # 2-D DCT-II of the source (ortho-normalized, Parseval identity preserved).
    # All intermediate computation stays in float32 to match the reference
    # (inference_progressive.py uses scipy float32 throughout).  bfloat16 has
    # only 7 mantissa bits; quantising the DCT coefficients before IDCT would
    # introduce mean absolute error ~0.8 against an output range of ±4.
    X_low = dct_2d(x.float(), norm="ortho")  # (..., H, W) float32

    # Fill 2N×2N grid with float32 white Gaussian noise of variance sigma_t²
    # per DCT bin, matching the reference's float32 noise path.
    if isinstance(seed, Sequence) and not isinstance(seed, (str, bytes)):
        if not leading or len(seed) != leading[0]:
            batch_dim = leading[0] if leading else 0
            raise ValueError(
                "seed list length must match leading batch dimension: "
                f"{len(seed)} vs {batch_dim}"
            )
        big = torch.cat(
            [
                torch.randn(
                    1,
                    *leading[1:],
                    H2,
                    W2,
                    generator=torch.Generator(device=x.device).manual_seed(int(item)),
                    dtype=torch.float32,
                    device=x.device,
                )
                for item in seed
            ],
            dim=0,
        )
    else:
        generator = torch.Generator(device=x.device).manual_seed(int(seed))
        big = torch.randn(
            *leading, H2, W2, generator=generator, dtype=torch.float32, device=x.device
        )
    big = big * sigma_t

    # Embed low-res DCT coefficients in the top-left corner (no precision loss).
    big[..., :H, :W] = X_low

    # 2-D IDCT-II → spatial domain, then cast back to original dtype.
    result = idct_2d(big, norm="ortho").to(x.dtype)

    if rewind:
        gamma = 1.0 + sigma_t
        result = result * (2.0 / gamma)
        t_eff = 2.0 * sigma_t / gamma
        return result, t_eff
    return result


def apply_upsample(
    x: torch.Tensor,
    sigma_t: float,
    seed: int | Sequence[int],
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
