# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

BETA_START = 0.00085
BETA_END = 0.0120
NUM_TRAIN_TIMESTEPS = 1000


def zero_snr_sigmas(
    *, num_timesteps: int = NUM_TRAIN_TIMESTEPS, device: torch.device | None = None
) -> torch.Tensor:
    """Must stay DESCENDING: index 220 selects 0.839, not the ascending table's 0.152 (inference_engine.py:873, :844)."""
    betas = (
        torch.linspace(
            BETA_START**0.5,
            BETA_END**0.5,
            num_timesteps,
            dtype=torch.float64,
        )
        ** 2
    )
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
    sqrt_alphas = alphas_cumprod.sqrt()

    first, last = sqrt_alphas[0].clone(), sqrt_alphas[-1].clone()
    sqrt_alphas = (sqrt_alphas - last) * (first / (first - last))

    return sqrt_alphas.to(device=device, dtype=torch.float32)


def upsample_latent(latent: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    # Time goes to 2 * T - 1 on the refiner grid.
    frames = latent.shape[2]
    return torch.nn.functional.interpolate(
        latent.float(),
        size=(2 * frames - 1, height, width),
        mode="trilinear",
        align_corners=True,
    )


def renoise(
    latent: torch.Tensor,
    *,
    noise: torch.Tensor,
    sigma_index: int,
    sigmas: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mix ``latent`` with ``noise``; ``sigma`` is a signal coefficient, not a noise level."""
    table = sigmas if sigmas is not None else zero_snr_sigmas(device=latent.device)
    sigma = table[sigma_index].to(latent.dtype)
    return sigma * latent + (1.0 - sigma**2).sqrt() * noise
