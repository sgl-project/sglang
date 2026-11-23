# Copied and adapted from: https://github.com/runwayml/praxis-1-oss

# SPDX-License-Identifier: Apache-2.0
"""The frozen Wan2.2 video autoencoder of Praxis-1 on SGLang's ``AutoencoderKLWan``.

Frames go in as ``(1, 3, T, H, W)`` in [0, 255] and come out as the 192-channel policy latent: the 48
latent channels, normalized, with each 2 x 2 spatial patch moved into channels. The weights are rounded
float32 to float16 to bfloat16 and the convolutions run under float16 autocast, as in the reference.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import Tensor

from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEConfig
from sglang.multimodal_gen.runtime.models.vaes.wanvae import AutoencoderKLWan

_WEIGHTS_FILE = "diffusion_pytorch_model.safetensors"
_Z_DIM = 48


def to_policy_latent(z: Tensor) -> Tensor:
    """``(B, 48, T, h, w)`` to ``(B, 192, T, h / 2, w / 2)``."""
    b, c, t, h, w = z.shape
    z = z.view(b, c, t, h // 2, 2, w // 2, 2)
    return z.permute(0, 1, 4, 6, 2, 3, 5).reshape(b, c * 4, t, h // 2, w // 2)


def from_policy_latent(z: Tensor) -> Tensor:
    b, c4, t, h, w = z.shape
    z = z.view(b, c4 // 4, 2, 2, t, h, w)
    return z.permute(0, 1, 4, 5, 2, 6, 3).reshape(b, c4 // 4, t, h * 2, w * 2)


class PraxisVAE:
    def __init__(self, path: str | Path, *, device: torch.device) -> None:
        path = Path(path)
        config = WanVAEConfig(load_encoder=True, load_decoder=True)
        config.update_model_arch(json.loads((path / "config.json").read_text()))
        arch = config.arch_config
        if arch.z_dim != _Z_DIM:
            raise ValueError(
                f"expected a Wan2.2 VAE with {_Z_DIM} channels, got {arch.z_dim}"
            )
        model = AutoencoderKLWan(config)
        weights = load_file(str(path / _WEIGHTS_FILE))
        model.load_state_dict(
            {k: v.to(torch.float16).to(torch.bfloat16) for k, v in weights.items()}
        )
        self.model = model.to(device, torch.bfloat16).eval().requires_grad_(False)
        self.device = device
        self._mean = arch.shift_factor.to(device=device, dtype=torch.float32)
        self._inv_std = arch.scaling_factor.to(device=device, dtype=torch.float32)

    @torch.no_grad()
    def encode(self, frames: Tensor) -> Tensor:
        """Policy latent ``(1, 192, t, h, w)`` in bfloat16 of ``(1, 3, T, H, W)`` frames in [0, 255]."""
        unit = (frames.float() / 255) * 2 - 1
        with torch.autocast("cuda", dtype=torch.float16):
            mu = self.model.encode(unit).mode()
        z = (mu.float() - self._mean) * self._inv_std
        return to_policy_latent(z.to(torch.bfloat16))

    @torch.no_grad()
    def decode(self, latent: Tensor) -> Tensor:
        """uint8 frames ``(T', 3, H, W)`` of a policy latent ``(1, 192, T, h, w)``."""
        z = from_policy_latent(latent.to(torch.bfloat16))
        z = z / self._inv_std + self._mean
        with torch.autocast("cuda", dtype=torch.float16):
            video = self.model.decode(z)
        frames = video[0].permute(1, 0, 2, 3).float().mul(0.5).add(0.5).clamp(0, 1)
        return (frames * 255).to(torch.uint8)
