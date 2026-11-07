# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
# (nodes/minimax_h3_latent_upscaler_3d.py, MIT License)
"""3D-convolution latent upscaler for MiniMax-H3 video latents.

A scale-conditioned ResBlock stack before and after a trilinear resize of the
24-channel latent; time is never resized. Weights:
LBH-123-AI/Minimax_h3_latent_Upscaler (also shipped by
Efficient-Large-Model/Open-H3-Regenerate-2K under ``latent_upsampler/``).
"""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from torch import nn


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels)


class _ResBlockEmb3D(nn.Module):
    def __init__(self, channels: int, emb_channels: int) -> None:
        super().__init__()
        self.in_layers = nn.Sequential(
            _norm(channels), nn.SiLU(), nn.Conv3d(channels, channels, 3, padding=1)
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * channels)
        )
        self.out_norm = _norm(channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Dropout(), nn.Conv3d(channels, channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        scale, shift = self.emb_layers(emb).type(h.dtype)[..., None, None, None].chunk(
            2, dim=1
        )
        h = self.out_layers(self.out_norm(h) * (1 + scale) + shift)
        return x + h


class _TemporalConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.norm = _norm(channels)
        self.dwconv = nn.Conv3d(
            channels,
            channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(kernel_size // 2, 0, 0),
            groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))


class MiniMaxH3LatentUpscaler(nn.Module):
    """Normalized latent [B, 24, T, h, w] -> normalized latent [B, 24, T, H, W]."""

    def __init__(
        self,
        *,
        in_channels: int = 24,
        in_blocks: int = 12,
        out_blocks: int = 12,
        channels: int = 512,
        temporal_every: int = 2,
        temporal_kernel: int = 5,
    ) -> None:
        super().__init__()
        embed_dim = 64
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim)
        )

        def stack(num_blocks: int) -> nn.ModuleList:
            blocks = nn.ModuleList()
            for index in range(num_blocks):
                blocks.append(_ResBlockEmb3D(channels, embed_dim))
                if temporal_every > 0 and index % temporal_every == 0:
                    blocks.append(_TemporalConv(channels, temporal_kernel))
            return blocks

        self.in_blocks = stack(in_blocks)
        self.out_blocks = stack(out_blocks)
        self.norm_out = _norm(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    @classmethod
    def from_state_dict(cls, state_dict: dict[str, torch.Tensor]):
        """Build the architecture the checkpoint describes (the node's detection)."""

        def count(pattern: str) -> int:
            return len({m.group(1) for k in state_dict if (m := re.match(pattern, k))})

        temporal = [k for k in state_dict if k.endswith("dwconv.weight")]
        model = cls(
            in_channels=state_dict["conv_in.weight"].shape[1],
            channels=state_dict["conv_in.weight"].shape[0],
            in_blocks=count(r"in_blocks\.(\d+)\.in_layers\."),
            out_blocks=count(r"out_blocks\.(\d+)\.in_layers\."),
            temporal_every=2 if temporal else 0,
            temporal_kernel=state_dict[temporal[0]].shape[2] if temporal else 5,
        )
        model.load_state_dict(state_dict, strict=True)
        return model

    def forward(
        self, x: torch.Tensor, *, size: tuple[int, int, int], scale: float
    ) -> torch.Tensor:
        emb = self.embed(
            torch.tensor([[scale - 1.0]], dtype=x.dtype, device=x.device)
        ).expand(x.shape[0], -1)
        x = self.conv_in(x)
        for block in self.in_blocks:
            x = block(x, emb) if isinstance(block, _ResBlockEmb3D) else block(x)
        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)
        for block in self.out_blocks:
            x = block(x, emb) if isinstance(block, _ResBlockEmb3D) else block(x)
        return self.conv_out(F.silu(self.norm_out(x)))

    @torch.inference_mode()
    def upscale(
        self,
        latent: torch.Tensor,
        *,
        target_hw: tuple[int, int],
        mean: torch.Tensor,
        std: torch.Tensor,
    ) -> torch.Tensor:
        """The node's TARGET_DIMENSIONS call: (x - mean) / std -> model -> back.

        ``latent`` is [B, 24, T, h, w]; ``mean`` / ``std`` broadcast over it.
        The scale fed to the embedding is the mean of the two axis ratios; the
        (de)normalization runs in the model dtype, as the node does.
        """
        h, w = latent.shape[-2:]
        scale = (target_hw[1] / w + target_hw[0] / h) / 2.0
        dtype = next(self.parameters()).dtype
        mean, std = mean.to(dtype), std.to(dtype)
        x = (latent.to(dtype) - mean) / std
        out = self(x, size=(latent.shape[2], *target_hw), scale=scale)
        return (out * std + mean).float()
