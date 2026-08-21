# SPDX-License-Identifier: Apache-2.0
"""Native SenseNova U1 vision embedding tower."""

from __future__ import annotations

import torch
from torch import nn


def precompute_rope_freqs_sincos(
    dim: int,
    max_position: int,
    *,
    base: float,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        base
        ** (
            torch.arange(
                0,
                dim,
                2,
                dtype=torch.float32,
                device=device,
            )
            / dim
        )
    )
    positions = torch.arange(
        max_position,
        dtype=torch.float32,
        device=device,
    )
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def build_abs_positions_from_grid_hw(
    grid_hw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid_hw = grid_hw.long()
    heights = grid_hw[:, 0]
    widths = grid_hw[:, 1]
    patch_counts = heights * widths
    total_patches = int(patch_counts.sum().item())
    patch_to_sample = torch.repeat_interleave(
        torch.arange(grid_hw.shape[0], device=grid_hw.device),
        patch_counts,
    )
    offsets = torch.cumsum(
        torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=grid_hw.device),
                patch_counts[:-1],
            ]
        ),
        dim=0,
    )
    patch_ids = torch.arange(total_patches, device=grid_hw.device)
    within_sample = patch_ids - offsets[patch_to_sample]
    width_per_patch = widths[patch_to_sample]
    return within_sample % width_per_patch, within_sample // width_per_patch


def _apply_rotary_1d(
    hidden_states: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    cos = cos_cache[positions]
    sin = sin_cache[positions]
    even = hidden_states[..., 0::2]
    odd = hidden_states[..., 1::2]
    output = torch.empty_like(hidden_states)
    output[..., 0::2] = even * cos - odd * sin
    output[..., 1::2] = even * sin + odd * cos
    return output


class NEOVisionModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.llm_embed_dim = config.llm_hidden_size
        self.patch_size = config.patch_size
        self.downsample_factor = int(1 / config.downsample_ratio)

        self.patch_embedding = nn.Conv2d(
            config.num_channels,
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.dense_embedding = nn.Conv2d(
            self.embed_dim,
            self.llm_embed_dim,
            kernel_size=self.downsample_factor,
            stride=self.downsample_factor,
        )
        self.activation = nn.GELU()

        rope_dim = self.embed_dim // 2
        cos_x, sin_x = precompute_rope_freqs_sincos(
            rope_dim,
            config.max_position_embeddings_vision,
            base=config.rope_theta_vision,
        )
        cos_y, sin_y = precompute_rope_freqs_sincos(
            rope_dim,
            config.max_position_embeddings_vision,
            base=config.rope_theta_vision,
        )
        self.register_buffer("cos_cached_x", cos_x, persistent=False)
        self.register_buffer("sin_cached_x", sin_x, persistent=False)
        self.register_buffer("cos_cached_y", cos_y, persistent=False)
        self.register_buffer("sin_cached_y", sin_y, persistent=False)
        self.rope_dim = rope_dim

    @property
    def device(self) -> torch.device:
        return self.patch_embedding.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embedding.weight.dtype

    def _apply_2d_rope(
        self,
        patch_embeddings: torch.Tensor,
        grid_hw: torch.Tensor,
    ) -> torch.Tensor:
        self._ensure_fp32_rope_cache()
        abs_x, abs_y = build_abs_positions_from_grid_hw(grid_hw)
        half_dim = patch_embeddings.shape[-1] // 2
        x_part = _apply_rotary_1d(
            patch_embeddings[..., :half_dim].float(),
            self.cos_cached_x.float(),
            self.sin_cached_x.float(),
            abs_x,
        )
        y_part = _apply_rotary_1d(
            patch_embeddings[..., half_dim:].float(),
            self.cos_cached_y.float(),
            self.sin_cached_y.float(),
            abs_y,
        )
        return torch.cat([x_part, y_part], dim=-1).to(self.dtype)

    def _ensure_fp32_rope_cache(self) -> None:
        expected_shape = (
            self.config.max_position_embeddings_vision,
            self.rope_dim // 2,
        )
        caches = (
            self.cos_cached_x,
            self.sin_cached_x,
            self.cos_cached_y,
            self.sin_cached_y,
        )
        if all(
            cache.device == self.device
            and cache.dtype == torch.float32
            and tuple(cache.shape) == expected_shape
            for cache in caches
        ):
            return

        cos_x, sin_x = precompute_rope_freqs_sincos(
            self.rope_dim,
            self.config.max_position_embeddings_vision,
            base=self.config.rope_theta_vision,
            device=self.device,
        )
        cos_y, sin_y = precompute_rope_freqs_sincos(
            self.rope_dim,
            self.config.max_position_embeddings_vision,
            base=self.config.rope_theta_vision,
            device=self.device,
        )
        self.cos_cached_x = cos_x
        self.sin_cached_x = sin_x
        self.cos_cached_y = cos_y
        self.sin_cached_y = sin_y

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_hw: torch.Tensor,
    ) -> torch.Tensor:
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
        grid_hw = grid_hw.to(device=self.device, dtype=torch.long)
        pixels = pixel_values.reshape(
            -1,
            self.config.num_channels,
            self.patch_size,
            self.patch_size,
        )
        patch_embeddings = self.activation(self.patch_embedding(pixels))
        patch_embeddings = patch_embeddings.flatten(1)
        patch_embeddings = self._apply_2d_rope(patch_embeddings, grid_hw)

        outputs = []
        offset = 0
        for height, width in grid_hw.tolist():
            patch_count = height * width
            image = patch_embeddings[offset : offset + patch_count]
            image = image.reshape(1, height, width, self.embed_dim)
            image = self.dense_embedding(image.permute(0, 3, 1, 2))
            outputs.append(image.permute(0, 2, 3, 1).flatten(0, 2))
            offset += patch_count
        return torch.cat(outputs, dim=0)


__all__ = [
    "NEOVisionModel",
    "build_abs_positions_from_grid_hw",
]
