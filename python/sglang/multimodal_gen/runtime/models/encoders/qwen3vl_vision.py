# SPDX-License-Identifier: Apache-2.0
"""Native Qwen3-VL vision encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.encoders.qwen_vl_vision import (
    PackedSequenceMetadata,
    QwenVLVisionAttention,
)
from sglang.srt.models.qwen3_vl import (
    Qwen3_VisionMLP,
    Qwen3VLMoeVisionPatchMerger,
    Qwen3VLVisionPatchEmbed,
)
from sglang.srt.runtime_context import get_parallel


@dataclass(frozen=True)
class Qwen3VLVisionOutput:
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor
    deepstack_features: list[torch.Tensor]


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, sequence_length: int) -> torch.Tensor:
        positions = torch.arange(
            sequence_length,
            device=self.inv_freq.device,
            dtype=self.inv_freq.dtype,
        )
        return torch.outer(positions, self.inv_freq)


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        parallel = get_parallel()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = QwenVLVisionAttention(
            config,
            prefix=f"visual.blocks.{layer_idx}.attn",
            model_name="Qwen3-VL",
        )
        self.mlp = Qwen3_VisionMLP(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            hidden_act=config.hidden_act,
            prefix=f"visual.blocks.{layer_idx}.mlp",
            tp_rank=parallel.tp_rank,
            tp_size=parallel.tp_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        metadata: PackedSequenceMetadata,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            metadata=metadata,
            position_embeddings=position_embeddings,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


def _vision_position_ids(
    grid_thw: torch.Tensor, spatial_merge_size: int
) -> torch.Tensor:
    position_ids = []
    device = grid_thw.device
    for temporal, height, width in grid_thw.tolist():
        merged_height = height // spatial_merge_size
        merged_width = width // spatial_merge_size
        block_rows = torch.arange(merged_height, device=device)
        block_cols = torch.arange(merged_width, device=device)
        intra_rows = torch.arange(spatial_merge_size, device=device)
        intra_cols = torch.arange(spatial_merge_size, device=device)
        rows = (
            block_rows[:, None, None, None] * spatial_merge_size
            + intra_rows[None, None, :, None]
        )
        cols = (
            block_cols[None, :, None, None] * spatial_merge_size
            + intra_cols[None, None, None, :]
        )
        rows = rows.expand(
            merged_height, merged_width, spatial_merge_size, spatial_merge_size
        ).reshape(-1)
        cols = cols.expand(
            merged_height, merged_width, spatial_merge_size, spatial_merge_size
        ).reshape(-1)
        coordinates = torch.stack((rows, cols), dim=-1)
        position_ids.append(coordinates.repeat(temporal, 1))
    return torch.cat(position_ids)


def _vision_bilinear_indices_and_weights(
    grid_thw: torch.Tensor,
    num_grid_per_side: int,
    spatial_merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    index_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
    device = grid_thw.device

    for temporal, height, width in grid_thw.tolist():
        height_positions = torch.linspace(
            0, num_grid_per_side - 1, height, device=device
        )
        width_positions = torch.linspace(0, num_grid_per_side - 1, width, device=device)
        height_floor = height_positions.int()
        width_floor = width_positions.int()
        height_ceil = (height_floor + 1).clip(max=num_grid_per_side - 1)
        width_ceil = (width_floor + 1).clip(max=num_grid_per_side - 1)
        height_fraction = height_positions - height_floor
        width_fraction = width_positions - width_floor
        base_height = height_floor * num_grid_per_side
        base_height_ceil = height_ceil * num_grid_per_side

        corner_indices = (
            (base_height[:, None] + width_floor[None]).flatten(),
            (base_height[:, None] + width_ceil[None]).flatten(),
            (base_height_ceil[:, None] + width_floor[None]).flatten(),
            (base_height_ceil[:, None] + width_ceil[None]).flatten(),
        )
        corner_weights = (
            ((1 - height_fraction)[:, None] * (1 - width_fraction)[None]).flatten(),
            ((1 - height_fraction)[:, None] * width_fraction[None]).flatten(),
            (height_fraction[:, None] * (1 - width_fraction)[None]).flatten(),
            (height_fraction[:, None] * width_fraction[None]).flatten(),
        )

        height_order = torch.arange(height, device=device).view(
            height // spatial_merge_size, spatial_merge_size
        )
        width_order = torch.arange(width, device=device).view(
            width // spatial_merge_size, spatial_merge_size
        )
        merge_order = (
            (height_order[:, :, None, None] * width + width_order[None, None, :, :])
            .transpose(1, 2)
            .flatten()
            .repeat(temporal)
        )
        for corner in range(4):
            index_parts[corner].append(corner_indices[corner][merge_order])
            weight_parts[corner].append(corner_weights[corner][merge_order])

    indices = torch.stack([torch.cat(parts) for parts in index_parts])
    weights = torch.stack([torch.cat(parts) for parts in weight_parts])
    return indices, weights


def _vision_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    return F.pad(cu_seqlens, (1, 0), value=0)


class Qwen3VLVisionTransformer(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        parallel = get_parallel()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size**2
        self.patch_size = config.patch_size
        self.patch_embed = Qwen3VLVisionPatchEmbed(config, disable_linear=True)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            Qwen3VLVisionBlock(config, layer_idx) for layer_idx in range(config.depth)
        )
        self.merger = Qwen3VLMoeVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            padded_context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            use_postshuffle_norm=False,
            prefix="visual.merger",
            tp_rank=parallel.tp_rank,
            tp_size=parallel.tp_size,
        )
        self.deepstack_visual_indexes = tuple(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            Qwen3VLMoeVisionPatchMerger(
                dim=config.out_hidden_size,
                context_dim=config.hidden_size,
                padded_context_dim=config.hidden_size,
                spatial_merge_size=config.spatial_merge_size,
                use_postshuffle_norm=True,
                prefix=f"visual.deepstack_merger_list.{merger_idx}",
                tp_rank=parallel.tp_rank,
                tp_size=parallel.tp_size,
            )
            for merger_idx, _ in enumerate(self.deepstack_visual_indexes)
        )
        self._deepstack_merger_by_layer = {
            layer_idx: merger_idx
            for merger_idx, layer_idx in enumerate(self.deepstack_visual_indexes)
        }

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def _interpolate_position_embeddings(self, grid_thw: torch.Tensor) -> torch.Tensor:
        indices, weights = _vision_bilinear_indices_and_weights(
            grid_thw,
            num_grid_per_side=self.num_grid_per_side,
            spatial_merge_size=self.spatial_merge_size,
        )
        return (self.pos_embed(indices) * weights[:, :, None]).sum(0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        **_: Any,
    ) -> Qwen3VLVisionOutput:
        hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
        grid_thw = grid_thw.to(self.device)
        hidden_states = self.patch_embed(hidden_states)
        position_embeddings = self._interpolate_position_embeddings(grid_thw)
        hidden_states = hidden_states + position_embeddings.to(hidden_states.dtype)

        position_ids = _vision_position_ids(grid_thw, self.spatial_merge_size)
        rotary = self.rotary_pos_emb(int(grid_thw[:, 1:].max()))[position_ids]
        rotary = rotary.flatten(1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (rotary.cos(), rotary.sin())
        metadata = PackedSequenceMetadata.from_cu_seqlens(_vision_cu_seqlens(grid_thw))

        deepstack_features = []
        for layer_idx, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                metadata=metadata,
                position_embeddings=position_embeddings,
            )
            merger_idx = self._deepstack_merger_by_layer.get(layer_idx)
            if merger_idx is not None:
                deepstack_features.append(
                    self.deepstack_merger_list[merger_idx](hidden_states)
                )

        return Qwen3VLVisionOutput(
            last_hidden_state=hidden_states,
            pooler_output=self.merger(hidden_states),
            deepstack_features=deepstack_features,
        )
