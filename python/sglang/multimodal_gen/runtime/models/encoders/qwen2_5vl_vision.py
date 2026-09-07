# SPDX-License-Identifier: Apache-2.0
"""Native Qwen2.5-VL vision encoder."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.models.encoders.qwen_vl_vision import (
    PackedSequenceMetadata,
    QwenVLVisionAttention,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.models.qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed as Qwen2_5VLVisionPatchEmbed,
)
from sglang.srt.models.qwen2_5_vl import (
    Qwen2_5_VisionPatchMerger as Qwen2_5VLVisionPatchMerger,
)
from sglang.srt.models.qwen2_5_vl import (
    Qwen2_5_VisionRotaryEmbedding as Qwen2_5VLVisionRotaryEmbedding,
)
from sglang.srt.models.qwen2_5_vl import (
    Qwen2_5_VLMLP,
)


class Qwen2_5VLVisionBlock(nn.Module):
    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.norm1 = RMSNorm(
            config.hidden_size,
            eps=1e-6,
            cast_x_before_out_mul=True,
            force_native=True,
        )
        self.norm2 = RMSNorm(
            config.hidden_size,
            eps=1e-6,
            cast_x_before_out_mul=True,
            force_native=True,
        )
        self.attn = QwenVLVisionAttention(
            config,
            prefix=f"visual.blocks.{layer_idx}.attn",
            model_name="Qwen2.5-VL",
        )
        self.mlp = Qwen2_5_VLMLP(
            config.hidden_size,
            config.intermediate_size,
            bias=True,
            hidden_act=config.hidden_act,
            prefix=f"visual.blocks.{layer_idx}.mlp",
            fuse_gate_up=False,
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
    for t, h, w in grid_thw.tolist():
        h_positions = torch.arange(h, device=grid_thw.device)[:, None].expand(h, w)
        h_positions = (
            h_positions.reshape(
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
            )
            .transpose(1, 2)
            .flatten()
        )
        w_positions = torch.arange(w, device=grid_thw.device)[None, :].expand(h, w)
        w_positions = (
            w_positions.reshape(
                h // spatial_merge_size,
                spatial_merge_size,
                w // spatial_merge_size,
                spatial_merge_size,
            )
            .transpose(1, 2)
            .flatten()
        )
        positions = torch.stack((h_positions, w_positions), dim=-1)
        position_ids.append(positions.repeat(t, 1))
    return torch.cat(position_ids, dim=0)


def _vision_window_index(
    grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
    window_size: int,
    patch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    window_indices = []
    cumulative_window_lengths = [0]
    window_index_offset = 0
    merger_window_size = window_size // spatial_merge_size // patch_size
    spatial_merge_unit = spatial_merge_size**2

    for grid_t, grid_h, grid_w in grid_thw.tolist():
        merged_height = grid_h // spatial_merge_size
        merged_width = grid_w // spatial_merge_size
        index = torch.arange(
            grid_t * merged_height * merged_width, device=grid_thw.device
        ).reshape(grid_t, merged_height, merged_width)
        pad_height = merger_window_size - merged_height % merger_window_size
        pad_width = merger_window_size - merged_width % merger_window_size
        num_windows_height = (merged_height + pad_height) // merger_window_size
        num_windows_width = (merged_width + pad_width) // merger_window_size
        index = F.pad(index, (0, pad_width, 0, pad_height), value=-100)
        index = index.reshape(
            grid_t,
            num_windows_height,
            merger_window_size,
            num_windows_width,
            merger_window_size,
        )
        index = index.permute(0, 1, 3, 2, 4).reshape(
            grid_t,
            num_windows_height * num_windows_width,
            merger_window_size,
            merger_window_size,
        )
        sequence_lengths = (index != -100).sum(dim=(2, 3)).reshape(-1)
        index = index.flatten()
        window_indices.append(index[index != -100] + window_index_offset)
        cumulative = (
            sequence_lengths.cumsum(0) * spatial_merge_unit
            + cumulative_window_lengths[-1]
        )
        cumulative_window_lengths.extend(cumulative.tolist())
        window_index_offset += grid_t * merged_height * merged_width

    window_index = torch.cat(window_indices)
    cu_window_seqlens = torch.tensor(
        cumulative_window_lengths, device=grid_thw.device, dtype=torch.int32
    )
    return window_index, torch.unique_consecutive(cu_window_seqlens)


class Qwen2_5VLVisionTransformer(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size**2
        self.patch_size = config.patch_size
        self.window_size = config.window_size
        self.full_attention_layers = frozenset(
            int(layer_idx) for layer_idx in config.fullatt_block_indexes
        )
        self.patch_embed = Qwen2_5VLVisionPatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.hidden_size,
            disable_linear=True,
        )
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5VLVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            Qwen2_5VLVisionBlock(config, layer_idx) for layer_idx in range(config.depth)
        )
        self.merger = Qwen2_5VLVisionPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            padded_context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            prefix="visual.merger",
            cast_x_before_out_mul=True,
            force_native_norm=True,
        )

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embed.proj.weight.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = hidden_states.to(device=self.device, dtype=self.dtype)
        grid_thw = grid_thw.to(self.device)
        hidden_states = self.patch_embed(hidden_states)

        position_ids = _vision_position_ids(grid_thw, self.spatial_merge_size)
        window_index, cu_window_seqlens = _vision_window_index(
            grid_thw,
            spatial_merge_size=self.spatial_merge_size,
            window_size=self.window_size,
            patch_size=self.patch_size,
        )

        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )[window_index]
        hidden_states = hidden_states.reshape(seq_len, -1)

        rotary = self.rotary_pos_emb(position_ids)
        rotary = rotary.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1
        )[window_index]
        rotary = rotary.reshape(seq_len, -1)
        rotary = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (
            rotary.cos(),
            rotary.sin(),
        )

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        full_metadata = PackedSequenceMetadata.from_cu_seqlens(cu_seqlens)
        window_metadata = PackedSequenceMetadata.from_cu_seqlens(cu_window_seqlens)

        for layer_idx, block in enumerate(self.blocks):
            metadata = (
                full_metadata
                if layer_idx in self.full_attention_layers
                else window_metadata
            )
            hidden_states = block(
                hidden_states,
                metadata=metadata,
                position_embeddings=position_embeddings,
            )

        merged = self.merger(hidden_states)
        return merged[torch.argsort(window_index)]
