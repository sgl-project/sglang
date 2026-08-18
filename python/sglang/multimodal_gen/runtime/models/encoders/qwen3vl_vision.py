# SPDX-License-Identifier: Apache-2.0
"""Native Qwen3-VL vision encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass(frozen=True)
class Qwen3VLVisionOutput:
    last_hidden_state: torch.Tensor
    pooler_output: torch.Tensor
    deepstack_features: list[torch.Tensor]


@dataclass(frozen=True)
class _PackedSequenceMetadata:
    cu_seqlens: torch.Tensor
    cu_seqlens_host: tuple[int, ...]
    max_seqlen: int

    @classmethod
    def from_cu_seqlens(cls, cu_seqlens: torch.Tensor) -> _PackedSequenceMetadata:
        bounds = tuple(int(value) for value in cu_seqlens.tolist())
        return cls(
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=bounds,
            max_seqlen=max(
                stop - start for start, stop in zip(bounds[:-1], bounds[1:])
            ),
        )


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel_size = (
            config.temporal_patch_size,
            config.patch_size,
            config.patch_size,
        )
        self.proj = nn.Conv3d(
            config.in_channels,
            config.hidden_size,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        return self.proj(hidden_states.to(self.proj.weight.dtype)).view(
            -1, self.embed_dim
        )


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


def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_vision_rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query_dtype = query.dtype
    key_dtype = key.dtype
    query = query.float()
    key = key.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    query = query * cos + _rotate_half(query) * sin
    key = key * cos + _rotate_half(key) * sin
    return query.to(query_dtype), key.to(key_dtype)


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Any, prefix: str) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        backend = get_attn_backend(self.head_dim, torch.get_default_dtype())
        self._attention_impl = None
        if backend.supports_packed_varlen():
            self._attention_impl = backend.get_impl_cls()(
                num_heads=self.num_heads,
                head_size=self.head_dim,
                num_kv_heads=self.num_heads,
                softmax_scale=self.scaling,
                causal=False,
                prefix=prefix,
            )
        else:
            logger.warning_once(
                "Qwen3-VL vision attention uses torch SDPA because "
                f"{backend.get_enum().name.lower()} does not support packed sequences"
            )

    def _packed_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: _PackedSequenceMetadata,
    ) -> torch.Tensor:
        if self._attention_impl is not None:
            return self._attention_impl.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=metadata.cu_seqlens,
                cu_seqlens_host=metadata.cu_seqlens_host,
                max_seqlen=metadata.max_seqlen,
            )

        output = torch.empty_like(query)
        for start, stop in zip(
            metadata.cu_seqlens_host[:-1], metadata.cu_seqlens_host[1:]
        ):
            if start == stop:
                continue
            segment = F.scaled_dot_product_attention(
                query[start:stop].transpose(0, 1).unsqueeze(0),
                key[start:stop].transpose(0, 1).unsqueeze(0),
                value[start:stop].transpose(0, 1).unsqueeze(0),
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
            )
            output[start:stop] = segment.squeeze(0).transpose(0, 1)
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        metadata: _PackedSequenceMetadata,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        sequence_length = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states)
            .reshape(sequence_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query, key = _apply_vision_rotary_embedding(query, key, *position_embeddings)
        output = self._packed_attention(query, key, value, metadata)
        return self.proj(output.reshape(sequence_length, -1).contiguous())


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        if config.hidden_act != "gelu_pytorch_tanh":
            raise ValueError(
                f"Unsupported Qwen3-VL vision activation: {config.hidden_act}"
            )
        self.linear_fc1 = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.linear_fc2 = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True
        )
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(
            config, prefix=f"visual.blocks.{layer_idx}.attn"
        )
        self.mlp = Qwen3VLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        metadata: _PackedSequenceMetadata,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            metadata=metadata,
            position_embeddings=position_embeddings,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config: Any, *, use_postshuffle_norm: bool) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size * config.spatial_merge_size**2
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_size = self.hidden_size if use_postshuffle_norm else config.hidden_size
        self.norm = nn.LayerNorm(norm_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = self.norm(hidden_states).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))


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
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.spatial_merge_unit = config.spatial_merge_size**2
        self.patch_size = config.patch_size
        self.patch_embed = Qwen3VLVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            Qwen3VLVisionBlock(config, layer_idx) for layer_idx in range(config.depth)
        )
        self.merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = tuple(config.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True)
            for _ in self.deepstack_visual_indexes
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
        metadata = _PackedSequenceMetadata.from_cu_seqlens(_vision_cu_seqlens(grid_thw))

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
