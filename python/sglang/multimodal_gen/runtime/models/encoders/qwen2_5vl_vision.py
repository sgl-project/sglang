# SPDX-License-Identifier: Apache-2.0
"""Native Qwen2.5-VL vision encoder."""

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


class Qwen2_5VLVisionRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        variance = hidden_states.square().mean(dim=-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen2_5VLVisionPatchEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
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


class Qwen2_5VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


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


class Qwen2_5VLVisionAttention(nn.Module):
    def __init__(self, config: Any, prefix: str) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scaling = self.head_dim**-0.5
        self.prefix = prefix
        self.qkv = nn.Linear(config.hidden_size, config.hidden_size * 3, bias=True)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self._attention_impl = None
        self._initialize_attention(torch.get_default_dtype())

    def _initialize_attention(self, dtype: torch.dtype) -> None:
        backend = get_attn_backend(self.head_dim, dtype)
        if backend.supports_packed_varlen():
            self._attention_impl = backend.get_impl_cls()(
                num_heads=self.num_heads,
                head_size=self.head_dim,
                num_kv_heads=self.num_heads,
                softmax_scale=self.scaling,
                causal=False,
                prefix=self.prefix,
            )
        else:
            logger.warning_once(
                "Qwen2.5-VL vision attention uses torch SDPA because "
                f"{backend.get_enum().name.lower()} does not support packed sequences"
            )

    def _packed_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...],
        max_seqlen: int,
    ) -> torch.Tensor:
        if self._attention_impl is not None:
            return self._attention_impl.forward_varlen(
                query,
                key,
                value,
                cu_seqlens=cu_seqlens,
                cu_seqlens_host=cu_seqlens_host,
                max_seqlen=max_seqlen,
            )

        output = torch.empty_like(query)
        for start, stop in zip(cu_seqlens_host[:-1], cu_seqlens_host[1:]):
            if start == stop:
                continue
            query_segment = query[start:stop].transpose(0, 1).unsqueeze(0)
            key_segment = key[start:stop].transpose(0, 1).unsqueeze(0)
            value_segment = value[start:stop].transpose(0, 1).unsqueeze(0)
            segment = F.scaled_dot_product_attention(
                query_segment,
                key_segment,
                value_segment,
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
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...],
        max_seqlen: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        seq_len = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states)
            .reshape(seq_len, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        query, key = _apply_vision_rotary_embedding(query, key, *position_embeddings)
        output = self._packed_attention(
            query,
            key,
            value,
            cu_seqlens,
            cu_seqlens_host,
            max_seqlen,
        )
        return self.proj(output.reshape(seq_len, -1).contiguous())


class Qwen2_5VLVisionMLP(nn.Module):
    def __init__(self, config: Any) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise ValueError(
                f"Unsupported Qwen2.5-VL vision activation: {config.hidden_act}"
            )
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=True
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=True
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class Qwen2_5VLVisionBlock(nn.Module):
    def __init__(self, config: Any, layer_idx: int) -> None:
        super().__init__()
        self.norm1 = Qwen2_5VLVisionRMSNorm(config.hidden_size)
        self.norm2 = Qwen2_5VLVisionRMSNorm(config.hidden_size)
        self.attn = Qwen2_5VLVisionAttention(
            config, prefix=f"visual.blocks.{layer_idx}.attn"
        )
        self.mlp = Qwen2_5VLVisionMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        cu_seqlens_host: tuple[int, ...],
        max_seqlen: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
        )
        return hidden_states + self.mlp(self.norm2(hidden_states))


class Qwen2_5VLVisionPatchMerger(nn.Module):
    def __init__(
        self,
        output_dim: int,
        context_dim: int,
        spatial_merge_size: int,
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * spatial_merge_size**2
        self.ln_q = Qwen2_5VLVisionRMSNorm(context_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, output_dim),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ln_q(hidden_states).view(-1, self.hidden_size)
        return self.mlp(hidden_states)


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
        )
        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen2_5VLVisionRotaryEmbedding(head_dim // 2)
        self.blocks = nn.ModuleList(
            Qwen2_5VLVisionBlock(config, layer_idx) for layer_idx in range(config.depth)
        )
        self.merger = Qwen2_5VLVisionPatchMerger(
            output_dim=config.out_hidden_size,
            context_dim=config.hidden_size,
            spatial_merge_size=config.spatial_merge_size,
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

        full_metadata = _PackedSequenceMetadata.from_cu_seqlens(cu_seqlens)
        window_metadata = _PackedSequenceMetadata.from_cu_seqlens(cu_window_seqlens)

        for layer_idx, block in enumerate(self.blocks):
            metadata = (
                full_metadata
                if layer_idx in self.full_attention_layers
                else window_metadata
            )
            hidden_states = block(
                hidden_states,
                cu_seqlens=metadata.cu_seqlens,
                cu_seqlens_host=metadata.cu_seqlens_host,
                max_seqlen=metadata.max_seqlen,
                position_embeddings=position_embeddings,
            )

        merged = self.merger(hidden_states)
        return merged[torch.argsort(window_index)]
