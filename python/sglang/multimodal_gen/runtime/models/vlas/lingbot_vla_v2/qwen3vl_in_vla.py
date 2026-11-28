# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

# Adapted for inference from huggingface/lerobot at f58f7fb9dbb241304df96e967aea0f5b42ff9a4d.
# Copyright 2026 HuggingFace Inc. and the Robbyant Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-VL backbone of LingBot-VLA 2.0: per-camera vision attention and split QKV/output text layers."""

from collections.abc import Callable

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration as _Qwen3VLForConditionalGeneration,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel as _Qwen3VLModel
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLPreTrainedModel as _Qwen3VLPreTrainedModel,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextAttention,
    Qwen3VLTextMLP,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextModel as _Qwen3VLTextModel,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    Qwen3VLVisionMLP,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionModel as _Qwen3VLVisionModel,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
)


class Qwen3VLPreTrainedModel(_Qwen3VLPreTrainedModel):
    """Vendored base class: weight init is a no-op."""

    def _init_weights(self, module) -> None:
        return


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config: Qwen3VLVisionConfig) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.num_key_value_groups = 1
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.scaling = self.head_dim**-0.5
        self.config = config
        self.is_causal = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        assert position_embeddings is not None
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        lengths_list = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        splits = [
            torch.split(tensor, lengths_list, dim=2)
            for tensor in (query_states, key_states, value_states)
        ]
        attn_outputs = [
            attention_interface(
                self,
                q,
                k,
                v,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0,
                is_causal=False,
            )[0]
            for q, k, v in zip(*splits, strict=False)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        attn_output = self.proj(attn_output.to(self.proj.weight.dtype))
        return attn_output


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLTextDecoderLayer(nn.Module):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3VLTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        att_output: torch.Tensor | None = None,
        start: int | None = 0,
        end: int | None = 0,
        compute_kqv: bool = False,
        output_atten: bool = False,
    ) -> (
        torch.FloatTensor
        | tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]
    ):
        param_dtype = self.self_attn.q_proj.weight.dtype
        hidden_states = hidden_states.to(param_dtype)
        if att_output is not None:
            att_output = att_output.to(param_dtype)

        if compute_kqv:
            hidden_states = self.input_layernorm(hidden_states)
            hidden_shape = (*hidden_states.shape[:-1], -1, self.self_attn.head_dim)
            query_state = self.self_attn.q_norm(
                self.self_attn.q_proj(hidden_states).view(hidden_shape)
            )
            key_state = self.self_attn.k_norm(
                self.self_attn.k_proj(hidden_states).view(hidden_shape)
            )
            value_state = self.self_attn.v_proj(hidden_states).view(hidden_shape)
            return query_state, key_state, value_state

        if output_atten:
            assert att_output is not None
            if att_output.dtype != self.self_attn.o_proj.weight.dtype:
                att_output = att_output.to(self.self_attn.o_proj.weight.dtype)
            out_emb = self.self_attn.o_proj(att_output[:, start:end])
            out_emb += hidden_states
            residual = out_emb  # the norm and MLP below do not write to their input
            out_emb = self.post_attention_layernorm(out_emb)
            out_emb = self.mlp(out_emb)
            out_emb += residual
            return out_emb

        raise ValueError(
            f"Invalid operation compute_kqv={compute_kqv} and output_atten={output_atten} "
            "with Qwen3VLTextDecoderLayer in LingBot-VLA"
        )


@torch.compiler.disable
def preprocess_grid_thw(self, grid_thw: torch.Tensor):
    rotary_pos_emb = self.rot_pos_emb(grid_thw)

    seq_len = int(torch.prod(grid_thw, dim=1).sum().item())
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
    return pos_embeds, position_embeddings, cu_seqlens


def vision_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor):
    hidden_states = self.patch_embed(hidden_states)
    pos_embeds, position_embeddings, cu_seqlens = self.preprocess_grid_thw(grid_thw)

    hidden_states = hidden_states + pos_embeds
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)

    deepstack_feature_lists = []
    for layer_num, blk in enumerate(self.blocks):
        hidden_states = blk(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        if layer_num in self.deepstack_visual_indexes:
            deepstack_feature = self.deepstack_merger_list[
                self.deepstack_visual_indexes.index(layer_num)
            ](hidden_states)
            deepstack_feature_lists.append(deepstack_feature)

    hidden_states = self.merger(hidden_states)
    return hidden_states, deepstack_feature_lists


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    """Rotary angles for (h, w) patch coordinates; the buffer is derived, not loaded from the checkpoint."""

    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


class Qwen3VLVisionModel(_Qwen3VLVisionModel):
    """Vision tower with per-observation grid metadata and LingBot attention blocks."""

    preprocess_grid_thw = preprocess_grid_thw
    forward = vision_forward

    def __init__(self, config, *inputs, **kwargs) -> None:
        super().__init__(config, *inputs, **kwargs)
        self.blocks = nn.ModuleList(
            [Qwen3VLVisionBlock(config) for _ in range(config.depth)]
        )
        # note (0xtoward): these position calculations preserve the checkpoint's Transformers 5.14 behavior.
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(
            config.hidden_size // config.num_heads // 2
        )

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Rotary angles per patch, laid out block-major over the spatial-merge blocks."""
        merge_size = self.spatial_merge_size
        position_ids = []
        for frames, height, width in grid_thw.tolist():
            row_ids, column_ids = torch.meshgrid(
                torch.arange(height, device=grid_thw.device),
                torch.arange(width, device=grid_thw.device),
                indexing="ij",
            )
            block_shape = (
                height // merge_size,
                merge_size,
                width // merge_size,
                merge_size,
            )
            row_ids = row_ids.reshape(block_shape).transpose(1, 2).flatten()
            column_ids = column_ids.reshape(block_shape).transpose(1, 2).flatten()
            position_ids.append(
                torch.stack([row_ids, column_ids], dim=-1).repeat(frames, 1)
            )
        return self.rotary_pos_emb(torch.cat(position_ids, dim=0))

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Bilinearly resample the learned position grid to each image's patches, block-major like rot_pos_emb."""
        side = self.num_grid_per_side
        merge_size = self.spatial_merge_size
        device = grid_thw.device
        corner_index_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
        corner_weight_parts: list[list[torch.Tensor]] = [[] for _ in range(4)]
        for frames, height, width in grid_thw.tolist():
            row_grid = torch.linspace(0, side - 1, height, device=device)
            column_grid = torch.linspace(0, side - 1, width, device=device)
            row_floor = row_grid.int()
            column_floor = column_grid.int()
            row_ceil = (row_floor + 1).clamp(max=side - 1)
            column_ceil = (column_floor + 1).clamp(max=side - 1)
            row_fraction = row_grid - row_floor
            column_fraction = column_grid - column_floor
            corner_indices = [
                (row_floor[:, None] * side + column_floor[None, :]).flatten(),
                (row_floor[:, None] * side + column_ceil[None, :]).flatten(),
                (row_ceil[:, None] * side + column_floor[None, :]).flatten(),
                (row_ceil[:, None] * side + column_ceil[None, :]).flatten(),
            ]
            corner_weights = [
                (
                    (1 - row_fraction)[:, None] * (1 - column_fraction)[None, :]
                ).flatten(),
                ((1 - row_fraction)[:, None] * column_fraction[None, :]).flatten(),
                (row_fraction[:, None] * (1 - column_fraction)[None, :]).flatten(),
                (row_fraction[:, None] * column_fraction[None, :]).flatten(),
            ]
            row_blocks = torch.arange(height, device=device).view(
                height // merge_size, merge_size
            )
            column_blocks = torch.arange(width, device=device).view(
                width // merge_size, merge_size
            )
            block_order = (
                row_blocks[:, :, None, None] * width + column_blocks[None, None, :, :]
            ).transpose(1, 2)
            block_order = block_order.flatten().repeat(frames)
            for corner in range(4):
                corner_index_parts[corner].append(corner_indices[corner][block_order])
                corner_weight_parts[corner].append(corner_weights[corner][block_order])
        indices = torch.stack([torch.cat(parts) for parts in corner_index_parts])
        weights = torch.stack([torch.cat(parts) for parts in corner_weight_parts])
        return (self.pos_embed(indices) * weights[:, :, None]).sum(0)


class Qwen3VLTextModel(_Qwen3VLTextModel):
    def __init__(self, config: Qwen3VLTextConfig):
        # note (0xtoward): skip allocation of the unused stock decoder layers.
        Qwen3VLPreTrainedModel.__init__(self, config)  # type: ignore[call-arg]
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen3VLTextDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
        self.post_init()


class Qwen3VLModel(_Qwen3VLModel):
    def __init__(self, config: Qwen3VLConfig):
        Qwen3VLPreTrainedModel.__init__(self, config)  # type: ignore[call-arg]
        self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()


class Qwen3VLForConditionalGeneration(_Qwen3VLForConditionalGeneration):
    """Keeps the checkpoint layout (qwenvl.model.*); the policy has no LM head, so nothing is tied."""

    _tied_weights_keys = None
    config_class = Qwen3VLConfig

    def __init__(self, config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.model = Qwen3VLModel(config)
        self.post_init()
