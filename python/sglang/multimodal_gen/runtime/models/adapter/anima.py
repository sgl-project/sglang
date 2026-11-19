# Copyright 2026 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Anima's learned T5-token queries over Qwen3 text states."""

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.configs.models.adapter.anima import (
    AnimaTextConditionerConfig,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)


class AnimaConditionerAttention(nn.Module):
    def __init__(self, config, context_dim, cross_attention=False):
        super().__init__()
        dim = config.model_dim
        self.heads = config.num_attention_heads
        self.head_dim = dim // self.heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(context_dim, dim, bias=False)
        self.v_proj = nn.Linear(context_dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.attn = LocalAttention(
            self.heads, self.head_dim, is_cross_attention=cross_attention
        )

    def forward(self, x, context, mask, rope, context_rope):
        q = self.q_norm(self.q_proj(x).unflatten(-1, (self.heads, self.head_dim)))
        k = self.k_norm(self.k_proj(context).unflatten(-1, (self.heads, self.head_dim)))
        v = self.v_proj(context).unflatten(-1, (self.heads, self.head_dim))
        cos, sin = rope
        q1, q2 = q.chunk(2, -1)
        q = q * cos + torch.cat((-q2, q1), -1) * sin
        cos, sin = context_rope
        k1, k2 = k.chunk(2, -1)
        k = k * cos + torch.cat((-k2, k1), -1) * sin
        return self.o_proj(self.attn(q, k, v, attn_mask=mask).flatten(2))


class AnimaConditionerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.use_self_attention = config.use_self_attention
        norm = nn.LayerNorm if config.use_layer_norm else nn.RMSNorm
        eps = 1e-5 if config.use_layer_norm else 1e-6
        dim = config.model_dim
        if self.use_self_attention:
            self.norm_self_attn = norm(dim, eps=eps)
            self.self_attn = AnimaConditionerAttention(config, dim)
        self.norm_cross_attn = norm(dim, eps=eps)
        self.cross_attn = AnimaConditionerAttention(
            config, config.source_dim, cross_attention=True
        )
        self.norm_mlp = norm(dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * config.mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * config.mlp_ratio), dim),
        )

    def forward(self, x, source, target_mask, source_mask, rope, source_rope):
        if self.use_self_attention:
            normed = self.norm_self_attn(x)
            x = x + self.self_attn(normed, normed, target_mask, rope, rope)
        x = x + self.cross_attn(
            self.norm_cross_attn(x), source, source_mask, rope, source_rope
        )
        return x + self.mlp(self.norm_mlp(x))


class AnimaTextConditioner(nn.Module, LayerwiseOffloadableModuleMixin):
    layerwise_offload_dit_group_enabled = False
    layer_names = ["blocks"]

    def __init__(self, config: AnimaTextConditionerConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.target_vocab_size, config.target_dim)
        self.in_proj = (
            nn.Linear(config.target_dim, config.model_dim)
            if config.target_dim != config.model_dim
            else nn.Identity()
        )
        self.blocks = nn.ModuleList(
            [AnimaConditionerBlock(config) for _ in range(config.num_layers)]
        )
        self.out_proj = nn.Linear(config.model_dim, config.target_dim)
        self.norm = nn.RMSNorm(config.target_dim, eps=1e-6)

    def _rope(self, length, device, dtype):
        dim = self.config.model_dim // self.config.num_attention_heads
        inv = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        freqs = torch.outer(torch.arange(length, device=device).float(), inv)
        freqs = torch.cat((freqs, freqs), dim=-1)[None, :, None, :]
        return freqs.cos().to(dtype), freqs.sin().to(dtype)

    def forward(
        self,
        source_hidden_states,
        target_input_ids,
        target_attention_mask,
        source_attention_mask,
    ):
        x = self.in_proj(self.embed(target_input_ids).to(source_hidden_states.dtype))
        rope = self._rope(x.shape[1], x.device, x.dtype)
        source_rope = self._rope(source_hidden_states.shape[1], x.device, x.dtype)
        # additive -inf preserves SDPA's zero output for fully masked rows
        target_mask = torch.zeros_like(target_attention_mask, dtype=x.dtype)
        target_mask.masked_fill_(target_attention_mask == 0, float("-inf"))
        source_mask = torch.zeros_like(source_attention_mask, dtype=x.dtype)
        source_mask.masked_fill_(source_attention_mask == 0, float("-inf"))
        for block in self.blocks:
            x = block(
                x, source_hidden_states, target_mask, source_mask, rope, source_rope
            )
        x = self.norm(self.out_proj(x)) * target_attention_mask.unsqueeze(-1).to(
            x.dtype
        )
        return F.pad(x, (0, 0, 0, max(0, self.config.min_sequence_length - x.shape[1])))


EntryClass = AnimaTextConditioner
