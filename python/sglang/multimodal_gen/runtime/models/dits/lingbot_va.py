# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 The Robbyant Team Authors.
"""Inference adaptation of Robbyant/lingbot-va at 7c6ffa9. KV belongs to the request."""

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        time_freq_dim,
        time_proj_dim,
        text_embed_dim,
        pos_embed_seq_len,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim
        )
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh"
        )

    def forward(
        self,
        timestep: torch.Tensor,
        dtype=None,
    ):
        B, L = timestep.shape
        timestep = timestep.reshape(-1)
        timestep = self.timesteps_proj(timestep)
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).to(dtype=dtype)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb.reshape(B, L, -1), timestep_proj.reshape(B, L, -1)


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        # Precompute and register buffers
        f_freqs_base, h_freqs_base, w_freqs_base = self._precompute_freqs_base()
        self.f_freqs_base = f_freqs_base
        self.h_freqs_base = h_freqs_base
        self.w_freqs_base = w_freqs_base

    def _precompute_freqs_base(self):
        # freqs_base = 1.0 / (theta ** (2k / dim))
        f_freqs_base = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.f_dim, 2)[: (self.f_dim // 2)].double()
                / self.f_dim
            )
        )
        h_freqs_base = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.h_dim, 2)[: (self.h_dim // 2)].double()
                / self.h_dim
            )
        )
        w_freqs_base = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.w_dim, 2)[: (self.w_dim // 2)].double()
                / self.w_dim
            )
        )
        return f_freqs_base, h_freqs_base, w_freqs_base

    def forward(self, grid_ids):
        with torch.no_grad():
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(
                grid_ids.device
            )
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(
                grid_ids.device
            )
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(
                grid_ids.device
            )
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis


class LingBotAttention(nn.Module):
    def __init__(self, dim, heads, eps):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])
        self.norm_q = nn.RMSNorm(dim, eps=eps)
        self.norm_k = nn.RMSNorm(dim, eps=eps)

    def forward(self, hidden, context, rotary=None, history=None, save=None):
        query = self.norm_q(self.to_q(hidden)).unflatten(-1, (self.heads, -1))
        key = self.norm_k(self.to_k(context)).unflatten(-1, (self.heads, -1))
        value = self.to_v(context).unflatten(-1, (self.heads, -1))
        if rotary is not None:

            def rotate(tensor):
                pairs = torch.view_as_complex(tensor.double().unflatten(-1, (-1, 2)))
                return torch.view_as_real(pairs * rotary).flatten(-2).to(tensor.dtype)

            query, key = rotate(query), rotate(key)
        if save is not None:
            save.append((key, value))
        if history is not None:
            key = torch.cat([history[0], key], dim=1)
            value = torch.cat([history[1], value], dim=1)
        out = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        )
        return self.to_out[0](out.transpose(1, 2).flatten(2))


class LingBotBlock(nn.Module):
    def __init__(self, dim, ffn_dim, num_heads, cross_attn_norm, eps):
        super().__init__()
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = LingBotAttention(dim, num_heads, eps)
        self.attn2 = LingBotAttention(dim, num_heads, eps)
        self.norm2 = FP32LayerNorm(dim, eps) if cross_attn_norm else nn.Identity()
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(torch.empty(1, 6, dim))

    def forward(self, hidden, text, time_proj, rotary, history, save):
        shift, scale, gate, ff_shift, ff_scale, ff_gate = (
            self.scale_shift_table[None] + time_proj.float()
        ).unbind(2)
        norm = (self.norm1(hidden.float()) * (1 + scale) + shift).to(hidden.dtype)
        attn = self.attn1(norm, norm, rotary, history, save)
        hidden = (hidden.float() + attn * gate).to(hidden.dtype)
        hidden = hidden + self.attn2(self.norm2(hidden.float()).to(hidden.dtype), text)
        norm = (self.norm3(hidden.float()) * (1 + ff_scale) + ff_shift).to(hidden.dtype)
        return (hidden.float() + self.ffn(norm).float() * ff_gate).to(hidden.dtype)


class LingBotVATransformer(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-06,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        attn_mode="torch",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding_mlp = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2], inner_dim
        )
        self.action_embedder = nn.Linear(action_dim, inner_dim)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        self.condition_embedder_action = deepcopy(self.condition_embedder)

        self.blocks = nn.ModuleList(
            [
                LingBotBlock(
                    inner_dim, ffn_dim, num_attention_heads, cross_attn_norm, eps
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.action_proj_out = nn.Linear(inner_dim, action_dim)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

    def forward(
        self, latents, timesteps, grid, text, *, action=False, history=None, save=None
    ):
        if action:
            hidden = self.action_embedder(
                rearrange(latents, "b c f h w -> b (f h w) c")
            )
        else:
            patches = rearrange(
                latents,
                "b c (f t) (h y) (w x) -> b (f h w) (c t y x)",
                t=self.patch_size[0],
                y=self.patch_size[1],
                x=self.patch_size[2],
            )
            hidden = self.patch_embedding_mlp(patches)
        text = self.condition_embedder.text_embedder(text)
        rotary = self.rope(grid)[:, :, None]
        embedder = self.condition_embedder_action if action else self.condition_embedder
        time = timesteps.repeat_interleave(hidden.shape[1] // timesteps.shape[1], dim=1)
        temb, proj = embedder(time, dtype=hidden.dtype)
        proj = proj.unflatten(2, (6, -1))
        for index, layer in enumerate(self.blocks):
            hidden = layer(
                hidden, text, proj, rotary, history[index] if history else None, save
            )
        shift, scale = (self.scale_shift_table[None] + temb[:, :, None]).unbind(2)
        hidden = (self.norm_out(hidden.float()) * (1 + scale) + shift).to(hidden.dtype)
        if action:
            return self.action_proj_out(hidden)
        return (
            self.proj_out(hidden)
            .unflatten(-1, (math.prod(self.patch_size), -1))
            .flatten(1, 2)
        )
