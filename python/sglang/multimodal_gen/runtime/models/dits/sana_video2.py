# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: Apache-2.0

"""SANA-Video 2.0 inference transformer with grouped Attention Residuals."""

import math
from typing import ClassVar

import torch
from torch import nn
from torch.nn import functional as F

from sglang.multimodal_gen.configs.models.dits.sana_video2 import SanaVideo2Config
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.models.dits.sana_wm_components import (
    CaptionEmbedder,
    PatchEmbedMS3D,
    T2IFinalLayer,
    TimestepEmbedder,
    WanRotaryPosEmbed,
    _RMSNorm,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


def _apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    rotated = torch.view_as_complex(x.to(torch.float64).unflatten(-1, (-1, 2)))
    return torch.view_as_real(rotated * freqs).flatten(-2).type_as(x)


class SanaVideo2RotaryPosEmbed(WanRotaryPosEmbed):
    def _apply(self, fn, recurse=True):
        # Keep complex128 frequencies; forward moves them to the input device.
        return self

    def forward(self, fhw, device, frame_index=None):
        if self._freqs.device != device:
            self._freqs = self._freqs.to(device)
        return super().forward(fhw, device, frame_index)


class ChannelRMSNorm(_RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = (
            x.float() * (x.float().square().mean(-2, keepdim=True) + self.eps).rsqrt()
        )
        return (self.weight[None, None, :, None] * normalized).type_as(x)


class GatedLinearAttention(nn.Module):
    def __init__(self, dim, head_dim, qk_norm=True, fp32_attention=True):
        super().__init__()
        self.heads = dim // head_dim
        self.dim = head_dim
        self.fp32_attention = fp32_attention
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = _RMSNorm(dim, eps=1e-5) if qk_norm else nn.Identity()
        self.k_norm = _RMSNorm(dim, eps=1e-5) if qk_norm else nn.Identity()
        self.beta_proj = nn.Linear(dim, self.heads)
        self.output_gate = nn.Linear(dim, dim)
        self.o_norm = ChannelRMSNorm(head_dim, eps=1e-5)

    def forward(self, x, rotary_emb=None):
        batch, tokens, channels = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, channels).unbind(2)
        dtype = q.dtype
        q = (
            self.q_norm(q)
            .transpose(-1, -2)
            .reshape(batch, self.heads, self.dim, tokens)
        )
        k = (
            self.k_norm(k)
            .transpose(-1, -2)
            .reshape(batch, self.heads, self.dim, tokens)
        )
        v = v.transpose(-1, -2).reshape(batch, self.heads, self.dim, tokens)
        if rotary_emb is not None:
            q = _apply_rope(q.transpose(-1, -2), rotary_emb).transpose(-1, -2)
            k = _apply_rope(k.transpose(-1, -2), rotary_emb).transpose(-1, -2)
        beta = self.beta_proj(x).sigmoid().transpose(1, 2).unsqueeze(2)
        k = k * beta
        if self.fp32_attention:
            q, k, v = q.float(), k.float(), v.float()
        out = (v @ k.transpose(-1, -2)) @ q
        out = self.o_norm(out.to(dtype))
        out = out.reshape(batch, channels, tokens).transpose(1, 2)
        return self.proj(out * self.output_gate(x).sigmoid())


class GatedSoftmaxAttention(nn.Module):
    def __init__(self, dim, head_dim, qk_norm=True, fp32_attention=True):
        super().__init__()
        self.heads = dim // head_dim
        self.dim = head_dim
        self.fp32_attention = fp32_attention
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = _RMSNorm(dim, eps=1e-5) if qk_norm else nn.Identity()
        self.k_norm = _RMSNorm(dim, eps=1e-5) if qk_norm else nn.Identity()
        self.output_gate = nn.Linear(dim, dim)

    def forward(self, x, rotary_emb=None):
        batch, tokens, channels = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, channels).unbind(2)
        dtype = q.dtype
        q = self.q_norm(q).reshape(batch, tokens, self.heads, self.dim).transpose(1, 2)
        k = self.k_norm(k).reshape(batch, tokens, self.heads, self.dim).transpose(1, 2)
        v = v.reshape(batch, tokens, self.heads, self.dim).transpose(1, 2)
        if rotary_emb is not None:
            q, k = _apply_rope(q, rotary_emb), _apply_rope(k, rotary_emb)
        if self.fp32_attention:
            q, k, v = q.float(), k.float(), v.float()
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(batch, tokens, channels).to(dtype)
        return self.proj(out * self.output_gate(x).sigmoid())


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, 2 * dim)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = _RMSNorm(dim, eps=1e-6) if qk_norm else nn.Identity()
        self.k_norm = _RMSNorm(dim, eps=1e-6) if qk_norm else nn.Identity()

    def forward(self, x, y, mask=None):
        batch, tokens, channels = x.shape
        q = self.q_norm(self.q_linear(x))
        k, v = self.kv_linear(y).reshape(batch, -1, 2, channels).unbind(2)
        q = q.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)
        k = (
            self.k_norm(k)
            .reshape(batch, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = v.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        if mask is not None:
            mask = (1 - mask.to(q.dtype))[:, None, None, :] * -10000.0
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, channels))


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SanaVideo2Block(nn.Module):
    def __init__(self, config, attention):
        super().__init__()
        dim = config.hidden_size
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        attention_cls = (
            GatedSoftmaxAttention if attention == "softmax" else GatedLinearAttention
        )
        head_dim = (
            config.softmax_head_dim
            if attention == "softmax"
            else config.linear_head_dim
        )
        self.attn = attention_cls(dim, head_dim, config.qk_norm, config.fp32_attention)
        self.cross_attn = CrossAttention(dim, config.num_heads, config.cross_norm)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = SwiGLU(dim, int(dim * config.mlp_ratio))
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def _modulation(self, t, batch):
        if t.ndim == 2:
            return (self.scale_shift_table[None] + t.reshape(batch, 6, -1)).chunk(
                6, dim=1
            )
        groups = t.shape[2]
        return (
            self.scale_shift_table[None, None] + t.reshape(batch, groups, 6, -1)
        ).chunk(6, dim=2)

    def forward_attn_sublayer(self, x, y, t, mask=None, rotary_emb=None):
        batch, _, channels = x.shape
        shift, scale, gate, _, _, _ = self._modulation(t, batch)
        grouped_shape = (
            (batch, -1, channels) if t.ndim == 2 else (batch, t.shape[2], -1, channels)
        )
        attn_input = self.norm1(x).reshape(grouped_shape) * (1 + scale) + shift
        delta = self.attn(attn_input.reshape_as(x), rotary_emb)
        delta = (gate * delta.reshape(grouped_shape)).reshape_as(x)
        return delta + self.cross_attn(x + delta, y, mask)

    def forward_mlp_sublayer(self, x, t):
        batch, _, channels = x.shape
        _, _, _, shift, scale, gate = self._modulation(t, batch)
        grouped_shape = (
            (batch, -1, channels) if t.ndim == 2 else (batch, t.shape[2], -1, channels)
        )
        mlp_input = self.norm2(x).reshape(grouped_shape) * (1 + scale) + shift
        delta = self.mlp(mlp_input.reshape_as(x)).reshape(grouped_shape)
        return (gate * delta).reshape_as(x)


class DepthRMSNorm(nn.Module):
    def forward(self, x):
        return x * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt()


class BlockAttentionResidual(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn_proj = nn.Linear(hidden_size, 1, bias=False)
        self.mlp_proj = nn.Linear(hidden_size, 1, bias=False)
        self.final_proj = nn.Linear(hidden_size, 1, bias=False)
        self.key_norm = DepthRMSNorm()
        for projection in (self.attn_proj, self.mlp_proj, self.final_proj):
            nn.init.zeros_(projection.weight)

    def attend_buffer(self, projection, values, keys, active_count, partial):
        source_count = active_count
        if partial is not None:
            values[active_count] = partial
            keys[active_count] = self.key_norm(partial.unsqueeze(0)).squeeze(0)
            source_count += 1
        if source_count == 1:
            return values[0]
        values, keys = values[:source_count], keys[:source_count]
        logits = torch.einsum("d,nbtd->nbt", projection.weight.squeeze(0), keys)
        return torch.einsum("nbt,nbtd->btd", logits.softmax(0), values)


class SanaVideo2Transformer3DModel(BaseDiT):
    _fsdp_shard_conditions: ClassVar[list] = []
    _compile_conditions: ClassVar[list] = []
    _supported_attention_backends: ClassVar[set] = {AttentionBackendEnum.TORCH_SDPA}
    param_names_mapping: ClassVar[dict] = {}
    reverse_param_names_mapping: ClassVar[dict] = {}
    lora_param_names_mapping: ClassVar[dict] = {}

    def __init__(self, config: SanaVideo2Config, hf_config=None, **kwargs):
        super().__init__(config, hf_config=hf_config or {}, **kwargs)
        c = self.config
        self.hidden_size = c.hidden_size
        self.num_attention_heads = c.num_heads
        self.num_channels_latents = c.in_channels
        self.register_buffer(
            "pos_embed", torch.zeros(1, c.input_size**2, c.hidden_size)
        )
        self.x_embedder = PatchEmbedMS3D(c.patch_size, c.in_channels, c.hidden_size)
        self.t_embedder = TimestepEmbedder(c.hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(), nn.Linear(c.hidden_size, 6 * c.hidden_size)
        )
        self.y_embedder = CaptionEmbedder(
            c.caption_channels, c.hidden_size, c.model_max_length
        )
        if c.y_norm:
            self.attention_y_norm = _RMSNorm(
                c.hidden_size, c.y_norm_scale_factor, c.norm_eps
            )
        self.rope_linear = SanaVideo2RotaryPosEmbed(c.linear_head_dim, c.patch_size)
        self.rope_softmax = SanaVideo2RotaryPosEmbed(c.softmax_head_dim, c.patch_size)
        anchor_count = max(1, int(c.depth * c.softmax_ratio))
        self.softmax_layer_indices = [
            int((i + 1) * c.depth / anchor_count) - 1 for i in range(anchor_count)
        ]
        self.block_attention_types = [
            "softmax" if i in self.softmax_layer_indices else "linear"
            for i in range(c.depth)
        ]
        self.blocks = nn.ModuleList(
            [SanaVideo2Block(c, attention) for attention in self.block_attention_types]
        )
        self.final_layer = T2IFinalLayer(c.hidden_size, c.patch_size, c.in_channels)
        self.attn_res = BlockAttentionResidual(c.hidden_size)

    @property
    def dtype(self):
        return self.x_embedder.proj.weight.dtype

    def post_load_weights(self):
        for rope in (self.rope_linear, self.rope_softmax):
            if rope._freqs.is_meta:
                rope._init_freqs_buffer()

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        **kwargs,
    ):
        c = self.config
        batch, _, frames, height, width = hidden_states.shape
        if timestep.ndim == 5:
            timestep = timestep.reshape(batch, 1, frames)
        if c.timestep_norm_scale_factor == 1.0:
            timestep = timestep.long().float()
        else:
            timestep = timestep.float() / c.timestep_norm_scale_factor
        x = self.x_embedder(hidden_states.to(self.dtype))
        ropes = {
            "linear": self.rope_linear((frames, height, width), x.device),
            "softmax": self.rope_softmax((frames, height, width), x.device),
        }
        t = self.t_embedder(timestep.flatten())
        t0 = self.t_block(t).unflatten(0, timestep.shape)
        t = t.unflatten(0, timestep.shape)
        y = self.y_embedder(encoder_hidden_states.to(self.dtype))
        if c.y_norm:
            y = self.attention_y_norm(y)
        mask = encoder_attention_mask
        if mask is not None:
            mask = mask.reshape(batch, -1)
        # Keep the grouped depth buffer layout stable across denoising steps.
        count = math.ceil(c.depth / c.attn_res_block_size)
        values = x.new_empty(count + 2, batch, x.shape[1], c.hidden_size)
        keys = torch.empty_like(values)
        values[0] = x
        keys[0] = self.attn_res.key_norm(x.unsqueeze(0)).squeeze(0)
        active_count, partial = 1, None
        for index, block in enumerate(self.blocks):
            hidden = self.attn_res.attend_buffer(
                self.attn_res.attn_proj, values, keys, active_count, partial
            )
            delta = block.forward_attn_sublayer(
                hidden, y, t0, mask, ropes[self.block_attention_types[index]]
            )
            partial = delta if partial is None else partial + delta
            hidden = self.attn_res.attend_buffer(
                self.attn_res.mlp_proj, values, keys, active_count, partial
            )
            partial = partial + block.forward_mlp_sublayer(hidden, t0)
            if (index + 1) % c.attn_res_block_size == 0 or index + 1 == c.depth:
                values[active_count] = partial
                keys[active_count] = self.attn_res.key_norm(
                    partial.unsqueeze(0)
                ).squeeze(0)
                active_count += 1
                partial = None
        x = self.attn_res.attend_buffer(
            self.attn_res.final_proj, values, keys, active_count, None
        )
        return (
            self.final_layer(x, t)
            .transpose(1, 2)
            .reshape(batch, c.in_channels, frames, height, width)
        )


EntryClass = SanaVideo2Transformer3DModel
