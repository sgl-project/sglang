# Copyright 2025 The HuggingFace Team and SANA-Video Team.
# SPDX-License-Identifier: Apache-2.0
"""Native SGLang implementation of the SANA-Video 3D transformer."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import PixArtAlphaTextProjection

from sglang.multimodal_gen.configs.models.dits.sana_video import SanaVideoConfig
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.linear import MergedColumnParallelLinear
from sglang.multimodal_gen.runtime.layers.rotary_embedding.mrope import (
    get_1d_rotary_pos_embed,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.models.dits.sana import SanaAdaLayerNormSingle


def apply_interleaved_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """Apply Diffusers-compatible interleaved real RoPE to ``[B, N, H, D]``."""
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2].to(device=hidden_states.device)
    sin = freqs_sin[..., 1::2].to(device=hidden_states.device)
    output = torch.empty_like(hidden_states)
    output[..., 0::2] = x1 * cos - x2 * sin
    output[..., 1::2] = x1 * sin + x2 * cos
    return output


class SanaVideoRotaryPosEmbed(nn.Module):
    """3D RoPE split across temporal, height, and width head dimensions."""

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta
        self._init_freqs_buffers()

    def _init_freqs_buffers(self) -> None:
        h_dim = w_dim = 2 * (self.attention_head_dim // 6)
        t_dim = self.attention_head_dim - h_dim - w_dim
        self.split_sizes = (t_dim, h_dim, w_dim)

        freqs_cos = []
        freqs_sin = []
        for dim in self.split_sizes:
            cos, sin = get_1d_rotary_pos_embed(
                dim,
                self.max_seq_len,
                theta=self.theta,
                dtype=torch.float64,
            )
            freqs_cos.append(cos.repeat_interleave(2, dim=-1))
            freqs_sin.append(sin.repeat_interleave(2, dim=-1))
        self.register_buffer(
            "freqs_cos", torch.cat(freqs_cos, dim=-1), persistent=False
        )
        self.register_buffer(
            "freqs_sin", torch.cat(freqs_sin, dim=-1), persistent=False
        )

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, num_frames, height, width = hidden_states.shape
        patch_t, patch_h, patch_w = self.patch_size
        frames = num_frames // patch_t
        height = height // patch_h
        width = width // patch_w

        cos_t, cos_h, cos_w = self.freqs_cos.split(self.split_sizes, dim=-1)
        sin_t, sin_h, sin_w = self.freqs_sin.split(self.split_sizes, dim=-1)

        def expand_axis(table, axis):
            if axis == 0:
                return (
                    table[:frames]
                    .view(frames, 1, 1, -1)
                    .expand(frames, height, width, -1)
                )
            if axis == 1:
                return (
                    table[:height]
                    .view(1, height, 1, -1)
                    .expand(frames, height, width, -1)
                )
            return table[:width].view(1, 1, width, -1).expand(frames, height, width, -1)

        cos = torch.cat(
            [
                expand_axis(cos_t, 0),
                expand_axis(cos_h, 1),
                expand_axis(cos_w, 2),
            ],
            dim=-1,
        ).reshape(1, frames * height * width, 1, -1)
        sin = torch.cat(
            [
                expand_axis(sin_t, 0),
                expand_axis(sin_h, 1),
                expand_axis(sin_w, 2),
            ],
            dim=-1,
        ).reshape(1, frames * height * width, 1, -1)
        return cos, sin


class GLUMBTempConv(nn.Module):
    """SANA-Video gated spatial MLP with temporal aggregation."""

    def __init__(self, channels: int, expand_ratio: float) -> None:
        super().__init__()
        hidden_channels = int(expand_ratio * channels)
        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(channels, hidden_channels * 2, 1)
        self.conv_depth = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            3,
            padding=1,
            groups=hidden_channels * 2,
        )
        self.conv_point = nn.Conv2d(hidden_channels, channels, 1, bias=False)
        self.conv_temp = nn.Conv2d(
            channels,
            channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, height, width, channels = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch_size * num_frames, height, width, channels
        ).permute(0, 3, 1, 2)
        hidden_states = self.nonlinearity(self.conv_inverted(hidden_states))
        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)
        hidden_states = self.conv_point(hidden_states)

        temporal = hidden_states.reshape(
            batch_size, num_frames, channels, height * width
        ).permute(0, 2, 1, 3)
        hidden_states = temporal + self.conv_temp(temporal)
        return hidden_states.permute(0, 2, 3, 1).reshape(
            batch_size, num_frames, height, width, channels
        )


class SanaVideoLinearAttention(nn.Module):
    """Diffusers-compatible ReLU linear attention with packed QKV."""

    def __init__(
        self, query_dim: int, num_heads: int, head_dim: int, bias: bool
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.to_qkv = MergedColumnParallelLinear(
            query_dim,
            [self.inner_dim, self.inner_dim, self.inner_dim],
            bias=bias,
            gather_output=True,
        )
        self.norm_q = RMSNorm(self.inner_dim, eps=1e-5)
        self.norm_k = RMSNorm(self.inner_dim, eps=1e-5)
        self.to_out = nn.ModuleList(
            [nn.Linear(self.inner_dim, query_dim, bias=True), nn.Identity()]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        original_dtype = hidden_states.dtype
        batch_size, sequence_length, _ = hidden_states.shape
        qkv, _ = self.to_qkv(hidden_states)
        query, key, value = qkv.split(self.inner_dim, dim=-1)
        query = self.norm_q(query).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        key = self.norm_k(key).view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        value = value.view(batch_size, sequence_length, self.num_heads, self.head_dim)

        query = F.relu(query)
        key = F.relu(key)
        query_rotate = apply_interleaved_rotary_emb(query, *rotary_emb)
        key_rotate = apply_interleaved_rotary_emb(key, *rotary_emb)

        query = query.permute(0, 2, 3, 1)
        key = key.permute(0, 2, 3, 1)
        query_rotate = query_rotate.permute(0, 2, 3, 1).float()
        key_rotate = key_rotate.permute(0, 2, 3, 1).float()
        value = value.permute(0, 2, 3, 1).float()

        normalizer = 1.0 / (
            key.sum(dim=-1, keepdim=True).transpose(-2, -1) @ query + 1e-15
        )
        scores = value @ key_rotate.transpose(-1, -2)
        hidden_states = (scores @ query_rotate) * normalizer
        hidden_states = hidden_states.flatten(1, 2).transpose(1, 2)
        hidden_states = hidden_states.to(original_dtype)
        return self.to_out[0](hidden_states)


class SanaVideoCrossAttention(nn.Module):
    """Text cross-attention with packed K/V projections."""

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int,
        num_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=True)
        self.to_kv = MergedColumnParallelLinear(
            cross_attention_dim,
            [self.inner_dim, self.inner_dim],
            bias=True,
            gather_output=True,
        )
        self.norm_q = RMSNorm(self.inner_dim, eps=1e-5)
        self.norm_k = RMSNorm(self.inner_dim, eps=1e-5)
        self.to_out = nn.ModuleList(
            [nn.Linear(self.inner_dim, query_dim, bias=True), nn.Identity()]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size, query_length, _ = hidden_states.shape
        key_length = encoder_hidden_states.shape[1]
        query = self.norm_q(self.to_q(hidden_states))
        key_value, _ = self.to_kv(encoder_hidden_states)
        key, value = key_value.split(self.inner_dim, dim=-1)
        key = self.norm_k(key)

        query = query.view(
            batch_size, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(batch_size, key_length, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        value = value.view(
            batch_size, key_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attention_mask = None
        if encoder_attention_mask is not None:
            attention_mask = encoder_attention_mask.to(torch.bool)[:, None, None, :]

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, query_length, self.inner_dim
        )
        return self.to_out[0](hidden_states)


class SanaVideoTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_cross_attention_heads: int,
        cross_attention_head_dim: int,
        cross_attention_dim: int,
        mlp_ratio: float,
        norm_eps: float,
        attention_bias: bool,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn1 = SanaVideoLinearAttention(
            dim, num_attention_heads, attention_head_dim, attention_bias
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn2 = SanaVideoCrossAttention(
            dim,
            cross_attention_dim,
            num_cross_attention_heads,
            cross_attention_head_dim,
        )
        self.ff = GLUMBTempConv(dim, mlp_ratio)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
        timestep: torch.Tensor,
        frames: int,
        height: int,
        width: int,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None, None]
            + timestep.reshape(batch_size, timestep.shape[1], 6, -1)
        ).unbind(dim=2)

        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        hidden_states = hidden_states + gate_msa * self.attn1(
            norm_hidden_states.to(hidden_states.dtype), rotary_emb
        )
        hidden_states = hidden_states + self.attn2(
            hidden_states, encoder_hidden_states, encoder_attention_mask
        )

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
        norm_hidden_states = norm_hidden_states.unflatten(1, (frames, height, width))
        ff_output = self.ff(norm_hidden_states).flatten(1, 3)
        return hidden_states + gate_mlp * ff_output


class SanaVideoModulatedNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedded_timestep: torch.Tensor,
        scale_shift_table: torch.Tensor,
    ) -> torch.Tensor:
        shift, scale = (
            scale_shift_table[None, None] + embedded_timestep[:, :, None]
        ).unbind(dim=2)
        return self.norm(hidden_states) * (1 + scale) + shift


class SanaVideoTransformer3DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    _fsdp_shard_conditions = [
        lambda _name, module: isinstance(module, SanaVideoTransformerBlock)
    ]
    _compile_conditions = [
        lambda _name, module: isinstance(module, SanaVideoTransformerBlock)
    ]
    param_names_mapping = SanaVideoConfig().arch_config.param_names_mapping
    reverse_param_names_mapping = {}

    def __init__(self, config: SanaVideoConfig, hf_config=None, **kwargs) -> None:
        super().__init__(config, hf_config=hf_config or {}, **kwargs)
        arch = config.arch_config
        self.out_channels = arch.out_channels
        self.patch_size = tuple(arch.patch_size)
        self.inner_dim = arch.num_attention_heads * arch.attention_head_dim
        self.hidden_size = self.inner_dim
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.caption_channels = arch.caption_channels
        self.cross_attention_dim = arch.cross_attention_dim

        self.rope = SanaVideoRotaryPosEmbed(
            arch.attention_head_dim, self.patch_size, arch.rope_max_seq_len
        )
        self.patch_embedding = nn.Conv3d(
            arch.in_channels,
            self.inner_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        if arch.guidance_embeds:
            raise NotImplementedError(
                "SANA-Video checkpoints with embedded guidance are not supported"
            )
        self.time_embed = SanaAdaLayerNormSingle(self.inner_dim)
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=arch.caption_channels, hidden_size=self.inner_dim
        )
        self.caption_norm = RMSNorm(self.inner_dim, eps=1e-5)
        self.transformer_blocks = nn.ModuleList(
            [
                SanaVideoTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=arch.num_attention_heads,
                    attention_head_dim=arch.attention_head_dim,
                    num_cross_attention_heads=arch.num_cross_attention_heads,
                    cross_attention_head_dim=arch.cross_attention_head_dim,
                    cross_attention_dim=arch.cross_attention_dim,
                    mlp_ratio=arch.mlp_ratio,
                    norm_eps=arch.norm_eps,
                    attention_bias=arch.attention_bias,
                )
                for _ in range(arch.num_layers)
            ]
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, self.inner_dim) / self.inner_dim**0.5
        )
        self.norm_out = SanaVideoModulatedNorm(self.inner_dim, arch.norm_eps)
        self.proj_out = nn.Linear(
            self.inner_dim, math.prod(self.patch_size) * self.out_channels
        )
        self.layer_names = ["transformer_blocks"]

    def post_load_weights(self) -> None:
        if self.rope.freqs_cos.is_meta or self.rope.freqs_sin.is_meta:
            self.rope._init_freqs_buffers()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        guidance: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del guidance, kwargs
        if encoder_hidden_states is None:
            raise ValueError("SANA-Video requires encoder_hidden_states")
        if isinstance(encoder_attention_mask, (list, tuple)):
            encoder_attention_mask = (
                encoder_attention_mask[0] if encoder_attention_mask else None
            )

        batch_size, _, num_frames, height, width = hidden_states.shape
        patch_t, patch_h, patch_w = self.patch_size
        post_patch_frames = num_frames // patch_t
        post_patch_height = height // patch_h
        post_patch_width = width // patch_w
        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        timestep, embedded_timestep = self.time_embed(
            timestep.flatten(), hidden_dtype=hidden_states.dtype
        )
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(
            batch_size, -1, embedded_timestep.shape[-1]
        )

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(
            batch_size, -1, hidden_states.shape[-1]
        )
        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                encoder_attention_mask,
                timestep,
                post_patch_frames,
                post_patch_height,
                post_patch_width,
                rotary_emb,
            )

        hidden_states = self.norm_out(
            hidden_states, embedded_timestep, self.scale_shift_table
        )
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_frames,
            post_patch_height,
            post_patch_width,
            patch_t,
            patch_h,
            patch_w,
            self.out_channels,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3).float()


EntryClass = SanaVideoTransformer3DModel
