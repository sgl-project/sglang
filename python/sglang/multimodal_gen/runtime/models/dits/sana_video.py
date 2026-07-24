# SPDX-License-Identifier: Apache-2.0
"""Native SANA-Video 3D diffusion transformer.

The architecture follows Diffusers ``SanaVideoTransformer3DModel`` while the
performance paths are adapted from Sol-Engine:

* self-attention Q/K/V and cross-attention K/V use SGLang packed linears;
* linear-attention aggregation can run in BF16;
* EasyCache skips the transformer block stack with request-local state;
* the block stack remains a stable torch.compile hot path.
"""

from __future__ import annotations

import math
from typing import ClassVar

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    get_1d_rotary_pos_embed,
)
from sglang.multimodal_gen.configs.models.dits.sana_video import SanaVideoConfig
from sglang.multimodal_gen.runtime.cache.easycache import EasyCacheController
from sglang.multimodal_gen.runtime.layers.layernorm import RMSNorm
from sglang.multimodal_gen.runtime.layers.linear import MergedColumnParallelLinear
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT
from sglang.multimodal_gen.runtime.models.dits.sana import SanaAdaLayerNormSingle
from torch import nn


class WanRotaryPosEmbed(nn.Module):
    """Three-dimensional RoPE with the frequency layout used by SANA-Video."""

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        height_dim = width_dim = 2 * (attention_head_dim // 6)
        time_dim = attention_head_dim - height_dim - width_dim
        self.split_sizes = (time_dim, height_dim, width_dim)

        # These tensors are derived data rather than checkpoint state. Keeping
        # them as lazy attributes also avoids meta-device materialization.
        self._freqs_cos: torch.Tensor | None = None
        self._freqs_sin: torch.Tensor | None = None

    def _ensure_freqs(self, device: torch.device) -> None:
        if self._freqs_cos is not None and self._freqs_cos.device == device:
            return

        cos_parts = []
        sin_parts = []
        for dim in self.split_sizes:
            cos, sin = get_1d_rotary_pos_embed(
                dim,
                self.max_seq_len,
                self.theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=torch.float64,
            )
            cos_parts.append(cos)
            sin_parts.append(sin)

        self._freqs_cos = torch.cat(cos_parts, dim=1).to(device)
        self._freqs_sin = torch.cat(sin_parts, dim=1).to(device)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_freqs(hidden_states.device)
        assert self._freqs_cos is not None
        assert self._freqs_sin is not None

        _, _, num_frames, height, width = hidden_states.shape
        patch_t, patch_h, patch_w = self.patch_size
        frames = num_frames // patch_t
        grid_height = height // patch_h
        grid_width = width // patch_w

        cos_t, cos_h, cos_w = self._freqs_cos.split(self.split_sizes, dim=1)
        sin_t, sin_h, sin_w = self._freqs_sin.split(self.split_sizes, dim=1)

        cos_t = cos_t[:frames].view(frames, 1, 1, -1)
        cos_h = cos_h[:grid_height].view(1, grid_height, 1, -1)
        cos_w = cos_w[:grid_width].view(1, 1, grid_width, -1)
        sin_t = sin_t[:frames].view(frames, 1, 1, -1)
        sin_h = sin_h[:grid_height].view(1, grid_height, 1, -1)
        sin_w = sin_w[:grid_width].view(1, 1, grid_width, -1)

        grid_shape = (frames, grid_height, grid_width, -1)
        freqs_cos = torch.cat(
            [
                cos_t.expand(grid_shape),
                cos_h.expand(grid_shape),
                cos_w.expand(grid_shape),
            ],
            dim=-1,
        )
        freqs_sin = torch.cat(
            [
                sin_t.expand(grid_shape),
                sin_h.expand(grid_shape),
                sin_w.expand(grid_shape),
            ],
            dim=-1,
        )
        sequence_length = frames * grid_height * grid_width
        return (
            freqs_cos.reshape(1, sequence_length, 1, -1),
            freqs_sin.reshape(1, sequence_length, 1, -1),
        )


def _apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    first, second = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    output = torch.empty_like(hidden_states)
    output[..., 0::2] = first * cos - second * sin
    output[..., 1::2] = first * sin + second * cos
    return output


class SanaVideoLinearAttention(nn.Module):
    """ReLU linear self-attention with packed QKV and three-dimensional RoPE."""

    def __init__(
        self,
        query_dim: int,
        num_heads: int,
        head_dim: int,
        *,
        qk_norm: bool,
        bias: bool,
        aggregation_precision: str,
    ) -> None:
        super().__init__()
        inner_dim = num_heads * head_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.aggregation_precision = aggregation_precision

        self.to_qkv = MergedColumnParallelLinear(
            query_dim,
            [inner_dim, inner_dim, inner_dim],
            bias=bias,
            gather_output=True,
        )
        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, query_dim, bias=True), nn.Identity()]
        )
        self.norm_q = RMSNorm(inner_dim) if qk_norm else None
        self.norm_k = RMSNorm(inner_dim) if qk_norm else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        original_dtype = hidden_states.dtype

        qkv, _ = self.to_qkv(hidden_states)
        query, key, value = qkv.split(
            [self.inner_dim, self.inner_dim, self.inner_dim], dim=-1
        )

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        key = key.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        value = value.view(batch_size, sequence_length, self.num_heads, self.head_dim)

        query = F.relu(query)
        key = F.relu(key)
        rotated_query = _apply_rotary_emb(query, *rotary_emb)
        rotated_key = _apply_rotary_emb(key, *rotary_emb)

        # [B, N, H, D] -> [B, H, D, N]
        query = query.permute(0, 2, 3, 1)
        key = key.permute(0, 2, 3, 1)
        rotated_query = rotated_query.permute(0, 2, 3, 1)
        rotated_key = rotated_key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 3, 1)

        aggregation_dtype = (
            torch.bfloat16 if self.aggregation_precision == "bf16" else torch.float32
        )
        rotated_query = rotated_query.to(aggregation_dtype)
        rotated_key = rotated_key.to(aggregation_dtype)
        value = value.to(aggregation_dtype)

        normalizer = 1.0 / (
            key.sum(dim=-1, keepdim=True).transpose(-2, -1) @ query + 1e-15
        )
        scores = value @ rotated_key.transpose(-1, -2)
        hidden_states = (scores @ rotated_query) * normalizer

        hidden_states = hidden_states.flatten(1, 2).transpose(1, 2)
        hidden_states = hidden_states.to(original_dtype)
        return self.to_out[1](self.to_out[0](hidden_states))


class SanaVideoCrossAttention(nn.Module):
    """Softmax cross-attention over Gemma2 text features."""

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int,
        num_heads: int,
        head_dim: int,
        *,
        qk_norm: bool,
    ) -> None:
        super().__init__()
        inner_dim = num_heads * head_dim
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.to_q = nn.Linear(query_dim, inner_dim, bias=True)
        self.to_kv = MergedColumnParallelLinear(
            cross_attention_dim,
            [inner_dim, inner_dim],
            bias=True,
            gather_output=True,
        )
        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, query_dim, bias=True), nn.Identity()]
        )
        self.norm_q = RMSNorm(inner_dim) if qk_norm else None
        self.norm_k = RMSNorm(inner_dim) if qk_norm else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        text_length = encoder_hidden_states.shape[1]

        query = self.to_q(hidden_states)
        key_value, _ = self.to_kv(encoder_hidden_states)
        key, value = key_value.split([self.inner_dim, self.inner_dim], dim=-1)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = query.view(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = key.view(
            batch_size, text_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = value.view(
            batch_size, text_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attention_mask = None
        if encoder_attention_mask is not None:
            if encoder_attention_mask.dtype == torch.bool:
                attention_mask = encoder_attention_mask[:, None, None, :].expand(
                    batch_size, self.num_heads, sequence_length, text_length
                )
            else:
                attention_mask = encoder_attention_mask.view(
                    batch_size, 1, -1, text_length
                ).expand(batch_size, self.num_heads, sequence_length, text_length)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, sequence_length, self.inner_dim
        )
        return self.to_out[1](self.to_out[0](hidden_states))


class GLUMBTempConv(nn.Module):
    """SANA gated convolutional FFN with temporal aggregation."""

    def __init__(
        self, in_channels: int, out_channels: int, expand_ratio: float
    ) -> None:
        super().__init__()
        hidden_channels = int(expand_ratio * in_channels)
        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(in_channels, hidden_channels * 2, kernel_size=1)
        self.conv_depth = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
        )
        self.conv_point = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=1, bias=False
        )
        self.conv_temp = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 1),
            padding=(1, 0),
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, height, width, num_channels = hidden_states.shape
        hidden_states = hidden_states.reshape(
            batch_size * num_frames, height, width, num_channels
        ).permute(0, 3, 1, 2)

        hidden_states = self.nonlinearity(self.conv_inverted(hidden_states))
        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = hidden_states.chunk(2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)
        hidden_states = self.conv_point(hidden_states)

        temporal = hidden_states.reshape(
            batch_size, num_frames, num_channels, height * width
        ).permute(0, 2, 1, 3)
        hidden_states = temporal + self.conv_temp(temporal)
        return hidden_states.permute(0, 2, 3, 1).reshape(
            batch_size, num_frames, height, width, num_channels
        )


class SanaVideoModulatedNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_embedding: torch.Tensor,
        scale_shift_table: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.norm(hidden_states)
        shift, scale = (
            scale_shift_table[None, None]
            + timestep_embedding[:, :, None].to(scale_shift_table.device)
        ).unbind(dim=2)
        return hidden_states * (1 + scale) + shift


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
        *,
        qk_norm: bool,
        attention_bias: bool,
        aggregation_precision: str,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn1 = SanaVideoLinearAttention(
            query_dim=dim,
            num_heads=num_attention_heads,
            head_dim=attention_head_dim,
            qk_norm=qk_norm,
            bias=attention_bias,
            aggregation_precision=aggregation_precision,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn2 = SanaVideoCrossAttention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            num_heads=num_cross_attention_heads,
            head_dim=cross_attention_head_dim,
            qk_norm=qk_norm,
        )
        self.ff = GLUMBTempConv(dim, dim, expand_ratio=mlp_ratio)
        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        frames: int,
        height: int,
        width: int,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = (
            self.scale_shift_table[None, None]
            + timestep.reshape(batch_size, timestep.shape[1], 6, -1)
        ).unbind(dim=2)

        normalized = self.norm1(hidden_states)
        normalized = normalized * (1 + scale_msa) + shift_msa
        attention_output = self.attn1(
            normalized.to(hidden_states.dtype), rotary_emb=rotary_emb
        )
        hidden_states = hidden_states + gate_msa * attention_output

        hidden_states = hidden_states + self.attn2(
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        normalized = self.norm2(hidden_states)
        normalized = normalized * (1 + scale_mlp) + shift_mlp
        feed_forward_output = self.ff(
            normalized.unflatten(1, (frames, height, width))
        ).flatten(1, 3)
        return hidden_states + gate_mlp * feed_forward_output


class SanaVideoTransformer3DModel(CachableDiT, LayerwiseOffloadableModuleMixin):
    """SANA-Video denoising transformer with Sol-Engine acceleration paths."""

    _fsdp_shard_conditions: ClassVar[list] = [
        lambda _name, module: isinstance(module, SanaVideoTransformerBlock),
    ]
    _compile_conditions: ClassVar[list] = [
        lambda _name, module: isinstance(module, SanaVideoTransformerBlock),
    ]
    param_names_mapping: ClassVar[dict] = (
        SanaVideoConfig().arch_config.param_names_mapping
    )
    reverse_param_names_mapping: ClassVar[dict] = {}

    def __init__(
        self, config: SanaVideoConfig, hf_config: dict | None = None, **kwargs
    ) -> None:
        super().__init__(config, hf_config=hf_config or {}, **kwargs)
        arch = config.arch_config

        self.out_channels = arch.out_channels
        self.patch_size = tuple(arch.patch_size)
        self.inner_dim = arch.num_attention_heads * arch.attention_head_dim
        self.hidden_size = self.inner_dim
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.out_channels
        self.enable_easycache = config.enable_easycache

        self.rope = WanRotaryPosEmbed(
            arch.attention_head_dim,
            self.patch_size,
            arch.rope_max_seq_len,
        )
        self.patch_embedding = nn.Conv3d(
            arch.in_channels,
            self.inner_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.time_embed = SanaAdaLayerNormSingle(self.inner_dim)
        self.caption_projection = PixArtAlphaTextProjection(
            in_features=arch.caption_channels,
            hidden_size=self.inner_dim,
        )
        self.caption_norm = RMSNorm(self.inner_dim)

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
                    qk_norm=arch.qk_norm is not None,
                    attention_bias=arch.attention_bias,
                    aggregation_precision=(
                        config.linear_attention_aggregation_precision
                    ),
                )
                for _ in range(arch.num_layers)
            ]
        )

        self.scale_shift_table = nn.Parameter(
            torch.randn(2, self.inner_dim) / self.inner_dim**0.5
        )
        self.norm_out = SanaVideoModulatedNorm(self.inner_dim, eps=arch.norm_eps)
        self.proj_out = nn.Linear(
            self.inner_dim,
            math.prod(self.patch_size) * self.out_channels,
        )
        self.layer_names = ["transformer_blocks"]

    def _run_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        frames: int,
        height: int,
        width: int,
        rotary_emb: tuple[torch.Tensor, torch.Tensor],
        encoder_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Stable block-stack hot path used by eager and compiled execution."""

        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep,
                frames,
                height,
                width,
                rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
            )
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor] | None = None,
        timestep: torch.LongTensor | None = None,
        guidance: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        del guidance, kwargs
        if encoder_hidden_states is None:
            raise ValueError("SANA-Video forward requires encoder_hidden_states")
        if timestep is None:
            raise ValueError("SANA-Video forward requires timestep")

        if isinstance(encoder_hidden_states, (list, tuple)):
            encoder_hidden_states = encoder_hidden_states[0]
        if isinstance(encoder_attention_mask, (list, tuple)):
            encoder_attention_mask = encoder_attention_mask[0]

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch_size, _, num_frames, height, width = hidden_states.shape
        patch_t, patch_h, patch_w = self.patch_size
        frames = num_frames // patch_t
        grid_height = height // patch_h
        grid_width = width // patch_w

        rotary_emb = self.rope(hidden_states)
        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        timestep_modulation, embedded_timestep = self.time_embed(
            timestep, hidden_dtype=hidden_states.dtype
        )
        timestep_modulation = timestep_modulation.reshape(
            batch_size, -1, timestep_modulation.shape[-1]
        )
        embedded_timestep = embedded_timestep.reshape(
            batch_size, -1, embedded_timestep.shape[-1]
        )

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        if encoder_hidden_states.shape[0] != batch_size:
            encoder_hidden_states = encoder_hidden_states.expand(
                batch_size, -1, -1
            ).contiguous()
        encoder_hidden_states = encoder_hidden_states.reshape(
            batch_size, -1, hidden_states.shape[-1]
        )
        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        cache_decision = (
            EasyCacheController.begin_forward(hidden_states)
            if self.enable_easycache
            else None
        )
        if cache_decision is None or cache_decision.should_compute:
            block_input = hidden_states
            hidden_states = self._run_blocks(
                hidden_states,
                encoder_hidden_states,
                timestep_modulation,
                frames,
                grid_height,
                grid_width,
                rotary_emb,
                encoder_attention_mask,
            )
            if cache_decision is not None:
                EasyCacheController.after_compute(
                    cache_decision, block_input, hidden_states
                )
        else:
            hidden_states = EasyCacheController.reuse(cache_decision, hidden_states)

        hidden_states = self.norm_out(
            hidden_states, embedded_timestep, self.scale_shift_table
        )
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            frames,
            grid_height,
            grid_width,
            patch_t,
            patch_h,
            patch_w,
            self.out_channels,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


EntryClass = SanaVideoTransformer3DModel
