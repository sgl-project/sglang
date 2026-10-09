# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
"""Kandinsky 6 transformer with channel-last video and [B, A, D] audio latents.

TP shards whole heads. SP shards video tokens; text and audio stay replicated.
Audio-to-video cross-attention gathers projected video K/V and removes padding.
NABLA attention is not supported.
"""

import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from sglang.kernels.ops.diffusion import apply_matrix_rope, residual_gate_fp32
from sglang.multimodal_gen.configs.models.dits.kandinsky6 import (
    Kandinsky6ArchConfig,
    Kandinsky6VideoAudioConfig,
)
from sglang.multimodal_gen.configs.models.fsdp import is_module_list_entry_in
from sglang.multimodal_gen.runtime.distributed import divide, get_tp_world_size
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_ctx,
    get_ulysses_ctx,
)
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    gather_seq,
    shard_like,
    shard_seq,
    tail_attn_meta,
)
from sglang.multimodal_gen.runtime.layers.activation import get_act_fn
from sglang.multimodal_gen.runtime.layers.attention import USPAttention
from sglang.multimodal_gen.runtime.layers.layernorm import LayerNormScaleShift
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)
from sglang.multimodal_gen.runtime.models.dits.base import BaseDiT
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.srt.utils import add_prefix

logger = init_logger(__name__)

_ARCH_CONFIG_DEFAULTS = Kandinsky6VideoAudioConfig().arch_config

# block containers for FSDP and compilation across video-only and joint models
_KANDINSKY6_BLOCK_CONTAINERS = (
    "text_transformer_blocks",
    "video_text_transformer_blocks",
    "audio_text_transformer_blocks",
    "visual_transformer_blocks",
)


def _is_kandinsky6_transformer_block(name: str, module: object) -> bool:
    return is_module_list_entry_in(name, _KANDINSKY6_BLOCK_CONTAINERS)


def _validate_parallelism(num_heads: int, **tp_dimensions: int) -> None:
    """TP splits heads/FFNs; Ulysses splits video heads remaining after TP."""
    tp_size = get_tp_world_size()
    ulysses_size, _ = get_ulysses_ctx()
    ring_size, _ = get_ring_ctx()
    for name, size in (("TP", tp_size), ("Ulysses", ulysses_size), ("ring", ring_size)):
        if size <= 0:
            raise ValueError(f"Kandinsky6 {name} size must be positive.")
    for name, value in dict(num_attention_heads=num_heads, **tp_dimensions).items():
        if value % tp_size:
            raise ValueError(
                f"Kandinsky6 {name}={value} must be divisible by TP size {tp_size}."
            )
    # ring rotates complete K/V shards, so it imposes no head divisibility constraint
    local_heads = num_heads // tp_size
    if local_heads % ulysses_size:
        raise ValueError(
            f"Kandinsky6 TP-local video attention heads {local_heads} must be "
            f"divisible by Ulysses size {ulysses_size} (total heads={num_heads}, TP={tp_size})."
        )


def _build_rotary_freqs(dim: int, max_period: float) -> torch.Tensor:
    return torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=dim, dtype=torch.float32)
        / dim
    )


class Kandinsky6TimeEmbeddings(nn.Module):
    """Sinusoidal timestep embedding -> Linear -> SiLU -> Linear."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        max_period: float = 10000.0,
        prefix: str = "",
    ):
        super().__init__()
        assert model_dim % 2 == 0
        self.model_dim = model_dim
        self.max_period = max_period
        # plain fp32 frequencies are rebuilt by post_load_weights after meta initialization
        self.freqs = _build_rotary_freqs(self.model_dim // 2, self.max_period)
        self.in_layer = ReplicatedLinear(
            model_dim, time_dim, bias=True, prefix=add_prefix("in_layer", prefix)
        )
        self.activation = nn.SiLU()
        self.out_layer = ReplicatedLinear(
            time_dim, time_dim, bias=True, prefix=add_prefix("out_layer", prefix)
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        # compute sinusoidal features in fp32, then cast to the loaded linear dtype
        args = torch.outer(time, self.freqs.to(device=time.device))
        time_embed = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        time_embed = time_embed.to(self.in_layer.weight.dtype)
        time_embed, _ = self.in_layer(time_embed)
        time_embed = self.activation(time_embed)
        time_embed, _ = self.out_layer(time_embed)
        return time_embed


class Kandinsky6TextEmbeddings(nn.Module):
    """Linear + LayerNorm projection, reused for text tokens and audio latents."""

    def __init__(self, in_dim: int, model_dim: int, prefix: str = ""):
        super().__init__()
        self.in_layer = ReplicatedLinear(
            in_dim, model_dim, bias=True, prefix=add_prefix("in_layer", prefix)
        )
        self.norm = nn.LayerNorm(model_dim, elementwise_affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.in_layer(x)
        return self.norm(x).type_as(x)


class Kandinsky6VisualEmbeddings(nn.Module):
    """Patchifies ``[B, T, H, W, C]`` video into non-overlapping ``patch_size`` patches."""

    def __init__(
        self,
        visual_dim: int,
        model_dim: int,
        patch_size: tuple[int, int, int],
        prefix: str = "",
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_layer = ReplicatedLinear(
            math.prod(patch_size) * visual_dim,
            model_dim,
            prefix=add_prefix("in_layer", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pt, ph, pw = self.patch_size
        x = rearrange(
            x,
            "b (t pt) (h ph) (w pw) c -> b t h w (pt ph pw c)",
            pt=pt,
            ph=ph,
            pw=pw,
        )
        x, _ = self.in_layer(x)
        return x


class Kandinsky6RoPE1D(nn.Module):
    """1D rotary embedding for text and audio sequences."""

    def __init__(
        self,
        dim: int,
        max_pos: int = 2048,
        max_period: float = 10000.0,
        freqs_scaling: float = 1.0,
    ):
        super().__init__()
        self.max_period = max_period
        self.dim = dim
        self.max_pos = max_pos
        self.freqs_scaling = freqs_scaling
        freq = _build_rotary_freqs(dim // 2, max_period) * freqs_scaling
        pos = torch.arange(max_pos, dtype=freq.dtype)
        self.register_buffer("args", torch.outer(pos, freq), persistent=False)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        args = self.args[pos]
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class Kandinsky6RoPE3D(nn.Module):
    """3D rotary embedding for video spatial-temporal (T, H, W) tokens."""

    def __init__(
        self,
        axes_dims: tuple[int, int, int],
        max_pos: tuple[int, int, int] = (128, 128, 128),
        max_period: float = 10000.0,
    ):
        super().__init__()
        self.axes_dims = axes_dims
        self.max_pos = max_pos
        self.max_period = max_period

        for i, (axes_dim, ax_max_pos) in enumerate(
            zip(axes_dims, max_pos, strict=True)
        ):
            freq = _build_rotary_freqs(axes_dim // 2, max_period)
            pos = torch.arange(ax_max_pos, dtype=freq.dtype)
            self.register_buffer(f"args_{i}", torch.outer(pos, freq), persistent=False)

    def forward(
        self,
        shape: tuple[int, int, int, int],
        pos: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | list[torch.Tensor],
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> torch.Tensor:
        batch_size, duration, height, width = shape
        args_t = self.args_0[pos[0]] / scale_factor[0]
        args_h = self.args_1[pos[1]] / scale_factor[1]
        args_w = self.args_2[pos[2]] / scale_factor[2]

        args = torch.cat(
            [
                args_t.view(1, duration, 1, 1, -1).expand(
                    batch_size, -1, height, width, -1
                ),
                args_h.view(1, 1, height, 1, -1).expand(
                    batch_size, duration, -1, width, -1
                ),
                args_w.view(1, 1, 1, width, -1).expand(
                    batch_size, duration, height, -1, -1
                ),
            ],
            dim=-1,
        )
        cosine = torch.cos(args)
        sine = torch.sin(args)
        rope = torch.stack([cosine, -sine, sine, cosine], dim=-1)
        rope = rope.view(*rope.shape[:-1], 2, 2)
        return rope.unsqueeze(-4)


class Kandinsky6Modulation(nn.Module):
    """AdaLN-style modulation generator: SiLU -> Linear (zero-initialized)."""

    def __init__(self, time_dim: int, model_dim: int, num_params: int):
        super().__init__()
        self.activation = nn.SiLU()
        self.out_layer = ReplicatedLinear(time_dim, num_params * model_dim, bias=True)
        self.out_layer.weight.data.zero_()
        self.out_layer.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # cast explicitly: fp32 autocast does not convert loaded bf16 weights
        x = x.to(self.out_layer.weight.dtype)
        x = self.activation(x)
        x, _ = self.out_layer(x)
        return x


def _apply_rotary(
    x: torch.Tensor, rope: torch.Tensor, dtype: torch.dtype | None = None
) -> torch.Tensor:
    dtype = x.dtype if dtype is None else dtype
    if (
        x.is_cuda
        and torch.version.hip is None
        and not torch.is_grad_enabled()
        and x.is_contiguous()
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and rope.dtype == torch.float32
    ):
        return apply_matrix_rope(x, rope, dtype)
    x_ = x.to(dtype).reshape(*x.shape[:-1], -1, 1, 2).to(torch.float32)
    x_out = (rope * x_).sum(dim=-1)
    return x_out.reshape(*x.shape).to(dtype)


class Kandinsky6QKNorm(nn.RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            x.is_cuda
            and torch.version.hip is None
            and x.dtype == torch.float32
            and self.normalized_shape == (128,)
            and not torch.is_grad_enabled()
        ):
            # matching dtypes enables native fusion without changing parameter
            # storage; other widths retain their original reduction order
            return F.rms_norm(x, self.normalized_shape, self.weight.float(), self.eps)
        return super().forward(x)


class Kandinsky6Attention(nn.Module):
    """TP-sharded attention with head-local QK norm and optional asymmetric K/V.

    Video self-attention uses Ulysses/ring. Text/audio self-attention stays
    replicated; cross-attention defaults to local K/V unless explicitly gathered."""

    def __init__(
        self,
        num_channels: int,
        head_dim: int,
        supported_attention_backends: set[AttentionBackendEnum] | None,
        prefix: str = "",
        kv_dim: int | None = None,
        quant_config: QuantizationConfig | None = None,
        is_cross_attention: bool = False,
        skip_sequence_parallel: bool | None = None,
    ):
        super().__init__()
        assert num_channels % head_dim == 0
        self.num_heads = num_channels // head_dim
        kv_dim = kv_dim or num_channels
        tp_size = get_tp_world_size()
        self.local_num_heads = divide(self.num_heads, tp_size)

        for name, width in (
            ("to_query", num_channels),
            ("to_key", kv_dim),
            ("to_value", kv_dim),
        ):
            self.add_module(
                name,
                ColumnParallelLinear(
                    width,
                    num_channels,
                    bias=True,
                    gather_output=False,
                    quant_config=quant_config,
                    prefix=add_prefix(name, prefix),
                ),
            )
        self.query_norm = Kandinsky6QKNorm(head_dim)
        self.key_norm = Kandinsky6QKNorm(head_dim)
        self.out_layer = RowParallelLinear(
            num_channels,
            num_channels,
            bias=True,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=add_prefix("out_layer", prefix),
        )
        self.attention = USPAttention(
            num_heads=self.local_num_heads,
            head_size=head_dim,
            causal=False,
            supported_attention_backends=supported_attention_backends,
            is_cross_attention=is_cross_attention,
            skip_sequence_parallel=(
                is_cross_attention
                if skip_sequence_parallel is None
                else skip_sequence_parallel
            ),
            prefix=add_prefix("attention", prefix),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        rotary_emb: torch.Tensor | None = None,
        rotary_emb_kv: torch.Tensor | None = None,
        attn_mask_meta: dict | None = None,
        context_seq_len: int | None = None,
        skip_sequence_parallel: bool = False,
    ) -> torch.Tensor:
        query, _ = self.to_query(hidden_states)

        kv_source = (
            hidden_states if encoder_hidden_states is None else encoder_hidden_states
        )
        key, _ = self.to_key(kv_source)
        value, _ = self.to_value(kv_source)

        # reshape rank-local whole heads after the TP projections
        shape, kv_shape = query.shape[:-1], key.shape[:-1]
        query = query.reshape(*shape, self.local_num_heads, -1)
        key = key.reshape(*kv_shape, self.local_num_heads, -1)
        value = value.reshape(*kv_shape, self.local_num_heads, -1)

        query_dtype, key_dtype = query.dtype, key.dtype
        query = self.query_norm(query.float())
        if rotary_emb is not None:
            query = _apply_rotary(query, rotary_emb, query_dtype)
        else:
            query = query.to(query_dtype)
        kv_rope = (
            rotary_emb_kv
            if rotary_emb_kv is not None
            else (rotary_emb if encoder_hidden_states is None else None)
        )
        key = self.key_norm(key.float())
        if kv_rope is not None:
            key = _apply_rotary(key, kv_rope, key_dtype)
        else:
            key = key.to(key_dtype)

        if context_seq_len is not None:
            key = gather_seq(key, context_seq_len)
            value = gather_seq(value, context_seq_len)

        hidden_states = self.attention(
            query,
            key,
            value,
            attn_mask_meta=attn_mask_meta,
            skip_sequence_parallel_override=skip_sequence_parallel,
        )
        hidden_states = hidden_states.flatten(-2, -1)
        hidden_states, _ = self.out_layer(hidden_states)
        return hidden_states


class Kandinsky6FeedForward(nn.Module):
    """Bias-free TP MLP with checkpoint-mapped mlp.fc_in/fc_out names.

    The shared MLP currently hardcodes bias=True, so it cannot be used here."""

    def __init__(
        self,
        dim: int,
        ff_dim: int,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        prefix = add_prefix("mlp", prefix)
        self.mlp = nn.Module()
        self.mlp.fc_in = ColumnParallelLinear(
            dim,
            ff_dim,
            bias=False,
            gather_output=False,
            quant_config=quant_config,
            prefix=add_prefix("fc_in", prefix),
        )
        self.mlp.act = get_act_fn("gelu")
        self.mlp.fc_out = RowParallelLinear(
            ff_dim,
            dim,
            bias=False,
            input_is_parallel=True,
            quant_config=quant_config,
            prefix=add_prefix("fc_out", prefix),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.mlp.fc_in(x)
        x = self.mlp.act(x)
        x, _ = self.mlp.fc_out(x)
        return x


def _norm_scale_shift(
    norm: LayerNormScaleShift, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """Compute LayerNorm and AdaLN affine in fp32, then restore the input dtype."""
    return norm(x.float(), shift=shift, scale=scale).type_as(x)


class Kandinsky6OutLayer(nn.Module):
    """Projects visual hidden states back to packed latent patches."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        visual_dim: int,
        patch_size: tuple[int, int, int],
    ):
        super().__init__()
        self.patch_size = patch_size
        self.modulation = Kandinsky6Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, eps=1e-5, elementwise_affine=False)
        self.out_layer = ReplicatedLinear(
            model_dim, math.prod(patch_size) * visual_dim, bias=True
        )

    def forward(
        self,
        visual_embed: torch.Tensor,
        time_embed: torch.Tensor,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        shift, scale = torch.chunk(
            self.modulation(time_embed).unsqueeze(dim=1), 2, dim=-1
        )
        visual_embed = (
            self.norm(visual_embed.float()) * (scale.float()[:, None, None] + 1.0)
            + shift.float()[:, None, None]
        ).to(dtype=compute_dtype or visual_embed.dtype)

        x, _ = self.out_layer(visual_embed)

        pt, ph, pw = self.patch_size
        return rearrange(
            x,
            "b t h w (c pt ph pw) -> b (t pt) (h ph) (w pw) c",
            pt=pt,
            ph=ph,
            pw=pw,
        )


class Kandinsky6OutLayerAudio(nn.Module):
    """Projects audio hidden states back to audio latent channels."""

    def __init__(self, model_dim: int, time_dim: int, audio_dim: int):
        super().__init__()
        self.modulation = Kandinsky6Modulation(time_dim, model_dim, 2)
        self.norm = nn.LayerNorm(model_dim, eps=1e-5, elementwise_affine=False)
        self.out_layer = ReplicatedLinear(model_dim, audio_dim, bias=True)

    def forward(
        self, audio_embed: torch.Tensor, time_embed: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = torch.chunk(
            self.modulation(time_embed).unsqueeze(dim=1), 2, dim=-1
        )
        x = (
            self.norm(audio_embed.float()) * (scale.float() + 1.0) + shift.float()
        ).type_as(audio_embed)
        # the reference applies a second norm after modulation; preserve its arithmetic
        x = self.norm(x.float()).type_as(audio_embed)
        out, _ = self.out_layer(x)
        return out


class Kandinsky6TransformerBlock(nn.Module):
    """Shared parameter layout for text, video and SR transformer blocks."""

    modulation_name = "visual_modulation"
    with_cross_attention = False
    skip_sequence_parallel = False

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.add_module(
            self.modulation_name,
            Kandinsky6Modulation(
                time_dim, model_dim, 9 if self.with_cross_attention else 6
            ),
        )
        attention_names = (
            ("self_attention", "cross_attention")
            if self.with_cross_attention
            else ("self_attention",)
        )
        for name in (*attention_names, "feed_forward"):
            self.add_module(
                f"{name}_norm",
                LayerNormScaleShift(
                    model_dim, eps=1e-5, elementwise_affine=False, dtype=torch.float32
                ),
            )
        for name in attention_names:
            self.add_module(
                name,
                Kandinsky6Attention(
                    model_dim,
                    head_dim,
                    supported_attention_backends,
                    prefix=add_prefix(name, prefix),
                    quant_config=quant_config,
                    is_cross_attention=name == "cross_attention",
                    skip_sequence_parallel=self.skip_sequence_parallel
                    or name == "cross_attention",
                ),
            )
        self.feed_forward = Kandinsky6FeedForward(
            model_dim,
            ff_dim,
            prefix=add_prefix("feed_forward", prefix),
            quant_config=quant_config,
        )


class Kandinsky6TransformerEncoderBlock(Kandinsky6TransformerBlock):
    """Replicated text self-attention and FFN."""

    modulation_name = "text_modulation"
    skip_sequence_parallel = True

    def forward(
        self, x: torch.Tensor, time_embed: torch.Tensor, rope: torch.Tensor
    ) -> torch.Tensor:
        self_attn_params, ff_params = torch.chunk(
            self.text_modulation(time_embed).unsqueeze(dim=1), 2, dim=-1
        )
        shift, scale, gate = torch.chunk(self_attn_params, 3, dim=-1)
        out = _norm_scale_shift(self.self_attention_norm, x, shift, scale)
        out = self.self_attention(out, rotary_emb=rope)
        x = (x.float() + gate.float() * out.float()).type_as(x)

        ff_shift, ff_scale, ff_gate = torch.chunk(ff_params, 3, dim=-1)
        out = _norm_scale_shift(self.feed_forward_norm, x, ff_shift, ff_scale)
        out = self.feed_forward(out)
        x = (x.float() + ff_gate.float() * out.float()).type_as(x)

        return x


class Kandinsky6TransformerDecoderBlock(Kandinsky6TransformerBlock):
    """Video self-attention, text cross-attention and FFN."""

    with_cross_attention = True

    def forward(
        self,
        visual_embed: torch.Tensor,
        text_embed: torch.Tensor,
        time_embed: torch.Tensor,
        rope: torch.Tensor | None,
        attn_mask_meta: dict | None = None,
    ) -> torch.Tensor:
        self_attn_params, cross_attn_params, ff_params = torch.chunk(
            self.visual_modulation(time_embed).unsqueeze(dim=1), 3, dim=-1
        )

        self_shift, self_scale, self_gate = torch.chunk(self_attn_params, 3, dim=-1)
        visual_out = _norm_scale_shift(
            self.self_attention_norm, visual_embed, self_shift, self_scale
        )
        visual_out = self.self_attention(
            visual_out,
            rotary_emb=rope,
            attn_mask_meta=attn_mask_meta,
        )
        visual_embed = (
            visual_embed.float() + self_gate.float() * visual_out.float()
        ).type_as(visual_embed)

        cross_shift, cross_scale, cross_gate = torch.chunk(cross_attn_params, 3, dim=-1)
        visual_out = _norm_scale_shift(
            self.cross_attention_norm, visual_embed, cross_shift, cross_scale
        )
        visual_out = self.cross_attention(visual_out, encoder_hidden_states=text_embed)
        visual_embed = (
            visual_embed.float() + cross_gate.float() * visual_out.float()
        ).type_as(visual_embed)

        ff_shift, ff_scale, ff_gate = torch.chunk(ff_params, 3, dim=-1)
        visual_out = _norm_scale_shift(
            self.feed_forward_norm, visual_embed, ff_shift, ff_scale
        )
        visual_out = self.feed_forward(visual_out)
        visual_embed = (
            visual_embed.float() + ff_gate.float() * visual_out.float()
        ).type_as(visual_embed)

        return visual_embed


def _apply_scale_shift(
    norm: nn.LayerNorm, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
) -> torch.Tensor:
    return (norm(x.float()) * (scale.float() + 1.0) + shift.float()).type_as(x)


class Kandinsky6FusedTransformerDecoderBlock(nn.Module):
    """Joint video/audio decoder with independently modulated bidirectional attention."""

    def __init__(
        self,
        model_dim: int,
        time_dim: int,
        ff_dim: int,
        head_dim: int,
        model_dim_a: int,
        time_dim_a: int,
        ff_dim_a: int,
        head_dim_a: int,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        ca_rope: bool = False,
        cross_gates: bool = False,
        fix_modulation: bool = False,
        quant_config: QuantizationConfig | None = None,
    ):
        super().__init__()
        self.videoT = Kandinsky6TransformerDecoderBlock(
            model_dim,
            time_dim,
            ff_dim,
            head_dim,
            supported_attention_backends,
            prefix=add_prefix("videoT", prefix),
            quant_config=quant_config,
        )
        self.audioT = Kandinsky6TransformerDecoderBlock(
            model_dim_a,
            time_dim_a,
            ff_dim_a,
            head_dim_a,
            supported_attention_backends,
            prefix=add_prefix("audioT", prefix),
            quant_config=quant_config,
        )
        self.va_cross_attention = Kandinsky6Attention(
            model_dim,
            head_dim,
            supported_attention_backends=supported_attention_backends,
            prefix=add_prefix("va_cross_attention", prefix),
            kv_dim=model_dim_a,
            quant_config=quant_config,
            is_cross_attention=True,
        )
        self.av_cross_attention = Kandinsky6Attention(
            model_dim_a,
            head_dim_a,
            supported_attention_backends=supported_attention_backends,
            prefix=add_prefix("av_cross_attention", prefix),
            kv_dim=model_dim,
            quant_config=quant_config,
            is_cross_attention=True,
        )
        self.va_modulation = Kandinsky6Modulation(
            time_dim,
            model_dim if not cross_gates else model_dim * 2 + model_dim_a,
            1 if cross_gates else 3,
        )
        self.av_modulation = Kandinsky6Modulation(
            time_dim_a,
            model_dim_a if not cross_gates else model_dim_a * 2 + model_dim,
            1 if cross_gates else 3,
        )
        self.va_normalization = nn.LayerNorm(model_dim, elementwise_affine=False)
        self.av_normalization = nn.LayerNorm(model_dim_a, elementwise_affine=False)
        self.ca_rope = ca_rope
        self.cross_gates = cross_gates
        self.fix_modulation = fix_modulation
        self.model_dim = model_dim
        self.model_dim_a = model_dim_a

    def forward(
        self,
        vis: torch.Tensor,
        aud: torch.Tensor,
        text_v: torch.Tensor,
        text_a: torch.Tensor,
        time_embed: tuple[torch.Tensor, torch.Tensor],
        vis_rope: torch.Tensor | None,
        aud_rope: torch.Tensor | None,
        va_gate_scale: float = 1.0,
        av_gate_scale: float = 1.0,
        video_seq_len: int | None = None,
        video_attn_meta: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t_v, t_a = time_embed

        sa_p, ca_p, ff_p = torch.chunk(
            self.videoT.visual_modulation(t_v).unsqueeze(dim=1), 3, dim=-1
        )
        shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
        vis = residual_gate_fp32(
            vis,
            self.videoT.self_attention(
                _norm_scale_shift(self.videoT.self_attention_norm, vis, shift, scale),
                rotary_emb=vis_rope,
                attn_mask_meta=video_attn_meta,
            ),
            gate,
        )
        shift, scale, gate_v = torch.chunk(ca_p, 3, dim=-1)
        vis_pre_ca = _norm_scale_shift(
            self.videoT.cross_attention_norm, vis, shift, scale
        )
        vis_out_t = self.videoT.cross_attention(
            vis_pre_ca, encoder_hidden_states=text_v
        )

        sa_p, ca_p, ff_p_a = torch.chunk(
            self.audioT.visual_modulation(t_a).unsqueeze(dim=1), 3, dim=-1
        )
        shift, scale, gate = torch.chunk(sa_p, 3, dim=-1)
        aud = residual_gate_fp32(
            aud,
            self.audioT.self_attention(
                _norm_scale_shift(self.audioT.self_attention_norm, aud, shift, scale),
                rotary_emb=aud_rope,
                skip_sequence_parallel=True,
            ),
            gate,
        )
        shift, scale, gate_a = torch.chunk(ca_p, 3, dim=-1)
        aud_pre_ca = _norm_scale_shift(
            self.audioT.cross_attention_norm, aud, shift, scale
        )
        aud_out_t = self.audioT.cross_attention(
            aud_pre_ca, encoder_hidden_states=text_a
        )
        aud = residual_gate_fp32(aud, aud_out_t, gate_a)

        t_va_mod = t_a if not self.fix_modulation else t_v
        t_av_mod = t_v if not self.fix_modulation else t_a
        va_params = self.va_modulation(t_va_mod).unsqueeze(dim=1)
        av_params = self.av_modulation(t_av_mod).unsqueeze(dim=1)
        if self.cross_gates:
            va_shift, va_scale, va_gate = torch.split(
                va_params, [self.model_dim, self.model_dim, self.model_dim_a], dim=-1
            )
            av_shift, av_scale, av_gate = torch.split(
                av_params, [self.model_dim_a, self.model_dim_a, self.model_dim], dim=-1
            )
        else:
            va_shift, va_scale, va_gate = torch.chunk(va_params, 3, dim=-1)
            av_shift, av_scale, av_gate = torch.chunk(av_params, 3, dim=-1)

        vis = residual_gate_fp32(vis, vis_out_t, gate_v)
        vis_for_va = _apply_scale_shift(self.va_normalization, vis, va_scale, va_shift)
        aud_for_av = _apply_scale_shift(self.av_normalization, aud, av_scale, av_shift)
        rq_v = vis_rope if self.ca_rope else None
        rk_a = aud_rope if self.ca_rope else None
        vis_from_aud = self.va_cross_attention(
            vis_for_va,
            encoder_hidden_states=aud_pre_ca,
            rotary_emb=rq_v,
            rotary_emb_kv=rk_a,
        )
        aud_from_vis = self.av_cross_attention(
            aud_for_av,
            encoder_hidden_states=vis_pre_ca,
            rotary_emb=rk_a,
            rotary_emb_kv=rq_v,
            context_seq_len=video_seq_len,
        )
        vis = residual_gate_fp32(
            vis,
            vis_from_aud,
            (va_gate if not self.cross_gates else av_gate) * va_gate_scale,
        )
        aud = residual_gate_fp32(
            aud,
            aud_from_vis,
            (av_gate if not self.cross_gates else va_gate) * av_gate_scale,
        )

        shift, scale, gate = torch.chunk(ff_p, 3, dim=-1)
        vis = residual_gate_fp32(
            vis,
            self.videoT.feed_forward(
                _norm_scale_shift(self.videoT.feed_forward_norm, vis, shift, scale)
            ),
            gate,
        )
        shift, scale, gate = torch.chunk(ff_p_a, 3, dim=-1)
        aud = residual_gate_fp32(
            aud,
            self.audioT.feed_forward(
                _norm_scale_shift(self.audioT.feed_forward_norm, aud, shift, scale)
            ),
            gate,
        )
        return vis, aud


@dataclass
class Kandinsky6TransformerOutput:
    sample: torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class Kandinsky6Transformer3DModel(BaseDiT, LayerwiseOffloadableModuleMixin):
    """Native sglang-diffusion implementation of the Kandinsky6 T2VA/IT2VA transformer."""

    _fsdp_shard_conditions = [_is_kandinsky6_transformer_block]
    _compile_conditions = [_is_kandinsky6_transformer_block]
    param_names_mapping = _ARCH_CONFIG_DEFAULTS.param_names_mapping
    reverse_param_names_mapping = _ARCH_CONFIG_DEFAULTS.reverse_param_names_mapping
    lora_param_names_mapping = _ARCH_CONFIG_DEFAULTS.lora_param_names_mapping
    # dense backends only: these attention roles do not supply sparsity metadata
    _supported_attention_backends = {
        AttentionBackendEnum.FA,
        AttentionBackendEnum.TORCH_SDPA,
    }

    def __init__(
        self,
        config: Kandinsky6VideoAudioConfig,
        hf_config: dict[str, Any],
        quant_config: QuantizationConfig | None = None,
    ) -> None:
        super().__init__(config=config, hf_config=hf_config)
        arch: Kandinsky6ArchConfig = self.config
        self.quant_config = quant_config

        if arch.attention_engine == "nabla":
            raise NotImplementedError(
                "Kandinsky6Transformer3DModel: attention_engine='nabla' (NABLA block-sparse "
                "attention) is not yet ported to sglang-diffusion. The only verified Kandinsky6 "
                "checkpoint uses attention_engine='sdpa'; set attention_engine to 'auto' or "
                "'sdpa' to use the dense attention path."
            )

        head_dim = sum(arch.axes_dims)
        head_dim_a = sum(arch.axes_dims_a)

        _validate_parallelism(
            arch.model_dim // head_dim,
            num_attention_heads_a=arch.model_dim_a // head_dim_a,
            ff_dim=arch.ff_dim,
            ff_dim_a=arch.ff_dim_a,
        )
        self.in_visual_dim = arch.in_visual_dim
        self.in_audio_dim = arch.in_audio_dim
        self.model_dim = arch.model_dim
        self.patch_size = arch.patch_size
        self.visual_cond = arch.visual_cond
        self.is_multimodal = arch.is_multimodal
        self.attention_engine = arch.attention_engine
        self.visual_token_type_num_embeddings = arch.visual_token_type_num_embeddings

        visual_embed_dim = (
            (2 * arch.in_visual_dim + 1) if arch.visual_cond else arch.in_visual_dim
        )

        self.visual_embeddings = Kandinsky6VisualEmbeddings(
            visual_embed_dim,
            arch.model_dim,
            arch.patch_size,
            prefix=add_prefix("visual_embeddings", self.prefix),
        )
        self.visual_token_type_embeddings: nn.Embedding | None = None
        if self.visual_token_type_num_embeddings > 0:
            self.visual_token_type_embeddings = nn.Embedding(
                self.visual_token_type_num_embeddings, arch.model_dim
            )
        self.visual_rope_embeddings = Kandinsky6RoPE3D(arch.axes_dims)
        self.out_layer = Kandinsky6OutLayer(
            arch.model_dim, arch.time_dim, arch.out_visual_dim, arch.patch_size
        )

        if not self.is_multimodal:
            self._build_text_tower(
                "", arch.model_dim, arch.time_dim, arch.ff_dim, head_dim
            )
            block_cls = Kandinsky6TransformerDecoderBlock
        else:
            self.audio_embeddings = Kandinsky6TextEmbeddings(
                arch.in_audio_dim, arch.model_dim_a
            )
            self.audio_rope_embeddings = Kandinsky6RoPE1D(
                head_dim_a, freqs_scaling=arch.audio_freqs_scaling
            )
            self.audio_out_layer = Kandinsky6OutLayerAudio(
                arch.model_dim_a,
                arch.time_dim_a,
                arch.out_audio_dim or arch.in_audio_dim,
            )

            self._build_text_tower(
                "video_", arch.model_dim, arch.time_dim, arch.ff_dim, head_dim
            )
            self._build_text_tower(
                "audio_", arch.model_dim_a, arch.time_dim_a, arch.ff_dim_a, head_dim_a
            )

            block_cls = partial(
                Kandinsky6FusedTransformerDecoderBlock,
                model_dim_a=arch.model_dim_a,
                time_dim_a=arch.time_dim_a,
                ff_dim_a=arch.ff_dim_a,
                head_dim_a=head_dim_a,
                ca_rope=arch.ca_rope,
                cross_gates=arch.cross_gates,
                fix_modulation=arch.fix_modulation,
            )
        self.visual_transformer_blocks = nn.ModuleList(
            block_cls(
                arch.model_dim,
                arch.time_dim,
                arch.ff_dim,
                head_dim,
                supported_attention_backends=self._supported_attention_backends,
                prefix=add_prefix(f"visual_transformer_blocks.{i}", self.prefix),
                quant_config=quant_config,
            )
            for i in range(arch.num_visual_blocks)
        )

        self.gradient_checkpointing = False
        self.hidden_size = arch.hidden_size
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.num_channels_latents
        self.layer_names = (
            ["video_text_transformer_blocks", "audio_text_transformer_blocks"]
            if self.is_multimodal
            else ["text_transformer_blocks"]
        ) + ["visual_transformer_blocks"]
        self.__post_init__()

    def _build_text_tower(self, prefix, model_dim, time_dim, ff_dim, head_dim):
        arch = self.config
        self.add_module(
            f"{prefix}time_embeddings", Kandinsky6TimeEmbeddings(model_dim, time_dim)
        )
        self.add_module(
            f"{prefix}text_embeddings",
            Kandinsky6TextEmbeddings(arch.in_text_dim, model_dim),
        )
        self.add_module(
            f"{prefix}pooled_text_embeddings",
            Kandinsky6TextEmbeddings(arch.in_text_dim2, time_dim),
        )
        self.add_module(f"{prefix}text_rope_embeddings", Kandinsky6RoPE1D(head_dim))
        self.add_module(
            f"{prefix}text_transformer_blocks",
            nn.ModuleList(
                Kandinsky6TransformerEncoderBlock(
                    model_dim,
                    time_dim,
                    ff_dim,
                    head_dim,
                    self._supported_attention_backends,
                    prefix=add_prefix(
                        f"{prefix}text_transformer_blocks.{i}", self.prefix
                    ),
                    quant_config=self.quant_config,
                )
                for i in range(arch.num_text_blocks)
            ),
        )

    def _encode_text(
        self,
        prefix: str | None,
        text_embed: torch.Tensor,
        pooled: torch.Tensor,
        time: torch.Tensor,
        text_rope_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = f"{prefix}_" if prefix else ""
        te = self.get_submodule(f"{prefix}text_embeddings")(text_embed)
        tm = self.get_submodule(f"{prefix}time_embeddings")(time)
        tm = tm + self.get_submodule(f"{prefix}pooled_text_embeddings")(pooled)
        rope_embeddings = self.get_submodule(f"{prefix}text_rope_embeddings")
        blocks = self.get_submodule(f"{prefix}text_transformer_blocks")
        # each text tower has its own head dimension and RoPE table
        text_rope = rope_embeddings(text_rope_pos).unsqueeze(dim=0)
        for block in blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                block = partial(checkpoint, block, use_reentrant=False)
            te = block(te, tm, text_rope)
        return te, tm

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        pooled_projections: torch.Tensor | None = None,
        hidden_states_audio: torch.Tensor | None = None,
        audio_timestep: torch.Tensor | None = None,
        visual_rope_pos: (
            tuple[torch.Tensor, torch.Tensor, torch.Tensor] | list[torch.Tensor] | None
        ) = None,
        audio_rope_pos: torch.Tensor | None = None,
        text_rope_pos: torch.Tensor | None = None,
        scale_factor: tuple[float, float, float] = (1.0, 1.0, 1.0),
        sparse_params: dict[str, Any] | None = None,
        visual_token_type_ids: torch.Tensor | None = None,
        va_gate_scale: float = 1.0,
        av_gate_scale: float = 1.0,
        return_dict: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | Kandinsky6TransformerOutput:
        if pooled_projections is None:
            if encoder_hidden_states_image is None:
                raise ValueError("pooled_projections must be provided for Kandinsky6.")
            pooled_projections = encoder_hidden_states_image
        if visual_rope_pos is None or text_rope_pos is None:
            raise ValueError(
                "visual_rope_pos and text_rope_pos are required for Kandinsky6."
            )
        if (
            visual_token_type_ids is not None
            and self.visual_token_type_num_embeddings == 0
        ):
            raise ValueError(
                "visual_token_type_ids requires visual_token_type_num_embeddings > 0."
            )

        if sparse_params is not None:
            raise NotImplementedError(
                "Kandinsky6 does not support NABLA sparse attention."
            )
        both = self.is_multimodal
        if both and (hidden_states is None or hidden_states_audio is None):
            raise NotImplementedError(
                "Multimodal Kandinsky6 checkpoints require both video and audio latents."
            )

        video_te, video_tm = self._encode_text(
            "video" if both else None,
            encoder_hidden_states,
            pooled_projections,
            timestep,
            text_rope_pos,
        )
        if both:
            audio_te, audio_tm = self._encode_text(
                "audio",
                encoder_hidden_states,
                pooled_projections,
                audio_timestep if audio_timestep is not None else timestep,
                text_rope_pos,
            )
            if audio_rope_pos is None:
                audio_rope_pos = torch.arange(
                    hidden_states_audio.shape[1], device=hidden_states_audio.device
                )

        visual_embed = self.visual_embeddings(hidden_states)
        if (
            self.visual_token_type_embeddings is not None
            and visual_token_type_ids is not None
        ):
            type_embed = self.visual_token_type_embeddings(visual_token_type_ids)
            visual_embed = visual_embed + type_embed[:, :, None, None, :]
        visual_shape = visual_embed.shape[:-1]
        visual_rope = self.visual_rope_embeddings(
            visual_shape, visual_rope_pos, scale_factor
        ).flatten(1, 3)
        visual_embed, shard = shard_seq(visual_embed.flatten(1, 3))
        visual_rope = shard_like(visual_rope, shard, pad_mode="repeat_last")
        attn_meta = tail_attn_meta(shard, visual_embed.shape[0], visual_embed.device)

        if both:
            audio_embed = self.audio_embeddings(hidden_states_audio)
            audio_rope = self.audio_rope_embeddings(audio_rope_pos).unsqueeze(dim=0)
            for block in self.visual_transformer_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    block = partial(checkpoint, block, use_reentrant=False)
                visual_embed, audio_embed = block(
                    visual_embed,
                    audio_embed,
                    video_te,
                    audio_te,
                    (video_tm, audio_tm),
                    visual_rope,
                    audio_rope,
                    va_gate_scale,
                    av_gate_scale,
                    video_seq_len=shard.orig_len,
                    video_attn_meta=attn_meta,
                )
        else:
            for block in self.visual_transformer_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    block = partial(checkpoint, block, use_reentrant=False)
                visual_embed = block(
                    visual_embed, video_te, video_tm, visual_rope, attn_meta
                )

        visual_embed = gather_seq(visual_embed, shard.orig_len).reshape(
            *visual_shape, -1
        )
        video_out = self.out_layer(visual_embed, video_tm)
        result = (
            (video_out, self.audio_out_layer(audio_embed, audio_tm))
            if both
            else video_out
        )

        if return_dict:
            return Kandinsky6TransformerOutput(sample=result)
        return result

    def post_load_weights(self) -> None:
        """Materialize non-checkpoint RoPE/time frequencies left on meta after loading."""
        device = next(self.parameters()).device

        for i, (axes_dim, ax_max_pos) in enumerate(
            zip(
                self.visual_rope_embeddings.axes_dims,
                self.visual_rope_embeddings.max_pos,
                strict=True,
            )
        ):
            name = f"args_{i}"
            buf = self.visual_rope_embeddings._buffers.get(name)
            if isinstance(buf, torch.Tensor) and buf.is_meta:
                freq = _build_rotary_freqs(
                    axes_dim // 2, self.visual_rope_embeddings.max_period
                ).to(device=device)
                pos = torch.arange(ax_max_pos, dtype=freq.dtype, device=device)
                self.visual_rope_embeddings._buffers[name] = torch.outer(pos, freq)

        for module in self.modules():
            if isinstance(module, Kandinsky6RoPE1D) and module.args.is_meta:
                freq = (
                    _build_rotary_freqs(module.dim // 2, module.max_period).to(
                        device=device
                    )
                    * module.freqs_scaling
                )
                pos = torch.arange(module.max_pos, dtype=freq.dtype, device=device)
                module._buffers["args"] = torch.outer(pos, freq)
            elif isinstance(module, Kandinsky6TimeEmbeddings) and module.freqs.is_meta:
                module.freqs = _build_rotary_freqs(
                    module.model_dim // 2, module.max_period
                ).to(device=device)


EntryClass = Kandinsky6Transformer3DModel
