# SPDX-License-Identifier: Apache-2.0
"""Wan2.2-5B causal transformer with MinWM discrete action conditioning."""

from __future__ import annotations

import inspect
import importlib.util
import logging
import os
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from sglang.multimodal_gen.configs.models.dits.minwm import MinWMVideoConfig
from sglang.multimodal_gen.runtime.distributed import (
    get_sp_group,
    get_sp_parallel_rank,
    get_sp_world_size,
    get_tp_world_size,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.distributed.sp_shard_utils import (
    compute_sequence_splits,
    gather_sequence_varlen,
    sequence_splits_are_uniform,
    shard_sequence_varlen,
)
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.layernorm import tensor_parallel_rms_norm
from sglang.multimodal_gen.runtime.layers.rotary_embedding import NDRotaryEmbedding
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_input_all_to_all_varlen,
    _usp_output_all_to_all,
    _usp_output_all_to_all_varlen,
)
from sglang.multimodal_gen.runtime.layers.visual_embedding import (
    PatchEmbed,
    TimestepEmbedder,
)
from sglang.multimodal_gen.runtime.managers.forward_context import get_forward_context
from sglang.multimodal_gen.runtime.models.dits.causal_wanvideo import (
    CausalWanSelfAttention,
    CausalWanTransformerBlock,
    CausalWanTransformer3DModel,
)
from sglang.multimodal_gen.runtime.models.dits.minwm_action import (
    PrimitiveTokenResidualActionEncoder,
)
from sglang.multimodal_gen.runtime.models.dits.wanvideo import WanT2VCrossAttention
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


_MINWM_ATTENTION_IMPL = os.environ.get("MINWM_ATTENTION_IMPL", "packed").strip().lower()
_MINWM_PACKED_ATTENTION_DETERMINISTIC = _env_flag(
    "MINWM_PACKED_ATTENTION_DETERMINISTIC", True
)
_MINWM_SEGMENT_COMPILE = _env_flag("MINWM_SEGMENT_COMPILE", True)
_MINWM_ANNOUNCED_ATTENTION_BACKENDS: set[tuple[str, str]] = set()


class _MinWMTimestepEmbedder(TimestepEmbedder):
    """Use minWM's float64 sinusoid construction before the checkpoint MLP."""

    def forward(
        self, timestep: torch.Tensor, timestep_seq_len: int | None = None
    ) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        position = timestep.to(torch.float64)
        sinusoid = torch.outer(
            position,
            torch.pow(
                10000,
                -torch.arange(half, device=timestep.device, dtype=torch.float64) / half,
            ),
        )
        embedding = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
        embedding = embedding.to(self.mlp.fc_in.weight.dtype)
        if timestep_seq_len is not None:
            embedding = embedding.unflatten(
                0, (embedding.shape[0] // timestep_seq_len, timestep_seq_len)
            )
        return self.mlp(embedding)


class MinWMPatchEmbed(PatchEmbed):
    """Use minWM main's native Conv3d instead of the linearized fast path."""

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        if self.flatten:
            hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return self.norm(hidden_states)


class MinWMRMSNorm(nn.Module):
    """Match minWM's BF16 rounding boundary before the learned weight."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return _MinWMSegmentCompile.get(MinWMRMSNorm._norm, hidden_states.is_cuda)(
            hidden_states, self.weight, self.eps
        )

    @staticmethod
    def _norm(
        hidden_states: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> torch.Tensor:
        hidden_states_float = hidden_states.float()
        normalized = hidden_states_float * torch.rsqrt(
            hidden_states_float.pow(2).mean(dim=-1, keepdim=True) + eps
        )
        return normalized.type_as(hidden_states) * weight


class _MinWMSegmentCompile:
    """Mirror minWM main's shared, dynamic segment-compile cache."""

    _compiled = {}

    @classmethod
    def get(cls, function, use_compile: bool):
        if not use_compile or not _MINWM_SEGMENT_COMPILE:
            return function
        if function not in cls._compiled:
            kwargs = {}
            if "recompile_limit" in inspect.signature(torch.compile).parameters:
                kwargs["recompile_limit"] = 64
            if torch.are_deterministic_algorithms_enabled():
                kwargs["options"] = {"deterministic": True}
            cls._compiled[function] = torch.compile(
                function, dynamic=True, mode=None, **kwargs
            )
        return cls._compiled[function]


def apply_minwm_rotary_embedding(
    hidden_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply minWM's explicit FP32 interleaved RoPE arithmetic."""
    sequence_length = hidden_states.shape[-3]
    half_head_dim = hidden_states.shape[-1] // 2
    broadcast_shape = (1,) * (hidden_states.dim() - 3) + (
        sequence_length,
        1,
        half_head_dim,
    )
    cos = cos.reshape(broadcast_shape)
    sin = sin.reshape(broadcast_shape)
    real = hidden_states[..., 0::2].float()
    imaginary = hidden_states[..., 1::2].float()
    return (
        torch.stack(
            (real * cos - imaginary * sin, real * sin + imaginary * cos),
            dim=-1,
        )
        .flatten(-2)
        .type_as(hidden_states)
    )


def _minwm_layer_norm(
    hidden_states: torch.Tensor,
    *,
    eps: float,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match ``WanLayerNorm._norm`` from minWM main."""
    return _MinWMSegmentCompile.get(_minwm_layer_norm_op, hidden_states.is_cuda)(
        hidden_states, weight, bias, eps
    )


def _minwm_layer_norm_op(
    hidden_states: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    return F.layer_norm(
        hidden_states.float(),
        (hidden_states.shape[-1],),
        weight.float() if weight is not None else None,
        bias.float() if bias is not None else None,
        eps,
    ).type_as(hidden_states)


def _minwm_adaln_op(
    x: torch.Tensor,
    m_shift: torch.Tensor | None = None,
    m_scale: torch.Tensor | None = None,
    e_shift: torch.Tensor | None = None,
    e_scale: torch.Tensor | None = None,
    eps: float = 1e-6,
    y: torch.Tensor | None = None,
    m_gate: torch.Tensor | None = None,
    e_gate: torch.Tensor | None = None,
    r: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    cast_norm: bool = False,
):
    """Source-shaped copy of minWM main's compiled ``adaln_op``."""
    if y is not None:
        x = (x.float() + y.float() * (m_gate.float() + e_gate.float())).type_as(x)
    if r is not None:
        x = x + r
    if m_shift is None:
        return x
    h = torch.nn.functional.layer_norm(
        x.float(),
        (x.shape[-1],),
        weight.float() if weight is not None else None,
        bias.float() if bias is not None else None,
        eps,
    )
    shift, scale = m_shift + e_shift, m_scale + e_scale
    if cast_norm:
        h = h.type_as(x)
        return x, h * (1 + scale) + shift
    return x, (h * (1 + scale.float()) + shift.float()).type_as(x)


def _minwm_adaln(hidden_states: torch.Tensor, *args, **kwargs):
    return _MinWMSegmentCompile.get(_minwm_adaln_op, hidden_states.is_cuda)(
        hidden_states, *args, **kwargs
    )


def _minwm_frame_indices(hidden_states: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Map each local token to its frame, including shards cut inside a frame."""
    forward_batch = get_forward_context().forward_batch
    if (
        forward_batch is not None
        and getattr(forward_batch, "enable_sequence_shard", False)
        and get_ulysses_parallel_world_size() > 1
    ):
        frame_indices = getattr(forward_batch, "sequence_shard_frame_indices", None)
        if frame_indices is None:
            raise ValueError(
                "MinWM sequence sharding requires "
                "forward_batch.sequence_shard_frame_indices."
            )
        if frame_indices.numel() != hidden_states.shape[1]:
            raise ValueError(
                "MinWM sequence shard frame indices do not match the local "
                f"sequence length: {frame_indices.numel()} vs "
                f"{hidden_states.shape[1]}."
            )
        return frame_indices

    if hidden_states.shape[1] % num_frames != 0:
        raise ValueError(
            f"MinWM sequence length {hidden_states.shape[1]} must be divisible "
            f"by num_frames {num_frames} when sequence sharding is disabled."
        )
    tokens_per_frame = hidden_states.shape[1] // num_frames
    return torch.arange(num_frames, device=hidden_states.device).repeat_interleave(
        tokens_per_frame
    )


def _minwm_qk_norm_rope_op(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    eps: float,
    rope: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = MinWMRMSNorm._norm(query, query_weight, eps)
    key = MinWMRMSNorm._norm(key, key_weight, eps)
    *leading, dim = query.shape
    query = query.reshape(*leading, num_heads, dim // num_heads)
    key = key.reshape(*leading, num_heads, dim // num_heads)

    def apply(hidden_states: torch.Tensor) -> torch.Tensor:
        sequence_length = hidden_states.shape[-3]
        head_dim = hidden_states.shape[-1]
        shaped_rope = rope.reshape(
            *((1,) * (hidden_states.dim() - 3)),
            sequence_length,
            1,
            head_dim // 2,
            2,
        )
        cos, sin = shaped_rope[..., 0], shaped_rope[..., 1]
        real = hidden_states[..., 0::2].float()
        imaginary = hidden_states[..., 1::2].float()
        return (
            torch.stack(
                (real * cos - imaginary * sin, real * sin + imaginary * cos),
                dim=-1,
            )
            .flatten(-2)
            .type_as(hidden_states)
        )

    return apply(query), apply(key)


def _minwm_packed_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Call the same device-selected packed-varlen backend as minWM main."""
    if query.device.type != "cuda":
        raise RuntimeError("MinWM packed-varlen attention requires CUDA")
    batch_size, query_length, num_heads, head_dim = query.shape
    key_length = key.shape[1]
    if key.shape != value.shape or key.shape[0] != batch_size:
        raise ValueError("MinWM attention key/value shapes must match")
    if key.shape[2:] != (num_heads, head_dim):
        raise ValueError("MinWM attention Q/K/V head geometry must match")

    query_lengths = torch.full(
        (batch_size,), query_length, dtype=torch.int32, device=query.device
    )
    key_lengths = torch.full(
        (batch_size,), key_length, dtype=torch.int32, device=key.device
    )
    cu_query = F.pad(query_lengths.cumsum(0), (1, 0)).to(torch.int32)
    cu_key = F.pad(key_lengths.cumsum(0), (1, 0)).to(torch.int32)
    backend = _minwm_packed_attention_backend(query.device)
    announce_key = (backend, str(query.device))
    if announce_key not in _MINWM_ANNOUNCED_ATTENTION_BACKENDS:
        logger.info(
            "MinWM packed-varlen attention backend=%s device=%s", backend, query.device
        )
        _MINWM_ANNOUNCED_ATTENTION_BACKENDS.add(announce_key)

    common_kwargs = {
        "q": query.reshape(batch_size * query_length, num_heads, head_dim),
        "k": key.reshape(batch_size * key_length, num_heads, head_dim),
        "v": value.reshape(batch_size * key_length, num_heads, head_dim),
        "cu_seqlens_q": cu_query,
        "cu_seqlens_k": cu_key,
        "max_seqlen_q": query_length,
        "max_seqlen_k": key_length,
        "softmax_scale": None,
        "causal": False,
        "deterministic": _MINWM_PACKED_ATTENTION_DETERMINISTIC,
    }
    if backend == "fa4":
        from flash_attn.cute import flash_attn_varlen_func

        output = flash_attn_varlen_func(
            **common_kwargs,
            window_size=(None, None),
            return_lse=False,
        )
    elif backend == "fa3":
        import flash_attn_interface

        output = flash_attn_interface.flash_attn_varlen_func(**common_kwargs)
    else:
        import flash_attn

        output = flash_attn.flash_attn_varlen_func(
            **common_kwargs,
            dropout_p=0.0,
            window_size=(-1, -1),
        )
    if isinstance(output, tuple):
        output = output[0]
    return output.reshape(batch_size, query_length, num_heads, head_dim)


def _minwm_packed_attention_backend(device: torch.device) -> str:
    """Mirror minWM main's FA4/FA3/FA2 device and availability fallback."""
    capability = torch.cuda.get_device_capability(device)[0]
    if capability >= 10 and importlib.util.find_spec("flash_attn.cute") is not None:
        return "fa4"
    if capability == 9 and importlib.util.find_spec("flash_attn_interface") is not None:
        return "fa3"
    if importlib.util.find_spec("flash_attn") is not None:
        return "fa2"
    raise RuntimeError("No minWM-compatible packed FlashAttention backend is available")


class MinWMCausalSelfAttention(CausalWanSelfAttention):
    """SGLang cache ownership with minWM main's FA4 call shape."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ulysses_world_size = max(get_ulysses_parallel_world_size(), 1)
        if self.num_heads % ulysses_world_size != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"ulysses_degree ({ulysses_world_size})."
            )
        self.ulysses_num_heads = self.num_heads // ulysses_world_size
        self.ulysses_attn = (
            self.attn
            if ulysses_world_size == 1
            else LocalAttention(
                num_heads=self.ulysses_num_heads,
                head_size=self.head_dim,
                dropout_rate=0,
                softmax_scale=None,
                causal=False,
                supported_attention_backends=(
                    AttentionBackendEnum.FA,
                    AttentionBackendEnum.AITER,
                    AttentionBackendEnum.TORCH_SDPA,
                ),
            )
        )

    def forward(
        self,
        query,
        key,
        value,
        freqs_cis,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        qk_already_roped=False,
    ):
        if kv_cache is None:
            return super().forward(
                query,
                key,
                value,
                freqs_cis,
                block_mask,
                kv_cache,
                current_start,
                cache_start,
                qk_already_roped=qk_already_roped,
            )
        if qk_already_roped:
            roped_query, roped_key = query.type_as(value), key.type_as(value)
        else:
            cos, sin = freqs_cis
            roped_query = apply_minwm_rotary_embedding(query, cos, sin).type_as(value)
            roped_key = apply_minwm_rotary_embedding(key, cos, sin).type_as(value)

        forward_batch = get_forward_context().forward_batch
        sequence_shard_enabled = (
            forward_batch is not None
            and getattr(forward_batch, "enable_sequence_shard", False)
            and get_ulysses_parallel_world_size() > 1
        )
        seq_splits = None
        uniform_seq_splits = False
        if sequence_shard_enabled:
            seq_splits = getattr(forward_batch, "sequence_shard_splits", None)
            if seq_splits is None:
                raise ValueError(
                    "MinWM causal sequence sharding requires "
                    "forward_batch.sequence_shard_splits."
                )
            seq_splits = list(seq_splits)
            uniform_seq_splits = sequence_splits_are_uniform(seq_splits)
            qkv = torch.cat([roped_query, roped_key, value], dim=-1)
            qkv = (
                _usp_input_all_to_all(qkv, head_dim=2)
                if uniform_seq_splits
                else _usp_input_all_to_all_varlen(qkv, seq_splits, head_dim=2)
            )
            roped_query, roped_key, value = qkv.chunk(3, dim=-1)

        if not sequence_shard_enabled and kv_cache.can_direct_current_attention(
            roped_key.shape[1]
        ):
            attention_key, attention_value = roped_key, value
        else:
            cache_view = kv_cache.update_and_get_attention_kv(
                key=roped_key,
                value=value,
                current_chunk_start=current_start,
                cache_head_start=0 if sequence_shard_enabled else self.head_start,
                debug_name="MinWM causal KV cache",
            )
            attention_key, attention_value = cache_view.k, cache_view.v
        if _MINWM_ATTENTION_IMPL == "dense":
            output = (self.ulysses_attn if sequence_shard_enabled else self.attn)(
                roped_query, attention_key, attention_value
            )
        else:
            output = _minwm_packed_varlen_attention(
                roped_query, attention_key, attention_value
            )
        if sequence_shard_enabled:
            assert seq_splits is not None
            output = (
                _usp_output_all_to_all(output, head_dim=2)
                if uniform_seq_splits
                else _usp_output_all_to_all_varlen(output, seq_splits, head_dim=2)
            )
        return output


class MinWMPackedCrossAttention(WanT2VCrossAttention):
    """Full-512 text attention with minWM main's packed-varlen FA4 call."""

    def forward(self, x, context, context_lens, crossattn_cache=None):
        del context_lens
        query, _ = self.to_q(x)
        if self.tp_rmsnorm:
            query = tensor_parallel_rms_norm(query, self.norm_q)
        else:
            query = self.norm_q(query)
        query = query.unflatten(2, (self.local_num_heads, self.head_dim))

        if crossattn_cache is not None and crossattn_cache.is_init:
            key, value = crossattn_cache.k, crossattn_cache.v
        else:
            key, _ = self.to_k(context)
            if self.tp_rmsnorm:
                key = tensor_parallel_rms_norm(key, self.norm_k)
            else:
                key = self.norm_k(key)
            key = key.unflatten(2, (self.local_num_heads, self.head_dim))
            value, _ = self.to_v(context)
            value = value.unflatten(2, (self.local_num_heads, self.head_dim))
            if crossattn_cache is not None:
                crossattn_cache.store(key, value)

        if _MINWM_ATTENTION_IMPL == "dense":
            output = self.attn(query, key, value).flatten(2)
        else:
            output = _minwm_packed_varlen_attention(query, key, value).flatten(2)
        output, _ = self.to_out(output)
        return output


def _frame_modulation(
    hidden_states: torch.Tensor,
    model_value: torch.Tensor,
    timestep_value: torch.Tensor,
    *,
    num_frames: int,
) -> torch.Tensor:
    """Add modulation in BF16, then broadcast it over each frame's tokens."""
    value = model_value.to(hidden_states.dtype) + timestep_value.to(hidden_states.dtype)
    return (
        value.unsqueeze(2)
        .expand(-1, -1, hidden_states.shape[1] // num_frames, -1)
        .flatten(1, 2)
    )


def _frame_gate(
    hidden_states: torch.Tensor,
    model_value: torch.Tensor,
    timestep_value: torch.Tensor,
    *,
    num_frames: int,
) -> torch.Tensor:
    """Promote each BF16 gate operand before adding, as minWM main does."""
    value = (
        model_value.to(hidden_states.dtype).float()
        + timestep_value.to(hidden_states.dtype).float()
    )
    return (
        value.unsqueeze(2)
        .expand(-1, -1, hidden_states.shape[1] // num_frames, -1)
        .flatten(1, 2)
    )


def _minwm_adaln_modulation(
    hidden_states: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Match minWM's non-Triton ``adaln_op`` normalization and final cast."""
    normalized = F.layer_norm(
        hidden_states.float(), (hidden_states.shape[-1],), eps=eps
    )
    return (normalized * (1 + scale.float()) + shift.float()).type_as(hidden_states)


class MinWMCausalTransformerBlock(CausalWanTransformerBlock):
    """Causal Wan block with minWM main's exact eager rounding order."""

    self_attention_cls = MinWMCausalSelfAttention
    cross_attention_cls = MinWMPackedCrossAttention

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start: int = 0,
        cache_start: int | None = None,
    ) -> torch.Tensor:
        if hidden_states.dim() == 4:
            hidden_states = hidden_states.squeeze(1)
        num_frames = temb.shape[1]
        orig_dtype = hidden_states.dtype
        modulation = self.scale_shift_table.to(orig_dtype)
        frame_index = _minwm_frame_indices(hidden_states, num_frames)
        # minWM main first expands the full [B, F, 6, D] tensor with advanced
        # indexing, then selects a modulation slice. Besides equal values, this
        # preserves its non-contiguous 6*D token stride for the compiled AdaLN.
        timestep_modulation = temb[:, frame_index]

        _, norm_hidden_states = _minwm_adaln(
            hidden_states,
            modulation[:, 0],
            modulation[:, 1],
            timestep_modulation.select(-2, 0),
            timestep_modulation.select(-2, 1),
            self.norm1.eps,
        )

        query, _ = self.to_q(norm_hidden_states)
        key, _ = self.to_k(norm_hidden_states)
        value, _ = self.to_v(norm_hidden_states)
        if self.tp_rmsnorm:
            query = tensor_parallel_rms_norm(query, self.norm_q)
            key = tensor_parallel_rms_norm(key, self.norm_k)
            query = query.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))
            key = key.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))
            qk_already_roped = False
        else:
            rope = torch.stack(freqs_cis, dim=-1)
            query, key = _MinWMSegmentCompile.get(
                _minwm_qk_norm_rope_op, query.is_cuda
            )(
                query.squeeze(1),
                key.squeeze(1),
                self.norm_q.weight,
                self.norm_k.weight,
                self.norm_q.eps,
                rope,
                self.local_num_heads,
            )
            qk_already_roped = True
        value = value.squeeze(1).unflatten(2, (self.local_num_heads, self.dim_head))
        attn_output = self.attn1(
            query,
            key,
            value,
            freqs_cis,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
            qk_already_roped=qk_already_roped,
        ).flatten(2)
        attn_output, _ = self.to_out(attn_output)
        attn_output = attn_output.squeeze(1)

        hidden_states = _minwm_adaln(
            hidden_states,
            y=attn_output,
            m_gate=modulation[:, 2],
            e_gate=timestep_modulation.select(-2, 2),
        )

        affine_norm = self.self_attn_residual_norm.norm
        norm_hidden_states = _minwm_layer_norm(
            hidden_states,
            eps=affine_norm.eps,
            weight=affine_norm.weight,
            bias=affine_norm.bias,
        )
        cross_output = self.attn2(
            norm_hidden_states,
            context=encoder_hidden_states,
            context_lens=None,
            crossattn_cache=crossattn_cache,
        )

        hidden_states, norm_hidden_states = _minwm_adaln(
            hidden_states,
            modulation[:, 3],
            modulation[:, 4],
            timestep_modulation.select(-2, 3),
            timestep_modulation.select(-2, 4),
            self.cross_attn_residual_norm.norm.eps,
            r=cross_output,
        )

        ff_output = self.ffn(norm_hidden_states)
        return _minwm_adaln(
            hidden_states,
            y=ff_output,
            m_gate=modulation[:, 5],
            e_gate=timestep_modulation.select(-2, 5),
        )


class MinWMCausalTransformer3DModel(CausalWanTransformer3DModel):
    transformer_block_cls = MinWMCausalTransformerBlock
    patch_embedding_cls = MinWMPatchEmbed
    _fsdp_shard_conditions = MinWMVideoConfig()._fsdp_shard_conditions
    _compile_conditions = MinWMVideoConfig()._compile_conditions
    _supported_attention_backends = MinWMVideoConfig()._supported_attention_backends
    param_names_mapping = MinWMVideoConfig().param_names_mapping
    reverse_param_names_mapping = MinWMVideoConfig().reverse_param_names_mapping
    lora_param_names_mapping = MinWMVideoConfig().lora_param_names_mapping

    def __init__(self, config, hf_config, quant_config=None) -> None:
        if _MINWM_ATTENTION_IMPL not in {"packed", "dense"}:
            raise ValueError(
                "MINWM_ATTENTION_IMPL must be 'packed' or 'dense', got "
                f"{_MINWM_ATTENTION_IMPL!r}"
            )
        logger.info(
            "MinWM execution profile: attention_impl=%s "
            "packed_deterministic=%s segment_compile=%s",
            _MINWM_ATTENTION_IMPL,
            _MINWM_PACKED_ATTENTION_DETERMINISTIC,
            _MINWM_SEGMENT_COMPILE,
        )
        deterministic = os.environ.get("MINWM_PARITY_DETERMINISTIC", "0")
        if deterministic.strip().lower() not in {"", "0", "false", "no", "off"}:
            torch.use_deterministic_algorithms(True)
        super().__init__(config, hf_config, quant_config)
        self.sp_size = get_sp_world_size()
        d = self.hidden_size // self.num_attention_heads
        self._sequence_shard_rotary_emb = NDRotaryEmbedding(
            rope_dim_list=[d - 4 * (d // 6), 2 * (d // 6), 2 * (d // 6)],
            rope_theta=10000,
            dtype=(
                torch.float64
                if current_platform.is_float64_supported()
                else torch.float32
            ),
        )
        old_time = self.condition_embedder.time_embedder
        exact_time = _MinWMTimestepEmbedder(
            self.hidden_size,
            act_layer="silu",
            frequency_embedding_size=config.freq_dim,
        )
        exact_time.mlp = old_time.mlp
        self.condition_embedder.time_embedder = exact_time
        for block in self.blocks:
            block.attn1.rotary_embedding_override = apply_minwm_rotary_embedding
            block.norm_q = MinWMRMSNorm(config.hidden_size, eps=config.eps)
            block.norm_k = MinWMRMSNorm(config.hidden_size, eps=config.eps)
            block.attn2.norm_q = MinWMRMSNorm(config.hidden_size, eps=config.eps)
            block.attn2.norm_k = MinWMRMSNorm(config.hidden_size, eps=config.eps)
        self.action_in = PrimitiveTokenResidualActionEncoder(
            self.hidden_size,
            embed_dim=config.action_embed_dim,
            hidden_dim=config.action_hidden_dim,
            kernel_size=config.action_kernel_size,
        )
        self.action_history_frames = config.action_history_frames
        expected_history = 2 * (config.action_kernel_size - 1)
        if self.action_history_frames != expected_history:
            raise ValueError(
                "MinWM action_history_frames must equal "
                f"2 * (action_kernel_size - 1) = {expected_history}"
            )
        self._install_parity_debug_hooks()

    @lru_cache(maxsize=16)
    def _compute_sequence_shard_rope(
        self,
        local_seq_len: int,
        token_start: int,
        frame_stride: int,
        width: int,
        start_frame: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_indices = torch.arange(
            token_start,
            token_start + local_seq_len,
            device=device,
            dtype=torch.long,
        )
        t_idx = token_indices // frame_stride + start_frame
        remainder = token_indices % frame_stride
        h_idx = remainder // width
        w_idx = remainder % width
        positions = torch.stack((t_idx, h_idx, w_idx), dim=1)
        return self._sequence_shard_rotary_emb.forward_uncached(positions)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | list[torch.Tensor],
        timestep: torch.LongTensor,
        encoder_hidden_states_image: torch.Tensor | list[torch.Tensor] | None = None,
        kv_cache=None,
        crossattn_cache=None,
        current_start: int = 0,
        cache_start: int = 0,
        start_frame: int = 0,
        action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        forward_batch = get_forward_context().forward_batch
        sequence_shard_enabled = (
            forward_batch is not None
            and getattr(forward_batch, "enable_sequence_shard", False)
            and self.sp_size > 1
        )
        if not sequence_shard_enabled:
            return super().forward(
                hidden_states,
                encoder_hidden_states,
                timestep,
                encoder_hidden_states_image=encoder_hidden_states_image,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                start_frame=start_frame,
                action=action,
            )

        ulysses_world_size = get_ulysses_parallel_world_size()
        if get_ring_parallel_world_size() > 1:
            raise NotImplementedError(
                "MinWM causal sequence sharding supports Ulysses with "
                "ring_degree = 1 only."
            )
        if ulysses_world_size <= 1 or ulysses_world_size != self.sp_size:
            raise ValueError(
                "MinWM causal sequence sharding requires "
                "sp_degree == ulysses_degree > 1."
            )
        if get_tp_world_size() > 1:
            raise NotImplementedError(
                "MinWM causal sequence sharding cannot be combined with tensor "
                "parallelism yet."
            )
        if kv_cache is None or crossattn_cache is None:
            raise ValueError(
                "MinWM causal sequence sharding requires self- and "
                "cross-attention KV caches."
            )

        orig_dtype = hidden_states.dtype
        if not isinstance(encoder_hidden_states, torch.Tensor):
            encoder_hidden_states = encoder_hidden_states[0]
        if (
            isinstance(encoder_hidden_states_image, list)
            and len(encoder_hidden_states_image) > 0
        ):
            encoder_hidden_states_image = encoder_hidden_states_image[0]
        else:
            encoder_hidden_states_image = None

        batch_size, _, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w
        frame_stride = post_patch_height * post_patch_width
        total_seq_len = post_patch_num_frames * frame_stride
        seq_splits = compute_sequence_splits(total_seq_len, self.sp_size)
        sp_rank = get_sp_parallel_rank()
        local_start = sum(seq_splits[:sp_rank])
        local_end = local_start + seq_splits[sp_rank]
        forward_batch.sequence_shard_splits = tuple(seq_splits)
        forward_batch.sequence_shard_frame_indices = (
            torch.arange(
                local_start,
                local_end,
                device=hidden_states.device,
                dtype=torch.long,
            )
            // frame_stride
        )

        freqs_cos, freqs_sin = self._compute_sequence_shard_rope(
            seq_splits[sp_rank],
            local_start,
            frame_stride,
            post_patch_width,
            start_frame=start_frame,
            device=hidden_states.device,
        )
        freqs_cis = (
            freqs_cos.float(),
            freqs_sin.float(),
        )

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self._apply_patch_token_condition(
            hidden_states,
            action=action,
            num_frames=post_patch_num_frames,
            height=post_patch_height,
            width=post_patch_width,
        )
        hidden_states = shard_sequence_varlen(hidden_states, seq_splits, sp_rank)

        (
            temb,
            timestep_proj,
            encoder_hidden_states,
            encoder_hidden_states_image,
        ) = self.condition_embedder(
            timestep.flatten(), encoder_hidden_states, encoder_hidden_states_image
        )
        timestep_proj = timestep_proj.unflatten(1, (6, self.hidden_size)).unflatten(
            dim=0, sizes=timestep.shape
        )
        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )
        if current_platform.is_mps():
            encoder_hidden_states = encoder_hidden_states.to(orig_dtype)
        if encoder_hidden_states.dtype != orig_dtype:
            raise ValueError(
                "MinWM encoder hidden-state dtype must match the latent dtype."
            )

        for block_index, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                freqs_cis,
                block_mask=self.block_mask,
                kv_cache=kv_cache[block_index],
                crossattn_cache=crossattn_cache[block_index],
                current_start=current_start,
                cache_start=cache_start,
            )

        hidden_states = self._apply_output_head(hidden_states, temb, timestep)
        hidden_states = gather_sequence_varlen(
            hidden_states,
            seq_splits,
            get_sp_group().device_group,
        )
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    def _install_parity_debug_hooks(self) -> None:
        dump_root = os.environ.get("MINWM_PARITY_DUMP_DIR")
        if not dump_root:
            return
        dump_dir = Path(dump_root) / "sglang"
        dump_dir.mkdir(parents=True, exist_ok=True)
        counters = {"patch": 0, "block0": 0}

        def dump(name: str, output) -> None:
            index = counters[name]
            if index < 2:
                if isinstance(output, tuple):
                    output = output[0]
                torch.save(
                    output.detach().cpu(),
                    dump_dir / f"{name}_output_{index:03d}.pt",
                )
            counters[name] = index + 1

        self.patch_embedding.register_forward_hook(
            lambda _module, _args, output: dump("patch", output)
        )

        def dump_first_block(_module, hook_args, output) -> None:
            index = counters["block0"]
            if index < 2:
                torch.save(
                    hook_args[0].detach().cpu(),
                    dump_dir / f"block0_input_{index:03d}.pt",
                )
            dump("block0", output)

        self.blocks[0].register_forward_hook(dump_first_block)

        def register_detail(name: str, module: nn.Module) -> None:
            counters[name] = 0

            def hook(_module, hook_args, output, detail_name=name):
                index = counters[detail_name]
                if index < 2 and detail_name == "self_q":
                    torch.save(
                        hook_args[0].detach().cpu(),
                        dump_dir / f"self_q_input_{index:03d}.pt",
                    )
                dump(detail_name, output)

            module.register_forward_hook(hook)

        block0 = self.blocks[0]
        detail_modules = {
            "time_embed": self.condition_embedder.time_embedder,
            "time_projection": self.condition_embedder.time_modulation,
            "text_embed": self.condition_embedder.text_embedder,
            "self_q": block0.to_q,
            "self_k": block0.to_k,
            "self_v": block0.to_v,
            "self_out": block0.to_out,
            "cross_q": block0.attn2.to_q,
            "cross_k": block0.attn2.to_k,
            "cross_v": block0.attn2.to_v,
            "cross_out": block0.attn2.to_out,
            "ffn": block0.ffn,
        }
        for detail_name, module in detail_modules.items():
            register_detail(detail_name, module)

    def _apply_patch_token_condition(
        self,
        hidden_states: torch.Tensor,
        *,
        action: torch.Tensor | None,
        num_frames: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if action is None:
            raise ValueError(
                "MinWM requires an action label for every latent frame; use 0 for noop"
            )
        residual = self.action_in.token_residual(
            action,
            num_current_frames=num_frames,
            tokens_per_frame=height * width,
            dtype=hidden_states.dtype,
        )
        # minWM materializes both patch and action token lists through
        # ``torch.cat`` before this add. Even for B=1 that makes the block input
        # contiguous; a channel-first stride selects a different compiled
        # LayerNorm reduction on B200 despite identical tensor values.
        return (hidden_states + residual).contiguous()

    def _apply_output_head(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        num_frames = timestep.shape[1]
        temb = temb.unflatten(dim=0, sizes=timestep.shape).to(hidden_states.dtype)
        modulation = self.scale_shift_table.to(hidden_states.dtype)
        frame_index = _minwm_frame_indices(hidden_states, num_frames)
        timestep_value = temb[:, frame_index]
        _, normalized = _minwm_adaln(
            hidden_states,
            modulation[:, 0],
            modulation[:, 1],
            timestep_value,
            timestep_value,
            self.norm_out.norm.eps,
        )
        return self.proj_out(normalized)


EntryClass = MinWMCausalTransformer3DModel
