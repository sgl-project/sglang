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
    _usp_input_all_to_all_qkv,
    _usp_input_all_to_all_varlen_qkv,
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
    PrimitiveRoPETokenResidualActionEncoder,
    PrimitiveTokenResidualActionEncoder,
)
from sglang.multimodal_gen.runtime.models.dits.minwm_kv_cache import (
    MinWMCausalAttentionKVPlan,
    MinWMCausalSelfAttentionKVCache,
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
_MINWM_CUDA_GRAPH_ACTIVE = False
_MINWM_CACHE_ROTATED_K = _env_flag("MINWM_CACHE_ROTATED_K", True)
_MINWM_PRECOMPUTE_CACHE_ROPE = _env_flag("MINWM_PRECOMPUTE_CACHE_ROPE", True)
_MINWM_CACHE_PACKED_METADATA = _env_flag("MINWM_CACHE_PACKED_METADATA", True)
_MINWM_ANNOUNCED_ATTENTION_BACKENDS: set[tuple[str, str]] = set()


class _MinWMUlyssesWorkspace:
    """Reusable communication buffers shared by the sequential transformer blocks."""

    def __init__(self) -> None:
        self._buffers: dict[str, torch.Tensor] = {}

    def get(
        self,
        name: str,
        reference: torch.Tensor,
        numel: int,
    ) -> torch.Tensor:
        buffer = self._buffers.get(name)
        if (
            buffer is None
            or buffer.numel() != numel
            or buffer.dtype != reference.dtype
            or buffer.device != reference.device
        ):
            buffer = reference.new_empty(numel)
            self._buffers[name] = buffer
        return buffer


@torch.compiler.disable
def _minwm_update_and_get_attention_kv(
    kv_cache,
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    current_chunk_start: int,
    cache_head_start: int | None,
):
    """Keep MinWM's stateful bounded-cache update outside whole-DiT compile."""
    return kv_cache.update_and_get_attention_kv(
        key=key,
        value=value,
        current_chunk_start=current_chunk_start,
        cache_head_start=cache_head_start,
        debug_name="MinWM causal KV cache",
    )


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
        if not use_compile or not _MINWM_SEGMENT_COMPILE or _MINWM_CUDA_GRAPH_ACTIVE:
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


def set_minwm_cuda_graph_active(enabled: bool) -> None:
    """Keep lazily compiled Inductor segments out of manual DiT capture.

    Segment compilation can allocate or compile on its first invocation, which
    is unsafe inside CUDA Graph capture. It can also change MinWM's BF16 rounding
    boundaries, so the graph and eager benchmark lanes both use eager segments.
    """
    global _MINWM_CUDA_GRAPH_ACTIVE
    enabled = bool(enabled)
    if enabled == _MINWM_CUDA_GRAPH_ACTIVE:
        return
    _MINWM_CUDA_GRAPH_ACTIVE = enabled
    if enabled and _MINWM_SEGMENT_COMPILE:
        logger.info("MinWM CUDA graph disables nested segment torch.compile for parity")


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
    # minWM casts the entire generator to BF16 before inference. Keep that cast
    # outside the compiled function so its graph and operand dtypes match
    # WanLayerNorm._norm exactly.
    weight = weight.to(hidden_states.dtype) if weight is not None else None
    bias = bias.to(hidden_states.dtype) if bias is not None else None
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
        if num_frames == 1:
            return torch.zeros_like(frame_indices)
        return frame_indices

    if hidden_states.shape[1] % num_frames != 0:
        raise ValueError(
            f"MinWM sequence length {hidden_states.shape[1]} must be divisible "
            f"by num_frames {num_frames} when sequence sharding is disabled."
        )
    return _minwm_uniform_frame_indices(
        hidden_states.shape[1], num_frames, hidden_states.device
    )


@lru_cache(maxsize=32)
def _minwm_uniform_frame_indices(
    sequence_length: int,
    num_frames: int,
    device: torch.device,
) -> torch.Tensor:
    tokens_per_frame = sequence_length // num_frames
    return torch.arange(num_frames, device=device).repeat_interleave(tokens_per_frame)


def _minwm_qk_norm_rope_op(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    eps: float,
    rope: torch.Tensor,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query, key = _minwm_qk_norm_op(query, key, query_weight, key_weight, eps, num_heads)

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
        return apply_minwm_rotary_embedding(hidden_states, cos, sin)

    return apply(query), apply(key)


def _minwm_qk_norm_op(
    query: torch.Tensor,
    key: torch.Tensor,
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    eps: float,
    num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = MinWMRMSNorm._norm(query, query_weight, eps)
    key = MinWMRMSNorm._norm(key, key_weight, eps)
    *leading, dim = query.shape
    query = query.reshape(*leading, num_heads, dim // num_heads)
    key = key.reshape(*leading, num_heads, dim // num_heads)
    return query, key


def _minwm_apply_qk_op(
    qk_op,
    qk_args: list,
    *,
    use_cache: bool,
    use_compile: bool,
):
    """Keep cache inference eager, matching minWM main's BF16 reduction."""
    if use_cache:
        return qk_op(*qk_args)
    return _MinWMSegmentCompile.get(qk_op, use_compile)(*qk_args)


@torch.compiler.disable
def _minwm_packed_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Call the same device-selected packed-varlen backend as minWM main.

    Keep the packed FlashAttention boundary eager when the enclosing DiT is
    compiled. FA4 already supplies the fused kernel; fixed-shape cu-seqlens are
    cached separately so this boundary performs no metadata kernels.
    """
    if query.device.type != "cuda":
        raise RuntimeError("MinWM packed-varlen attention requires CUDA")
    batch_size, query_length, num_heads, head_dim = query.shape
    key_length = key.shape[1]
    if key.shape != value.shape or key.shape[0] != batch_size:
        raise ValueError("MinWM attention key/value shapes must match")
    if key.shape[2:] != (num_heads, head_dim):
        raise ValueError("MinWM attention Q/K/V head geometry must match")

    if _MINWM_CACHE_PACKED_METADATA:
        cu_query = _minwm_uniform_cu_seqlens(batch_size, query_length, query.device)
        cu_key = _minwm_uniform_cu_seqlens(batch_size, key_length, key.device)
    else:
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


@lru_cache(maxsize=128)
def _minwm_uniform_cu_seqlens(
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Cache fixed-shape packed-attention metadata outside the hot path."""
    return (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device) * sequence_length
    )


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
        self.ulysses_workspace: _MinWMUlyssesWorkspace | None = None
        ulysses_world_size = max(get_ulysses_parallel_world_size(), 1)
        if self.num_heads % ulysses_world_size != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by "
                f"ulysses_degree ({ulysses_world_size})."
            )
        self.ulysses_num_heads = self.num_heads // ulysses_world_size
        self._minwm_rotary_emb = NDRotaryEmbedding(
            rope_dim_list=[
                self.head_dim - 4 * (self.head_dim // 6),
                2 * (self.head_dim // 6),
                2 * (self.head_dim // 6),
            ],
            rope_theta=10000,
            dtype=(
                torch.float64
                if current_platform.is_float64_supported()
                else torch.float32
            ),
        )
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
        if not isinstance(kv_cache, MinWMCausalSelfAttentionKVCache):
            raise TypeError("MinWM inference requires its position-aware raw-K cache")
        if qk_already_roped:
            raise ValueError("MinWM inference cache must receive unrotated Q/K")

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
            workspace = getattr(self, "ulysses_workspace", None)
            if uniform_seq_splits:
                input_buffer = output_buffer = None
                if workspace is not None:
                    packed_numel = 3 * query.numel()
                    input_buffer = workspace.get("qkv_send", query, packed_numel)
                    output_buffer = workspace.get("qkv_recv", query, packed_numel)
                qkv = _usp_input_all_to_all_qkv(
                    query,
                    key,
                    value,
                    input_buffer=input_buffer,
                    output_buffer=output_buffer,
                )
            else:
                qkv = _usp_input_all_to_all_varlen_qkv(query, key, value, seq_splits)
            query, key, value = qkv.chunk(3, dim=-1)

        cache_view = _minwm_update_and_get_attention_kv(
            kv_cache,
            key=key,
            value=value,
            current_chunk_start=current_start,
            cache_head_start=0 if sequence_shard_enabled else self.head_start,
        )
        if cache_view.query_cos is None or cache_view.query_sin is None:
            query_cos, query_sin = self._minwm_rotary_emb.forward_uncached(
                cache_view.query_position_ids
            )
        else:
            query_cos, query_sin = cache_view.query_cos, cache_view.query_sin
        roped_query = apply_minwm_rotary_embedding(query, query_cos, query_sin).type_as(
            value
        )
        if (
            _MINWM_CACHE_ROTATED_K
            and cache_view.rotated_k_is_valid
            and cache_view.is_recompute
        ):
            rotated_current_key = apply_minwm_rotary_embedding(
                key, query_cos, query_sin
            ).type_as(value)
            cache_view.rotated_k[
                :, cache_view.current_local_start : cache_view.current_local_end
            ].copy_(rotated_current_key)
            attention_key = cache_view.rotated_k
        else:
            if cache_view.key_cos is None or cache_view.key_sin is None:
                key_cos, key_sin = self._minwm_rotary_emb.forward_uncached(
                    cache_view.key_position_ids
                )
            else:
                key_cos, key_sin = cache_view.key_cos, cache_view.key_sin
            attention_key = apply_minwm_rotary_embedding(
                cache_view.k, key_cos, key_sin
            ).type_as(value)
            if _MINWM_CACHE_ROTATED_K:
                cache_view.rotated_k.copy_(attention_key)
                kv_cache.rotated_k_is_valid = True
        attention_value = cache_view.v
        parity_dump_dir = getattr(self, "_minwm_parity_dump_dir", None)
        parity_index = getattr(self, "_minwm_parity_forward_index", 0)
        if parity_dump_dir is not None and parity_index < 2:
            torch.save(
                query.detach().cpu(),
                parity_dump_dir / f"self_q_norm_{parity_index:03d}.pt",
            )
            torch.save(
                cache_view.k.detach().cpu(),
                parity_dump_dir / f"self_k_norm_{parity_index:03d}.pt",
            )
            torch.save(
                roped_query.detach().cpu(),
                parity_dump_dir / f"self_q_roped_{parity_index:03d}.pt",
            )
            torch.save(
                attention_key.detach().cpu(),
                parity_dump_dir / f"self_k_roped_{parity_index:03d}.pt",
            )
        if _MINWM_ATTENTION_IMPL == "dense":
            output = (self.ulysses_attn if sequence_shard_enabled else self.attn)(
                roped_query, attention_key, attention_value
            )
        else:
            output = _minwm_packed_varlen_attention(
                roped_query, attention_key, attention_value
            )
        if parity_dump_dir is not None:
            if parity_index < 2:
                torch.save(
                    output.detach().cpu(),
                    parity_dump_dir / f"self_attention_output_{parity_index:03d}.pt",
                )
            self._minwm_parity_forward_index = parity_index + 1
        if sequence_shard_enabled:
            assert seq_splits is not None
            if uniform_seq_splits:
                output_buffer = None
                workspace = getattr(self, "ulysses_workspace", None)
                if workspace is not None:
                    output_buffer = workspace.get(
                        "attention_recv", output, output.numel()
                    )
                output = _usp_output_all_to_all(
                    output, head_dim=2, output_buffer=output_buffer
                )
            else:
                output = _usp_output_all_to_all_varlen(output, seq_splits, head_dim=2)
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
            qk_op = _minwm_qk_norm_rope_op if kv_cache is None else _minwm_qk_norm_op
            qk_args = [
                query.squeeze(1),
                key.squeeze(1),
                self.norm_q.weight,
                self.norm_k.weight,
                self.norm_q.eps,
            ]
            if kv_cache is None:
                qk_args.append(torch.stack(freqs_cis, dim=-1))
            qk_args.append(self.local_num_heads)
            # minWM main's inference/cache path calls qk_norm_op eagerly.
            # Compiling this reduction changes its BF16 rounding boundary.
            query, key = _minwm_apply_qk_op(
                qk_op,
                qk_args,
                use_cache=kv_cache is not None,
                use_compile=query.is_cuda,
            )
            qk_already_roped = kv_cache is None
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
        parity_dump_dir = getattr(self, "_minwm_parity_dump_dir", None)
        parity_index = getattr(self, "_minwm_parity_forward_index", 0)
        if parity_dump_dir is not None:
            if parity_index < 2:
                torch.save(
                    hidden_states.detach().cpu(),
                    parity_dump_dir / f"self_residual_norm_input_{parity_index:03d}.pt",
                )
            self._minwm_parity_forward_index = parity_index + 1

        affine_norm = self.self_attn_residual_norm.norm
        norm_hidden_states = _minwm_layer_norm(
            hidden_states,
            eps=affine_norm.eps,
            weight=affine_norm.weight,
            bias=affine_norm.bias,
        )
        if parity_dump_dir is not None and parity_index < 2:
            torch.save(
                norm_hidden_states.detach().cpu(),
                parity_dump_dir / f"self_residual_norm_{parity_index:03d}.pt",
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
            "packed_deterministic=%s segment_compile=%s cache_rotated_k=%s "
            "precompute_cache_rope=%s cache_packed_metadata=%s",
            _MINWM_ATTENTION_IMPL,
            _MINWM_PACKED_ATTENTION_DETERMINISTIC,
            _MINWM_SEGMENT_COMPILE,
            _MINWM_CACHE_ROTATED_K,
            _MINWM_PRECOMPUTE_CACHE_ROPE,
            _MINWM_CACHE_PACKED_METADATA,
        )
        deterministic = os.environ.get("MINWM_PARITY_DETERMINISTIC", "0")
        if deterministic.strip().lower() not in {"", "0", "false", "no", "off"}:
            torch.use_deterministic_algorithms(True)
        super().__init__(config, hf_config, quant_config)
        self.sp_size = get_sp_world_size()
        ulysses_workspace = _MinWMUlyssesWorkspace() if self.sp_size > 1 else None
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
            block.attn1.ulysses_workspace = ulysses_workspace
            block.norm_q = MinWMRMSNorm(config.hidden_size, eps=config.eps)
            block.norm_k = MinWMRMSNorm(config.hidden_size, eps=config.eps)
            block.attn2.norm_q = MinWMRMSNorm(config.hidden_size, eps=config.eps)
            block.attn2.norm_k = MinWMRMSNorm(config.hidden_size, eps=config.eps)
        action_encoder_cls = (
            PrimitiveRoPETokenResidualActionEncoder
            if config.action_type == "primitive_rope_token_residual"
            else PrimitiveTokenResidualActionEncoder
        )
        self.action_in = action_encoder_cls(
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

    @lru_cache(maxsize=64)
    def _compute_cache_position_ids(
        self,
        num_frames: int,
        grid_height: int,
        grid_width: int,
        start_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        temporal = (
            torch.arange(
                num_frames,
                device=device,
                dtype=torch.long,
            )
            + start_frame
        )
        temporal = temporal[:, None, None].expand(num_frames, grid_height, grid_width)
        height_ids = torch.arange(grid_height, device=device, dtype=torch.long)[
            None, :, None
        ].expand(num_frames, grid_height, grid_width)
        width_ids = torch.arange(grid_width, device=device, dtype=torch.long)[
            None, None, :
        ].expand(num_frames, grid_height, grid_width)
        return torch.stack([temporal, height_ids, width_ids], dim=-1).reshape(-1, 3)

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
        precomputed_attention_plan: MinWMCausalAttentionKVPlan | None = None,
    ) -> torch.Tensor:
        if kv_cache is not None:
            attention_plan = precomputed_attention_plan
            if attention_plan is None:
                attention_plan = self.prepare_causal_attention_plan(
                    hidden_states,
                    kv_cache=kv_cache,
                    current_start=current_start,
                    start_frame=start_frame,
                )
            for cache_block in kv_cache:
                if not isinstance(cache_block, MinWMCausalSelfAttentionKVCache):
                    raise TypeError(
                        "MinWM transformer requires position-aware raw-K caches"
                    )
                cache_block.set_prepared_attention_plan(attention_plan)

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

        hidden_states = self._apply_output_head(
            hidden_states,
            temb,
            timestep,
            sequence_shard_splits=seq_splits,
            sequence_shard_rank=sp_rank,
        )
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

    def prepare_causal_attention_plan(
        self,
        hidden_states: torch.Tensor,
        *,
        kv_cache,
        current_start: int,
        start_frame: int,
    ) -> MinWMCausalAttentionKVPlan:
        """Prepare host-side cache metadata before an eager or graphed forward."""
        _, _, num_frames, latent_height, latent_width = hidden_states.shape
        _, patch_height, patch_width = self.patch_size
        grid_height = latent_height // patch_height
        grid_width = latent_width // patch_width
        position_ids = self._compute_cache_position_ids(
            num_frames,
            grid_height,
            grid_width,
            int(start_frame),
            hidden_states.device,
        )
        metadata_cache = kv_cache[0]
        if not isinstance(metadata_cache, MinWMCausalSelfAttentionKVCache):
            raise TypeError("MinWM transformer requires position-aware raw-K caches")
        attention_plan = metadata_cache.prepare_attention_plan(
            current_chunk_start=current_start,
            position_ids=position_ids,
        )
        if _MINWM_PRECOMPUTE_CACHE_ROPE and attention_plan.query_cos is None:
            (
                attention_plan.query_cos,
                attention_plan.query_sin,
            ) = self._sequence_shard_rotary_emb.forward_uncached(
                attention_plan.query_position_ids
            )
            (
                attention_plan.key_cos,
                attention_plan.key_sin,
            ) = self._sequence_shard_rotary_emb.forward_uncached(
                attention_plan.key_position_ids
            )
        return attention_plan

    def _install_parity_debug_hooks(self) -> None:
        dump_root = os.environ.get("MINWM_PARITY_DUMP_DIR")
        if not dump_root:
            return
        dump_dir = (
            Path(dump_root)
            / "sglang"
            / f"sp_{get_sp_world_size():02d}_rank_{get_sp_parallel_rank():02d}"
        )
        dump_dir.mkdir(parents=True, exist_ok=True)
        counters = {"patch": 0}

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

        def dump_block(
            _module,
            hook_args,
            output,
            *,
            block_name: str,
            include_input: bool,
        ) -> None:
            index = counters[block_name]
            if include_input and index < 2:
                torch.save(
                    hook_args[0].detach().cpu(),
                    dump_dir / f"{block_name}_input_{index:03d}.pt",
                )
            dump(block_name, output)

        dump_all_blocks = os.environ.get("MINWM_PARITY_DUMP_ALL_BLOCKS", "0") == "1"
        debug_blocks = self.blocks if dump_all_blocks else self.blocks[:1]
        for block_index, block in enumerate(debug_blocks):
            block_name = f"block{block_index}"
            counters[block_name] = 0
            block.register_forward_hook(
                lambda module, args, output, name=block_name, include=block_index == 0: (
                    dump_block(
                        module,
                        args,
                        output,
                        block_name=name,
                        include_input=include,
                    )
                )
            )

        def register_detail(name: str, module: nn.Module) -> None:
            counters[name] = 0

            def hook(_module, hook_args, output, detail_name=name):
                index = counters[detail_name]
                if index < 2 and detail_name in {
                    "self_q",
                    "self_out",
                    "cross_q",
                    "cross_out",
                    "output_proj",
                }:
                    torch.save(
                        hook_args[0].detach().cpu(),
                        dump_dir / f"{detail_name}_input_{index:03d}.pt",
                    )
                if index == 0 and detail_name == "self_q":
                    torch.save(
                        _module.weight.detach().cpu(),
                        dump_dir / "self_q_weight.pt",
                    )
                    torch.save(
                        _module.bias.detach().cpu(),
                        dump_dir / "self_q_bias.pt",
                    )
                dump(detail_name, output)

            module.register_forward_hook(hook)

        block0 = self.blocks[0]
        block0._minwm_parity_dump_dir = dump_dir
        block0._minwm_parity_forward_index = 0
        block0.attn1._minwm_parity_dump_dir = dump_dir
        block0.attn1._minwm_parity_forward_index = 0
        detail_modules = {
            "time_embed": self.condition_embedder.time_embedder,
            "time_projection": self.condition_embedder.time_modulation,
            "text_embed": self.condition_embedder.text_embedder,
            "self_q": block0.to_q,
            "self_k": block0.to_k,
            "self_v": block0.to_v,
            "self_norm_q": block0.norm_q,
            "self_norm_k": block0.norm_k,
            "self_out": block0.to_out,
            "cross_q": block0.attn2.to_q,
            "cross_k": block0.attn2.to_k,
            "cross_v": block0.attn2.to_v,
            "cross_norm_q": block0.attn2.norm_q,
            "cross_norm_k": block0.attn2.norm_k,
            "cross_out": block0.attn2.to_out,
            "ffn": block0.ffn,
            "output_proj": self.proj_out,
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
        *,
        sequence_shard_splits: list[int] | tuple[int, ...] | None = None,
        sequence_shard_rank: int = 0,
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
        if _minwm_should_restore_reference_output_projection(
            normalized,
            sequence_shard_splits,
        ):
            return _minwm_project_output_in_reference_row_bucket(
                self.proj_out,
                normalized,
                sequence_shard_splits,
                sequence_shard_rank,
            )
        return self.proj_out(normalized)


def _minwm_should_restore_reference_output_projection(
    hidden_states: torch.Tensor,
    sequence_shard_splits: list[int] | tuple[int, ...] | None,
) -> bool:
    return bool(
        sequence_shard_splits is not None
        and len(sequence_shard_splits) == 8
        and hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and torch.version.hip is None
        and torch.cuda.get_device_capability(hidden_states.device) == (9, 0)
    )


def _minwm_project_output_in_reference_row_bucket(
    projection: nn.Module,
    hidden_states: torch.Tensor,
    sequence_shard_splits: list[int] | tuple[int, ...],
    sequence_shard_rank: int,
) -> torch.Tensor:
    """Run Hopper SP8 output projection with the reference SP1 row layout.

    Hopper can select a different BF16 GEMM reduction for the short SP8 row
    bucket. Padding each local shard back to its exact global row positions
    keeps the projection bitwise aligned with SP1; the extra output-head work
    is negligible relative to the transformer blocks.
    """
    if not 0 <= sequence_shard_rank < len(sequence_shard_splits):
        raise ValueError(
            f"sequence shard rank {sequence_shard_rank} is outside "
            f"{len(sequence_shard_splits)} splits"
        )
    local_seq_len = sequence_shard_splits[sequence_shard_rank]
    if hidden_states.shape[1] != local_seq_len:
        raise ValueError(
            f"local output-head sequence length {hidden_states.shape[1]} "
            f"does not match split {local_seq_len}"
        )
    row_start = sum(sequence_shard_splits[:sequence_shard_rank])
    row_end = row_start + local_seq_len
    global_seq_len = sum(sequence_shard_splits)
    padded = F.pad(
        hidden_states,
        (0, 0, row_start, global_seq_len - row_end),
    )
    projected = projection(padded)
    return projected.narrow(1, row_start, local_seq_len)


EntryClass = MinWMCausalTransformer3DModel
