"""Fused QK norm / RoPE / FP8 KV-cache write for Muse Spark attention on ROCm (decode and target verify).

``MuseSpark1xAttention`` hands K/V to the cache itself when one of these kernels applies and then calls the
attention layer with ``save_kv_cache=False``, the usual SGLang pattern for producers that fuse the cache store
(bailing_moe, gemma4):

* ``qkv_proj_norm_rope_kv_write`` (``SGLANG_MUSE_FUSED_QKV_CACHE``): up to 32 rows, the QKV + output-gate
  projection's split-K partials feed the QK-norm / RoPE / cache-store epilogue directly.
* ``qk_norm_rope_kv_write`` (``SGLANG_MUSE_QK_ROPE_CACHE``): up to 64 rows, from the projected Q/K/V.

Both write exactly where and what the backend's own store would (FP8 with the layer's per-tensor scales, through
the full -> SWA slot mapping on sliding layers), so they only engage when the backend reads K/V from the
16-token NHD FP8 pages of the aiter unified-attention backend.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch

from sglang.kernels.ops.attention.qk_norm_rope_cache import (
    qk_norm_rope_cache,
    qkv_proj_qk_norm_rope_cache,
)
from sglang.srt.environ import envs
from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_attn_backend

logger = logging.getLogger(__name__)

QK_ROPE_CACHE = envs.SGLANG_MUSE_QK_ROPE_CACHE.get()
FUSED_QKV_CACHE = envs.SGLANG_MUSE_FUSED_QKV_CACHE.get()
_logged: set = set()


class _KVWriteTarget(NamedTuple):
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    locations: torch.Tensor
    swa_mapping: torch.Tensor | None
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    cos_sin: torch.Tensor | None


def _attention_shape_ok(attention, forward_batch: ForwardBatch) -> bool:
    return (
        attention.head_dim == 64
        and attention.num_heads == 16
        and attention.num_kv_heads == 2
        and attention.use_qk_norm
        and not attention.use_attn_temperature_tuning
        and (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        )
        and not forward_batch.attn_attend_prefix_cache
        and forward_batch.dcp_kv_mask is None
    )


def _kv_write_target(attention, forward_batch: ForwardBatch) -> _KVWriteTarget | None:
    """Where the backend reads this layer's K/V from, if the fused kernels can write it there."""
    backend = get_attn_backend()
    layer = attention.attn
    if not (
        isinstance(backend, AiterAttnBackend)
        and not backend.use_mla
        and backend.page_size == 16
        and not backend.kv_cache_is_vectorized_5d
        and backend.use_triton_unified_attention
        and not layer.is_cross_attention
        and backend.kv_cache_dtype == torch.float8_e4m3fn
    ):
        return None
    rope = attention.rotary_emb
    if rope is not None and not (
        type(rope).__name__ == "RotaryEmbedding"
        and not rope.is_neox_style
        and rope.rotary_dim == 64
        and rope.position_start == 0
        and rope.cos_sin_cache.dtype == torch.float32
    ):
        return None
    k_cache, v_cache = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    if not (
        k_cache.dtype == v_cache.dtype == torch.float8_e4m3fn
        and k_cache.is_contiguous()
        and v_cache.is_contiguous()
    ):
        return None
    # The same per-tensor scales the backend's own store would divide by.
    k_scale = layer.k_scale if layer.k_scale is not None else backend.k_scale
    v_scale = layer.v_scale if layer.v_scale is not None else backend.v_scale
    if not (
        k_scale is not None
        and v_scale is not None
        and k_scale.numel() == v_scale.numel() == 1
    ):
        return None
    swa_mapping = (
        backend.swa_kv_pool.full_to_swa_index_mapping
        if backend.use_sliding_window_kv_pool and layer.sliding_window_size > 0
        else None
    )
    return _KVWriteTarget(
        k_cache,
        v_cache,
        forward_batch.out_cache_loc,
        swa_mapping,
        k_scale,
        v_scale,
        rope.cos_sin_cache if rope is not None else None,
    )


def _note(name: str, rows: int) -> None:
    if name not in _logged:
        _logged.add(name)
        logger.info("Muse fast path: %s, rows=%d", name, rows)


def qkv_proj_norm_rope_kv_write(
    attention, positions: torch.Tensor, x: torch.Tensor, forward_batch: ForwardBatch
):
    """(q, k, v, o_gate) with K/V already in the cache, or None when the fused projection does not apply."""
    projection = attention.qkv_proj
    if not (
        x.ndim == 2
        and 0 < x.shape[0] <= 32
        and x.shape[1] == 8192
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and projection.weight.shape == (2304, 8192)
        and projection.weight.dtype == torch.bfloat16
        and projection.weight.is_contiguous()
        and projection.bias is None
        and not projection.gather_output
        and attention.use_o_gate
        and _attention_shape_ok(attention, forward_batch)
    ):
        return None
    target = _kv_write_target(attention, forward_batch)
    if target is None:
        return None
    _note("QKV projection/QK norm/RoPE/FP8 cache", x.shape[0])
    return qkv_proj_qk_norm_rope_cache(
        x,
        projection.weight,
        positions,
        target.cos_sin,
        target.k_cache,
        target.v_cache,
        target.locations,
        target.swa_mapping,
        attention.norm_eps,
        attention.scale_query_by,
        target.k_scale,
        target.v_scale,
    )


def qk_norm_rope_kv_write(
    attention,
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    forward_batch: ForwardBatch,
):
    """(q, k) normalized and rotated with K/V already in the cache, or None when the kernel does not apply."""
    if not (
        0 < q.shape[0] <= 64
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.stride(1) == k.stride(1) == v.stride(1) == 1
        and _attention_shape_ok(attention, forward_batch)
    ):
        return None
    target = _kv_write_target(attention, forward_batch)
    if target is None:
        return None
    _note("QK norm/RoPE/FP8 cache", q.shape[0])
    return qk_norm_rope_cache(
        q,
        k,
        v,
        positions,
        target.cos_sin,
        target.k_cache,
        target.v_cache,
        target.locations,
        target.swa_mapping,
        attention.norm_eps,
        attention.scale_query_by,
        target.k_scale,
        target.v_scale,
    )
