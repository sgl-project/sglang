"""Muse QKV fusion shape and cache-layout guards."""

import logging

import torch

from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.rocm_muse_spark.muse_qk_rope_cache import fused_qk_rope_cache
from sglang.srt.rocm_muse_spark.muse_qkv_fused import project_qkv_cache

_LOGGED = set()


def _note(name, m):
    if name not in _LOGGED:
        logging.getLogger(__name__).info("Muse Day0 fast path: %s, rows=%d", name, m)
        _LOGGED.add(name)


def try_qk_rope_cache(attention, positions, q, k, v, batch):
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

    if not (
        0 < q.shape[0] <= 64
        and attention.head_dim == 64
        and attention.num_heads == 16
        and attention.num_kv_heads == 2
        and attention.use_qk_norm
        and not attention.use_attn_temperature_tuning
        and q.dtype == k.dtype == v.dtype == torch.bfloat16
        and q.stride(1) == k.stride(1) == v.stride(1) == 1
        and (batch.forward_mode.is_decode() or batch.forward_mode.is_target_verify())
        and not batch.attn_attend_prefix_cache
        and batch.dcp_kv_mask is None
    ):
        return None
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
    kc, vc = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    if not (
        kc.dtype == vc.dtype == torch.float8_e4m3fn
        and kc.is_contiguous()
        and vc.is_contiguous()
    ):
        return None
    ks = layer.k_scale if layer.k_scale is not None else backend.k_scale
    vs = layer.v_scale if layer.v_scale is not None else backend.v_scale
    if not (ks is not None and vs is not None and ks.numel() == vs.numel() == 1):
        return None
    mapping = (
        backend.swa_kv_pool.full_to_swa_index_mapping
        if backend.use_sliding_window_kv_pool and layer.sliding_window_size > 0
        else None
    )
    _note("QK norm/RoPE/FP8 cache", q.shape[0])
    return fused_qk_rope_cache(
        q,
        k,
        v,
        positions,
        rope.cos_sin_cache if rope is not None else None,
        kc,
        vc,
        batch.out_cache_loc,
        mapping,
        attention.norm_eps,
        attention.scale_query_by,
        ks,
        vs,
    )


def try_project_qkv_cache(attention, positions, x, batch):
    from sglang.srt.layers.attention.aiter_backend import AiterAttnBackend

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
        and attention.head_dim == 64
        and attention.num_heads == 16
        and attention.num_kv_heads == 2
        and attention.use_o_gate
        and attention.use_qk_norm
        and not attention.use_attn_temperature_tuning
        and (batch.forward_mode.is_decode() or batch.forward_mode.is_target_verify())
        and not batch.attn_attend_prefix_cache
        and batch.dcp_kv_mask is None
    ):
        return None
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
    kc, vc = backend.token_to_kv_pool.get_kv_buffer(layer.layer_id)
    if not (
        kc.dtype == vc.dtype == torch.float8_e4m3fn
        and kc.is_contiguous()
        and vc.is_contiguous()
    ):
        return None
    ks = layer.k_scale if layer.k_scale is not None else backend.k_scale
    vs = layer.v_scale if layer.v_scale is not None else backend.v_scale
    if not (ks is not None and vs is not None and ks.numel() == vs.numel() == 1):
        return None
    mapping = (
        backend.swa_kv_pool.full_to_swa_index_mapping
        if backend.use_sliding_window_kv_pool and layer.sliding_window_size > 0
        else None
    )
    _note("QKV projection/QK norm/RoPE/FP8 cache", x.shape[0])
    return project_qkv_cache(
        x,
        projection.weight,
        positions,
        rope.cos_sin_cache if rope is not None else None,
        kc,
        vc,
        batch.out_cache_loc,
        mapping,
        attention.norm_eps,
        attention.scale_query_by,
        ks,
        vs,
    )
