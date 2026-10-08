"""Opt-in Kimi-K3 absorbed key projection and FP8 MLA cache writer.

The generated kernel fuses ``q_nope @ w_kc``, query/key concatenation, and
the physical page-size-one FP8 cache write.  Selection is deliberately strict:
only the qualified Kimi-K3 gfx950 TP8 Triton MLA layout is admitted.  A launch
that may have written cache state is never retried through the native path.
"""

from __future__ import annotations

import os

import torch

from sglang.kernels.ops.attention.kda_whole_layer_gluon_hip import (
    _has_required_gluon_api,
    _rocm_arch,
)
from sglang.srt.utils import is_hip


def enabled() -> bool:
    return os.environ.get("SGLANG_ROCM_K3_MLA_KC_FUSED_BACKEND", "").lower() == "gluon"


def entrypoint_name(rows: int) -> str | None:
    if rows in (1, 4, 8):
        return "mla_kc_cache_m1_4_8"
    if rows in (2, 16):
        return "mla_kc_cache_m2_16"
    if rows in (32, 64, 128):
        return f"mla_kc_cache_m{rows}"
    if rows == 256 or 1024 <= rows <= 8192:
        return "mla_kc_cache_m256_1024_8192"
    return None


def _unit_cache_scale(scale) -> bool:
    # Do not synchronize a device scalar merely to qualify the fast path.
    return scale is None or type(scale) in (float, int) and scale == 1.0


def can_prepare(attn, parallel, server_args) -> bool:
    """Admit only the loaded TP8 BF16 Kimi MLA projection contract."""
    if not enabled() or not is_hip() or not _has_required_gluon_api():
        return False
    weight = getattr(attn, "w_kc", None)
    return (
        isinstance(weight, torch.Tensor)
        and weight.is_cuda
        and _rocm_arch(weight.device.index) == "gfx950"
        and type(attn).__name__ == "KimiK3MLAAttention"
        and attn.rotary_emb is None
        and not attn.use_dsa
        and not attn.use_deep_gemm_bmm
        and parallel.attn_tp_size == 8
        and not parallel.dcp_enabled
        and not parallel.dcp_replicate_q_proj
        and not getattr(server_args, "enable_lora", False)
        and not getattr(server_args, "speculative_algorithm", None)
        and type(attn.w_scale) in (float, int)
        and attn.w_scale == 1.0
        and _unit_cache_scale(attn.attn_mqa.k_scale)
        and (
            attn.num_local_heads,
            attn.qk_nope_head_dim,
            attn.qk_rope_head_dim,
            attn.kv_lora_rank,
        )
        == (12, 128, 64, 512)
        and weight.dtype == torch.bfloat16
        and tuple(weight.shape) == (12, 128, 512)
        and weight.stride() == (65536, 1, 128)
        and weight.storage_offset() == 0
    )


def _physical_triton_pool():
    from sglang.srt.model_executor.forward_context import (
        get_attn_backend,
        get_token_to_kv_pool,
    )

    backend = get_attn_backend()
    full = getattr(backend, "full_attn_backend", backend)
    pool = get_token_to_kv_pool()
    metadata = getattr(full, "forward_metadata", None)
    translator = getattr(full, "kv_index_translator", None)
    if not (
        type(full).__name__ == "TritonAttnBackend"
        and type(full).__module__ == "sglang.srt.layers.attention.triton_backend"
        and getattr(full, "token_to_kv_pool", None) is pool
        and translator is not None
        and not translator.is_translating
        and getattr(pool, "layer_transfer_counter", None) is None
        and metadata is not None
        and getattr(metadata, "out_cache_loc_full_physical", None) is None
        and getattr(metadata, "swa_out_cache_loc", None) is None
    ):
        return None
    return pool


def covered(attn, query, latent, key_tail, forward_batch) -> bool:
    """Validate mutable tensors and the physical page-size-one cache ABI."""
    rows = query.shape[0] if isinstance(query, torch.Tensor) and query.ndim else 0
    locations = getattr(forward_batch, "out_cache_loc", None)
    if not (
        entrypoint_name(rows) is not None
        and attn.current_attention_backend == "triton"
        and tuple(query.shape) == (rows, 12, 192)
        and query.dtype == torch.bfloat16
        and query.device == attn.w_kc.device
        and query.stride(2) == 1
        and query.storage_offset() == 0
        and isinstance(latent, torch.Tensor)
        and tuple(latent.shape) == (rows, 1, 512)
        and latent.dtype == query.dtype
        and latent.device == query.device
        and latent.stride(-1) == 1
        and isinstance(key_tail, torch.Tensor)
        and tuple(key_tail.shape) == (rows, 1, 64)
        and key_tail.dtype == query.dtype
        and key_tail.device == query.device
        and key_tail.stride(-1) == 1
        and isinstance(locations, torch.Tensor)
        and tuple(locations.shape) == (rows,)
        and locations.dtype == torch.int64
        and locations.device == query.device
        and locations.is_contiguous()
    ):
        return False

    pool = _physical_triton_pool()
    if pool is None or getattr(pool, "page_size", None) != 1:
        return False
    cache = pool.get_key_buffer(attn.attn_mqa.layer_id)
    value = pool.get_value_buffer(attn.attn_mqa.layer_id)
    return (
        cache.dtype == torch.float8_e4m3fn
        and tuple(cache.shape) == (655361, 1, 576)
        and cache.stride() == (576, 576, 1)
        and cache.storage_offset() == 0
        and cache.device == query.device
        and tuple(value.shape) == (655361, 1, 512)
        and value.stride() == (576, 576, 1)
        and value.storage_offset() == 0
        and value.dtype == cache.dtype
        and value.untyped_storage().data_ptr() == cache.untyped_storage().data_ptr()
    )


def run(query, latent, key_tail, weight, locations, cache):
    from sglang.kernels.ops.attention.mla_gluon import kc_cache

    name = entrypoint_name(query.shape[0])
    if name is None:
        raise ValueError(f"Unqualified Kimi-K3 MLA KC/cache M={query.shape[0]}")
    return getattr(kc_cache, name)(query, latent, key_tail, weight, locations, cache)


def apply(attn, query, latent, key_tail, forward_batch):
    """Run once and validate outputs after the cache-mutating launch."""
    from sglang.srt.model_executor.forward_context import get_token_to_kv_pool

    rows = query.shape[0]
    cache = get_token_to_kv_pool().get_key_buffer(attn.attn_mqa.layer_id)
    qcat, fresh = run(
        query,
        latent.view(rows, 512),
        key_tail.view(rows, 64),
        attn.w_kc,
        forward_batch.out_cache_loc,
        cache.view(655361, 576),
    )
    if not (
        qcat.dtype == fresh.dtype == torch.bfloat16
        and tuple(qcat.shape) == (rows, 12, 576)
        and tuple(fresh.shape) == (rows, 576)
        and qcat.is_contiguous()
        and fresh.is_contiguous()
        and qcat.device == fresh.device == query.device
        and qcat.untyped_storage().data_ptr()
        not in {
            query.untyped_storage().data_ptr(),
            latent.untyped_storage().data_ptr(),
            key_tail.untyped_storage().data_ptr(),
            attn.w_kc.untyped_storage().data_ptr(),
            cache.untyped_storage().data_ptr(),
        }
        and fresh.untyped_storage().data_ptr()
        not in {
            query.untyped_storage().data_ptr(),
            latent.untyped_storage().data_ptr(),
            key_tail.untyped_storage().data_ptr(),
            attn.w_kc.untyped_storage().data_ptr(),
            cache.untyped_storage().data_ptr(),
        }
    ):
        raise RuntimeError("Kimi-K3 MLA KC/cache output ABI changed")
    return qcat, fresh, latent
