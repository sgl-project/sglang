"""Fused MiniMax-M3 sparse-cache store with the fp8 cast folded in.

One Triton launch per layer scales, casts and scatters the main K/V heads, the
index-K head and the optional index-V head into their token-major caches; it
serves the fp8-pool case that the raw-byte CUDA store (`minimax_store_kv_index`)
cannot, on CUDA and ROCm alike.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _store_kv_index_quant_kernel(
    k_ptr,
    v_ptr,
    kc_ptr,
    vc_ptr,
    ik_ptr,
    ikc_ptr,
    iv_ptr,
    ivc_ptr,
    loc_ptr,
    k_scale,
    v_scale,
    ik_scale,
    iv_scale,
    sk_t,
    sk_h,
    sv_t,
    sv_h,
    sik_t,
    siv_t,
    skc_t,
    skc_h,
    svc_t,
    svc_h,
    sikc_t,
    sivc_t,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    V_HEAD_DIM: tl.constexpr,
    IDX_DIM: tl.constexpr,
    HAS_IDX_V: tl.constexpr,
):
    t = tl.program_id(0)
    loc = tl.load(loc_ptr + t).to(tl.int64)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_dv = tl.arange(0, V_HEAD_DIM)
    for h in tl.static_range(NUM_KV_HEADS):
        k = tl.load(k_ptr + t * sk_t + h * sk_h + offs_d).to(tl.float32) / k_scale
        tl.store(
            kc_ptr + loc * skc_t + h * skc_h + offs_d, k.to(kc_ptr.dtype.element_ty)
        )
        v = tl.load(v_ptr + t * sv_t + h * sv_h + offs_dv).to(tl.float32) / v_scale
        tl.store(
            vc_ptr + loc * svc_t + h * svc_h + offs_dv, v.to(vc_ptr.dtype.element_ty)
        )
    offs_i = tl.arange(0, IDX_DIM)
    ik = tl.load(ik_ptr + t * sik_t + offs_i).to(tl.float32) / ik_scale
    tl.store(ikc_ptr + loc * sikc_t + offs_i, ik.to(ikc_ptr.dtype.element_ty))
    if HAS_IDX_V:
        iv = tl.load(iv_ptr + t * siv_t + offs_i).to(tl.float32) / iv_scale
        tl.store(ivc_ptr + loc * sivc_t + offs_i, iv.to(ivc_ptr.dtype.element_ty))


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


_SUPPORTED_CACHE_DTYPES = (
    torch.bfloat16,
    torch.float16,
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
)


def can_store_kv_index_quant(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_k: torch.Tensor,
    idx_k_cache: torch.Tensor,
    idx_v: Optional[torch.Tensor],
    idx_v_cache: Optional[torch.Tensor],
) -> bool:
    """Whether every (input, cache) pair has a layout the kernel can address."""

    def _pair_storable(x: torch.Tensor, cache: torch.Tensor) -> bool:
        return (
            x.dtype in (torch.bfloat16, torch.float16)
            and cache.dtype in _SUPPORTED_CACHE_DTYPES
            and x.dim() == 3
            and cache.dim() == 3
            and _is_pow2(cache.shape[2])
            and x.shape[2] == cache.shape[2]
            and x.stride(2) == 1
            and cache.stride(2) == 1
        )

    if not (_pair_storable(k, k_cache) and _pair_storable(v, v_cache)):
        return False
    if not _pair_storable(idx_k, idx_k_cache):
        return False
    if (idx_v is None) != (idx_v_cache is None):
        return False
    if idx_v is not None and not _pair_storable(idx_v, idx_v_cache):
        return False
    return True


def store_kv_index_quant(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_k: torch.Tensor,
    idx_k_cache: torch.Tensor,
    idx_v: Optional[torch.Tensor],
    idx_v_cache: Optional[torch.Tensor],
    loc: torch.Tensor,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    idx_k_scale: Optional[float] = None,
    idx_v_scale: Optional[float] = None,
) -> None:
    """`cache[loc] = (x / scale).to(cache.dtype)` for K, V, index-K and index-V in one launch.

    Inputs are `[tokens, heads, dim]` (index tensors have one head), caches
    `[slots, heads, dim]`, `loc` one slot per token. As in
    `MHATokenToKVPool.set_kv_buffer`, a scale applies only where the store
    casts; a `None` scale is unit.
    """
    T, H, D = k.shape
    Dv = v.shape[2]
    Di = idx_k.shape[2]
    if T == 0:
        return
    has_idx_v = idx_v is not None
    if not has_idx_v:
        idx_v, idx_v_cache = idx_k, idx_k_cache

    def _scale_if_cast(scale: Optional[float], x: torch.Tensor, cache: torch.Tensor):
        if scale is None or x.dtype == cache.dtype:
            return 1.0
        return float(scale)

    k_scale = _scale_if_cast(k_scale, k, k_cache)
    v_scale = _scale_if_cast(v_scale, v, v_cache)
    idx_k_scale = _scale_if_cast(idx_k_scale, idx_k, idx_k_cache)
    idx_v_scale = _scale_if_cast(idx_v_scale, idx_v, idx_v_cache)
    _store_kv_index_quant_kernel[(T,)](
        k,
        v,
        k_cache,
        v_cache,
        idx_k,
        idx_k_cache,
        idx_v,
        idx_v_cache,
        loc,
        k_scale,
        v_scale,
        idx_k_scale,
        idx_v_scale,
        k.stride(0),
        k.stride(1),
        v.stride(0),
        v.stride(1),
        idx_k.stride(0),
        idx_v.stride(0),
        k_cache.stride(0),
        k_cache.stride(1),
        v_cache.stride(0),
        v_cache.stride(1),
        idx_k_cache.stride(0),
        idx_v_cache.stride(0),
        NUM_KV_HEADS=H,
        HEAD_DIM=D,
        V_HEAD_DIM=Dv,
        IDX_DIM=Di,
        HAS_IDX_V=has_idx_v,
        # one program per token with a static head loop: sized for the few KV heads per rank
        num_warps=1,
    )
