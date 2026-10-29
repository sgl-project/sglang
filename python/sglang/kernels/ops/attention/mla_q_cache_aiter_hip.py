"""Fail-closed adapter for Kimi-K3's fused MLA Q materialization and cache write."""

from __future__ import annotations

import functools

import torch

# Kimi-K3 MLA widths (kv_lora_rank, qk_rope_head_dim); other widths are not covered.
_KV_LORA_RANK = 512
_PE_DIM = 64
_HEAD_DIM = _KV_LORA_RANK + _PE_DIM
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


@functools.cache
def _op():
    try:
        import aiter

        return aiter.fused_qk_rope_concat_and_cache_mla
    except (AttributeError, ImportError):
        return None


@functools.cache
def _is_gfx950() -> bool:
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        return get_gfx_runtime() == "gfx950"
    except (AssertionError, ImportError, KeyError, RuntimeError):
        return False


@functools.cache
def available() -> bool:
    return (
        torch.version.hip is not None
        and torch.cuda.is_available()
        and _is_gfx950()
        and _op() is not None
    )


def covered(
    *,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    k_scale: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    out: torch.Tensor,
    q_scale: torch.Tensor | None = None,
) -> bool:
    if not available() or q_nope.device.type != "cuda" or q_nope.ndim != 3:
        return False
    tokens, heads, _ = q_nope.shape
    q_scale = k_scale if q_scale is None else q_scale
    tensors = (
        q_pe,
        k_nope,
        k_pe,
        kv_cache,
        slot_mapping,
        positions,
        k_scale,
        q_scale,
        cos_cache,
        sin_cache,
        out,
    )
    return (
        tokens > 0
        and q_nope.shape == (tokens, heads, _KV_LORA_RANK)
        and q_pe.shape == (tokens, heads, _PE_DIM)
        and k_nope.shape == (tokens, 1, _KV_LORA_RANK)
        and k_pe.shape == (tokens, 1, _PE_DIM)
        and all(
            x.dtype == torch.bfloat16 and x.stride(-1) == 1
            for x in (q_nope, q_pe, k_nope, k_pe)
        )
        and all(x.device == q_nope.device for x in tensors)
        and kv_cache.ndim in (3, 4)
        and kv_cache.shape[-1] == _HEAD_DIM
        and kv_cache.dtype in (torch.bfloat16, *_FP8_DTYPES)
        and slot_mapping.shape == (tokens,)
        # AITER's gfx950 kernels take one int64 slot and position per token.
        and slot_mapping.dtype == torch.int64
        and slot_mapping.stride(0) == 1
        and positions.shape == (tokens,)
        and positions.dtype == torch.int64
        and positions.stride(0) == 1
        and k_scale.numel() == 1
        and k_scale.dtype == torch.float32
        and k_scale.device == q_nope.device
        and q_scale.numel() == 1
        and q_scale.dtype == torch.float32
        and q_scale.device == q_nope.device
        and cos_cache.shape == (1, _PE_DIM // 2)
        and sin_cache.shape == (1, _PE_DIM // 2)
        and cos_cache.dtype == torch.bfloat16
        and sin_cache.dtype == torch.bfloat16
        and cos_cache.device == q_nope.device
        and sin_cache.device == q_nope.device
        and out.ndim == 3
        and out.shape[0] == tokens
        # Extra out heads stay untouched,
        # so 12-head K3 can write a 16-head MLA pad buffer.
        and out.shape[1] >= heads
        and out.shape[2] == _HEAD_DIM
        # AITER decode wants FP8 Q with FP8 KV,
        # Triton decode wants BF16 Q with FP8 KV.
        and out.dtype in (q_nope.dtype, kv_cache.dtype)
        and out.is_contiguous()
    )


def run(
    *,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    k_scale: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    out: torch.Tensor,
    q_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Write the latent KV cache and MLA query for the supported gfx950 contract."""
    if not available():
        raise RuntimeError("fused MLA Q/cache requires AITER on gfx950")
    if not covered(
        q_nope=q_nope,
        q_pe=q_pe,
        k_nope=k_nope,
        k_pe=k_pe,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        positions=positions,
        k_scale=k_scale,
        cos_cache=cos_cache,
        sin_cache=sin_cache,
        out=out,
        q_scale=q_scale,
    ):
        raise ValueError("unsupported inputs for fused MLA Q/cache")
    op = _op()
    if op is None:
        raise RuntimeError("fused MLA Q/cache AITER operator is unavailable")
    op(
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        kv_cache,
        out,
        slot_mapping,
        k_scale,
        k_scale if q_scale is None else q_scale,
        positions,
        cos_cache,
        sin_cache,
        True,  # is_neox
        True,  # is_nope_first
        compute_all_q_rope=False,
    )
    return out
