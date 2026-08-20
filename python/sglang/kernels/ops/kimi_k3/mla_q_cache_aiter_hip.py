"""Fail-closed adapter for Kimi-K3's fused MLA Q materialization and cache write."""

from __future__ import annotations

import os

import torch

from sglang.srt.utils import is_hip

_KV_LORA_RANK = 512
_PE_DIM = 64
_HEAD_DIM = _KV_LORA_RANK + _PE_DIM
_FP8_DTYPES = tuple(
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
    )
    if dtype is not None
)


def enabled() -> bool:
    return os.environ.get("SGLANG_K3_AITER_MLA_Q_CACHE_FUSION", "0").lower() in (
        "1",
        "true",
    )


def _op():
    try:
        import aiter

        return aiter.fused_qk_rope_concat_and_cache_mla
    except (AttributeError, ImportError, ModuleNotFoundError):
        return None


def supports_compute_all_q_rope() -> bool:
    try:
        schema = torch.ops.aiter.fused_qk_rope_concat_and_cache_mla.default._schema
    except (AttributeError, RuntimeError):
        return False
    return "compute_all_q_rope" in str(schema)


def _is_gfx950() -> bool:
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        return get_gfx_runtime() == "gfx950"
    except (AssertionError, ImportError, KeyError, RuntimeError):
        return False


def available(device: torch.device | None = None) -> bool:
    if (
        not enabled()
        or not is_hip()
        or not torch.cuda.is_available()
        or not _is_gfx950()
        or _op() is None
        or not supports_compute_all_q_rope()
    ):
        return False
    return device is None or device.type == "cuda"


def covered(
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
    if not available(q_nope.device) or q_nope.ndim != 3:
        return False
    tokens, heads, _ = q_nope.shape
    q_scale = k_scale if q_scale is None else q_scale
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
        and kv_cache.ndim in (3, 4)
        and kv_cache.shape[-1] == _HEAD_DIM
        and kv_cache.dtype in (torch.bfloat16, *_FP8_DTYPES)
        and slot_mapping.shape == (tokens,)
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
        and out.shape == (tokens, heads, _HEAD_DIM)
        # AITER decode consumes FP8 Q alongside FP8 KV, while Triton decode
        # keeps Q in BF16 and only stores KV as FP8. The AITER fused operator
        # supports q_out and kv_cache with independent dtypes.
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
    q_scale = k_scale if q_scale is None else q_scale
    if not covered(
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        kv_cache,
        slot_mapping,
        positions,
        k_scale,
        cos_cache,
        sin_cache,
        out,
        q_scale,
    ):
        raise NotImplementedError(
            "Kimi-K3 fused MLA Q/cache requires contiguous gfx950 BF16 inputs, "
            "a page-compatible 576-wide BF16/FP8 cache, int64 slot/position "
            "vectors, scalar FP32 scale, identity RoPE buffers and BF16/FP8 output"
        )
    op = _op()
    if op is None:
        raise RuntimeError("AITER fused MLA Q/cache operator is unavailable")
    op(
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        kv_cache,
        out,
        slot_mapping,
        k_scale,
        q_scale,
        positions,
        cos_cache,
        sin_cache,
        True,
        True,
        compute_all_q_rope=False,
    )
    return out
