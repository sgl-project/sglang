"""Fail-closed AITER Triton adapter for Kimi-K3's fused MLA Q materialization and cache write.

The no-RoPE form of the gfx950 HIP op in mla_q_cache_aiter_hip, for the ROCm
targets that lack it. Covers BF16 inputs, kv_lora_rank 512, rope dim 64, one KV
head, a [slots, 1, 576] BF16 or platform-e4m3 cache and int32/int64 slots; not
gfx1250, where AITER has no no-RoPE kernel. run() rejects anything else.
"""

from __future__ import annotations

import functools

import torch

from sglang.kernels.arch import current_arch

_KV_LORA_RANK = 512
_PE_DIM = 64
_HEAD_DIM = _KV_LORA_RANK + _PE_DIM


@functools.cache
def _op():
    """(fused_qk_cat_and_cache_mla, AITER's e4m3 dtype), or None for an older AITER."""
    try:
        from aiter.ops.triton.fusions.fused_kv_cache import fused_qk_cat_and_cache_mla
        from aiter.ops.triton.utils.types import e4m3_dtype
    except ImportError:
        return None
    return fused_qk_cat_and_cache_mla, e4m3_dtype


@functools.cache
def available() -> bool:
    if torch.version.hip is None or not torch.cuda.is_available():
        return False
    hip = current_arch().hip
    # AITER asserts on gfx1250: its gluon kernel stages cos/sin and has no no-RoPE form.
    return hip is not None and hip.gcn_arch != "gfx1250" and _op() is not None


def _covered(
    *,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
) -> bool:
    if not available() or q_nope.device.type != "cuda" or q_nope.ndim != 3:
        return False
    tokens, heads, _ = q_nope.shape
    return (
        tokens > 0
        and q_nope.shape == (tokens, heads, _KV_LORA_RANK)
        and q_pe.shape == (tokens, heads, _PE_DIM)
        and k_nope.shape == (tokens, 1, _KV_LORA_RANK)
        and k_pe.shape == (tokens, 1, _PE_DIM)
        and all(x.dtype == torch.bfloat16 for x in (q_nope, q_pe, k_nope, k_pe))
        and all(
            x.device == q_nope.device
            for x in (q_pe, k_nope, k_pe, kv_cache, slot_mapping, k_scale)
        )
        # AITER asserts on any other cache dtype or layout.
        and kv_cache.ndim == 3
        and kv_cache.shape[1] == 1
        and kv_cache.shape[2] == _HEAD_DIM
        and kv_cache.stride(2) == 1
        and kv_cache.dtype in (torch.bfloat16, _op()[1])
        and slot_mapping.shape == (tokens,)
        and slot_mapping.dtype in (torch.int32, torch.int64)
        # The kernel indexes slot_mapping without a stride.
        and slot_mapping.stride(0) == 1
        and k_scale.numel() == 1
        and k_scale.dtype == torch.float32
    )


def run(
    *,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
) -> torch.Tensor | None:
    """Return BF16 cat(q_nope, q_pe) after writing cat(k_nope, k_pe) / k_scale
    into kv_cache at every slot >= 0.

    Returns None without launching when the kernel is unavailable or the inputs are
    outside the contract, so the caller keeps its unfused path.
    """
    if not _covered(
        q_nope=q_nope,
        q_pe=q_pe,
        k_nope=k_nope,
        k_pe=k_pe,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        k_scale=k_scale,
    ):
        return None
    q, _, _, _ = _op()[0](
        q_nope,
        q_pe,
        k_nope,
        k_pe,
        kv_cache,
        slot_mapping,
        k_scale,
        q_out_dtype=q_nope.dtype,
    )
    return q
