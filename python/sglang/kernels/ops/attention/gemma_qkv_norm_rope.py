# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0.
"""In-place Gemma QKV normalization and partial NeoX RoPE."""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.fused_qk_rmsnorm_rope_gate import (
    _fused_qk_rmsnorm_rope_gate_kernel,
)
from sglang.kernels.ops.layernorm.gemma4_fused_ops import _gemma_qkv_rmsnorm_store


@triton.jit
def _gemma_qkv_norm_rope(
    Q,
    K,
    V,
    QW,
    KW,
    CACHE,
    POS,
    SQ: tl.constexpr,
    SK: tl.constexpr,
    SV: tl.constexpr,
    SC: tl.constexpr,
    HQ: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    R: tl.constexpr,
    EPS: tl.constexpr,
    FP16: tl.constexpr,
    CUDA_ROPE_FMA: tl.constexpr,
    BLOCK: tl.constexpr,
    ROT_BLOCK: tl.constexpr,
):
    head = tl.program_id(1)
    if head < HQ + HK:
        _fused_qk_rmsnorm_rope_gate_kernel(
            Q,
            K,
            Q,
            K,
            Q,
            QW,
            KW,
            CACHE,
            POS,
            POS,
            SQ,
            SK,
            SQ,
            SK,
            SQ,
            SC,
            1,
            HQ,
            HK,
            D,
            R,
            R // 2,
            BLOCK,
            ROT_BLOCK,
            EPS,
            FP16,
            R < D,
            False,
            False,
            False,
            WEIGHT_SHIFT=0.0,
            EXPLICIT_ROPE_FMA=CUDA_ROPE_FMA,
            EXPLICIT_XPU_ROPE_FMA=not CUDA_ROPE_FMA,
        )
    else:
        cols = tl.arange(0, BLOCK)
        _gemma_qkv_rmsnorm_store(
            V,
            QW,
            SV,
            tl.program_id(0),
            head - HQ - HK,
            cols,
            cols < D,
            D,
            EPS,
            False,
        )


def gemma_qkv_norm_rope(
    q,
    k,
    v,
    q_weight,
    k_weight,
    cache,
    positions,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    eps,
):
    """Normalize Q/K with weights and V without weights, then rotate Q/K.

    When K/V are shared from an earlier layer, pass None for k, v, and
    k_weight to normalize and rotate only Q. BF16/FP16 rounding after the
    normalization is preserved before the FP32 rotary calculation.
    """
    if (k is None) != (v is None):
        raise ValueError("k and v must both be tensors or both be None")
    if k is None:
        num_kv_heads = 0
    is_cuda = q.device.type == "cuda"
    _gemma_qkv_norm_rope[(q.shape[0], num_q_heads + 2 * num_kv_heads)](
        q,
        k if k is not None else q,
        v if v is not None else q,
        q_weight,
        k_weight if k_weight is not None else q_weight,
        cache,
        positions,
        q.stride(0),
        k.stride(0) if k is not None else 0,
        v.stride(0) if v is not None else 0,
        cache.stride(0),
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        eps,
        q.dtype == torch.float16,
        is_cuda,
        triton.next_power_of_2(head_dim),
        triton.next_power_of_2(rotary_dim // 2),
        enable_fp_fusion=is_cuda,
    )
