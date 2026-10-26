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
            EXPLICIT_ROPE_FMA=True,
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
    """Normalize Q/K with weights and V without weights, then rotate Q/K."""
    _gemma_qkv_norm_rope[(q.shape[0], num_q_heads + 2 * num_kv_heads)](
        q,
        k,
        v,
        q_weight,
        k_weight,
        cache,
        positions,
        q.stride(0),
        k.stride(0),
        v.stride(0),
        cache.stride(0),
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        eps,
        q.dtype == torch.float16,
        triton.next_power_of_2(head_dim),
        triton.next_power_of_2(rotary_dim // 2),
    )
