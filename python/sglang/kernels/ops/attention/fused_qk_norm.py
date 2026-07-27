"""Fused Q/K RMSNorm in a single Triton kernel launch.

Ported from ATOM (atom/model_ops/layernorm.py). Fuses per-head Q RMSNorm
(optionally weightless) and KV RMSNorm into one kernel, halving the number
of norm kernel launches per attention layer.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_qk_norm_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    eps,
    num_tokens,
    head_dim,
    q_in_stride0,
    k_in_stride0,
    q_out_stride0,
    k_out_stride0,
    num_q_heads,
    num_k_heads,
    Q_HAS_WEIGHT: tl.constexpr,
    RBLOCK: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    num_q_rows = num_tokens * num_q_heads
    total_rows = num_tokens * (num_q_heads + num_k_heads)

    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < total_rows
    cols = tl.arange(0, RBLOCK)[None, :]
    col_mask = cols < head_dim

    is_q = xindex < num_q_rows
    row_in_section = tl.where(is_q, xindex, xindex - num_q_rows)
    cur_num_heads = tl.where(is_q, num_q_heads, num_k_heads)

    tokens = row_in_section // cur_num_heads
    heads = row_in_section % cur_num_heads

    in_stride = tl.where(is_q, q_in_stride0, k_in_stride0)
    in_bases = tokens * in_stride + heads * head_dim

    out_stride0 = tl.where(is_q, q_out_stride0, k_out_stride0)
    out_bases = tokens * out_stride0 + heads * head_dim

    mask = xmask & col_mask

    if Q_HAS_WEIGHT:
        qw = tl.load(
            q_weight_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
    else:
        qw = tl.full((RBLOCK,), 1.0, tl.float32)
    kw = tl.load(
        k_weight_ptr + cols, mask=col_mask, other=0.0, eviction_policy="evict_last"
    ).to(tl.float32)
    w = tl.where(is_q, qw, kw)

    x = tl.load(
        q_ptr + in_bases + cols,
        mask=mask & is_q,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    x = x + tl.load(
        k_ptr + in_bases + cols,
        mask=mask & ~is_q,
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)

    var = tl.sum(x * x, 1)[:, None]
    rstd = tl.rsqrt(var / head_dim + eps)

    out = (x * rstd * w).to(q_out_ptr.dtype.element_ty)
    tl.store(
        q_out_ptr + out_bases + cols,
        out,
        mask=mask & is_q,
        eviction_policy="evict_first",
    )
    tl.store(
        k_out_ptr + out_bases + cols,
        out,
        mask=mask & ~is_q,
        eviction_policy="evict_first",
    )


def fused_qk_norm(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: Optional[torch.Tensor],
    k_weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused Q/K RMSNorm in a single Triton kernel launch.

    Args:
        q: [num_tokens, num_heads, head_dim]
        k: [num_tokens, num_kv_heads, head_dim]
        q_weight: [head_dim] norm weight, or None for weightless Q norm
        k_weight: [head_dim] norm weight (always required)
        eps: epsilon for numerical stability

    Returns:
        (q_normed, k_normed) same shapes as inputs
    """
    head_dim = k_weight.shape[0]
    if q_weight is not None:
        assert q_weight.shape[0] == head_dim
    num_tokens = q.shape[0]
    num_q_heads = q.shape[1]
    num_k_heads = k.shape[1]
    total_rows = num_tokens * (num_q_heads + num_k_heads)
    RBLOCK = triton.next_power_of_2(head_dim)

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    XBLOCK = 2 if total_rows > 8192 else 1
    NUM_WARPS = 1
    q_weight_arg = q_weight if q_weight is not None else k_weight
    _fused_qk_norm_kernel[((total_rows + XBLOCK - 1) // XBLOCK,)](
        q,
        k,
        q_out,
        k_out,
        q_weight_arg,
        k_weight,
        eps,
        num_tokens,
        head_dim,
        q.stride(0),
        k.stride(0),
        q_out.stride(0),
        k_out.stride(0),
        num_q_heads,
        num_k_heads,
        Q_HAS_WEIGHT=q_weight is not None,
        RBLOCK=RBLOCK,
        XBLOCK=XBLOCK,
        num_warps=NUM_WARPS,
    )
    return q_out, k_out
