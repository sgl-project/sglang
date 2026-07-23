"""
Optimized Triton MLA Decode Kernels for DeepSeek V4.

This module provides optimized sparse attention decode.

Key optimizations:
1. Fused gather+dequant+attention kernels (eliminates intermediate buffers)
2. Split-K for better GPU parallelism on small batches
3. Proper dispatch: no-splitk for large batches, split-K for small batches
4. All paths use fused kernels (no 2-phase fallback)

Note: This implementation assumes KV cache is always FP8 quantized.
"""

from typing import Optional, Tuple

import torch

from .triton_mla_kernels_decode_fused import (
    DSV4_D_QK,
    fused_gather_attn_decode_dsv4,
    fused_gather_attn_decode_dsv4_dual_scope,
    fused_gather_attn_decode_dsv4_dual_scope_low_overhead,
)


def _should_use_fused_splitk(total_tokens: int, h_q: int, total_topk: int) -> bool:
    """Determine whether to use fused split-K kernel (low overhead).

    The fused split-K kernel is preferred for small batch sizes because
    split-K provides better GPU utilization when the grid is small.

    This matches the original _should_use_fused_dual_scope() thresholds.
    """
    if total_tokens <= 4:
        return True
    if h_q <= 64 and total_topk <= 800:
        return total_tokens <= 256
    if h_q <= 64 and total_topk >= 1024:
        return total_tokens <= 128
    # h_q > 64 (e.g. h_q=128 when q is padded to full n_heads).
    if h_q > 64:
        if total_topk >= 400:
            return total_tokens <= 32
        else:
            return total_tokens <= 128
    return True


def triton_sparse_attn_decode(
    q: torch.Tensor,
    kv_scope,
    extra_kv_scope,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimized sparse attention decode for DeepSeek V4 (d_qk=512)."""
    d_qk = q.shape[-1]

    if d_qk != DSV4_D_QK:
        raise ValueError(
            f"Unsupported d_qk: {d_qk}. Expected {DSV4_D_QK} (DeepSeek V4)"
        )

    return _triton_sparse_attn_decode_dsv4(
        q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink
    )


def _triton_sparse_attn_decode_dsv4(
    q: torch.Tensor,
    kv_scope,
    extra_kv_scope,
    sm_scale: float,
    d_v: int,
    attn_sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse attention decode for DeepSeek V4 (d_qk=512).

    All paths use fused kernels (no 2-phase fallback).

    Dispatch logic:
    - Single scope: always use fused kernel
    - Dual scope, small total_tokens: fused split-K kernel (low overhead)
    - Dual scope, otherwise: fused no-splitk kernel
    """
    b, s_q, h_q, d_qk = q.shape
    total_tokens = b * s_q

    topk_main = kv_scope.indices_in_kvcache.shape[-1]
    kv_quantized_main = kv_scope.blocked_k_quantized
    block_size_main = kv_scope.blocked_k.shape[1]

    # Single scope case
    if extra_kv_scope is None:
        q_reshaped = q.reshape(total_tokens, h_q, d_qk).contiguous()
        indices_main = kv_scope.indices_in_kvcache.reshape(
            total_tokens, topk_main
        ).contiguous()

        output, lse = fused_gather_attn_decode_dsv4(
            q_reshaped,
            kv_quantized_main,
            indices_main,
            block_size_main,
            sm_scale,
            topk_length=kv_scope.topk_length,
            attn_sink=attn_sink,
            s_q=s_q,
        )
        return output.view(b, s_q, h_q, d_v), lse.view(b, s_q, h_q).transpose(1, 2)

    # Dual scope case
    topk_extra = extra_kv_scope.indices_in_kvcache.shape[-1]
    total_topk = topk_main + topk_extra
    block_size_extra = extra_kv_scope.blocked_k.shape[1]

    q_reshaped = q.reshape(total_tokens, h_q, d_qk).contiguous()
    indices_main = kv_scope.indices_in_kvcache.reshape(
        total_tokens, topk_main
    ).contiguous()
    indices_extra = extra_kv_scope.indices_in_kvcache.reshape(
        total_tokens, topk_extra
    ).contiguous()

    # Dispatch: use split-K for small batches, no-splitk for everything else.
    if _should_use_fused_splitk(total_tokens, h_q, total_topk):
        # Small batch: fused split-K kernel (better GPU utilization)
        output, lse = fused_gather_attn_decode_dsv4_dual_scope_low_overhead(
            q_reshaped,
            kv_quantized_main,
            indices_main,
            block_size_main,
            extra_kv_scope.blocked_k_quantized,
            indices_extra,
            block_size_extra,
            sm_scale,
            topk_length_main=kv_scope.topk_length,
            topk_length_extra=extra_kv_scope.topk_length,
            attn_sink=attn_sink,
            s_q=s_q,
        )
    else:
        # Large batch / extend / prefill: fused no-splitk kernel
        output, lse = fused_gather_attn_decode_dsv4_dual_scope(
            q_reshaped,
            kv_quantized_main,
            indices_main,
            block_size_main,
            extra_kv_scope.blocked_k_quantized,
            indices_extra,
            block_size_extra,
            sm_scale,
            topk_length_main=kv_scope.topk_length,
            topk_length_extra=extra_kv_scope.topk_length,
            attn_sink=attn_sink,
            s_q=s_q,
            force_no_splitk=True,
        )

    return output.view(b, s_q, h_q, d_v), lse.view(b, s_q, h_q).transpose(1, 2)
