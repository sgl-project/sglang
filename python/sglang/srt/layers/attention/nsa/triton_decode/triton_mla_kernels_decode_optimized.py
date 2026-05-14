"""
Optimized Triton MLA Decode Kernels for DeepSeek V4.

This module provides optimized sparse attention decode with reduced Python overhead.

Key optimizations:
1. Fused gather+dequant+attention kernels (eliminates intermediate buffers)
2. Split-K for better GPU parallelism on small batches
3. Pre-allocated buffer pool for splitk intermediate results
4. Pre-computed strides to reduce tensor metadata operations

Note: This implementation assumes KV cache is always FP8 quantized.
"""

from typing import Optional, Tuple

import torch
import triton

from .triton_mla_kernels_decode_common import (
    _bucket_total_tokens,
    _unified_sparse_decode_kernel,
    compute_token_ranges,
)
from .triton_mla_kernels_decode_dsv4 import (
    DSV4_D_QK,
    fused_gather_dequant_fp8_dsv4,
)
from .triton_mla_kernels_decode_fused import (
    fused_gather_attn_decode_dsv4,
    fused_gather_attn_decode_dsv4_dual_scope_low_overhead,
)


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


def _should_use_fused_dual_scope(total_tokens: int, h_q: int, total_topk: int) -> bool:
    """Determine whether to use fused kernel for dual-scope cases.

    The fused kernel avoids allocating a large intermediate gathered_kv
    buffer and eliminates a separate gather kernel launch.  However, for
    h_q > 64 with medium-to-large batch sizes and larger topk, the
    non-splitk fused kernel suffers from low GPU utilization (the grid
    has only cdiv(h_q, BLOCK_H) blocks in the H dimension).  In those
    cases the fallback (separate gather + attention) can be faster on
    the GPU, though it incurs extra torch.empty() overhead in CUDA
    graphs.

    The thresholds below were determined empirically on MI355X (256 CUs).
    """
    if total_tokens <= 4:
        return True
    if h_q <= 64 and total_topk <= 800:
        return total_tokens <= 256
    if h_q <= 64 and total_topk >= 1024:
        return total_tokens <= 128
    # h_q > 64 (e.g. h_q=128 when q is padded to full n_heads).
    # For small topk (c128 layers, topk~192), fused always wins.
    # For larger topk (c4 layers, topk~640), fused wins at small bs
    # but the fallback catches up at bs>=16 due to better GPU utilization.
    # However, the fallback has 4 extra torch.empty() calls that add
    # ~30us CUDA-graph replay overhead, roughly cancelling the GPU gain.
    # So we route to fused for all practical batch sizes.
    if h_q > 64:
        return total_tokens <= 256
    return True


def _triton_sparse_attn_decode_dsv4(
    q: torch.Tensor,
    kv_scope,
    extra_kv_scope,
    sm_scale: float,
    d_v: int,
    attn_sink: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Optimized sparse attention decode for DeepSeek V4 (d_qk=512)."""
    b, s_q, h_q, d_qk = q.shape
    total_tokens = b * s_q
    device = q.device

    topk_main = kv_scope.indices_in_kvcache.shape[-1]
    kv_quantized_main = kv_scope.blocked_k_quantized
    block_size_main = kv_scope.blocked_k.shape[1]

    # Single scope case
    if extra_kv_scope is None:
        if topk_main < 8192:
            q_reshaped = q.reshape(total_tokens, h_q, d_qk)
            if not q_reshaped.is_contiguous():
                q_reshaped = q_reshaped.contiguous()

            indices_main = kv_scope.indices_in_kvcache.reshape(total_tokens, topk_main)
            if not indices_main.is_contiguous():
                indices_main = indices_main.contiguous()

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
        else:
            from .triton_mla_kernels_decode_dsv4 import triton_sparse_attn_decode_dsv4

            return triton_sparse_attn_decode_dsv4(
                q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink
            )

    # Dual scope case
    topk_extra = extra_kv_scope.indices_in_kvcache.shape[-1]
    total_topk = topk_main + topk_extra

    # Check if chunking needed (fall back to original implementation)
    token_ranges = compute_token_ranges(total_tokens, total_topk, d_qk)
    if len(token_ranges) > 1:
        from .triton_mla_kernels_decode_dsv4 import triton_sparse_attn_decode_dsv4

        return triton_sparse_attn_decode_dsv4(
            q, kv_scope, extra_kv_scope, sm_scale, d_v, attn_sink
        )

    # Use fused dual-scope kernel with low-overhead buffer pool
    if _should_use_fused_dual_scope(total_tokens, h_q, total_topk):
        q_reshaped = q.reshape(total_tokens, h_q, d_qk)
        if not q_reshaped.is_contiguous():
            q_reshaped = q_reshaped.contiguous()

        indices_main = kv_scope.indices_in_kvcache.reshape(total_tokens, topk_main)
        if not indices_main.is_contiguous():
            indices_main = indices_main.contiguous()

        block_size_extra = extra_kv_scope.blocked_k.shape[1]
        indices_extra = extra_kv_scope.indices_in_kvcache.reshape(
            total_tokens, topk_extra
        )
        if not indices_extra.is_contiguous():
            indices_extra = indices_extra.contiguous()

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
        return output.view(b, s_q, h_q, d_v), lse.view(b, s_q, h_q).transpose(1, 2)

    # Fallback: Separate gather + attention path
    return _fallback_gather_attention(
        q,
        kv_scope,
        extra_kv_scope,
        sm_scale,
        d_v,
        attn_sink,
        total_tokens,
        h_q,
        d_qk,
        topk_main,
        topk_extra,
        block_size_main,
        kv_quantized_main,
        fused_gather_dequant_fp8_dsv4,
    )


def _fallback_gather_attention(
    q: torch.Tensor,
    kv_scope,
    extra_kv_scope,
    sm_scale: float,
    d_v: int,
    attn_sink: Optional[torch.Tensor],
    total_tokens: int,
    h_q: int,
    d_qk: int,
    topk_main: int,
    topk_extra: int,
    block_size_main: int,
    kv_quantized_main,
    fused_gather_fn,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback path: separate gather + attention kernels."""
    b = q.shape[0]
    s_q = q.shape[1]
    device = q.device
    total_topk = topk_main + topk_extra

    gathered_kv = torch.empty(
        total_tokens, total_topk, d_qk, dtype=torch.bfloat16, device=device
    )
    invalid_mask = torch.empty(
        total_tokens, total_topk, dtype=torch.bool, device=device
    )
    output = torch.empty(total_tokens, h_q, d_v, dtype=torch.bfloat16, device=device)
    lse = torch.empty(total_tokens, h_q, dtype=torch.float32, device=device)

    indices_main = kv_scope.indices_in_kvcache.reshape(total_tokens, topk_main)
    block_size_extra = extra_kv_scope.blocked_k.shape[1]
    indices_extra = extra_kv_scope.indices_in_kvcache.reshape(total_tokens, topk_extra)

    fused_gather_fn(
        kv_quantized_main,
        indices_main,
        block_size_main,
        kv_scope.topk_length,
        extra_kv_scope.blocked_k_quantized,
        indices_extra,
        block_size_extra,
        extra_kv_scope.topk_length,
        gathered_kv,
        invalid_mask,
        s_q,
    )

    if q.dtype == torch.bfloat16 and q.is_contiguous():
        q_reshaped = q.view(total_tokens, h_q, d_qk)
    else:
        q_reshaped = q.to(torch.bfloat16).reshape(total_tokens, h_q, d_qk)
        if not q_reshaped.is_contiguous():
            q_reshaped = q_reshaped.contiguous()

    HAS_ATTN_SINK = attn_sink is not None
    attn_sink_tensor = attn_sink if HAS_ATTN_SINK else lse[:1]

    grid = lambda meta: (total_tokens, triton.cdiv(h_q, meta["BLOCK_H"]))
    _unified_sparse_decode_kernel[grid](
        q_reshaped,
        gathered_kv,
        invalid_mask,
        attn_sink_tensor,
        output,
        lse,
        sm_scale,
        total_tokens,
        _bucket_total_tokens(total_tokens),
        h_q,
        total_topk,
        d_qk,
        d_v,
        q_reshaped.stride(0),
        q_reshaped.stride(1),
        q_reshaped.stride(2),
        gathered_kv.stride(0),
        gathered_kv.stride(1),
        gathered_kv.stride(2),
        invalid_mask.stride(0),
        invalid_mask.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        HAS_ATTN_SINK=HAS_ATTN_SINK,
    )

    return output.view(b, s_q, h_q, d_v), lse.view(b, s_q, h_q).transpose(1, 2)
