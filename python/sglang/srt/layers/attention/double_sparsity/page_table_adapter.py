"""Double Sparsity token-to-physical-slot adapter.

Translates the DS selector's logical-token output ``selected_indices`` into the
physical token index tensor that the FlashMLA ``flashmla_kv`` sparse path
consumes, via a single ``req_to_token`` gather:
    physical_slots[b, k] = req_to_token[req_pool_indices[b], logical_topk[b, k]]
with ``-1`` preserved for padding slots and for rows whose ``req_pool_indices``
is out of range (mapped to ``-1`` rather than read out of bounds).
"""

from __future__ import annotations

import torch

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _logical_to_physical_kernel(
        selected_ptr,  # [bs, max_top_k] int32 (logical positions, -1 padded)
        rpi_ptr,  # [bs] int32 OR int64 (Triton casts on load)
        rtt_ptr,  # [num_pools, max_seqlen] int32
        out_ptr,  # [bs, max_top_k] int32 — written in place
        num_pools: tl.constexpr,
        max_seqlen: tl.constexpr,
        bs: tl.constexpr,
        max_top_k: tl.constexpr,
        sel_stride_b: tl.constexpr,
        rtt_stride_p: tl.constexpr,
        out_stride_b: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        b = tl.program_id(0)
        kb = tl.program_id(1)
        k_offs = kb * BLOCK_K + tl.arange(0, BLOCK_K)
        k_in_range = k_offs < max_top_k

        s = tl.load(
            selected_ptr + b * sel_stride_b + k_offs,
            mask=k_in_range,
            other=-1,
        ).to(tl.int32)
        is_pad = s < 0

        pool = tl.load(rpi_ptr + b).to(tl.int64)
        bad_pool = (pool < 0) | (pool >= num_pools)
        safe_pool = tl.minimum(tl.maximum(pool, 0), num_pools - 1)
        safe_s = tl.maximum(s, 0).to(tl.int64)
        phys = tl.load(
            rtt_ptr + safe_pool * rtt_stride_p + safe_s,
            mask=k_in_range,
            other=-1,
        ).to(tl.int32)
        result = tl.where(is_pad | bad_pool, tl.full(phys.shape, -1, tl.int32), phys)
        tl.store(
            out_ptr + b * out_stride_b + k_offs,
            result,
            mask=k_in_range,
        )


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def logical_to_physical(
    selected_indices: torch.Tensor,  # int32 [bs, max_top_k] logical positions, -1 padded
    req_pool_indices: torch.Tensor,  # int32 or int64 [bs]
    req_to_token: torch.Tensor,  # int32 [num_pools, max_seqlen]
    out: torch.Tensor,  # int32 [bs, max_top_k]  pre-allocated output
) -> None:
    """Convert logical token positions to physical KV-cache slot indices.

    Writes into the pre-allocated ``out`` tensor (capture-safe, no allocation,
    no host sync). On CUDA with Triton, runs an allocation-free kernel; otherwise
    a torch fallback. Out-of-range ``req_pool_indices`` rows are mapped to ``-1``.
    """
    bs, max_top_k = selected_indices.shape
    if bs == 0:
        out.fill_(-1)
        return

    if _TRITON_AVAILABLE and out.is_cuda:
        block_k = _next_pow2(max_top_k)
        grid = (bs, (max_top_k + block_k - 1) // block_k)
        _logical_to_physical_kernel[grid](
            selected_indices,
            req_pool_indices,
            req_to_token,
            out,
            num_pools=int(req_to_token.shape[0]),
            max_seqlen=int(req_to_token.shape[1]),
            bs=bs,
            max_top_k=max_top_k,
            sel_stride_b=selected_indices.stride(0),
            rtt_stride_p=req_to_token.stride(0),
            out_stride_b=out.stride(0),
            BLOCK_K=block_k,
        )
        return

    # Torch fallback (allocating; used on CPU + when Triton is unavailable).
    is_valid = selected_indices >= 0
    safe_positions = selected_indices.clamp(min=0)
    num_pools = req_to_token.shape[0]
    bad_pool = (req_pool_indices < 0) | (req_pool_indices >= num_pools)
    safe_pool = req_pool_indices.clamp(0, max(num_pools - 1, 0)).long()
    pool_expanded = safe_pool.unsqueeze(1).expand(-1, max_top_k)
    physical = req_to_token[pool_expanded, safe_positions.long()]
    pad_mask = ~is_valid | bad_pool.unsqueeze(1)
    out.copy_(torch.where(pad_mask, torch.full_like(physical, -1), physical))
