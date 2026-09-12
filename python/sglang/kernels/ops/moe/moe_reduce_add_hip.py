"""MoE top-k reduction with the shared-expert add folded in (ROCm): fp32 over the slots in
ascending order as aiter's ``moe_reduction_kernel`` sums them, one rounding at the end."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_topk_reduce_add_kernel(
    x_ptr,  # [M * TOPK, D] per-slot expert outputs
    shared_ptr,  # [M, D]
    out_ptr,  # [M, D]
    topk_ids_ptr,  # [M, TOPK] (HAS_MASK)
    expert_mask_ptr,  # [num_experts] (HAS_MASK): nonzero = local expert
    D,
    stride_xm,
    stride_sm,
    stride_om,
    alpha,
    TOPK: tl.constexpr,
    HAS_MASK: tl.constexpr,
    HAS_ALPHA: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < D
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        v = tl.load(
            x_ptr + (row * TOPK + k) * stride_xm + offs, mask=mask, other=0.0
        ).to(tl.float32)
        if HAS_MASK:
            # rows of experts this rank does not hold were never written
            e = tl.load(topk_ids_ptr + row * TOPK + k)
            local = tl.load(expert_mask_ptr + e) != 0
            v = tl.where(local, v, 0.0)
        acc += v
    if HAS_ALPHA:
        acc *= alpha
    acc += tl.load(shared_ptr + row * stride_sm + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    tl.store(
        out_ptr + row * stride_om + offs, acc.to(out_ptr.dtype.element_ty), mask=mask
    )


def moe_topk_reduce_add(
    per_slot: torch.Tensor,
    shared: torch.Tensor,
    out: torch.Tensor,
    topk: int,
    topk_ids: Optional[torch.Tensor] = None,
    expert_mask: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
) -> None:
    """``out[t] = alpha * sum_k per_slot[t * topk + k] + shared[t]``, skipping slots whose expert
    ``topk_ids`` maps outside ``expert_mask``; fp32 accumulation rounded once to ``out``'s dtype."""
    M, D = out.shape
    assert shared.shape == (M, D), (shared.shape, out.shape)
    assert per_slot.numel() == M * topk * D, (per_slot.shape, M, topk, D)
    assert out.stride(1) == 1 and shared.stride(1) == 1
    x = per_slot.view(M * topk, D) if per_slot.dim() != 2 else per_slot
    assert x.stride(1) == 1
    has_mask = expert_mask is not None
    if has_mask:
        assert topk_ids is not None and topk_ids.shape == (M, topk)
        topk_ids = topk_ids.contiguous()
        expert_mask = expert_mask.contiguous()
    if M == 0:
        return
    block_d = min(1024, triton.next_power_of_2(D))
    _moe_topk_reduce_add_kernel[(M, triton.cdiv(D, block_d))](
        x,
        shared,
        out,
        topk_ids if has_mask else out,
        expert_mask if has_mask else out,
        D,
        x.stride(0),
        shared.stride(0),
        out.stride(0),
        float(alpha),
        TOPK=topk,
        HAS_MASK=has_mask,
        HAS_ALPHA=float(alpha) != 1.0,
        BLOCK_D=block_d,
        num_warps=4,
    )
