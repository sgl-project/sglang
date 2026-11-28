from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _map_topk_ids_to_local_experts_kernel(
    topk_ids_ptr,
    mapping_ptr,
    out_ptr,
    numel,
    NUM_EXPERTS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    idx = tl.load(topk_ids_ptr + offs, mask=mask, other=-1)
    valid = mask & (idx >= 0) & (idx < NUM_EXPERTS)
    mapped = tl.load(mapping_ptr + tl.where(valid, idx, 0), mask=valid, other=-1)
    tl.store(out_ptr + offs, mapped.to(out_ptr.dtype.element_ty), mask=mask)


def map_topk_ids_to_local_experts(
    topk_ids: torch.Tensor,
    local_expert_mapping: torch.Tensor,
) -> torch.Tensor:
    """Map global expert ids to local ids via ``local_expert_mapping`` on CUDA."""
    assert topk_ids.is_cuda and topk_ids.is_contiguous()
    assert local_expert_mapping.is_cuda and local_expert_mapping.is_contiguous()
    numel = topk_ids.numel()
    out = torch.empty_like(topk_ids)
    if numel > 0:
        BLOCK_SIZE = 256
        grid = (triton.cdiv(numel, BLOCK_SIZE),)
        _map_topk_ids_to_local_experts_kernel[grid](
            topk_ids_ptr=topk_ids,
            mapping_ptr=local_expert_mapping,
            out_ptr=out,
            numel=numel,
            NUM_EXPERTS=local_expert_mapping.numel(),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=1,
        )
    return out


__all__ = ["map_topk_ids_to_local_experts"]
