from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _eplb_map_record_hist_kernel(
        topk_ids_ptr,
        dispatch_ptr,
        out_ids_ptr,
        load_ptr,
        num_valid_tokens_ptr,
        num_logical,
        num_physical,
        numel,
        HAS_DISPATCH: tl.constexpr,
        HAS_VALID_COUNT: tl.constexpr,
        BLOCK: tl.constexpr,
        NUM_BINS: tl.constexpr,
        TOP_K: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < numel
        logical_id = tl.load(topk_ids_ptr + offs, mask=mask, other=-1).to(tl.int64)
        valid_logical = (logical_id >= 0) & (logical_id < num_logical)
        safe_logical = tl.where(valid_logical, logical_id, 0)
        if HAS_DISPATCH:
            physical_id = tl.load(
                dispatch_ptr + safe_logical,
                mask=mask & valid_logical,
                other=0,
            ).to(tl.int64)
        else:
            physical_id = logical_id
        output_id = tl.where(valid_logical, physical_id, logical_id)
        tl.store(out_ids_ptr + offs, output_id, mask=mask)

        valid_row = mask
        if HAS_VALID_COUNT:
            num_valid_tokens = tl.load(num_valid_tokens_ptr).to(tl.int64)
            valid_row &= (offs // TOP_K) < num_valid_tokens
        in_range = (
            valid_row
            & valid_logical
            & (physical_id >= 0)
            & (physical_id < num_physical)
        )
        bin_idx = tl.where(in_range, physical_id, num_physical).to(tl.int32)
        histogram = tl.histogram(bin_idx, NUM_BINS)
        bins = tl.arange(0, NUM_BINS)
        tl.atomic_add(load_ptr + bins, histogram, mask=bins < num_physical)


def eplb_map_and_record_fused(
    topk_ids: torch.Tensor,
    dispatch_info: Optional[ExpertLocationDispatchInfo],
    load_buffer: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if not _HAS_TRITON or not topk_ids.is_cuda or topk_ids.numel() == 0:
        return None
    if dispatch_info is not None and dispatch_info.ep_dispatch_algorithm != "static":
        return None

    dispatch = (
        dispatch_info.partial_logical_to_rank_dispatch_physical_map
        if dispatch_info is not None
        else None
    )
    if dispatch_info is not None and dispatch is None:
        return None
    num_physical = int(load_buffer.numel())
    num_logical = int(dispatch.numel()) if dispatch is not None else num_physical
    top_k = int(topk_ids.shape[-1])
    numel = topk_ids.numel()
    output = torch.empty_like(topk_ids)
    dispatch_ptr = dispatch if dispatch is not None else topk_ids
    valid_count_ptr = (
        num_token_non_padded if num_token_non_padded is not None else topk_ids
    )
    grid = (triton.cdiv(numel, 256),)
    _eplb_map_record_hist_kernel[grid](
        topk_ids.contiguous(),
        dispatch_ptr,
        output,
        load_buffer,
        valid_count_ptr,
        num_logical,
        num_physical,
        numel,
        HAS_DISPATCH=dispatch is not None,
        HAS_VALID_COUNT=num_token_non_padded is not None,
        BLOCK=256,
        NUM_BINS=1 << num_physical.bit_length(),
        TOP_K=top_k,
    )
    return output


__all__ = ["eplb_map_and_record_fused"]
