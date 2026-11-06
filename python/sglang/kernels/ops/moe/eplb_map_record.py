from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


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
    WRITE_IDS: tl.constexpr,
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
    if WRITE_IDS:
        output_id = tl.where(valid_logical, physical_id, logical_id)
        tl.store(out_ids_ptr + offs, output_id, mask=mask)

    valid_row = mask
    if HAS_VALID_COUNT:
        num_valid_tokens = tl.load(num_valid_tokens_ptr).to(tl.int64)
        valid_row &= (offs // TOP_K) < num_valid_tokens
    in_range = (
        valid_row & valid_logical & (physical_id >= 0) & (physical_id < num_physical)
    )
    bin_idx = tl.where(in_range, physical_id, num_physical).to(tl.int32)
    histogram = tl.histogram(bin_idx, NUM_BINS)
    bins = tl.arange(0, NUM_BINS)
    tl.atomic_add(load_ptr + bins, histogram, mask=bins < num_physical)


def _launch_map_record_hist(
    topk_ids: torch.Tensor,
    load_buffer: torch.Tensor,
    *,
    dispatch: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    num_token_non_padded: Optional[torch.Tensor] = None,
) -> None:
    num_physical = load_buffer.numel()
    numel = topk_ids.numel()
    _eplb_map_record_hist_kernel[(triton.cdiv(numel, 256),)](
        topk_ids,
        dispatch if dispatch is not None else topk_ids,
        output if output is not None else topk_ids,
        load_buffer,
        num_token_non_padded if num_token_non_padded is not None else topk_ids,
        dispatch.numel() if dispatch is not None else num_physical,
        num_physical,
        numel,
        HAS_DISPATCH=dispatch is not None,
        HAS_VALID_COUNT=num_token_non_padded is not None,
        WRITE_IDS=output is not None,
        BLOCK=256,
        NUM_BINS=1 << num_physical.bit_length(),
        TOP_K=topk_ids.shape[-1],
    )


def eplb_map_and_record(
    topk_ids: torch.Tensor,
    dispatch: Optional[torch.Tensor],
    load_buffer: torch.Tensor,
    num_token_non_padded: Optional[torch.Tensor],
) -> torch.Tensor:
    topk_ids = topk_ids.contiguous()
    output = torch.empty_like(topk_ids)
    _launch_map_record_hist(
        topk_ids,
        load_buffer,
        dispatch=dispatch,
        output=output,
        num_token_non_padded=num_token_non_padded,
    )
    return output


def record_physical_counts(topk_ids: torch.Tensor, load_buffer: torch.Tensor) -> None:
    _launch_map_record_hist(topk_ids.contiguous(), load_buffer)


__all__ = ["eplb_map_and_record", "record_physical_counts"]
