from itertools import accumulate
from typing import List, Optional

import torch
import triton
import triton.language as tl


def transform_index_page_table_prefill(**kwargs):
    return transform_index_page_table_prefill_fast(**kwargs)


def transform_index_page_table_decode(**kwargs):
    return transform_index_page_table_decode_fast(**kwargs)


@triton.jit
def translate_owner_sharded_slots_kernel(
    slots_ptr: torch.Tensor,
    result_ptr: torch.Tensor,
    num_slots,
    OWNER_CP_SIZE: tl.constexpr,
    OWNER_PAGE_SIZE: tl.constexpr,
    OWNER_PAGES_PER_RANK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_slots
    logical_slots = tl.load(slots_ptr + offsets, mask=mask, other=-1)
    valid = logical_slots >= 0
    safe_slots = tl.where(valid, logical_slots, 0)
    logical_pages = safe_slots // OWNER_PAGE_SIZE
    page_offsets = safe_slots % OWNER_PAGE_SIZE
    owners = logical_pages % OWNER_CP_SIZE
    owner_pages = logical_pages // OWNER_CP_SIZE
    physical_pages = owners * OWNER_PAGES_PER_RANK + owner_pages
    physical_slots = physical_pages * OWNER_PAGE_SIZE + page_offsets
    tl.store(
        result_ptr + offsets,
        tl.where(valid, physical_slots, logical_slots),
        mask=mask,
    )


@triton.jit
def translate_owner_sharded_slots_with_current_kernel(
    slots_ptr: torch.Tensor,
    result_ptr: torch.Tensor,
    current_row_locs_ptr: torch.Tensor,
    num_slots,
    OWNER_CP_SIZE: tl.constexpr,
    OWNER_PAGE_SIZE: tl.constexpr,
    OWNER_PAGES_PER_RANK: tl.constexpr,
    CURRENT_ROWS_PER_REQUEST: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_slots
    logical_slots = tl.load(slots_ptr + offsets, mask=mask, other=-1)
    valid = logical_slots >= 0
    safe_slots = tl.where(valid, logical_slots, 0)
    logical_pages = safe_slots // OWNER_PAGE_SIZE
    page_offsets = safe_slots % OWNER_PAGE_SIZE
    owners = logical_pages % OWNER_CP_SIZE
    owner_pages = logical_pages // OWNER_CP_SIZE
    physical_pages = owners * OWNER_PAGES_PER_RANK + owner_pages
    physical_slots = physical_pages * OWNER_PAGE_SIZE + page_offsets

    current_row_markers = tl.full(offsets.shape, -1, tl.int32)
    query_row = offsets // TOPK
    current_row_position = query_row % CURRENT_ROWS_PER_REQUEST
    current_row_begin = query_row - current_row_position
    for current_row_slot in tl.static_range(0, 4):
        if current_row_slot < CURRENT_ROWS_PER_REQUEST:
            current_row_loc = tl.load(
                current_row_locs_ptr + current_row_begin + current_row_slot,
                mask=mask & (current_row_slot <= current_row_position),
                other=-1,
            )
            is_current_row = valid & (logical_slots == current_row_loc)
            current_row_markers = tl.where(
                is_current_row, -2 - current_row_slot, current_row_markers
            )
    translated_slots = tl.where(
        current_row_markers <= -2, current_row_markers, physical_slots
    )
    tl.store(
        result_ptr + offsets,
        tl.where(valid, translated_slots, logical_slots),
        mask=mask,
    )


def translate_owner_sharded_slots(
    slot_indices: torch.Tensor,
    *,
    owner_cp_size: int,
    owner_page_size: int,
    owner_pages_per_rank: int,
    result: Optional[torch.Tensor] = None,
    current_row_locs: Optional[torch.Tensor] = None,
    current_rows_per_request: int = 0,
) -> torch.Tensor:
    """Translate logical slots to the rank-major Shared-KV VMM layout."""
    assert slot_indices.dtype == torch.int32
    assert slot_indices.is_contiguous()
    assert owner_cp_size > 1
    assert owner_page_size > 0
    assert owner_pages_per_rank > 0
    if result is None:
        result = torch.empty_like(slot_indices)
    else:
        assert result.dtype == torch.int32
        assert result.shape == slot_indices.shape
        assert result.is_contiguous()
    has_current_rows = current_row_locs is not None
    if has_current_rows:
        assert slot_indices.dim() == 2
        assert current_row_locs is not None
        assert current_row_locs.dtype in (torch.int32, torch.int64)
        assert current_row_locs.is_contiguous()
        assert current_row_locs.numel() == slot_indices.shape[0]
        assert 1 <= current_rows_per_request <= 4
        assert slot_indices.shape[0] % current_rows_per_request == 0
    else:
        assert current_rows_per_request == 0
    if slot_indices.numel() == 0:
        return result
    block_size = 256
    grid = (triton.cdiv(slot_indices.numel(), block_size),)
    if has_current_rows:
        translate_owner_sharded_slots_with_current_kernel[grid](
            slot_indices,
            result,
            current_row_locs,
            slot_indices.numel(),
            OWNER_CP_SIZE=owner_cp_size,
            OWNER_PAGE_SIZE=owner_page_size,
            OWNER_PAGES_PER_RANK=owner_pages_per_rank,
            CURRENT_ROWS_PER_REQUEST=current_rows_per_request,
            TOPK=slot_indices.shape[1],
            BLOCK_SIZE=block_size,
        )
    else:
        translate_owner_sharded_slots_kernel[grid](
            slot_indices,
            result,
            slot_indices.numel(),
            OWNER_CP_SIZE=owner_cp_size,
            OWNER_PAGE_SIZE=owner_page_size,
            OWNER_PAGES_PER_RANK=owner_pages_per_rank,
            BLOCK_SIZE=block_size,
        )
    return result


def _allocate_prefill_result(
    topk_indices: torch.Tensor,
    real_num_tokens: int,
    output_num_tokens: Optional[int],
) -> torch.Tensor:
    topk_num_tokens = topk_indices.shape[0]
    if output_num_tokens is None:
        output_num_tokens = topk_num_tokens

    assert real_num_tokens <= topk_num_tokens, (
        f"sum(extend_lens_cpu) ({real_num_tokens}) exceeds "
        f"topk_indices rows ({topk_num_tokens})"
    )
    assert topk_num_tokens <= output_num_tokens, (
        f"topk_indices rows ({topk_num_tokens}) exceeds "
        f"output_num_tokens ({output_num_tokens})"
    )

    result = torch.empty(
        (output_num_tokens, topk_indices.shape[1]),
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if real_num_tokens < output_num_tokens:
        result[real_num_tokens:].fill_(-1)
    return result


@triton.jit
def transform_index_page_table_decode_kernel(
    page_table_ptr: torch.Tensor,
    topk_indices_ptr: torch.Tensor,
    result_ptr: torch.Tensor,
    page_size: tl.constexpr,
    page_table_row_stride: tl.constexpr,
    OWNER_CP_SIZE: tl.constexpr,
    OWNER_PAGE_SIZE: tl.constexpr,
    OWNER_PAGES_PER_RANK: tl.constexpr,
):
    TOPK: tl.constexpr = 2048
    req_id = tl.program_id(0)
    page_table_ptr = page_table_ptr + req_id * page_table_row_stride
    topk_indices_ptr = topk_indices_ptr + req_id * TOPK
    result_ptr = result_ptr + req_id * TOPK

    offset = tl.arange(0, TOPK)  # topk should be 2048
    loaded_topk_indices = tl.load(topk_indices_ptr + offset)
    mask = loaded_topk_indices >= 0
    loaded_kv_indices = tl.load(page_table_ptr + loaded_topk_indices, mask=mask)
    if OWNER_CP_SIZE > 1:
        logical_page = loaded_kv_indices // OWNER_PAGE_SIZE
        page_offset = loaded_kv_indices % OWNER_PAGE_SIZE
        owner_rank = logical_page % OWNER_CP_SIZE
        owner_page = logical_page // OWNER_CP_SIZE
        physical_page = owner_rank * OWNER_PAGES_PER_RANK + owner_page
        loaded_kv_indices = physical_page * OWNER_PAGE_SIZE + page_offset
    tl.store(result_ptr + offset, loaded_kv_indices, mask=mask)
    tl.store(result_ptr + offset, -1, mask=~mask)


@triton.jit
def transform_index_page_table_prefill_kernel(
    page_table_ptr: torch.Tensor,
    topk_indices_ptr: torch.Tensor,
    cu_seqlens_q_ptr: torch.Tensor,
    result_ptr: torch.Tensor,
    page_table_stride_0: tl.constexpr,
    page_table_stride_1: tl.constexpr,
    topk_indices_stride_0: tl.constexpr,
    topk_indices_stride_1: tl.constexpr,
    result_stride_0: tl.constexpr,
    result_stride_1: tl.constexpr,
    PAGE_TABLE_IS_EXPANDED: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
):
    request_id = tl.program_id(0)
    query_offsets = tl.program_id(1) * BLOCK_Q + tl.arange(0, BLOCK_Q)
    topk_offsets = tl.program_id(2) * BLOCK_TOPK + tl.arange(0, BLOCK_TOPK)

    query_start = tl.load(cu_seqlens_q_ptr + request_id)
    query_end = tl.load(cu_seqlens_q_ptr + request_id + 1)
    # Grid axis 1 spans the batch-max extend len; fully-masked blocks store nothing.
    if query_start + tl.program_id(1) * BLOCK_Q >= query_end:
        return
    token_indices = query_start + query_offsets
    mask = (token_indices[:, None] < query_end) & (topk_offsets[None, :] < TOPK)

    loaded_topk_indices = tl.load(
        topk_indices_ptr
        + token_indices[:, None] * topk_indices_stride_0
        + topk_offsets[None, :] * topk_indices_stride_1,
        mask=mask,
        other=-1,
    )
    valid_topk_mask = mask & (loaded_topk_indices >= 0)

    if PAGE_TABLE_IS_EXPANDED:
        page_table_rows = token_indices
    else:
        page_table_rows = token_indices * 0 + request_id
    loaded_kv_indices = tl.load(
        page_table_ptr
        + page_table_rows[:, None] * page_table_stride_0
        + loaded_topk_indices * page_table_stride_1,
        mask=valid_topk_mask,
        other=-1,
    )
    tl.store(
        result_ptr
        + token_indices[:, None] * result_stride_0
        + topk_offsets[None, :] * result_stride_1,
        loaded_kv_indices,
        mask=mask,
    )


def transform_index_page_table_decode_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
    owner_cp_size: int = 1,
    owner_page_size: int = 1,
    owner_pages_per_rank: int = 0,
) -> torch.Tensor:
    """
    Transform the page table according to topk indices for sparse topk attention.
    Args:
        page_table: [qo_len, max_seqlen_k], the original page table
        topk_indices: [qo_len, topk], the topk indices for each query position
    Returns:
        transformed_page_table: [qo_len, topk], the transformed page table
        For out-of-bound indices in topk_indices, this should be filled with -1.
    """
    assert page_size == 1
    assert page_table.shape[0] == topk_indices.shape[0]
    assert topk_indices.shape[1] == 2048
    assert owner_cp_size >= 1
    if owner_cp_size > 1:
        assert owner_page_size > 0
        assert owner_pages_per_rank > 0
    qo_len = topk_indices.shape[0]
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    # Launch triton kernel
    grid = (qo_len,)
    transform_index_page_table_decode_kernel[grid](
        page_table,
        topk_indices,
        result,
        page_size,
        page_table_row_stride=page_table.stride(0),
        OWNER_CP_SIZE=owner_cp_size,
        OWNER_PAGE_SIZE=owner_page_size,
        OWNER_PAGES_PER_RANK=owner_pages_per_rank,
    )
    return result


def transform_index_page_table_prefill_fast(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
    output_num_tokens: Optional[int] = None,
    page_table_is_expanded: bool = False,
    cu_seqlens_q: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert page_size == 1
    assert topk_indices.shape[1] == 2048
    real_num_tokens = sum(extend_lens_cpu)
    result = _allocate_prefill_result(topk_indices, real_num_tokens, output_num_tokens)
    if real_num_tokens == 0:
        return result

    max_extend_len = max(extend_lens_cpu)
    block_q = 1 if max_extend_len == 1 else 2 if max_extend_len == 2 else 4
    block_topk = 256
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.tensor(
            [0, *accumulate(extend_lens_cpu)],
            dtype=torch.int32,
            device=topk_indices.device,
        )
    grid = (
        cu_seqlens_q.shape[0] - 1,
        triton.cdiv(max_extend_len, block_q),
        triton.cdiv(topk_indices.shape[1], block_topk),
    )
    transform_index_page_table_prefill_kernel[grid](
        page_table,
        topk_indices,
        cu_seqlens_q,
        result,
        page_table.stride(0),
        page_table.stride(1),
        topk_indices.stride(0),
        topk_indices.stride(1),
        result.stride(0),
        result.stride(1),
        PAGE_TABLE_IS_EXPANDED=page_table_is_expanded,
        TOPK=topk_indices.shape[1],
        BLOCK_Q=block_q,
        BLOCK_TOPK=block_topk,
        num_warps=4,
    )
    return result


def transform_index_page_table_decode_ref(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    result: Optional[torch.Tensor] = None,
    page_size: int = 1,
) -> torch.Tensor:
    assert page_size == 1
    assert page_table.shape[0] == topk_indices.shape[0]
    if result is None:
        result = torch.empty_like(topk_indices, dtype=torch.int32)
    assert result.shape == topk_indices.shape
    torch.gather(
        page_table.to(result.dtype),
        dim=1,
        index=topk_indices.clamp(min=0),
        out=result,
    )
    result[topk_indices < 0] = -1
    return result


def transform_index_page_table_prefill_ref(
    page_table: torch.Tensor,
    topk_indices: torch.Tensor,
    extend_lens_cpu: List[int],
    page_size: int = 1,
    output_num_tokens: Optional[int] = None,
    page_table_is_expanded: bool = False,
) -> torch.Tensor:
    assert page_size == 1
    real_num_tokens = sum(extend_lens_cpu)
    result = _allocate_prefill_result(topk_indices, real_num_tokens, output_num_tokens)

    if page_table_is_expanded:
        if real_num_tokens > 0:
            transform_index_page_table_decode_ref(
                page_table[:real_num_tokens],
                topk_indices[:real_num_tokens],
                result=result[:real_num_tokens],
            )
        return result

    offset = 0
    for i, l in enumerate(extend_lens_cpu):
        transform_index_page_table_decode_ref(
            page_table[i].unsqueeze(0).expand(l, -1),
            topk_indices[offset : offset + l],
            result=result[offset : offset + l],
        )
        offset += l
    return result


if __name__ == "__main__":
    bs, topk, max_seqlen = 10, 2048, 3000
    page_table = torch.randint(0, 100, (bs, max_seqlen), device="cuda")
    topk_indices = torch.full((bs, topk), -1, device="cuda")
    topk_indices[:, :1600] = torch.arange(1600).unsqueeze(0).repeat(bs, 1)
    ref_result = transform_index_page_table_decode_ref(page_table, topk_indices)
    result = transform_index_page_table_decode_fast(page_table, topk_indices)
    assert torch.all(result == ref_result)
    print("Passed")
