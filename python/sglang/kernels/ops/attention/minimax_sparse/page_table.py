"""MiniMax sparse-attention page-table snapshot and lookup helpers."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def load_token_slots(
    page_table_ptr,
    batch_idx,
    positions,
    stride_page_table_b,
    mask,
    PAGE_SIZE: tl.constexpr,
):
    """Resolve logical token positions through a batch-local physical page table."""
    physical_pages = tl.load(
        page_table_ptr + batch_idx * stride_page_table_b + positions // PAGE_SIZE,
        mask=mask,
        other=0,
    ).to(tl.int64)
    return physical_pages * PAGE_SIZE + positions % PAGE_SIZE


@triton.jit
def _build_page_table_kernel(
    req_to_token_ptr,
    req_pool_indices_ptr,
    seq_lens_ptr,
    page_table_ptr,
    stride_r2t_b,
    stride_pt_b,
    max_num_pages,
    max_slots,
    SEQ_LEN_DELTA: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    page_offsets = pid_c * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    seq_len = tl.load(seq_lens_ptr + pid_b).to(tl.int64) + SEQ_LEN_DELTA
    num_pages = tl.cdiv(seq_len, PAGE_SIZE)
    in_bounds = page_offsets < max_num_pages
    active = in_bounds & (page_offsets < num_pages)

    req_idx = tl.load(req_pool_indices_ptr + pid_b).to(tl.int64)
    logical_first = page_offsets * PAGE_SIZE
    physical_slots = tl.load(
        req_to_token_ptr + req_idx * stride_r2t_b + logical_first,
        mask=active,
        other=0,
    ).to(tl.int64)
    physical_slots = (physical_slots + max_slots) % max_slots
    tl.store(
        page_table_ptr + pid_b * stride_pt_b + page_offsets,
        physical_slots // PAGE_SIZE,
        mask=in_bounds,
    )


@torch.no_grad()
def build_page_table_snapshot(
    page_table: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
    max_slots: int,
    *,
    seq_len_delta: int = 0,
) -> None:
    """Snapshot active request rows into graph-owned physical page ids."""
    assert page_table.dtype == torch.int32 and page_table.dim() == 2
    assert req_to_token.dtype == torch.int32 and req_to_token.dim() == 2
    assert page_table.shape[0] == req_pool_indices.shape[0] == seq_lens.shape[0]
    assert page_size > 0
    assert max_slots > 0 and max_slots % page_size == 0
    if page_table.shape[0] == 0:
        return

    block_size = 256
    grid = (
        page_table.shape[0],
        triton.cdiv(page_table.shape[1], block_size),
    )
    _build_page_table_kernel[grid](
        req_to_token,
        req_pool_indices,
        seq_lens,
        page_table,
        req_to_token.stride(0),
        page_table.stride(0),
        page_table.shape[1],
        max_slots,
        SEQ_LEN_DELTA=seq_len_delta,
        PAGE_SIZE=page_size,
        BLOCK_SIZE=block_size,
    )
