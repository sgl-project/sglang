"""Triton Ascend metadata preparation for LIM + Copy-SFA."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _lim_copy_sfa_metadata_kernel(
    source_ids_ptr,
    valid_rows_ptr,
    sparse_rows_ptr,
    request_slots_ptr,
    actual_seq_lengths_kv_ptr,
    hbm_block_table_ptr,
    seq_lens_ptr,
    K: tl.constexpr,
    Q: tl.constexpr,
    C: tl.constexpr,
    HBM_PAGES: tl.constexpr,
):
    row_id = tl.program_id(0).to(tl.int32)

    # Triton Ascend exposes torch.bool loads as int8; compare explicitly so
    # masks passed to tl.store and tl.where have boolean scalar type.
    is_valid = tl.load(valid_rows_ptr + row_id) != 0
    is_sparse = tl.load(sparse_rows_ptr + row_id) != 0
    request_slot = tl.load(request_slots_ptr + row_id).to(tl.int32)
    base_request = tl.where(is_valid, request_slot, 0).to(tl.int32)

    col = tl.arange(0, K).to(tl.int32)
    for route in range(Q):
        source_base = (row_id.to(tl.int64) * Q + route) * K
        tl.store(
            source_ids_ptr + source_base + col.to(tl.int64),
            -1,
            mask=~is_valid,
        )

    hbm_base = row_id.to(tl.int64) * HBM_PAGES
    hbm_values = base_request * HBM_PAGES + col
    tl.store(
        hbm_block_table_ptr + hbm_base + col.to(tl.int64),
        hbm_values,
        mask=col < HBM_PAGES,
    )

    seq_len = tl.load(seq_lens_ptr + row_id)
    # Native MTP subtracts Q - 1 - query_index from this cache-relative length.
    sparse_length = C + Q - 1
    logical_length = tl.where(is_sparse, sparse_length, seq_len)
    tl.store(
        actual_seq_lengths_kv_ptr + row_id,
        tl.where(is_valid, logical_length, 0).to(tl.int32),
    )


def prepare_copy_sfa_metadata(
    *,
    source_ids: torch.Tensor,
    valid_rows: torch.Tensor,
    sparse_rows: torch.Tensor,
    request_slots: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    hbm_block_table: torch.Tensor,
    capacity: int,
    seq_lens: torch.Tensor,
    query_width: int = 1,
) -> None:
    """Prepare native scheduling lengths, HBM pages, and invalid source IDs."""
    batch = source_ids.shape[0] // query_width
    hbm_pages = capacity // 128
    _lim_copy_sfa_metadata_kernel[(batch,)](
        source_ids,
        valid_rows,
        sparse_rows,
        request_slots,
        actual_seq_lengths_kv,
        hbm_block_table,
        seq_lens,
        2048,
        query_width,
        capacity,
        hbm_pages,
    )


__all__ = ["prepare_copy_sfa_metadata"]
