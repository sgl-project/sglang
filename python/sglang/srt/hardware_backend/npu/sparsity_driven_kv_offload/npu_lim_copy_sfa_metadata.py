"""Triton Ascend metadata preparation for LIM + Copy-SFA."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _lim_copy_sfa_metadata_kernel(
    source_ids_ptr,
    topk_misses_ptr,
    miss_counts_ptr,
    valid_rows_ptr,
    sparse_rows_ptr,
    request_slots_ptr,
    next_prefix_ptr,
    last_prefix_ptr,
    actual_seq_lengths_kv_ptr,
    num_cache_tokens_ptr,
    hbm_block_table_ptr,
    dram_block_table_ptr,
    B: tl.constexpr,
    K: tl.constexpr,
    Q: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    R: tl.constexpr,
    HBM_PAGES: tl.constexpr,
    DRAM_PAGES: tl.constexpr,
    B_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile_id = tl.program_id(0).to(tl.int32)
    row_id = tl.program_id(1).to(tl.int32)

    # Triton Ascend exposes torch.bool loads as int8; compare explicitly so
    # masks passed to tl.store and tl.where have boolean scalar type.
    is_valid = tl.load(valid_rows_ptr + row_id) != 0
    is_sparse = tl.load(sparse_rows_ptr + row_id) != 0
    request_slot = tl.load(request_slots_ptr + row_id).to(tl.int32)
    active = is_valid & is_sparse
    base_request = tl.where(active, request_slot, 0).to(tl.int32)

    col = tile_id * BLOCK + tl.arange(0, BLOCK).to(tl.int32)
    if tile_id * BLOCK < K:
        for route in range(Q):
            source_base = (row_id.to(tl.int64) * Q + route) * K
            tl.store(
                source_ids_ptr + source_base + col.to(tl.int64),
                -1,
                mask=(col < K) & ~is_valid,
            )

    if tile_id * BLOCK < HBM_PAGES:
        hbm_base = row_id.to(tl.int64) * HBM_PAGES
        hbm_values = base_request * HBM_PAGES + col
        tl.store(
            hbm_block_table_ptr + hbm_base + col.to(tl.int64),
            hbm_values,
            mask=col < HBM_PAGES,
        )

    if tile_id * BLOCK < DRAM_PAGES:
        dram_base = row_id.to(tl.int64) * DRAM_PAGES
        dram_values = base_request * DRAM_PAGES + col
        tl.store(
            dram_block_table_ptr + dram_base + col.to(tl.int64),
            dram_values,
            mask=col < DRAM_PAGES,
        )

    if tile_id == 0:
        for route in range(Q):
            tl.store(topk_misses_ptr + row_id * Q + route, 0, mask=~active)
        tl.store(miss_counts_ptr + row_id, 0, mask=~active)
        cache_length = tl.where(active, C, 0).to(tl.int32)
        tl.store(
            actual_seq_lengths_kv_ptr + row_id,
            tl.where(active, C + Q - 1, 0).to(tl.int32),
        )
        tl.store(num_cache_tokens_ptr + row_id, cache_length)
        tl.store(
            last_prefix_ptr + request_slot.to(tl.int64),
            tl.load(next_prefix_ptr + row_id).to(tl.int32),
            mask=is_valid,
        )

    # Only one program writes the invalid-request sentinel. This avoids a
    # cross-program race while preserving its sticky value when all rows valid.
    if (row_id == 0) & (tile_id == 0):
        valid_offsets = tl.arange(0, B_PAD).to(tl.int32)
        valid_values = (
            tl.load(
                valid_rows_ptr + valid_offsets,
                mask=valid_offsets < B,
                other=1,
            )
            != 0
        )
        invalid_count = tl.sum(
            ((valid_offsets < B) & ~valid_values).to(tl.int32), axis=0
        )
        tl.store(last_prefix_ptr + R, -1, mask=invalid_count > 0)


def _validate_tensor(name: str, tensor: torch.Tensor, *, dtype, device) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}.")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}.")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous.")


def fused_lim_copy_sfa_metadata(
    *,
    source_ids: torch.Tensor,
    topk_misses: torch.Tensor,
    miss_counts: torch.Tensor,
    valid_rows: torch.Tensor,
    sparse_rows: torch.Tensor,
    request_slots: torch.Tensor,
    next_prefix: torch.Tensor,
    last_prefix: torch.Tensor,
    actual_seq_lengths_kv: torch.Tensor,
    num_cache_tokens: torch.Tensor,
    hbm_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    capacity: int,
    host_context_len: int,
    request_rows: int,
    query_width: int = 1,
) -> None:
    """Update preallocated LIM / Copy-SFA metadata in one NPU kernel launch.

    ``source_ids`` keeps its full contiguous ``[B * query_width, 2048]``
    LIM stride. Valid short rows retain their selected IDs; only invalid
    request rows are reset to -1 across all their query rows.
    The miss source/slot buffers and all KV payloads are deliberately untouched.
    """
    if query_width not in (1, 4):
        raise ValueError("query_width must be 1 or 4.")
    if not isinstance(capacity, int) or not 2048 <= capacity <= 32640:
        raise ValueError("capacity must be an integer in [2048, 32640].")
    if capacity % 128:
        raise ValueError("capacity must be aligned to 128 tokens.")
    if not isinstance(host_context_len, int) or host_context_len <= 0:
        raise ValueError("host_context_len must be a positive integer.")
    if host_context_len % 128:
        raise ValueError("host_context_len must be aligned to 128 tokens.")
    if not isinstance(request_rows, int) or request_rows <= 0:
        raise ValueError("request_rows must be a positive integer.")
    if request_rows * (capacity // 128) > 2**31:
        raise ValueError("HBM page-table values must fit in int32.")
    if request_rows * (host_context_len // 128) > 2**31:
        raise ValueError("DRAM page-table values must fit in int32.")

    tensors = (
        source_ids,
        topk_misses,
        miss_counts,
        valid_rows,
        sparse_rows,
        request_slots,
        next_prefix,
        last_prefix,
        actual_seq_lengths_kv,
        num_cache_tokens,
        hbm_block_table,
        dram_block_table,
    )
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise TypeError("all metadata inputs and outputs must be torch.Tensor values.")
    if source_ids.ndim != 2 or source_ids.shape[1] != 2048:
        raise ValueError("source_ids must have shape [B * query_width, 2048].")
    if source_ids.shape[0] % query_width:
        raise ValueError("source_ids rows must be divisible by query_width.")
    batch = source_ids.shape[0] // query_width
    if batch <= 0:
        raise ValueError("batch must be positive.")
    if topk_misses.ndim != 1 or topk_misses.shape[0] != batch * query_width:
        raise ValueError("topk_misses must have shape [B * query_width].")
    row_tensors = (
        miss_counts,
        valid_rows,
        sparse_rows,
        request_slots,
        next_prefix,
        actual_seq_lengths_kv,
        num_cache_tokens,
    )
    if any(tensor.ndim != 1 or tensor.shape[0] != batch for tensor in row_tensors):
        raise ValueError("all per-row metadata tensors must have shape [B].")
    if last_prefix.ndim != 1 or last_prefix.shape[0] != request_rows + 1:
        raise ValueError("last_prefix must have shape [request_rows + 1].")
    if hbm_block_table.shape != (batch, capacity // 128):
        raise ValueError("hbm_block_table must have shape [B, capacity // 128].")
    if dram_block_table.shape != (batch, host_context_len // 128):
        raise ValueError(
            "dram_block_table must have shape [B, host_context_len // 128]."
        )

    device = source_ids.device
    if device.type != "npu":
        raise ValueError("fused_lim_copy_sfa_metadata requires NPU tensors.")
    int32_tensors = (
        source_ids,
        topk_misses,
        miss_counts,
        request_slots,
        next_prefix,
        last_prefix,
        actual_seq_lengths_kv,
        num_cache_tokens,
        hbm_block_table,
        dram_block_table,
    )
    for name, tensor in zip(
        (
            "source_ids",
            "topk_misses",
            "miss_counts",
            "request_slots",
            "next_prefix",
            "last_prefix",
            "actual_seq_lengths_kv",
            "num_cache_tokens",
            "hbm_block_table",
            "dram_block_table",
        ),
        int32_tensors,
    ):
        _validate_tensor(name, tensor, dtype=torch.int32, device=device)
    _validate_tensor("valid_rows", valid_rows, dtype=torch.bool, device=device)
    _validate_tensor("sparse_rows", sparse_rows, dtype=torch.bool, device=device)

    hbm_pages = capacity // 128
    dram_pages = host_context_len // 128
    tiles = triton.cdiv(max(2048, hbm_pages, dram_pages), 2048)
    _lim_copy_sfa_metadata_kernel[(tiles, batch)](
        source_ids,
        topk_misses,
        miss_counts,
        valid_rows,
        sparse_rows,
        request_slots,
        next_prefix,
        last_prefix,
        actual_seq_lengths_kv,
        num_cache_tokens,
        hbm_block_table,
        dram_block_table,
        batch,
        2048,
        query_width,
        capacity,
        host_context_len,
        request_rows,
        hbm_pages,
        dram_pages,
        triton.next_power_of_2(batch),
        2048,
    )


__all__ = ["fused_lim_copy_sfa_metadata"]
