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
    actual_seq_lengths_kv_ptr,
    num_cache_tokens_ptr,
    hbm_block_table_ptr,
    dram_block_table_ptr,
    host_block_table_ptr,
    seq_lens_ptr,
    K: tl.constexpr,
    Q: tl.constexpr,
    C: tl.constexpr,
    HBM_PAGES: tl.constexpr,
    DRAM_PAGES: tl.constexpr,
    BLOCK: tl.constexpr,
    PAGED_HOST: tl.constexpr,
    HOST_COLS: tl.constexpr,
    HOST_S0: tl.constexpr,
    HOST_S1: tl.constexpr,
):
    tile_id = tl.program_id(0).to(tl.int32)
    row_id = tl.program_id(1).to(tl.int32)

    # Triton Ascend exposes torch.bool loads as int8; compare explicitly so
    # masks passed to tl.store and tl.where have boolean scalar type.
    is_valid = tl.load(valid_rows_ptr + row_id) != 0
    is_sparse = tl.load(sparse_rows_ptr + row_id) != 0
    request_slot = tl.load(request_slots_ptr + row_id).to(tl.int32)
    active = is_valid & is_sparse
    base_request = tl.where(is_valid, request_slot, 0).to(tl.int32)

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
        if PAGED_HOST:
            pages = tl.load(
                host_block_table_ptr + row_id * HOST_S0 + col * HOST_S1,
                mask=col < HOST_COLS,
                other=0,
            ).to(tl.int32)
            dram_values = tl.where(is_valid, pages, 0)
        else:
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
        seq_len = tl.load(seq_lens_ptr + row_id)
        sparse_length = C + Q - 1
        logical_length = tl.where(is_sparse, sparse_length, seq_len)
        tl.store(
            actual_seq_lengths_kv_ptr + row_id,
            tl.where(is_valid, logical_length, 0).to(tl.int32),
        )
        tl.store(num_cache_tokens_ptr + row_id, cache_length)


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
    actual_seq_lengths_kv: torch.Tensor,
    num_cache_tokens: torch.Tensor,
    hbm_block_table: torch.Tensor,
    dram_block_table: torch.Tensor,
    capacity: int,
    host_context_len: int,
    request_rows: int,
    seq_lens: torch.Tensor,
    query_width: int = 1,
    host_block_table: torch.Tensor | None = None,
) -> None:
    """Update preallocated LIM / Copy-SFA metadata in one NPU kernel launch.

    ``source_ids`` keeps its full contiguous ``[B * query_width, 2048]``
    LIM stride. Valid short rows retain their selected IDs; only invalid
    request rows are reset to -1 across all their query rows.
    The miss source/slot buffers and all KV payloads are deliberately untouched.
    """
    if not 1 <= query_width <= 14:
        raise ValueError("query_width must be in [1,14].")
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
        actual_seq_lengths_kv,
        num_cache_tokens,
    )
    if any(tensor.ndim != 1 or tensor.shape[0] != batch for tensor in row_tensors):
        raise ValueError("all per-row metadata tensors must have shape [B].")
    if (
        seq_lens.shape != (batch,)
        or seq_lens.dtype != torch.int32
        or seq_lens.device != source_ids.device
        or not seq_lens.is_contiguous()
    ):
        raise ValueError("Sequence lengths must be contiguous same-device int32[B].")
    if hbm_block_table.shape != (batch, capacity // 128):
        raise ValueError("hbm_block_table must have shape [B, capacity // 128].")
    if dram_block_table.shape != (batch, host_context_len // 128):
        raise ValueError(
            "dram_block_table must have shape [B, host_context_len // 128]."
        )

    device = source_ids.device
    if device.type != "npu":
        raise ValueError("fused_lim_copy_sfa_metadata requires NPU tensors.")
    if host_block_table is not None:
        if (
            host_block_table.dtype != torch.int32
            or host_block_table.device != device
            or host_block_table.ndim != 2
            or host_block_table.shape[0] != batch
            or host_block_table.shape[1] > dram_block_table.shape[1]
            or any(s < 0 for s in host_block_table.stride())
        ):
            raise ValueError("host_block_table must be same-device int32 [B, pages].")
    int32_tensors = (
        source_ids,
        topk_misses,
        miss_counts,
        request_slots,
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
        actual_seq_lengths_kv,
        num_cache_tokens,
        hbm_block_table,
        dram_block_table,
        host_block_table if host_block_table is not None else dram_block_table,
        seq_lens,
        2048,
        query_width,
        capacity,
        hbm_pages,
        dram_pages,
        2048,
        host_block_table is not None,
        host_block_table.shape[1] if host_block_table is not None else 0,
        host_block_table.stride(0) if host_block_table is not None else 0,
        host_block_table.stride(1) if host_block_table is not None else 0,
    )


__all__ = ["fused_lim_copy_sfa_metadata"]
