"""Opt-in pre-LIM metadata preparation for Q1 LIM + Copy-SFA."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_kernel(
    seq_ptr,
    slot_ptr,
    last_ptr,
    src_table_ptr,
    out_table_ptr,
    safe_slot_ptr,
    length_ptr,
    prefix_ptr,
    sparse_ptr,
    valid_ptr,
    state_ptr,
    next_prefix_ptr,
    cache_size_ptr,
    B: tl.constexpr,
    ROWS: tl.constexpr,
    CAPACITY: tl.constexpr,
    FULL_VISIBLE: tl.constexpr,
    SRC_COLS: tl.constexpr,
    DST_COLS: tl.constexpr,
    SRC_S0: tl.constexpr,
    SRC_S1: tl.constexpr,
    DST_S0: tl.constexpr,
    DST_S1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    tile = tl.program_id(0).to(tl.int32)
    row = tl.program_id(1).to(tl.int32)
    cols = tile * BLOCK + tl.arange(0, BLOCK).to(tl.int32)

    # Every destination column is assigned once: preserve every source column,
    # including PAD columns, and zero only the destination's trailing columns.
    if tile * BLOCK < DST_COLS:
        value = tl.load(
            src_table_ptr + row * SRC_S0 + cols * SRC_S1,
            mask=(cols < SRC_COLS) & (cols < DST_COLS),
            other=0,
        ).to(tl.int32)
        tl.store(
            out_table_ptr + row * DST_S0 + cols * DST_S1, value, mask=cols < DST_COLS
        )

    if tile == 0:
        raw_slot = tl.load(slot_ptr + row).to(tl.int64)
        raw_length = tl.load(seq_ptr + row).to(tl.int64)
        valid = (raw_slot > 0) & (raw_slot < ROWS) & (raw_length > 0)
        safe_slot = tl.where(valid, raw_slot, ROWS).to(tl.int32)
        length = tl.where(valid, raw_length, 0).to(tl.int32)
        prefix = tl.where(
            FULL_VISIBLE, length, ((tl.maximum(length - 1, 0) // 128) * 128)
        ).to(tl.int32)
        sparse = valid & (prefix >= CAPACITY)
        previous = tl.load(last_ptr + safe_slot).to(tl.int32)
        warm = sparse & (previous >= 0) & (prefix >= previous)
        state = tl.where(sparse, tl.where(warm, -1, -2), -3).to(tl.int32)
        next_prefix = tl.where(sparse, prefix, -1).to(tl.int32)
        cache_size = tl.where(sparse, CAPACITY, 0).to(tl.int32)
        tl.store(safe_slot_ptr + row, safe_slot)
        tl.store(length_ptr + row, length)
        tl.store(prefix_ptr + row, prefix)
        tl.store(sparse_ptr + row, sparse)
        tl.store(valid_ptr + row, valid)
        tl.store(state_ptr + row, state)
        tl.store(next_prefix_ptr + row, next_prefix)
        tl.store(cache_size_ptr + row, cache_size)


def fused_prepare_lim_metadata(
    *,
    seq_lens,
    request_slots,
    last_prefix,
    block_table,
    out_block_table,
    safe_slots,
    lengths,
    prefixes,
    sparse_rows,
    valid_rows,
    request_states,
    next_prefixes,
    cache_sizes,
    capacity: int,
    request_rows: int,
    full_visible: bool,
) -> None:
    """Fill fixed-address Q1 inputs and LIM source table in one Triton launch."""
    if seq_lens.dtype not in (torch.int32, torch.int64):
        raise TypeError("seq_lens must be int32 or int64")
    if request_slots.dtype not in (torch.int32, torch.int64):
        raise TypeError("request_slots must be int32 or int64")
    if seq_lens.ndim != 1 or request_slots.ndim != 1:
        raise ValueError("seq_lens and request_slots must be one-dimensional")
    if not seq_lens.is_contiguous() or not request_slots.is_contiguous():
        raise ValueError("seq_lens and request_slots must be contiguous")
    if not isinstance(capacity, int) or capacity <= 0 or capacity % 128:
        raise ValueError("capacity must be positive and aligned to 128")
    if not isinstance(request_rows, int) or request_rows <= 0:
        raise ValueError("request_rows must be positive")
    batch = seq_lens.numel()
    if batch <= 0 or request_slots.numel() != batch:
        raise ValueError("seq_lens and request_slots must have equal nonzero length")
    if seq_lens.device.type != "npu" or request_slots.device != seq_lens.device:
        raise ValueError("pre-metadata kernel requires same-device NPU inputs")
    if (
        last_prefix.dtype != torch.int32
        or last_prefix.ndim != 1
        or last_prefix.numel() != request_rows + 1
    ):
        raise ValueError("last_prefix must be int32 [request_rows + 1]")
    if last_prefix.device != seq_lens.device or not last_prefix.is_contiguous():
        raise ValueError("last_prefix must be contiguous on the input device")
    if (
        block_table.device != seq_lens.device
        or out_block_table.device != seq_lens.device
    ):
        raise ValueError("block tables must be on the input device")
    if block_table.dtype != torch.int32 or out_block_table.dtype != torch.int32:
        raise TypeError("block tables must be int32")
    if block_table.ndim != 2 or out_block_table.ndim != 2:
        raise ValueError("block tables must be rank two")
    if block_table.shape[0] != batch or out_block_table.shape[0] < batch:
        raise ValueError("block-table batch mismatch")
    if out_block_table.shape[1] < block_table.shape[1]:
        raise ValueError("destination block table cannot truncate source columns")
    if out_block_table.shape[1] <= 0 or not out_block_table.is_contiguous():
        raise ValueError("destination block table must be nonempty and contiguous")
    if any(stride < 0 for stride in block_table.stride()) or any(
        stride < 0 for stride in out_block_table.stride()
    ):
        raise ValueError("negative block-table strides are unsupported")
    outputs = (
        safe_slots,
        lengths,
        prefixes,
        sparse_rows,
        valid_rows,
        request_states,
        next_prefixes,
        cache_sizes,
    )
    expected_dtypes = (
        torch.int32,
        torch.int32,
        torch.int32,
        torch.bool,
        torch.bool,
        torch.int32,
        torch.int32,
        torch.int32,
    )
    for output, dtype in zip(outputs, expected_dtypes):
        if output.dtype != dtype or output.device != seq_lens.device:
            raise TypeError(
                "all preallocated outputs must have the specified dtype/device"
            )
        if output.ndim != 1 or output.numel() < batch or not output.is_contiguous():
            raise ValueError("preallocated outputs must be contiguous [batch_capacity]")
    cols = out_block_table.shape[1]
    block = 128
    _prepare_kernel[(triton.cdiv(cols, block), batch)](
        seq_lens,
        request_slots,
        last_prefix,
        block_table,
        out_block_table,
        safe_slots,
        lengths,
        prefixes,
        sparse_rows,
        valid_rows,
        request_states,
        next_prefixes,
        cache_sizes,
        batch,
        request_rows,
        capacity,
        full_visible,
        block_table.shape[1],
        cols,
        block_table.stride(0),
        block_table.stride(1),
        out_block_table.stride(0),
        out_block_table.stride(1),
        block,
    )


__all__ = ["fused_prepare_lim_metadata"]
