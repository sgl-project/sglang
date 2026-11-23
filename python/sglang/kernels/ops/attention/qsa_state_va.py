# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse VA QSA cache store; adapt its group pooling to SGLang group addresses.

Source: the vLLM NPU plugin 24487f4f48ee99cf45d86d1c504e645549b8c1ef,
ops/triton/qwen4_exp/qsa.py.
The cache-store kernel/launcher are copied verbatim. The group-pooling adapter
uses SGLang's precomputed member slots instead of VA's raw-chunk/ring lookup;
no tile tuning, state reordering, or approximation is introduced.
"""

import torch
import triton
import triton.language as tl

HAS_TRITON = True


@triton.jit
def _compress_sg_group_slots(
    keys, rope, groups, pooled, first_positions,
    KEY_ROW: tl.constexpr, KEY_COL: tl.constexpr,
    ROPE_ROW: tl.constexpr, ROPE_COL: tl.constexpr,
    GROUP_ROW: tl.constexpr, GROUP_COL: tl.constexpr,
    WIDTH: tl.constexpr, BLOCK_D: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    # Same group-order FP32 accumulation and BF16 output boundary as VA.
    # The caller already owns the write plan; preserve its exact member slots,
    # including clipped short prefixes, padding, and current ring contents.
    for offset in tl.range(0, COMPRESS_RATIO):
        slot = tl.load(groups + row * GROUP_ROW + offset * GROUP_COL).to(tl.int64)
        values = tl.load(keys + slot * KEY_ROW + dims * KEY_COL,
                         mask=dims < WIDTH, other=0.0).to(tl.float32)
        accumulator += values
    tl.store(pooled + row * WIDTH + dims, accumulator / COMPRESS_RATIO,
             mask=dims < WIDTH)
    first_slot = tl.load(groups + row * GROUP_ROW).to(tl.int64)
    axes = tl.arange(0, 4)
    position = tl.load(rope + first_slot * ROPE_ROW + axes * ROPE_COL,
                       mask=axes < 3, other=0)
    tl.store(first_positions + row * 3 + axes, position, mask=axes < 3)


def compress_sg_group_slots(keys, positions, groups):
    if keys.ndim == 3:
        assert keys.shape[1] == 1
        keys = keys[:, 0]
    assert keys.ndim == 2 and keys.dtype == torch.bfloat16 and keys.shape[1] == 128
    assert positions.shape == (keys.shape[0], 3) and positions.dtype == torch.int64
    assert groups.ndim == 2 and groups.shape[1] == 4
    pooled = torch.empty((groups.shape[0], 1, keys.shape[1]), dtype=keys.dtype, device=keys.device)
    first = torch.empty((groups.shape[0], 3), dtype=positions.dtype, device=positions.device)
    if groups.shape[0]:
        _compress_sg_group_slots[(groups.shape[0],)](
            keys, positions, groups, pooled, first,
            KEY_ROW=keys.stride(0), KEY_COL=keys.stride(1),
            ROPE_ROW=positions.stride(0), ROPE_COL=positions.stride(1),
            GROUP_ROW=groups.stride(0), GROUP_COL=groups.stride(1),
            WIDTH=128, BLOCK_D=128, COMPRESS_RATIO=4, num_warps=4,
        )
    return pooled, first

@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    cache_capacity = num_blocks * PAGE_SIZE
    valid = (row < num_rows) & (slot >= 0) & (slot < cache_capacity)
    # The NPU Triton backend can validate addresses before applying the store mask.
    # Keep every lane in bounds even when graph padding supplies an invalid
    # positive slot. Cast before multiplying by a potentially large stride so
    # the physical cache offset cannot overflow int32 either.
    safe_slot = tl.minimum(tl.maximum(slot, 0), cache_capacity - 1).to(tl.int64)
    block = safe_slot // PAGE_SIZE
    token = safe_slot % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr + block * stride_cache_block + token * stride_cache_token + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store fixed-width rows in a strided QSA cache."""
    if cache.device.type != "npu" or not HAS_TRITON:
        raise RuntimeError("QSA NPU cache stores require Triton")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.ndim != 2 or rows.shape[1] != cache.shape[3]:
        raise ValueError("QSA cache rows have an incompatible width")

    slot_mapping = slot_mapping.reshape(-1)
    num_updates = min(slot_mapping.shape[0], rows.shape[0])
    slot_mapping = slot_mapping[:num_updates]
    rows = rows[:num_updates]
    if not num_updates:
        return
    _store_qsa_rows_kernel[(num_updates,)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        num_updates,
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]),
        num_warps=4,
    )
