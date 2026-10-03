"""LoRA-A shrink over aligned groups or raw pairs.

Dense slots use ``lora_ranks`` to bound live width at rank * STACK;
MoE slots use the full N.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
import triton
import triton.language as tl

from sglang.srt.lora.kernels.routing import (
    grouped_tile_coords,
    route_bucket_ids,
)
from sglang.srt.lora.route_view import RouteView


@triton.jit
def _grouped_lora_a_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    partial_ptr,
    counter_ptr,
    pair_to_row_ptr,
    sorted_pair_ids_ptr,
    block_bucket_ids_ptr,
    num_pairs_post_padded_ptr,
    lora_ranks_ptr,
    num_input_rows,
    num_pairs,
    stride_im,
    stride_ik,
    stride_we,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    stride_pp,
    stride_pm,
    stride_pn,
    WIDTH: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    PAIR_INPUT: tl.constexpr,
    STACK: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SPLIT_MODE: tl.constexpr,
    GROUPS_PER_SLOT: tl.constexpr,
    WEIGHT_DIV: tl.constexpr,
    NUM_M_BLOCKS: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    WINDOWED: tl.constexpr = False,
):
    # None pointers select pair IDs as rows, full rank, or no split-K scratch.
    # SPLIT_MODE: "serial" = the last arrival at a tile sums the fp32 planes in
    # fixed order (deterministic); "planes" = planes only, summed by the expand on load.
    # WINDOWED: each N // STACK rank block shrinks its own K-wide input window.
    # Tiles cannot straddle blocks. Ignore live rank: it would bound each block,
    # not a prefix of N.
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    if WINDOWED:
        N_BLOCK: tl.constexpr = N // STACK
        TILES_PER_BLOCK: tl.constexpr = (N_BLOCK + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        num_pid_n = STACK * TILES_PER_BLOCK
    else:
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m, pid_n = grouped_tile_coords(pid, num_pid_n, NUM_M_BLOCKS, GROUP_SIZE_M)
    if pid_m * BLOCK_SIZE_M >= tl.load(num_pairs_post_padded_ptr):
        return
    bucket_id = tl.load(block_bucket_ids_ptr + pid_m)
    if bucket_id == -1:
        return
    if WINDOWED:
        block = pid_n // TILES_PER_BLOCK
        n_begin = block * N_BLOCK + (pid_n % TILES_PER_BLOCK) * BLOCK_SIZE_N
        n_end = (block + 1) * N_BLOCK
        k_base = (block * K).to(tl.int64)
    else:
        n_begin = pid_n * BLOCK_SIZE_N
        n_end = N
        k_base = 0
    if lora_ranks_ptr is not None and not WINDOWED:
        n_live = (
            tl.load(lora_ranks_ptr + bucket_id // GROUPS_PER_SLOT).to(tl.int32) * STACK
        )
        if n_begin >= n_live:
            return
    else:
        n_live = n_end

    pair_slots = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    pair_ids = tl.load(sorted_pair_ids_ptr + pair_slots).to(tl.int64)
    pair_mask = pair_ids < num_pairs
    if pair_to_row_ptr is not None:
        input_rows = tl.load(
            pair_to_row_ptr + pair_ids,
            mask=pair_mask,
            other=-1,
        ).to(tl.int64)
    elif PAIR_INPUT:
        input_rows = pair_ids
    else:
        input_rows = pair_ids // WIDTH
    input_mask = pair_mask & (input_rows >= 0) & (input_rows < num_input_rows)
    n_offsets = n_begin + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = n_offsets < n_live
    plane = (bucket_id // WEIGHT_DIV).to(tl.int64)

    K_PER_SPLIT: tl.constexpr = (K + SPLIT_K - 1) // SPLIT_K
    if SPLIT_K == 1:
        k_start = 0
        k_end = K
    else:
        k_start = pid_k * K_PER_SPLIT
        k_end = tl.minimum(k_start + K_PER_SPLIT, K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_begin in range(k_start, k_end, BLOCK_SIZE_K):
        k_offsets = k_begin + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        k_mask = k_offsets < k_end
        lhs = tl.load(
            input_ptr
            + input_rows[:, None] * stride_im
            + (k_base + k_offsets)[None, :] * stride_ik,
            mask=input_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        rhs = tl.load(
            weight_ptr
            + plane * stride_we
            + n_offsets[None, :] * stride_wn
            + k_offsets[:, None] * stride_wk,
            mask=n_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        accumulator += tl.dot(lhs, rhs, out_dtype=tl.float32)

    out_mask = pair_mask[:, None] & n_mask[None, :]
    output_ptrs = (
        output_ptr + pair_ids[:, None] * stride_om + n_offsets[None, :] * stride_on
    )
    if SPLIT_K == 1:
        tl.store(
            output_ptrs, accumulator.to(output_ptr.dtype.element_ty), mask=out_mask
        )
    else:
        plane_ptrs = (
            partial_ptr + pair_ids[:, None] * stride_pm + n_offsets[None, :] * stride_pn
        )
        tl.store(
            plane_ptrs + pid_k.to(tl.int64) * stride_pp, accumulator, mask=out_mask
        )
        if SPLIT_MODE == "serial":
            # The barrier orders this program's plane stores before its release;
            # the last arrival's acquire makes every plane visible.
            tl.debug_barrier()
            tile = pid_m * num_pid_n + pid_n
            arrived = tl.atomic_add(counter_ptr + tile, 1, sem="acq_rel", scope="gpu")
            if arrived == SPLIT_K - 1:
                total = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for s in tl.static_range(SPLIT_K):
                    total += tl.load(
                        plane_ptrs + s * stride_pp, mask=out_mask, other=0.0
                    )
                tl.store(
                    output_ptrs, total.to(output_ptr.dtype.element_ty), mask=out_mask
                )
                tl.atomic_xchg(counter_ptr + tile, 0, sem="release", scope="gpu")


def grouped_lora_a(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    routing: RouteView,
    *,
    config: Mapping[str, int],
    pair_input: bool = False,
    pair_to_row: torch.Tensor | None = None,
    lora_ranks: torch.Tensor | None = None,
    stack: int = 1,
    partial: torch.Tensor | None = None,
    counters: torch.Tensor | None = None,
    weight_div: int = 1,
    windowed: bool = False,
) -> None:
    """output[pair, :n] = input[row(pair)] @ weight[plane(block)].T over the
    aligned route. ``lora_ranks`` bounds a block at rank * stack (None = the
    full N); the slot of a block is its bucket // ``routing.groups_per_slot``,
    its weight plane the bucket // ``weight_div``. ``windowed``: rank block b of
    the ``stack`` shrinks input columns [b*K, (b+1)*K) at its full width
    (``lora_ranks`` is not applied). Split-K: both modes need ``partial``
    [SPLIT_K, rows, N] fp32; serial also needs zeroed ``counters``, planes
    leaves the sum to the expand."""
    num_pairs = routing.num_rows
    if num_pairs == 0:
        return
    block_size_n, num_n_blocks = shrink_column_tiles(
        int(config["BLOCK_SIZE_N"]), weight.shape[1], stack, windowed
    )
    split_k = int(config.get("SPLIT_K", 1))
    split_mode = config.get("SPLIT_MODE", "serial")
    num_m_blocks = triton.cdiv(routing.sorted_pair_ids.numel(), routing.block_size)
    if split_k > 1:
        if partial is None or (split_mode == "serial" and counters is None):
            raise ValueError("a split-K shrink needs partial planes (and counters)")
        if split_mode == "serial" and counters.numel() < num_m_blocks * num_n_blocks:
            raise ValueError("split-K tile counters are smaller than the tile grid")
        stride_pp, stride_pm, stride_pn = partial.stride()
    else:
        partial = counters = None
        stride_pp = stride_pm = stride_pn = 0
    _grouped_lora_a_kernel[(num_m_blocks * num_n_blocks, split_k)](
        input,
        weight,
        output,
        partial,
        counters,
        pair_to_row,
        routing.sorted_pair_ids,
        routing.block_bucket_ids,
        routing.num_pairs_post_padded,
        lora_ranks,
        input.shape[0],
        num_pairs,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        stride_pp,
        stride_pm,
        stride_pn,
        WIDTH=routing.width,
        N=weight.shape[1],
        K=weight.shape[2],
        PAIR_INPUT=pair_input,
        STACK=stack,
        SPLIT_K=split_k,
        SPLIT_MODE=split_mode,
        GROUPS_PER_SLOT=routing.groups_per_slot,
        WEIGHT_DIV=weight_div,
        NUM_M_BLOCKS=num_m_blocks,
        BLOCK_SIZE_M=routing.block_size,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=int(config["BLOCK_SIZE_K"]),
        GROUP_SIZE_M=int(config["GROUP_SIZE_M"]),
        WINDOWED=windowed,
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )


def shrink_column_tiles(
    block_size_n: int, n: int, stack: int, windowed: bool
) -> tuple[int, int]:
    """Return (tile width, tile count), keeping windowed rank blocks separate."""
    if not windowed:
        return block_size_n, triton.cdiv(n, block_size_n)
    n_block = n // stack
    block_size_n = max(16, min(block_size_n, triton.next_power_of_2(n_block)))
    return block_size_n, stack * triton.cdiv(n_block, block_size_n)


@triton.jit
def _per_row_lora_a_kernel(
    input_ptr,
    weight_ptr,
    group_ids_ptr,
    token_slots_ptr,
    output_ptr,
    stride_im,
    stride_ik,
    stride_wg,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUPS_PER_SLOT: tl.constexpr,
    MAX_LORAS: tl.constexpr,
    WIDTH: tl.constexpr,
    HAS_GROUPS: tl.constexpr,
    PAIR_INPUT: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    STACK: tl.constexpr = 1,
    WINDOWED: tl.constexpr = False,
):
    # One program per row: the grid is exactly the row count, so every
    # program's row is in range. WINDOWED: see the grouped kernel.
    pair_id = tl.program_id(0)
    pid_n = tl.program_id(1)
    bucket_id = route_bucket_ids(
        group_ids_ptr,
        token_slots_ptr,
        pair_id,
        pair_id >= 0,
        GROUPS_PER_SLOT=GROUPS_PER_SLOT,
        MAX_LORAS=MAX_LORAS,
        WIDTH=WIDTH,
        HAS_GROUPS=HAS_GROUPS,
    )
    valid = bucket_id != -1
    safe_bucket_id = tl.maximum(bucket_id, 0).to(tl.int64)
    pair64 = pair_id.to(tl.int64)
    input_row = pair64 if PAIR_INPUT else pair64 // WIDTH

    if WINDOWED:
        N_BLOCK: tl.constexpr = N // STACK
        TILES_PER_BLOCK: tl.constexpr = (N_BLOCK + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
        block = pid_n // TILES_PER_BLOCK
        n_begin = block * N_BLOCK + (pid_n % TILES_PER_BLOCK) * BLOCK_SIZE_N
        n_end = (block + 1) * N_BLOCK
        k_base = (block * K).to(tl.int64)
    else:
        n_begin = pid_n * BLOCK_SIZE_N
        n_end = N
        k_base = 0
    n_offsets = n_begin.to(tl.int64) + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    n_mask = n_offsets < n_end
    accumulator = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)
    for k_begin in range(0, K, BLOCK_SIZE_K):
        k_offsets = k_begin + tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
        k_mask = k_offsets < K
        lhs = tl.load(
            input_ptr + input_row * stride_im + (k_base + k_offsets) * stride_ik,
            mask=valid & k_mask,
            other=0.0,
        )
        rhs = tl.load(
            weight_ptr
            + safe_bucket_id * stride_wg
            + n_offsets[:, None] * stride_wn
            + k_offsets[None, :] * stride_wk,
            mask=valid & n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        accumulator += tl.sum(rhs.to(tl.float32) * lhs[None, :].to(tl.float32), axis=1)

    # B zeros sentinel destinations without reading these bridge rows.
    tl.store(
        output_ptr + pair64 * stride_om + n_offsets * stride_on,
        accumulator.to(output_ptr.dtype.element_ty),
        mask=valid & n_mask,
    )


def per_row_lora_a(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    routing: RouteView,
    *,
    config: Mapping[str, int],
    pair_input: bool = False,
    stack: int = 1,
    windowed: bool = False,
) -> None:
    num_pairs = routing.num_rows
    if num_pairs == 0:
        return

    block_size_n, num_n_blocks = shrink_column_tiles(
        int(config["BLOCK_SIZE_N"]), weight.shape[1], stack, windowed
    )
    _per_row_lora_a_kernel[(num_pairs, num_n_blocks)](
        input,
        weight,
        routing.kernel_groups,
        routing.token_slots,
        output,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        N=weight.shape[1],
        K=weight.shape[2],
        GROUPS_PER_SLOT=routing.groups_per_slot,
        MAX_LORAS=routing.max_loras,
        WIDTH=routing.width,
        HAS_GROUPS=routing.group_ids is not None,
        PAIR_INPUT=pair_input,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=int(config["BLOCK_SIZE_K"]),
        STACK=stack,
        WINDOWED=windowed,
        num_warps=int(config["num_warps"]),
        num_stages=int(config["num_stages"]),
    )
