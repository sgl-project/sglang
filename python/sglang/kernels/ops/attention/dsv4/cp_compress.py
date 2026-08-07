# SPDX-License-Identifier: Apache-2.0
"""Consumer-direct DSV4 compressor for interleaved context parallelism.

The normal CP path all-gathers the full FP32 ``[kv, score]`` projection before
compressing it.  For an aligned single-sequence round-robin split, compression
is separable by CP rank: each rank publishes one online-softmax state per
compression window, and every consumer merges the four peer states directly.

This module deliberately implements an object contract, not another all-gather:
the shared object is ``[max, exp_sum, weighted_kv_sum]`` per window.  No rank
ever materializes a peer's token-level score tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    _blockwise_barrier,
)


@triton.jit
def _peer_barrier_kernel(
    signal_pad_ptrs,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
):
    _blockwise_barrier(signal_pad_ptrs, RANK, WORLD_SIZE, sem="acq_rel")


@triton.jit
def _online_update(max_value, exp_sum, weighted_sum, score, value):
    new_max = tl.maximum(max_value, score)
    old_scale = tl.exp(max_value - new_max)
    new_scale = tl.exp(score - new_max)
    return (
        new_max,
        exp_sum * old_scale + new_scale,
        weighted_sum * old_scale + value * new_scale,
    )


@triton.jit
def _c128_local_state_kernel(
    kv_score,
    ape,
    partial,
    num_windows,
    slot_offset,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    window = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < HEAD_DIM

    max_value = tl.full((BLOCK,), float("-inf"), tl.float32)
    exp_sum = tl.zeros((BLOCK,), tl.float32)
    weighted_sum = tl.zeros((BLOCK,), tl.float32)
    local_per_window: tl.constexpr = 128 // WORLD_SIZE
    for token in tl.static_range(local_per_window):
        local_row = window * local_per_window + token
        position = RANK + token * WORLD_SIZE
        value = tl.load(kv_score + local_row * (2 * HEAD_DIM) + offsets, mask=mask)
        score = tl.load(
            kv_score + local_row * (2 * HEAD_DIM) + HEAD_DIM + offsets,
            mask=mask,
        )
        score += tl.load(ape + position * HEAD_DIM + offsets, mask=mask)
        max_value, exp_sum, weighted_sum = _online_update(
            max_value, exp_sum, weighted_sum, score, value
        )

    base = partial + slot_offset + window * (3 * HEAD_DIM) + offsets
    tl.store(base, max_value, mask=mask)
    tl.store(base + HEAD_DIM, exp_sum, mask=mask)
    tl.store(base + 2 * HEAD_DIM, weighted_sum, mask=mask)


@triton.jit
def _c4_local_state_kernel(
    kv_score,
    ape,
    carry,
    plan_c,
    kv_score_buffer,
    partial,
    num_windows,
    slot_offset,
    HAS_PREFIX: tl.constexpr,
    USE_PAGED_PREFIX: tl.constexpr,
    STORE_CARRY: tl.constexpr,
    RANK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    window = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < HEAD_DIM
    row_base = window * (4 * HEAD_DIM)

    # The C4 projection layout is
    # [kv_overlap, kv_normal, score_overlap, score_normal].  Each CP4 rank
    # owns one token in the previous overlap half and one in the current half.
    value = tl.load(kv_score + row_base + HEAD_DIM + offsets, mask=mask)
    score = tl.load(kv_score + row_base + 3 * HEAD_DIM + offsets, mask=mask)
    score += tl.load(ape + (4 + RANK) * HEAD_DIM + offsets, mask=mask)
    max_value = score
    exp_sum = tl.full((BLOCK,), 1.0, tl.float32)
    weighted_sum = value

    if window == 0:
        if HAS_PREFIX:
            if USE_PAGED_PREFIX:
                plan_i32 = plan_c.to(tl.pointer_type(tl.int32))
                read_page_0 = tl.load(plan_i32 + 2)
                previous_base = (read_page_0 * 4 + RANK) * (4 * HEAD_DIM)
                previous_value = tl.load(
                    kv_score_buffer + previous_base + offsets, mask=mask
                )
                previous_score = tl.load(
                    kv_score_buffer + previous_base + 2 * HEAD_DIM + offsets,
                    mask=mask,
                )
            else:
                previous_value = tl.load(carry + offsets, mask=mask)
                previous_score = tl.load(carry + HEAD_DIM + offsets, mask=mask)
            previous_score += tl.load(ape + RANK * HEAD_DIM + offsets, mask=mask)
            max_value, exp_sum, weighted_sum = _online_update(
                max_value,
                exp_sum,
                weighted_sum,
                previous_score,
                previous_value,
            )
    else:
        previous_base = (window - 1) * (4 * HEAD_DIM)
        previous_value = tl.load(kv_score + previous_base + offsets, mask=mask)
        previous_score = tl.load(
            kv_score + previous_base + 2 * HEAD_DIM + offsets, mask=mask
        )
        previous_score += tl.load(ape + RANK * HEAD_DIM + offsets, mask=mask)
        max_value, exp_sum, weighted_sum = _online_update(
            max_value,
            exp_sum,
            weighted_sum,
            previous_score,
            previous_value,
        )

    base = partial + slot_offset + window * (3 * HEAD_DIM) + offsets
    tl.store(base, max_value, mask=mask)
    tl.store(base + HEAD_DIM, exp_sum, mask=mask)
    tl.store(base + 2 * HEAD_DIM, weighted_sum, mask=mask)

    # Persist this rank's final overlap projection for the first output of the
    # next aligned prefill chunk.  Peer consumers never read this local object.
    if STORE_CARRY and window == num_windows - 1:
        tl.store(
            carry + offsets,
            tl.load(kv_score + row_base + offsets, mask=mask),
            mask=mask,
        )
        tl.store(
            carry + HEAD_DIM + offsets,
            tl.load(kv_score + row_base + 2 * HEAD_DIM + offsets, mask=mask),
            mask=mask,
        )


@triton.jit
def _merge_peer_states_kernel(
    peer_ptrs,
    output,
    num_windows,
    slot_offset,
    RATIO: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    window = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < HEAD_DIM
    object_offset = slot_offset + window * (3 * HEAD_DIM)

    global_max = tl.full((BLOCK,), float("-inf"), tl.float32)
    for rank in tl.static_range(WORLD_SIZE):
        peer = tl.load(peer_ptrs + rank).to(tl.pointer_type(tl.float32))
        rank_max = tl.load(peer + object_offset + offsets, mask=mask)
        global_max = tl.maximum(global_max, rank_max)

    global_sum = tl.zeros((BLOCK,), tl.float32)
    global_weighted_sum = tl.zeros((BLOCK,), tl.float32)
    for rank in tl.static_range(WORLD_SIZE):
        peer = tl.load(peer_ptrs + rank).to(tl.pointer_type(tl.float32))
        rank_max = tl.load(peer + object_offset + offsets, mask=mask)
        rank_sum = tl.load(peer + object_offset + HEAD_DIM + offsets, mask=mask)
        rank_weighted_sum = tl.load(
            peer + object_offset + 2 * HEAD_DIM + offsets, mask=mask
        )
        scale = tl.exp(rank_max - global_max)
        global_sum += rank_sum * scale
        global_weighted_sum += rank_weighted_sum * scale

    tl.store(
        output + window * HEAD_DIM + offsets,
        global_weighted_sum / global_sum,
        mask=mask,
    )


@triton.jit
def _stage_c4_writes_kernel(
    kv_score,
    plan_w,
    write_stage,
    num_write,
    num_local_rows,
    write_slot_offset,
    RANK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    write_id = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    width: tl.constexpr = 4 * HEAD_DIM
    mask = offsets < width
    plan_i32 = plan_w.to(tl.pointer_type(tl.int32))
    ragged_id = tl.load(plan_i32 + write_id * 2)
    valid = (
        (write_id < num_write)
        & (ragged_id >= 0)
        & (ragged_id < num_local_rows * 4)
        & (ragged_id % 4 == RANK)
    )
    local_row = ragged_id // 4
    values = tl.load(kv_score + local_row * width + offsets, mask=mask & valid)
    tl.store(
        write_stage + write_slot_offset + write_id * width + offsets,
        values,
        mask=mask & valid,
    )


@triton.jit
def _consume_c4_writes_kernel(
    peer_write_ptrs,
    plan_w,
    kv_score_buffer,
    num_write,
    num_state_rows,
    write_slot_offset,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    write_id = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK + tl.arange(0, BLOCK)
    width: tl.constexpr = 4 * HEAD_DIM
    mask = offsets < width
    plan_i32 = plan_w.to(tl.pointer_type(tl.int32))
    ragged_id = tl.load(plan_i32 + write_id * 2)
    write_loc = tl.load(plan_i32 + write_id * 2 + 1)
    valid = (
        (write_id < num_write)
        & (ragged_id >= 0)
        & (write_loc >= 0)
        & (write_loc < num_state_rows)
    )
    owner = tl.where(valid, ragged_id % 4, 0)
    peer = tl.load(peer_write_ptrs + owner).to(tl.pointer_type(tl.float32))
    values = tl.load(
        peer + write_slot_offset + write_id * width + offsets,
        mask=mask & valid,
    )
    tl.store(kv_score_buffer + write_loc * width + offsets, values, mask=mask & valid)


@dataclass
class CPCompressorState:
    group: dist.ProcessGroup
    rank: int
    world_size: int
    ratio: int
    head_dim: int
    max_global_tokens: int
    partial: torch.Tensor
    peer_buffers: list[torch.Tensor]
    peer_ptrs: torch.Tensor
    write_stage: torch.Tensor | None
    write_handle: Any | None
    write_peer_buffers: list[torch.Tensor] | None
    peer_write_ptrs: torch.Tensor | None
    max_c4_writes: int
    handle: Any
    generation: int = 0


def create_cp_compressor_state(
    group: dist.ProcessGroup,
    rank: int,
    ratio: int,
    head_dim: int,
    max_global_tokens: int,
    device: torch.device,
) -> CPCompressorState:
    """Collectively allocate a double-buffered partial-state object."""
    world_size = group.size()
    if world_size != 4:
        raise ValueError(f"CP compressor currently requires CP4, got {world_size}")
    if ratio not in (4, 128):
        raise ValueError(f"unsupported compression ratio {ratio}")
    if max_global_tokens % ratio:
        raise ValueError("max_global_tokens must align to the compression ratio")
    max_windows = max_global_tokens // ratio
    symm_mem.set_signal_pad_size(max(symm_mem.get_signal_pad_size(), world_size * 4))
    with torch.inference_mode(False), torch.no_grad():
        partial = symm_mem.empty(
            (2, max_windows, 3, head_dim), dtype=torch.float32, device=device
        )
    handle = symm_mem.rendezvous(partial, group=group)
    peers = [
        handle.get_buffer(peer, list(partial.shape), torch.float32)
        for peer in range(world_size)
    ]
    peer_ptrs = torch.tensor(
        [peer.data_ptr() for peer in peers], dtype=torch.uint64, device=device
    )
    write_stage = None
    write_handle = None
    write_peers = None
    peer_write_ptrs = None
    max_c4_writes = 0
    if ratio == 4:
        # Aligned C4 writes four tail rows per 128-token SWA page plus the final
        # four rows. Keep this sparse write-set separate from token projections.
        max_c4_writes = max_global_tokens // 32 + 4
        with torch.inference_mode(False), torch.no_grad():
            write_stage = symm_mem.empty(
                (2, max_c4_writes, 4, head_dim),
                dtype=torch.float32,
                device=device,
            )
        write_handle = symm_mem.rendezvous(write_stage, group=group)
        write_peers = [
            write_handle.get_buffer(peer, list(write_stage.shape), torch.float32)
            for peer in range(world_size)
        ]
        peer_write_ptrs = torch.tensor(
            [peer.data_ptr() for peer in write_peers],
            dtype=torch.uint64,
            device=device,
        )
    return CPCompressorState(
        group=group,
        rank=rank,
        world_size=world_size,
        ratio=ratio,
        head_dim=head_dim,
        max_global_tokens=max_global_tokens,
        partial=partial,
        peer_buffers=peers,
        peer_ptrs=peer_ptrs,
        write_stage=write_stage,
        write_handle=write_handle,
        write_peer_buffers=write_peers,
        peer_write_ptrs=peer_write_ptrs,
        max_c4_writes=max_c4_writes,
        handle=handle,
    )


def cp_compress_aligned(
    state: CPCompressorState,
    kv_score_local: torch.Tensor,
    ape: torch.Tensor,
    *,
    prefix_tokens: int = 0,
    c4_carry: torch.Tensor | None = None,
    c4_plan_c: torch.Tensor | None = None,
    c4_plan_w: torch.Tensor | None = None,
    c4_state_buffer: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compress one aligned, single-request, round-robin CP4 token shard.

    Returns the compact FP32 ``[num_windows, head_dim]`` output expected by
    ``compress_forward`` and ``compress_norm_rope_store``.
    """
    ratio, head_dim, world_size = state.ratio, state.head_dim, state.world_size
    if kv_score_local.dtype != torch.float32 or not kv_score_local.is_contiguous():
        raise ValueError("kv_score_local must be contiguous FP32")
    expected_width = (4 if ratio == 4 else 2) * head_dim
    if kv_score_local.ndim != 2 or kv_score_local.shape[1] != expected_width:
        raise ValueError(
            f"expected local shape [tokens,{expected_width}], got {tuple(kv_score_local.shape)}"
        )
    global_tokens = kv_score_local.shape[0] * world_size
    if not 0 < global_tokens <= state.max_global_tokens or global_tokens % ratio:
        raise ValueError(f"unaligned global token count {global_tokens}")
    if prefix_tokens % ratio:
        raise ValueError(f"unaligned prefix token count {prefix_tokens}")
    if ape.shape != ((8 if ratio == 4 else 128), head_dim):
        raise ValueError(f"unexpected APE shape {tuple(ape.shape)}")
    if ape.dtype != torch.float32 or not ape.is_contiguous():
        raise ValueError("APE must be contiguous FP32")

    num_windows = global_tokens // ratio
    max_windows = state.max_global_tokens // ratio
    slot = state.generation & 1
    slot_offset = slot * max_windows * 3 * head_dim
    block = min(256, triton.next_power_of_2(head_dim))
    grid = (num_windows, triton.cdiv(head_dim, block))
    if ratio == 128:
        _c128_local_state_kernel[grid](
            kv_score_local,
            ape,
            state.partial,
            num_windows,
            slot_offset,
            RANK=state.rank,
            WORLD_SIZE=world_size,
            HEAD_DIM=head_dim,
            BLOCK=block,
            num_warps=4,
        )
    else:
        use_paged_prefix = c4_plan_c is not None and c4_state_buffer is not None
        store_carry = c4_carry is not None
        if prefix_tokens and not use_paged_prefix:
            if c4_carry is None or c4_carry.shape != (2, head_dim):
                raise ValueError(f"C4 prefix requires carry shape (2,{head_dim})")
            if c4_carry.dtype != torch.float32 or not c4_carry.is_contiguous():
                raise ValueError("C4 carry must be contiguous FP32")
        if c4_carry is None:
            c4_carry = kv_score_local
        if c4_plan_c is not None:
            if (
                c4_plan_c.dtype != torch.uint8
                or c4_plan_c.shape != (num_windows, 16)
                or not c4_plan_c.is_contiguous()
            ):
                raise ValueError("C4 plan_c must be contiguous [num_windows,16] uint8")
        if use_paged_prefix:
            assert c4_state_buffer is not None
            if c4_state_buffer.ndim != 2 or c4_state_buffer.shape[1] != 4 * head_dim:
                raise ValueError("C4 state buffer must have shape [rows,4*head_dim]")
        _c4_local_state_kernel[grid](
            kv_score_local,
            ape,
            c4_carry,
            c4_plan_c if c4_plan_c is not None else kv_score_local,
            c4_state_buffer if c4_state_buffer is not None else kv_score_local,
            state.partial,
            num_windows,
            slot_offset,
            HAS_PREFIX=prefix_tokens > 0,
            USE_PAGED_PREFIX=use_paged_prefix,
            STORE_CARRY=store_carry,
            RANK=state.rank,
            HEAD_DIM=head_dim,
            BLOCK=block,
            num_warps=4,
        )
        if (c4_plan_w is None) != (c4_state_buffer is None):
            raise ValueError("C4 plan_w and state buffer must be provided together")
        if c4_plan_w is not None:
            if c4_plan_w.dtype != torch.uint8 or c4_plan_w.shape[-1] != 8:
                raise ValueError("C4 plan_w must have shape [W,8] uint8")
            if (
                c4_state_buffer.dtype != torch.float32
                or not c4_state_buffer.is_contiguous()
            ):
                raise ValueError("C4 state buffer must be contiguous FP32")
            num_write = c4_plan_w.shape[0]
            if num_write > state.max_c4_writes:
                raise ValueError(
                    f"C4 plan has {num_write} writes, capacity is {state.max_c4_writes}"
                )
            assert state.write_stage is not None
            write_slot_offset = slot * state.max_c4_writes * 4 * head_dim
            if num_write:
                write_grid = (num_write, triton.cdiv(4 * head_dim, block))
                _stage_c4_writes_kernel[write_grid](
                    kv_score_local,
                    c4_plan_w,
                    state.write_stage,
                    num_write,
                    kv_score_local.shape[0],
                    write_slot_offset,
                    RANK=state.rank,
                    HEAD_DIM=head_dim,
                    BLOCK=block,
                    num_warps=4,
                )

    # Release local partial writes and acquire all peer publications. Double
    # buffering makes this the only barrier needed for safe reuse.
    _peer_barrier_kernel[(1,)](
        state.handle.signal_pad_ptrs_dev,
        RANK=state.rank,
        WORLD_SIZE=world_size,
        num_warps=1,
    )
    output = kv_score_local.new_empty((num_windows, head_dim))
    _merge_peer_states_kernel[grid](
        state.peer_ptrs,
        output,
        num_windows,
        slot_offset,
        RATIO=ratio,
        WORLD_SIZE=world_size,
        HEAD_DIM=head_dim,
        BLOCK=block,
        num_warps=4,
    )
    if ratio == 4 and c4_plan_w is not None and num_write:
        assert state.peer_write_ptrs is not None
        _consume_c4_writes_kernel[write_grid](
            state.peer_write_ptrs,
            c4_plan_w,
            c4_state_buffer,
            num_write,
            c4_state_buffer.shape[0],
            write_slot_offset,
            HEAD_DIM=head_dim,
            BLOCK=block,
            num_warps=4,
        )
    # No read-complete barrier is needed: before this parity is reused two
    # calls later, every rank must enter and leave the intervening publication
    # barrier, which happens after its reads from this call. Double buffering
    # therefore makes that next publication the reuse fence.
    state.generation += 1
    return output
