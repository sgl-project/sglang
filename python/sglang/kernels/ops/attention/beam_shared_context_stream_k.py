# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Stream-K shared-context attention for wide beam decoding.

Opt-in with SGLANG_BEAM_SHARED_CONTEXT_KERNEL=stream_k. Every beam of a group
attends to the group's shared prompt and to its own decode tokens. A tile is
BLOCK_M beams of one group and one query head. Its work units are the prompt's
BLOCK_N-token blocks (tensor-core products) and the group's decode steps
(CUDA-core dot products), after SETUP_UNITS that stand for the cost of starting
a tile.

The grid is persistent (stream-K): one CTA per SM, each taking an equal run of
the units of all tiles in order, so a run may end inside a tile and continue
in the next one. A tile covered by one CTA is written directly. A tile split
across CTAs leaves one partial output and log-sum-exp per CTA, which the
merge kernel combines. Unit counts come from the runtime prompt and decode
lengths, while the grid and workspace sizes depend only on static shapes, so
CUDA graphs replay the schedule for any lengths. Q and the output rows are read
and written through `row_indices`, like the default kernel.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.beam_shared_context import (
    attend_beam_token,
    attend_prompt_block,
    check_shared_context_inputs,
    empty_softmax_state,
    max_element_offset,
    num_sms,
    wave_ctas,
)

# Split-KV's tile, not re-swept here; RTX PRO 6000, 256 beams, head_dim 128.
# Shared memory: 64 KB at head_dim <= 128; 144 KB up to 256, past SM86/89/120's 99 KB.
_BLOCK_M = 128
_BLOCK_N = 64
_NUM_WARPS = 8
_NUM_STAGES = 2
# Measured on RTX PRO 6000: one unit per tile start balances best (0-3 tried);
# starting a segment loads Q and ends in an output store.
_SETUP_UNITS = 1
_MERGE_ROWS = 16
# Triton computes these offsets in 32 bits.
_MAX_OFFSET = 2**31


@triton.jit
def _units_per_tile(
    context_cu_seqlens,  # [groups + 1] int32 offsets into context_slots
    decode_lens,  # [groups] int32, decode steps per group
    group,  # the group whose tiles are counted
    BLOCK_N: tl.constexpr,  # prompt tokens per work unit
    SETUP_UNITS: tl.constexpr,  # units charged for starting a tile
):
    context_length = tl.load(context_cu_seqlens + group + 1) - tl.load(
        context_cu_seqlens + group
    )
    return SETUP_UNITS + tl.cdiv(context_length, BLOCK_N) + tl.load(decode_lens + group)


@triton.jit
def _total_units(
    context_cu_seqlens,  # [groups + 1] int32 offsets into context_slots
    decode_lens,  # [groups] int32, decode steps per group
    num_groups,  # groups in the bucket
    TILES_PER_GROUP: tl.constexpr,  # Q_HEADS * BEAM_BLOCKS
    BLOCK_N: tl.constexpr,  # prompt tokens per work unit
    SETUP_UNITS: tl.constexpr,  # units charged for starting a tile
):
    total = 0
    for group in range(0, num_groups):
        total += TILES_PER_GROUP * _units_per_tile(
            context_cu_seqlens, decode_lens, group, BLOCK_N, SETUP_UNITS
        )
    return total


@triton.jit
def _locate_unit(
    unit,  # index into the units of all tiles, in order
    context_cu_seqlens,  # [groups + 1] int32 offsets into context_slots
    decode_lens,  # [groups] int32, decode steps per group
    num_groups,  # groups in the bucket
    TILES_PER_GROUP: tl.constexpr,  # Q_HEADS * BEAM_BLOCKS
    BLOCK_N: tl.constexpr,  # prompt tokens per work unit
    SETUP_UNITS: tl.constexpr,  # units charged for starting a tile
):
    """Group of a unit, the group's first unit and its units per tile."""
    group_first_unit = 0
    found_group = 0
    found_first_unit = 0
    found_units_per_tile = 1
    for group in range(0, num_groups):
        units_per_tile = _units_per_tile(
            context_cu_seqlens, decode_lens, group, BLOCK_N, SETUP_UNITS
        )
        group_units = TILES_PER_GROUP * units_per_tile
        found = (unit >= group_first_unit) & (unit < group_first_unit + group_units)
        found_group = tl.where(found, group, found_group)
        found_first_unit = tl.where(found, group_first_unit, found_first_unit)
        found_units_per_tile = tl.where(found, units_per_tile, found_units_per_tile)
        group_first_unit += group_units
    return found_group, found_first_unit, found_units_per_tile


@triton.jit
def _unit_owner(
    unit,  # index into the units of all tiles, in order
    total_units,  # units of all tiles
    num_ctas,  # CTAs in the persistent grid
):
    # CTA c owns units [c * total // num_ctas, (c + 1) * total // num_ctas).
    return ((unit + 1) * num_ctas - 1) // total_units


@triton.jit
def _shared_context_stream_k_kernel(
    q,  # [rows, q_heads, HEAD_DIM], contiguous along HEAD_DIM
    k_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's K cache
    v_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's V cache
    row_indices,  # [groups, BEAM_WIDTH] int32, each beam's row of q and output
    context_slots,  # [>= total prompt tokens] int32 KV slots, groups back to back
    context_cu_seqlens,  # [groups + 1] int32, each group's start in context_slots
    decode_slots,  # [groups, capacity * BEAM_WIDTH] int32 KV slots, step-major
    decode_lens,  # [groups] int32, decode steps each beam attends to
    output,  # like q; written here for tiles that one CTA covers
    partial_output,  # fp32 [tiles + num_ctas, BLOCK_M, HEAD_DIM], contiguous
    partial_lse,  # fp32 [tiles + num_ctas, BLOCK_M], log2 softmax sums
    softmax_scale,  # float, scales q.k before the softmax
    num_groups,  # groups in the bucket, row_indices.shape[0]
    num_ctas,  # CTAs in the persistent grid
    stride_q_row,  # q.stride(0); all strides are in elements
    stride_q_head,  # q.stride(1)
    stride_row_group,  # row_indices.stride(0)
    stride_row_beam,  # row_indices.stride(1)
    stride_k_slot,  # k_buffer.stride(0)
    stride_k_head,  # k_buffer.stride(1)
    stride_v_slot,  # v_buffer.stride(0)
    stride_v_head,  # v_buffer.stride(1)
    stride_decode_group,  # decode_slots.stride(0); stride(1) must be 1
    stride_out_row,  # output.stride(0)
    stride_out_head,  # output.stride(1)
    BEAM_WIDTH: tl.constexpr,  # beams per group, row_indices.shape[1]
    Q_PER_KV: tl.constexpr,  # query heads per KV head
    Q_HEADS: tl.constexpr,  # query heads
    BEAM_BLOCKS: tl.constexpr,  # tiles per group and query head
    HEAD_DIM: tl.constexpr,  # q.shape[2]
    BLOCK_M: tl.constexpr,  # beams per tile
    BLOCK_N: tl.constexpr,  # prompt tokens per work unit
    BLOCK_D: tl.constexpr,  # HEAD_DIM rounded up to a power of 2
    SETUP_UNITS: tl.constexpr,  # units charged for starting a tile
    WIDE_KV_OFFSETS: tl.constexpr,  # K/V offsets past 2^31: slot * stride needs 64 bits
):
    TILES_PER_GROUP: tl.constexpr = Q_HEADS * BEAM_BLOCKS
    cta = tl.program_id(0)
    total_units = _total_units(
        context_cu_seqlens,
        decode_lens,
        num_groups,
        TILES_PER_GROUP,
        BLOCK_N,
        SETUP_UNITS,
    )
    run_end_unit = ((cta + 1) * total_units) // num_ctas

    beams_in_tile = tl.arange(0, BLOCK_M)
    context_offsets = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    negative_infinity = -float("inf")
    softmax_scale_log2 = softmax_scale * 1.4426950408889634

    # Units number every tile's work in order: SETUP_UNITS, then one per prompt block,
    # then one per decode step. Each CTA owns an equal run of units across tiles;
    # each loop pass handles one segment, the part of the run inside one tile.
    # work_*_in_tile index that tile's prompt blocks, then its decode steps, from 0
    # (setup units excluded); partial_idx = tile + cta holds a split tile's partials.
    segment_start_unit = (cta * total_units) // num_ctas
    while segment_start_unit < run_end_unit:
        group, group_first_unit, units_per_tile = _locate_unit(
            segment_start_unit,
            context_cu_seqlens,
            decode_lens,
            num_groups,
            TILES_PER_GROUP,
            BLOCK_N,
            SETUP_UNITS,
        )
        tile_in_group = (segment_start_unit - group_first_unit) // units_per_tile
        tile_first_unit = group_first_unit + tile_in_group * units_per_tile
        tile_end_unit = tile_first_unit + units_per_tile
        segment_end_unit = tl.minimum(run_end_unit, tile_end_unit)
        # A segment of setup units only has no work items, and is skipped.
        work_start_in_tile = (
            tl.maximum(segment_start_unit - tile_first_unit, SETUP_UNITS) - SETUP_UNITS
        )
        work_end_in_tile = segment_end_unit - tile_first_unit - SETUP_UNITS
        if work_start_in_tile < work_end_in_tile:
            q_head = tile_in_group // BEAM_BLOCKS
            kv_head = q_head // Q_PER_KV
            beams = (tile_in_group % BEAM_BLOCKS) * BLOCK_M + beams_in_tile
            beam_mask = beams < BEAM_WIDTH
            rows = tl.load(
                row_indices + group * stride_row_group + beams * stride_row_beam,
                mask=beam_mask,
                other=0,
            )
            row_mask = beam_mask[:, None] & dim_mask[None, :]

            q_values = tl.load(
                q
                + rows[:, None] * stride_q_row
                + q_head * stride_q_head
                + dims[None, :],
                mask=row_mask,
                other=0.0,
            )
            max_values, normalizers, accumulator = empty_softmax_state(
                beam_mask, BLOCK_M, BLOCK_D
            )

            context_start = tl.load(context_cu_seqlens + group)
            context_length = tl.load(context_cu_seqlens + group + 1) - context_start
            num_prompt_blocks = tl.cdiv(context_length, BLOCK_N)
            for block in tl.range(
                work_start_in_tile, tl.minimum(work_end_in_tile, num_prompt_blocks)
            ):
                tokens = block * BLOCK_N + context_offsets
                token_mask = tokens < context_length
                slots = tl.load(
                    context_slots + context_start + tokens,
                    mask=token_mask,
                    other=0,
                )
                if WIDE_KV_OFFSETS:
                    slots = slots.to(tl.int64)
                kv_mask = token_mask[:, None] & dim_mask[None, :]
                k_values = tl.load(
                    k_buffer
                    + slots[:, None] * stride_k_slot
                    + kv_head * stride_k_head
                    + dims[None, :],
                    mask=kv_mask,
                    other=0.0,
                )
                v_values = tl.load(
                    v_buffer
                    + slots[:, None] * stride_v_slot
                    + kv_head * stride_v_head
                    + dims[None, :],
                    mask=kv_mask,
                    other=0.0,
                )

                scores = tl.dot(q_values, tl.trans(k_values))
                scores = tl.where(token_mask[None, :], scores, negative_infinity)
                max_values, normalizers, accumulator = attend_prompt_block(
                    scores,
                    v_values,
                    max_values,
                    normalizers,
                    accumulator,
                    softmax_scale_log2,
                )

            num_decode_steps = tl.load(decode_lens + group)
            for step in range(
                tl.maximum(work_start_in_tile - num_prompt_blocks, 0),
                tl.minimum(work_end_in_tile - num_prompt_blocks, num_decode_steps),
            ):
                slots = tl.load(
                    decode_slots
                    + group * stride_decode_group
                    + step * BEAM_WIDTH
                    + beams,
                    mask=beam_mask,
                    other=0,
                )
                if WIDE_KV_OFFSETS:
                    slots = slots.to(tl.int64)
                k_values = tl.load(
                    k_buffer
                    + slots[:, None] * stride_k_slot
                    + kv_head * stride_k_head
                    + dims[None, :],
                    mask=row_mask,
                    other=0.0,
                )
                v_values = tl.load(
                    v_buffer
                    + slots[:, None] * stride_v_slot
                    + kv_head * stride_v_head
                    + dims[None, :],
                    mask=row_mask,
                    other=0.0,
                )
                scores = tl.sum(
                    q_values.to(tl.float32) * k_values.to(tl.float32),
                    axis=1,
                )
                scores = tl.where(beam_mask, scores, negative_infinity)
                max_values, normalizers, accumulator = attend_beam_token(
                    scores,
                    v_values,
                    max_values,
                    normalizers,
                    accumulator,
                    softmax_scale_log2,
                )

            if (work_start_in_tile == 0) & (segment_end_unit == tile_end_unit):
                tl.store(
                    output
                    + rows[:, None] * stride_out_row
                    + q_head * stride_out_head
                    + dims[None, :],
                    accumulator / normalizers[:, None],
                    mask=row_mask,
                )
            else:
                # partial_idx is unique: along the units, each new segment
                # advances the tile, the CTA or both.
                partial_idx = group * TILES_PER_GROUP + tile_in_group + cta
                # Every work unit holds a token, so every segment has softmax mass.
                tl.store(
                    partial_output
                    + ((partial_idx * BLOCK_M + beams_in_tile) * HEAD_DIM)[:, None]
                    + dims[None, :],
                    accumulator / normalizers[:, None],
                    mask=row_mask,
                )
                tl.store(
                    partial_lse + partial_idx * BLOCK_M + beams_in_tile,
                    max_values * softmax_scale_log2 + tl.log2(normalizers),
                    mask=beam_mask,
                )
        segment_start_unit = segment_end_unit


@triton.jit
def _merge_segments_kernel(
    partial_output,  # fp32 [tiles + num_ctas, BLOCK_M, HEAD_DIM], contiguous
    partial_lse,  # fp32 [tiles + num_ctas, BLOCK_M], log2 softmax sums
    row_indices,  # [groups, BEAM_WIDTH] int32, each beam's row of output
    output,  # [rows, Q_HEADS, HEAD_DIM]; written for tiles split across CTAs
    context_cu_seqlens,  # [groups + 1] int32 offsets into context_slots
    decode_lens,  # [groups] int32, decode steps per group
    num_groups,  # groups in the bucket, row_indices.shape[0]
    num_ctas,  # CTAs of the stream-K grid that wrote the partials
    stride_row_group,  # row_indices.stride(0); all strides are in elements
    stride_row_beam,  # row_indices.stride(1)
    stride_out_row,  # output.stride(0)
    stride_out_head,  # output.stride(1)
    BEAM_WIDTH: tl.constexpr,  # beams per group, row_indices.shape[1]
    Q_HEADS: tl.constexpr,  # query heads
    BEAM_BLOCKS: tl.constexpr,  # tiles per group and query head
    HEAD_DIM: tl.constexpr,  # output.shape[2]
    BLOCK_M: tl.constexpr,  # beams per tile
    BLOCK_N: tl.constexpr,  # prompt tokens per work unit
    BLOCK_D: tl.constexpr,  # HEAD_DIM rounded up to a power of 2
    SETUP_UNITS: tl.constexpr,  # units charged for starting a tile
    MERGE_ROWS: tl.constexpr,  # beams merged per program
):
    TILES_PER_GROUP: tl.constexpr = Q_HEADS * BEAM_BLOCKS
    tile = tl.program_id(0)
    group = tile // TILES_PER_GROUP
    tile_in_group = tile % TILES_PER_GROUP
    total_units = _total_units(
        context_cu_seqlens,
        decode_lens,
        num_groups,
        TILES_PER_GROUP,
        BLOCK_N,
        SETUP_UNITS,
    )
    group_first_unit = 0
    for previous_group in range(0, group):
        group_first_unit += TILES_PER_GROUP * _units_per_tile(
            context_cu_seqlens, decode_lens, previous_group, BLOCK_N, SETUP_UNITS
        )
    units_per_tile = _units_per_tile(
        context_cu_seqlens, decode_lens, group, BLOCK_N, SETUP_UNITS
    )
    tile_first_work_unit = (
        group_first_unit + tile_in_group * units_per_tile + SETUP_UNITS
    )
    tile_end_unit = group_first_unit + (tile_in_group + 1) * units_per_tile
    first_cta = _unit_owner(tile_first_work_unit, total_units, num_ctas)
    last_cta = _unit_owner(tile_end_unit - 1, total_units, num_ctas)
    if first_cta != last_cta:
        q_head = tile_in_group // BEAM_BLOCKS
        beams_in_tile = tl.program_id(1) * MERGE_ROWS + tl.arange(0, MERGE_ROWS)
        beams = (tile_in_group % BEAM_BLOCKS) * BLOCK_M + beams_in_tile
        beam_mask = beams < BEAM_WIDTH
        dims = tl.arange(0, BLOCK_D)
        mask = beam_mask[:, None] & (dims < HEAD_DIM)[None, :]
        negative_infinity = -float("inf")

        max_lse = tl.full((MERGE_ROWS,), negative_infinity, tl.float32)
        total = tl.zeros((MERGE_ROWS,), tl.float32)
        accumulator = tl.zeros((MERGE_ROWS, BLOCK_D), tl.float32)
        for cta in range(first_cta, last_cta + 1):
            # Only when units < CTAs can a CTA in the range own no work unit here.
            segment_start_unit = tl.maximum(
                (cta * total_units) // num_ctas, tile_first_work_unit
            )
            segment_end_unit = tl.minimum(
                ((cta + 1) * total_units) // num_ctas, tile_end_unit
            )
            has_segment = segment_start_unit < segment_end_unit
            partial_idx = tile + cta
            lse = tl.load(
                partial_lse + partial_idx * BLOCK_M + beams_in_tile,
                mask=beam_mask & has_segment,
                other=negative_infinity,
            )
            part = tl.load(
                partial_output
                + ((partial_idx * BLOCK_M + beams_in_tile) * HEAD_DIM)[:, None]
                + dims[None, :],
                mask=mask & has_segment,
                other=0.0,
            )
            new_max = tl.maximum(max_lse, lse)
            safe_max = tl.where(new_max == negative_infinity, 0.0, new_max)
            previous_scale = tl.exp2(max_lse - safe_max)
            weight = tl.exp2(lse - safe_max)
            accumulator = accumulator * previous_scale[:, None] + part * weight[:, None]
            total = total * previous_scale + weight
            max_lse = new_max

        rows = tl.load(
            row_indices + group * stride_row_group + beams * stride_row_beam,
            mask=beam_mask,
            other=0,
        )
        tl.store(
            output
            + rows[:, None] * stride_out_row
            + q_head * stride_out_head
            + dims[None, :],
            accumulator / total[:, None],
            mask=mask,
        )


def shared_context_attention_stream_k(
    q: torch.Tensor,  # [rows, q_heads, head_dim]
    k_buffer: torch.Tensor,  # [kv_slots, kv_heads, head_dim], the layer's K cache
    v_buffer: torch.Tensor,  # [kv_slots, kv_heads, head_dim], the layer's V cache
    row_indices: torch.Tensor,  # [groups, beam_width] int32, each beam's row of q
    context_slots: torch.Tensor,  # [>= total prompt tokens] int32 KV slots
    context_cu_seqlens: torch.Tensor,  # [groups + 1] int32 offsets into context_slots
    decode_slots: torch.Tensor,  # [groups, capacity * beam_width] int32, step-major
    decode_lens: torch.Tensor,  # [groups] int32, decode steps per group
    *,
    output: torch.Tensor | None = None,  # like q; allocated when None
    softmax_scale: float,  # scales q.k before the softmax
    num_ctas: Optional[int] = None,  # persistent grid size
) -> torch.Tensor:
    """Shared prompt and per-beam suffix attention on a persistent stream-K grid.

    Same contract as `shared_context_attention`: `row_indices` [groups, beam_width]
    maps each beam to its row of Q and of the output, and `decode_slots` is
    [groups, capacity * beam_width], step-major. Only those rows are written.
    Every group needs a prompt token or a decode step; the output of a group with
    neither is undefined. `num_ctas` defaults to the CTAs the GPU holds at once.
    """
    output = check_shared_context_inputs(
        q,
        k_buffer,
        v_buffer,
        row_indices,
        context_cu_seqlens,
        decode_slots,
        decode_lens,
        output,
    )
    if q.stride(2) != 1 or k_buffer.stride(2) != 1 or v_buffer.stride(2) != 1:
        raise ValueError("Q and K/V buffers must be contiguous along head_dim")
    if output.stride(2) != 1:
        raise ValueError("The output must be contiguous along head_dim")
    _, q_heads, head_dim = q.shape
    if k_buffer.shape[2] != head_dim:
        raise ValueError("Q and K/V buffers must have the same head_dim")
    if q.numel() >= _MAX_OFFSET:
        raise ValueError("Q must have fewer than 2^31 elements")
    if decode_slots.stride(1) != 1:
        raise ValueError("Decode slots must be contiguous [groups, steps * beam_width]")
    num_groups, beam_width = row_indices.shape
    kv_heads = k_buffer.shape[1]

    beam_blocks = triton.cdiv(beam_width, _BLOCK_M)
    block_d = triton.next_power_of_2(head_dim)
    constexprs = dict(
        BEAM_WIDTH=beam_width,
        Q_PER_KV=q_heads // kv_heads,
        Q_HEADS=q_heads,
        BEAM_BLOCKS=beam_blocks,
        HEAD_DIM=head_dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=block_d,
        SETUP_UNITS=_SETUP_UNITS,
        WIDE_KV_OFFSETS=max(map(max_element_offset, (k_buffer, v_buffer)))
        >= _MAX_OFFSET,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )

    def kernel_args(partial_output, partial_lse, ctas):
        return (
            q,
            k_buffer,
            v_buffer,
            row_indices,
            context_slots,
            context_cu_seqlens,
            decode_slots,
            decode_lens,
            output,
            partial_output,
            partial_lse,
            softmax_scale,
            num_groups,
            ctas,
            q.stride(0),
            q.stride(1),
            row_indices.stride(0),
            row_indices.stride(1),
            k_buffer.stride(0),
            k_buffer.stride(1),
            v_buffer.stride(0),
            v_buffer.stride(1),
            decode_slots.stride(0),
            output.stride(0),
            output.stride(1),
        )

    if num_ctas is None:
        # Persistent: one full wave. The probe passes the partials' dtype before they
        # are allocated, since their size depends on the CTA count.
        num_ctas = wave_ctas(
            key=("stream_k", q.device.index, q.dtype, k_buffer.dtype)
            + tuple(sorted(constexprs.items())),
            compile_kernel=lambda: _shared_context_stream_k_kernel.warmup(
                *kernel_args(torch.float32, torch.float32, num_sms(q.device.index)),
                grid=(1,),
                **constexprs,
            ),
            device_index=q.device.index,
        )
    num_tiles = num_groups * q_heads * beam_blocks
    # Unit counts are bounded by the static slot capacities, which CUDA graphs fix.
    decode_capacity = decode_slots.shape[1] // beam_width
    max_units = (
        q_heads
        * beam_blocks
        * (
            num_groups * (_SETUP_UNITS + 1 + decode_capacity)
            + triton.cdiv(context_slots.numel(), _BLOCK_N)
        )
    )
    if (max_units + 1) * num_ctas >= _MAX_OFFSET:
        raise ValueError(
            f"Up to {max_units} work units on {num_ctas} CTAs overflow 32-bit unit offsets"
        )
    if (num_tiles + num_ctas) * _BLOCK_M * head_dim >= _MAX_OFFSET:
        raise ValueError("The stream-K partials must have fewer than 2^31 elements")
    # One partial row per (tile, CTA) segment: partial_idx = tile + cta < tiles + CTAs.
    partial_output = torch.empty(
        (num_tiles + num_ctas, _BLOCK_M, head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    partial_lse = torch.empty(
        (num_tiles + num_ctas, _BLOCK_M),
        dtype=torch.float32,
        device=q.device,
    )

    _shared_context_stream_k_kernel[(num_ctas,)](
        *kernel_args(partial_output, partial_lse, num_ctas),
        **constexprs,
    )
    _merge_segments_kernel[(num_tiles, triton.cdiv(_BLOCK_M, _MERGE_ROWS))](
        partial_output,
        partial_lse,
        row_indices,
        output,
        context_cu_seqlens,
        decode_lens,
        num_groups,
        num_ctas,
        row_indices.stride(0),
        row_indices.stride(1),
        output.stride(0),
        output.stride(1),
        BEAM_WIDTH=beam_width,
        Q_HEADS=q_heads,
        BEAM_BLOCKS=beam_blocks,
        HEAD_DIM=head_dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=block_d,
        SETUP_UNITS=_SETUP_UNITS,
        MERGE_ROWS=_MERGE_ROWS,
        num_warps=4,
    )
    return output
