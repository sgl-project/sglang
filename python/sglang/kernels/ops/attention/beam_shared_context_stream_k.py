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
    num_sms,
)

# Measured on RTX PRO 6000 (SM120) at 256 beams, head_dim 128, 1-8 groups.
_BLOCK_M = 128
_BLOCK_N = 64
_NUM_WARPS = 8
_NUM_STAGES = 2
# Measured: one unit per tile start balances best (0-3 tried); starting a
# segment loads Q and ends in an output store.
_SETUP_UNITS = 1
_MERGE_ROWS = 16
# Triton computes these offsets in 32 bits.
_MAX_OFFSET = 2**31


@triton.jit
def _tile_units(
    context_cu_seqlens,
    decode_lens,
    group,
    BLOCK_N: tl.constexpr,
    SETUP_UNITS: tl.constexpr,
):
    context_length = tl.load(context_cu_seqlens + group + 1) - tl.load(
        context_cu_seqlens + group
    )
    return SETUP_UNITS + tl.cdiv(context_length, BLOCK_N) + tl.load(decode_lens + group)


@triton.jit
def _total_units(
    context_cu_seqlens,
    decode_lens,
    num_groups,
    TILES_PER_GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SETUP_UNITS: tl.constexpr,
):
    total = 0
    for group in range(0, num_groups):
        total += TILES_PER_GROUP * _tile_units(
            context_cu_seqlens, decode_lens, group, BLOCK_N, SETUP_UNITS
        )
    return total


@triton.jit
def _locate_unit(
    unit,
    context_cu_seqlens,
    decode_lens,
    num_groups,
    TILES_PER_GROUP: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SETUP_UNITS: tl.constexpr,
):
    """Group of a unit, the group's first unit and its units per tile."""
    group_base = 0
    found_group = 0
    found_base = 0
    found_units = 1
    for group in range(0, num_groups):
        units = _tile_units(
            context_cu_seqlens, decode_lens, group, BLOCK_N, SETUP_UNITS
        )
        span = TILES_PER_GROUP * units
        found = (unit >= group_base) & (unit < group_base + span)
        found_group = tl.where(found, group, found_group)
        found_base = tl.where(found, group_base, found_base)
        found_units = tl.where(found, units, found_units)
        group_base += span
    return found_group, found_base, found_units


@triton.jit
def _unit_owner(unit, total_units, num_ctas):
    # CTA c owns units [c * total // num_ctas, (c + 1) * total // num_ctas).
    return ((unit + 1) * num_ctas - 1) // total_units


@triton.jit
def _shared_context_stream_k_kernel(
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
    num_ctas,
    stride_q_row,
    stride_q_head,
    stride_row_group,
    stride_row_beam,
    stride_k_slot,
    stride_k_head,
    stride_v_slot,
    stride_v_head,
    stride_decode_group,
    stride_out_row,
    stride_out_head,
    BEAM_WIDTH: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    Q_HEADS: tl.constexpr,
    BEAM_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SETUP_UNITS: tl.constexpr,
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
    run_end = ((cta + 1) * total_units) // num_ctas

    tile_beams = tl.arange(0, BLOCK_M)
    context_offsets = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    negative_infinity = -float("inf")
    softmax_scale_log2 = softmax_scale * 1.4426950408889634

    unit = (cta * total_units) // num_ctas
    while unit < run_end:
        group, group_base, tile_units = _locate_unit(
            unit,
            context_cu_seqlens,
            decode_lens,
            num_groups,
            TILES_PER_GROUP,
            BLOCK_N,
            SETUP_UNITS,
        )
        tile_in_group = (unit - group_base) // tile_units
        tile_start = group_base + tile_in_group * tile_units
        tile_end = tile_start + tile_units
        segment_end = tl.minimum(run_end, tile_end)
        # Unit offsets inside the tile; a run of setup units only is skipped.
        work_start = tl.maximum(unit - tile_start, SETUP_UNITS) - SETUP_UNITS
        work_end = segment_end - tile_start - SETUP_UNITS
        if work_start < work_end:
            q_head = tile_in_group // BEAM_BLOCKS
            kv_head = q_head // Q_PER_KV
            beams = (tile_in_group % BEAM_BLOCKS) * BLOCK_M + tile_beams
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
            num_blocks = tl.cdiv(context_length, BLOCK_N)
            for block in tl.range(work_start, tl.minimum(work_end, num_blocks)):
                tokens = block * BLOCK_N + context_offsets
                token_mask = tokens < context_length
                slots = tl.load(
                    context_slots + context_start + tokens,
                    mask=token_mask,
                    other=0,
                )
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

            decode_length = tl.load(decode_lens + group)
            for step in range(
                tl.maximum(work_start - num_blocks, 0),
                tl.minimum(work_end - num_blocks, decode_length),
            ):
                slots = tl.load(
                    decode_slots
                    + group * stride_decode_group
                    + step * BEAM_WIDTH
                    + beams,
                    mask=beam_mask,
                    other=0,
                )
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

            if (work_start == 0) & (segment_end == tile_end):
                tl.store(
                    output
                    + rows[:, None] * stride_out_row
                    + q_head * stride_out_head
                    + dims[None, :],
                    accumulator / normalizers[:, None],
                    mask=row_mask,
                )
            else:
                # Slot tile + cta is unique: along the units, each new segment
                # advances the tile, the CTA or both.
                slot = group * TILES_PER_GROUP + tile_in_group + cta
                # Every work unit holds a token, so every segment has softmax mass.
                tl.store(
                    partial_output
                    + ((slot * BLOCK_M + tile_beams) * HEAD_DIM)[:, None]
                    + dims[None, :],
                    accumulator / normalizers[:, None],
                    mask=row_mask,
                )
                tl.store(
                    partial_lse + slot * BLOCK_M + tile_beams,
                    max_values * softmax_scale_log2 + tl.log2(normalizers),
                    mask=beam_mask,
                )
        unit = segment_end


@triton.jit
def _merge_segments_kernel(
    partial_output,
    partial_lse,
    row_indices,
    output,
    context_cu_seqlens,
    decode_lens,
    num_groups,
    num_ctas,
    stride_row_group,
    stride_row_beam,
    stride_out_row,
    stride_out_head,
    BEAM_WIDTH: tl.constexpr,
    Q_HEADS: tl.constexpr,
    BEAM_BLOCKS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SETUP_UNITS: tl.constexpr,
    MERGE_ROWS: tl.constexpr,
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
    group_base = 0
    for previous_group in range(0, group):
        group_base += TILES_PER_GROUP * _tile_units(
            context_cu_seqlens, decode_lens, previous_group, BLOCK_N, SETUP_UNITS
        )
    tile_units = _tile_units(
        context_cu_seqlens, decode_lens, group, BLOCK_N, SETUP_UNITS
    )
    work_start = group_base + tile_in_group * tile_units + SETUP_UNITS
    tile_end = group_base + (tile_in_group + 1) * tile_units
    first_cta = _unit_owner(work_start, total_units, num_ctas)
    last_cta = _unit_owner(tile_end - 1, total_units, num_ctas)
    if first_cta != last_cta:
        q_head = tile_in_group // BEAM_BLOCKS
        tile_beams = tl.program_id(1) * MERGE_ROWS + tl.arange(0, MERGE_ROWS)
        beams = (tile_in_group % BEAM_BLOCKS) * BLOCK_M + tile_beams
        beam_mask = beams < BEAM_WIDTH
        dims = tl.arange(0, BLOCK_D)
        mask = beam_mask[:, None] & (dims < HEAD_DIM)[None, :]
        negative_infinity = -float("inf")

        max_lse = tl.full((MERGE_ROWS,), negative_infinity, tl.float32)
        total = tl.zeros((MERGE_ROWS,), tl.float32)
        accumulator = tl.zeros((MERGE_ROWS, BLOCK_D), tl.float32)
        for cta in range(first_cta, last_cta + 1):
            # Only when units < CTAs can a CTA in the range own no work unit here.
            run_start = tl.maximum((cta * total_units) // num_ctas, work_start)
            run_end = tl.minimum(((cta + 1) * total_units) // num_ctas, tile_end)
            has_segment = run_start < run_end
            slot = tile + cta
            lse = tl.load(
                partial_lse + slot * BLOCK_M + tile_beams,
                mask=beam_mask & has_segment,
                other=negative_infinity,
            )
            part = tl.load(
                partial_output
                + ((slot * BLOCK_M + tile_beams) * HEAD_DIM)[:, None]
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
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    row_indices: torch.Tensor,
    context_slots: torch.Tensor,
    context_cu_seqlens: torch.Tensor,
    decode_slots: torch.Tensor,
    decode_lens: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    softmax_scale: float,
    num_ctas: Optional[int] = None,
) -> torch.Tensor:
    """Shared prompt and per-beam suffix attention on a persistent stream-K grid.

    Same contract as `shared_context_attention`: `row_indices` [groups, beam_width]
    maps each beam to its row of Q and of the output, and `decode_slots` is
    [groups, capacity * beam_width], step-major. Only those rows are written.
    Every group needs a prompt token or a decode step; the output of a group with
    neither is undefined. `num_ctas` defaults to one CTA per SM.
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
    if max(k_buffer.numel(), v_buffer.numel(), q.numel()) >= _MAX_OFFSET:
        raise ValueError("Q and K/V buffers must have fewer than 2^31 elements")
    if decode_slots.stride(1) != 1:
        raise ValueError("Decode slots must be contiguous [groups, steps * beam_width]")
    num_groups, beam_width = row_indices.shape
    kv_heads = k_buffer.shape[1]

    if num_ctas is None:
        # One CTA per SM: 8 warps of this kernel fill an SM's register file.
        num_ctas = num_sms(q.device.index)
    beam_blocks = triton.cdiv(beam_width, _BLOCK_M)
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
    # One slot per (tile, CTA) segment; slot = tile + cta < num_tiles + num_ctas.
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

    block_d = triton.next_power_of_2(head_dim)
    _shared_context_stream_k_kernel[(num_ctas,)](
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
        num_ctas,
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
        BEAM_WIDTH=beam_width,
        Q_PER_KV=q_heads // kv_heads,
        Q_HEADS=q_heads,
        BEAM_BLOCKS=beam_blocks,
        HEAD_DIM=head_dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=block_d,
        SETUP_UNITS=_SETUP_UNITS,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
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
