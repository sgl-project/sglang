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
"""Split-KV shared-context attention for wide beam decoding.

Opt-in with SGLANG_BEAM_SHARED_CONTEXT_KERNEL=split_kv. Every beam of a group
attends to the group's shared prompt and to its own decode tokens. The prompt
part runs as tensor-core tiles of BLOCK_M beams of one query head, split along
the prompt (split-KV) so that a few groups still fill the GPU; the last split
adds the beams' own decode tokens with CUDA-core dot products, and a merge
kernel combines the splits' partial results. Q and the output rows are read and
written through `row_indices`, like the default kernel.
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

# Measured on RTX PRO 6000 (SM120) at 256 beams, head_dim 128: fastest at 1-8
# groups and 256-2048 prompt tokens.
_BLOCK_M = 128
_BLOCK_N = 64
_NUM_WARPS = 8
_NUM_STAGES = 2
_MAX_SPLITS = 8
# Measured: the smallest split count that fills this share of the last wave is
# within 1-2% of the best one at 1-8 groups; more splits only add merge traffic.
_MIN_WAVE_EFFICIENCY = 0.8


@triton.jit
def _shared_context_split_kv_kernel(
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
    num_splits,
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
    stride_partial_split,
    stride_partial_row,
    stride_partial_head,
    stride_lse_split,
    stride_lse_row,
    BEAM_WIDTH: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WRITE_PARTIALS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    group = tl.program_id(0) // num_splits
    split = tl.program_id(0) % num_splits
    q_head = tl.program_id(1)
    kv_head = q_head // Q_PER_KV

    beams = tl.program_id(2) * BLOCK_M + tl.arange(0, BLOCK_M)
    beam_mask = beams < BEAM_WIDTH
    rows = tl.load(
        row_indices + group * stride_row_group + beams * stride_row_beam,
        mask=beam_mask,
        other=0,
    )
    context_offsets = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    row_mask = beam_mask[:, None] & dim_mask[None, :]

    q_values = tl.load(
        q + rows[:, None] * stride_q_row + q_head * stride_q_head + dims[None, :],
        mask=row_mask,
        other=0.0,
    )

    negative_infinity = -float("inf")
    max_values, normalizers, accumulator = empty_softmax_state(
        beam_mask, BLOCK_M, BLOCK_D
    )
    softmax_scale_log2 = softmax_scale * 1.4426950408889634

    context_start = tl.load(context_cu_seqlens + group)
    context_length = tl.load(context_cu_seqlens + group + 1) - context_start
    # Split bounds come from the runtime length: a graph replays shorter prompts
    # than it was captured with, which leaves trailing splits empty.
    split_blocks = tl.cdiv(tl.cdiv(context_length, BLOCK_N), num_splits)
    split_start = split * split_blocks * BLOCK_N
    split_end = tl.minimum(split_start + split_blocks * BLOCK_N, context_length)

    for block_start in tl.range(split_start, split_end, BLOCK_N):
        tokens = block_start + context_offsets
        token_mask = tokens < split_end
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

    if split == num_splits - 1:
        decode_length = tl.load(decode_lens + group)
        for step in range(0, decode_length):
            slots = tl.load(
                decode_slots + group * stride_decode_group + step * BEAM_WIDTH + beams,
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

    if WRITE_PARTIALS:
        # Partials are indexed by the bucket's own rows (group * BEAM_WIDTH + beam).
        local_rows = group * BEAM_WIDTH + beams
        # An empty split has no mass; its -inf log-sum-exp drops it from the merge.
        has_mass = normalizers > 0
        result = accumulator / tl.where(has_mass, normalizers, 1.0)[:, None]
        lse = tl.where(
            has_mass,
            max_values * softmax_scale_log2 + tl.log2(normalizers),
            negative_infinity,
        )
        tl.store(
            partial_output
            + split * stride_partial_split
            + local_rows[:, None] * stride_partial_row
            + q_head * stride_partial_head
            + dims[None, :],
            result,
            mask=row_mask,
        )
        tl.store(
            partial_lse
            + split * stride_lse_split
            + local_rows * stride_lse_row
            + q_head,
            lse,
            mask=beam_mask,
        )
    else:
        tl.store(
            output
            + rows[:, None] * stride_out_row
            + q_head * stride_out_head
            + dims[None, :],
            accumulator / normalizers[:, None],
            mask=row_mask,
        )


@triton.jit
def _merge_splits_kernel(
    partial_output,
    partial_lse,
    row_indices,
    output,
    num_splits,
    stride_partial_split,
    stride_partial_row,
    stride_partial_head,
    stride_lse_split,
    stride_lse_row,
    stride_row_group,
    stride_row_beam,
    stride_out_row,
    stride_out_head,
    BEAM_WIDTH: tl.constexpr,
    Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    local_row = tl.program_id(0)
    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    head_mask = heads < Q_HEADS
    mask = head_mask[:, None] & (dims < HEAD_DIM)[None, :]
    lse_pointers = partial_lse + local_row * stride_lse_row + heads

    negative_infinity = -float("inf")
    max_lse = tl.full((BLOCK_H,), negative_infinity, tl.float32)
    for split in range(0, num_splits):
        lse = tl.load(
            lse_pointers + split * stride_lse_split,
            mask=head_mask,
            other=negative_infinity,
        )
        max_lse = tl.maximum(max_lse, lse)
    max_lse = tl.where(max_lse == negative_infinity, 0.0, max_lse)

    total = tl.zeros((BLOCK_H,), tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)
    for split in range(0, num_splits):
        lse = tl.load(
            lse_pointers + split * stride_lse_split,
            mask=head_mask,
            other=negative_infinity,
        )
        weight = tl.exp2(lse - max_lse)
        part = tl.load(
            partial_output
            + split * stride_partial_split
            + local_row * stride_partial_row
            + heads[:, None] * stride_partial_head
            + dims[None, :],
            mask=mask,
            other=0.0,
        )
        accumulator += part * weight[:, None]
        total += weight

    row = tl.load(
        row_indices
        + (local_row // BEAM_WIDTH) * stride_row_group
        + (local_row % BEAM_WIDTH) * stride_row_beam
    )
    tl.store(
        output
        + row * stride_out_row
        + heads[:, None] * stride_out_head
        + dims[None, :],
        accumulator / total[:, None],
        mask=mask,
    )


def num_kv_splits(num_tiles: int, num_sms: int) -> int:
    """Pick the prompt split count from static shapes only, so graphs keep it."""
    best_splits, best_efficiency = 1, 0.0
    for splits in range(1, _MAX_SPLITS + 1):
        ctas = num_tiles * splits
        efficiency = ctas / (triton.cdiv(ctas, num_sms) * num_sms)
        if efficiency >= _MIN_WAVE_EFFICIENCY:
            return splits
        if efficiency > best_efficiency:
            best_splits, best_efficiency = splits, efficiency
    return best_splits


def shared_context_attention_split_kv(
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
    num_splits: Optional[int] = None,
) -> torch.Tensor:
    """Shared prompt and per-beam suffix attention, split along the prompt.

    Same contract as `shared_context_attention`: `row_indices` [groups, beam_width]
    maps each beam to its row of Q and of the output, and `decode_slots` is
    [groups, capacity * beam_width], step-major. Only those rows are written.
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
    if decode_slots.stride(1) != 1:
        raise ValueError("Decode slots must be contiguous [groups, steps * beam_width]")
    num_groups, beam_width = row_indices.shape
    _, q_heads, head_dim = q.shape
    kv_heads = k_buffer.shape[1]

    beam_blocks = triton.cdiv(beam_width, _BLOCK_M)
    if num_splits is None:
        num_splits = num_kv_splits(
            num_groups * q_heads * beam_blocks,
            num_sms(q.device.index),
        )
    write_partials = num_splits > 1
    if write_partials:
        partial_output = torch.empty(
            (num_splits, num_groups * beam_width, q_heads, head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        partial_lse = torch.empty(
            (num_splits, num_groups * beam_width, q_heads),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        partial_output = partial_lse = output.new_empty((1, 1, 1))

    block_d = triton.next_power_of_2(head_dim)
    _shared_context_split_kv_kernel[(num_groups * num_splits, q_heads, beam_blocks)](
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
        num_splits,
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
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        BEAM_WIDTH=beam_width,
        Q_PER_KV=q_heads // kv_heads,
        HEAD_DIM=head_dim,
        WRITE_PARTIALS=write_partials,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=block_d,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    if write_partials:
        _merge_splits_kernel[(num_groups * beam_width,)](
            partial_output,
            partial_lse,
            row_indices,
            output,
            num_splits,
            partial_output.stride(0),
            partial_output.stride(1),
            partial_output.stride(2),
            partial_lse.stride(0),
            partial_lse.stride(1),
            row_indices.stride(0),
            row_indices.stride(1),
            output.stride(0),
            output.stride(1),
            BEAM_WIDTH=beam_width,
            Q_HEADS=q_heads,
            HEAD_DIM=head_dim,
            BLOCK_H=triton.next_power_of_2(q_heads),
            BLOCK_D=block_d,
            num_warps=4,
        )
    return output
