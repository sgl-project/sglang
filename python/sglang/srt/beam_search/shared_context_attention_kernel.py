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
"""Native Triton shared-context attention for wide beam decoding."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _shared_context_attention_kernel(
    q,
    k_buffer,
    v_buffer,
    row_indices,
    context_slots,
    context_cu_seqlens,
    decode_slots,
    decode_lens,
    output,
    softmax_scale,
    stride_q_row,
    stride_q_head,
    stride_q_dim,
    stride_row_group,
    stride_row_beam,
    stride_k_slot,
    stride_k_head,
    stride_k_dim,
    stride_v_slot,
    stride_v_head,
    stride_v_dim,
    stride_decode_group,
    stride_decode_token,
    stride_out_row,
    stride_out_head,
    stride_out_dim,
    BEAM_WIDTH: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    MAX_DECODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    group = tl.program_id(0)
    q_head = tl.program_id(1)
    beam_block = tl.program_id(2)
    kv_head = q_head // Q_PER_KV

    beam_offsets = beam_block * BLOCK_M + tl.arange(0, BLOCK_M)
    context_offsets = tl.arange(0, BLOCK_N)
    dim_offsets = tl.arange(0, BLOCK_D)
    beam_mask = beam_offsets < BEAM_WIDTH
    dim_mask = dim_offsets < HEAD_DIM

    rows = tl.load(
        row_indices + group * stride_row_group + beam_offsets * stride_row_beam,
        mask=beam_mask,
        other=0,
    )
    q_offsets = (
        rows[:, None] * stride_q_row
        + q_head * stride_q_head
        + dim_offsets[None, :] * stride_q_dim
    )
    q_values = tl.load(
        q + q_offsets,
        mask=beam_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    negative_infinity = -float("inf")
    max_values = tl.where(beam_mask, negative_infinity, 0.0)
    normalizers = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    softmax_scale_log2 = softmax_scale * 1.4426950408889634

    context_start = tl.load(context_cu_seqlens + group)
    context_end = tl.load(context_cu_seqlens + group + 1)
    context_length = context_end - context_start

    for block_start in tl.range(0, context_length, BLOCK_N):
        token_offsets = block_start + context_offsets
        token_mask = token_offsets < context_length
        slots = tl.load(
            context_slots + context_start + token_offsets,
            mask=token_mask,
            other=0,
        )
        k_offsets = (
            slots[:, None] * stride_k_slot
            + kv_head * stride_k_head
            + dim_offsets[None, :] * stride_k_dim
        )
        v_offsets = (
            slots[:, None] * stride_v_slot
            + kv_head * stride_v_head
            + dim_offsets[None, :] * stride_v_dim
        )
        k_values = tl.load(
            k_buffer + k_offsets,
            mask=token_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        v_values = tl.load(
            v_buffer + v_offsets,
            mask=token_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )

        scores = tl.dot(q_values, tl.trans(k_values))
        scores = tl.where(
            beam_mask[:, None] & token_mask[None, :],
            scores,
            negative_infinity,
        )
        block_max = tl.max(scores, axis=1)
        new_max = tl.maximum(max_values, block_max)
        previous_scale = tl.exp2((max_values - new_max) * softmax_scale_log2)
        probabilities = tl.exp2((scores - new_max[:, None]) * softmax_scale_log2)
        normalizers = normalizers * previous_scale + tl.sum(probabilities, axis=1)
        accumulator = accumulator * previous_scale[:, None] + tl.dot(
            probabilities.to(v_values.dtype),
            v_values,
        )
        max_values = new_max

    decode_length = tl.load(decode_lens + group)
    for step in range(0, MAX_DECODE):
        step_valid = step < decode_length
        decode_offsets = (
            group * stride_decode_group
            + (step * BEAM_WIDTH + beam_offsets) * stride_decode_token
        )
        slots = tl.load(
            decode_slots + decode_offsets,
            mask=beam_mask & step_valid,
            other=0,
        )
        k_offsets = (
            slots[:, None] * stride_k_slot
            + kv_head * stride_k_head
            + dim_offsets[None, :] * stride_k_dim
        )
        v_offsets = (
            slots[:, None] * stride_v_slot
            + kv_head * stride_v_head
            + dim_offsets[None, :] * stride_v_dim
        )
        k_values = tl.load(
            k_buffer + k_offsets,
            mask=beam_mask[:, None] & step_valid & dim_mask[None, :],
            other=0.0,
        )
        v_values = tl.load(
            v_buffer + v_offsets,
            mask=beam_mask[:, None] & step_valid & dim_mask[None, :],
            other=0.0,
        )
        score_matrix = tl.dot(q_values, tl.trans(k_values))
        diagonal_mask = tl.arange(0, BLOCK_M)[:, None] == tl.arange(0, BLOCK_M)[None, :]
        scores = tl.sum(
            tl.where(diagonal_mask, score_matrix, 0.0),
            axis=1,
        )
        scores = tl.where(beam_mask & step_valid, scores, negative_infinity)

        new_max = tl.maximum(max_values, scores)
        previous_scale = tl.exp2((max_values - new_max) * softmax_scale_log2)
        current_scale = tl.exp2((scores - new_max) * softmax_scale_log2)
        normalizers = normalizers * previous_scale + current_scale
        accumulator = (
            accumulator * previous_scale[:, None] + v_values * current_scale[:, None]
        )
        max_values = new_max

    result = accumulator / normalizers[:, None]
    output_offsets = (
        rows[:, None] * stride_out_row
        + q_head * stride_out_head
        + dim_offsets[None, :] * stride_out_dim
    )
    tl.store(
        output + output_offsets,
        result,
        mask=beam_mask[:, None] & dim_mask[None, :],
    )


def shared_context_attention(
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
) -> torch.Tensor:
    """Apply shared prompt and per-beam suffix attention in one Triton kernel."""
    if q.ndim != 3:
        raise ValueError(f"Expected Q with 3 dimensions, got {q.shape}")
    if k_buffer.ndim != 3 or v_buffer.shape != k_buffer.shape:
        raise ValueError("K/V buffers must have matching [slots, heads, dim] shapes")

    if row_indices.ndim != 2:
        raise ValueError("Row indices must have shape [groups, beam_width]")
    num_groups, beam_width = row_indices.shape
    _, q_heads, head_dim = q.shape
    kv_heads = k_buffer.shape[1]
    if q_heads % kv_heads:
        raise ValueError(f"Q heads {q_heads} must be divisible by KV heads {kv_heads}")
    if decode_slots.shape[0] != num_groups:
        raise ValueError("Decode slot groups must match grouped Q")
    if decode_slots.shape[1] % beam_width:
        raise ValueError("Decode slot capacity must be divisible by beam width")
    if decode_lens.shape != (num_groups,):
        raise ValueError("Decode lengths must contain one value per beam group")
    if context_cu_seqlens.shape != (num_groups + 1,):
        raise ValueError("Context offsets must contain one boundary per beam group")
    if output is None:
        output = torch.empty_like(q)
    elif output.shape != q.shape:
        raise ValueError("Output must match Q shape")

    high_group_count = num_groups >= 6
    block_m = 64 if high_group_count else 128
    block_n = 64 if high_group_count else 128
    num_warps = 4 if high_group_count else 8
    num_stages = 4 if high_group_count else 3

    block_d = triton.next_power_of_2(head_dim)
    grid = (
        num_groups,
        q_heads,
        triton.cdiv(beam_width, block_m),
    )
    _shared_context_attention_kernel[grid](
        q,
        k_buffer,
        v_buffer,
        row_indices,
        context_slots,
        context_cu_seqlens,
        decode_slots,
        decode_lens,
        output,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        row_indices.stride(0),
        row_indices.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        k_buffer.stride(2),
        v_buffer.stride(0),
        v_buffer.stride(1),
        v_buffer.stride(2),
        decode_slots.stride(0),
        decode_slots.stride(1),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        BEAM_WIDTH=beam_width,
        Q_PER_KV=q_heads // kv_heads,
        HEAD_DIM=head_dim,
        MAX_DECODE=decode_slots.shape[1] // beam_width,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
