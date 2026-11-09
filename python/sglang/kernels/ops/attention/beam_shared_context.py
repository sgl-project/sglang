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
"""Native Triton shared-context attention for wide beam decoding.

The online-softmax steps, the input checks and the SM count are shared with the
opt-in split-KV and stream-K kernels (beam_shared_context_split_kv.py and
beam_shared_context_stream_k.py).
"""

from __future__ import annotations

import functools
from typing import Callable

import torch
import triton
import triton.language as tl


@triton.jit
def empty_softmax_state(
    beam_mask,  # [BLOCK_M] bool, the tile's beams that exist
    BLOCK_M: tl.constexpr,  # beams per tile
    BLOCK_D: tl.constexpr,  # head_dim rounded up to a power of 2
):
    max_values = tl.where(beam_mask, -float("inf"), 0.0)
    normalizers = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    return max_values, normalizers, accumulator


@triton.jit
def attend_prompt_block(
    scores,  # [BLOCK_M, BLOCK_N] fp32 unscaled q.k, -inf where masked
    v_values,  # [BLOCK_N, BLOCK_D] V of the block's prompt tokens
    max_values,  # [BLOCK_M] fp32 running max of the unscaled scores
    normalizers,  # [BLOCK_M] fp32 running softmax denominator
    accumulator,  # [BLOCK_M, BLOCK_D] fp32 weighted sum of V, not yet normalized
    softmax_scale_log2,  # softmax_scale * log2(e), for exp2
):
    # Online softmax step over a block of prompt tokens; masked scores are -inf.
    new_max = tl.maximum(max_values, tl.max(scores, axis=1))
    previous_scale = tl.exp2((max_values - new_max) * softmax_scale_log2)
    probabilities = tl.exp2((scores - new_max[:, None]) * softmax_scale_log2)
    normalizers = normalizers * previous_scale + tl.sum(probabilities, axis=1)
    accumulator = accumulator * previous_scale[:, None] + tl.dot(
        probabilities.to(v_values.dtype),
        v_values,
    )
    return new_max, normalizers, accumulator


@triton.jit
def attend_beam_token(
    scores,  # [BLOCK_M] fp32 unscaled q.k with each beam's own token, -inf if masked
    v_values,  # [BLOCK_M, BLOCK_D] V of each beam's own token
    max_values,  # [BLOCK_M] fp32 running max of the unscaled scores
    normalizers,  # [BLOCK_M] fp32 running softmax denominator
    accumulator,  # [BLOCK_M, BLOCK_D] fp32 weighted sum of V, not yet normalized
    softmax_scale_log2,  # softmax_scale * log2(e), for exp2
):
    # Online softmax step over one decode token per beam; masked scores are -inf.
    new_max = tl.maximum(max_values, scores)
    previous_scale = tl.exp2((max_values - new_max) * softmax_scale_log2)
    current_scale = tl.exp2((scores - new_max) * softmax_scale_log2)
    normalizers = normalizers * previous_scale + current_scale
    accumulator = (
        accumulator * previous_scale[:, None]
        + v_values.to(tl.float32) * current_scale[:, None]
    )
    return new_max, normalizers, accumulator


@triton.jit
def _shared_context_attention_kernel(
    q,  # [rows, q_heads, HEAD_DIM]; only the rows in row_indices are read
    k_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's K cache
    v_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's V cache
    row_indices,  # [groups, BEAM_WIDTH] int32, each beam's row of q and output
    context_slots,  # [>= total prompt tokens] int32 KV slots, groups back to back
    context_cu_seqlens,  # [groups + 1] int32, each group's start in context_slots
    decode_slots,  # [groups, MAX_DECODE * BEAM_WIDTH] int32 KV slots, step-major
    decode_lens,  # [groups] int32, decode steps each beam attends to
    output,  # like q; only the rows in row_indices are written
    softmax_scale,  # float, scales q.k before the softmax
    stride_q_row,  # q.stride(0); all strides are in elements
    stride_q_head,  # q.stride(1)
    stride_q_dim,  # q.stride(2)
    stride_row_group,  # row_indices.stride(0)
    stride_row_beam,  # row_indices.stride(1)
    stride_k_slot,  # k_buffer.stride(0)
    stride_k_head,  # k_buffer.stride(1)
    stride_k_dim,  # k_buffer.stride(2)
    stride_v_slot,  # v_buffer.stride(0)
    stride_v_head,  # v_buffer.stride(1)
    stride_v_dim,  # v_buffer.stride(2)
    stride_decode_group,  # decode_slots.stride(0)
    stride_decode_token,  # decode_slots.stride(1)
    stride_out_row,  # output.stride(0)
    stride_out_head,  # output.stride(1)
    stride_out_dim,  # output.stride(2)
    BEAM_WIDTH: tl.constexpr,  # beams per group, row_indices.shape[1]
    Q_PER_KV: tl.constexpr,  # query heads per KV head
    HEAD_DIM: tl.constexpr,  # q.shape[2]
    MAX_DECODE: tl.constexpr,  # decode_slots.shape[1] // BEAM_WIDTH, the step capacity
    BLOCK_M: tl.constexpr,  # beams per program
    BLOCK_N: tl.constexpr,  # prompt tokens per block
    BLOCK_D: tl.constexpr,  # HEAD_DIM rounded up to a power of 2
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
    max_values, normalizers, accumulator = empty_softmax_state(
        beam_mask, BLOCK_M, BLOCK_D
    )
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
        max_values, normalizers, accumulator = attend_prompt_block(
            scores,
            v_values,
            max_values,
            normalizers,
            accumulator,
            softmax_scale_log2,
        )

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
        max_values, normalizers, accumulator = attend_beam_token(
            scores,
            v_values,
            max_values,
            normalizers,
            accumulator,
            softmax_scale_log2,
        )

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


@functools.lru_cache(maxsize=None)
def num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def resident_ctas_per_sm(kernel) -> int:
    """CTAs of a compiled Triton kernel that one SM holds at once, as the driver counts.

    On RTX PRO 6000 the split-KV and stream-K kernels use all 64K registers of an SM,
    so this is 1 there; GPUs or compilers that need fewer registers per CTA fit more.
    """
    from cuda.bindings import driver

    # Loads the module and sets kernel.function, which a first launch would also do.
    kernel._init_handles()
    error, count = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        kernel.function, kernel.metadata.num_warps * 32, kernel.metadata.shared
    )
    if error != driver.CUresult.CUDA_SUCCESS or count < 1:
        raise RuntimeError(f"{kernel.name} fits no CTA on an SM: {error}, {count}")
    return count


def max_element_offset(tensor: torch.Tensor) -> int:
    """Largest element offset from the tensor's base that a kernel can address."""
    return sum(
        (size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())
    )


_WAVE_CTAS: dict[tuple, int] = {}


def wave_ctas(
    *,
    key: tuple,  # what changes the compiled code: kernel, device, dtypes, constexprs
    compile_kernel: Callable,  # compiles without launching (JITFunction.warmup)
    device_index: int,
) -> int:
    """CTAs the whole GPU holds at once for one kernel: SMs times CTAs per SM."""
    wave = _WAVE_CTAS.get(key)
    if wave is None:
        wave = num_sms(device_index) * resident_ctas_per_sm(compile_kernel())
        _WAVE_CTAS[key] = wave
    return wave


def check_shared_context_inputs(
    q: torch.Tensor,  # [rows, q_heads, head_dim]
    k_buffer: torch.Tensor,  # [kv_slots, kv_heads, head_dim]
    v_buffer: torch.Tensor,  # same shape as k_buffer
    row_indices: torch.Tensor,  # [groups, beam_width]
    context_cu_seqlens: torch.Tensor,  # [groups + 1]
    decode_slots: torch.Tensor,  # [groups, capacity * beam_width]
    decode_lens: torch.Tensor,  # [groups]
    output: torch.Tensor | None,  # like q, or None to allocate one
) -> torch.Tensor:
    """Check the inputs every kernel reads; returns the output to write."""
    if q.ndim != 3:
        raise ValueError(f"Expected Q with 3 dimensions, got {q.shape}")
    if k_buffer.ndim != 3 or v_buffer.shape != k_buffer.shape:
        raise ValueError("K/V buffers must have matching [slots, heads, dim] shapes")

    if row_indices.ndim != 2:
        raise ValueError("Row indices must have shape [groups, beam_width]")
    num_groups, beam_width = row_indices.shape
    q_heads = q.shape[1]
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
        return torch.empty_like(q)
    if output.shape != q.shape:
        raise ValueError("Output must match Q shape")
    return output


def shared_context_attention(
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
) -> torch.Tensor:
    """Apply shared prompt and per-beam suffix attention in one Triton kernel."""
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
    num_groups, beam_width = row_indices.shape
    _, q_heads, head_dim = q.shape
    kv_heads = k_buffer.shape[1]

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
