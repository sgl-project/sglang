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

Opt-in with SGLANG_USE_BEAM_SHARED_CONTEXT_SPLIT_KV=1. Every beam of a group
attends to the group's shared prompt and to its own decode tokens. The prompt
part runs as tensor-core tiles of BLOCK_M beams of one query head, split along
the prompt (split-KV) so that a few groups still fill the GPU; the last split
adds the beams' own decode tokens with CUDA-core dot products, and a merge
kernel combines the splits' partial results. Q and the output rows are read and
written through `row_indices`, like the default kernel.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

import msgspec
import torch
import triton
import triton.language as tl

from sglang.kernels.arch import current_arch
from sglang.kernels.ops.attention.beam_shared_context import (
    attend_beam_token,
    attend_prompt_block,
    check_shared_context_inputs,
    empty_softmax_state,
    max_element_offset,
)

# Fastest of ten tiles on RTX PRO 6000 at 256 beams, head_dim 128; untuned elsewhere.
# Shared memory: 64 KB at head_dim <= 128; 144 KB up to 256, past SM86/89/120's 99 KB.
_BLOCK_M = 128
_BLOCK_N = 64
_NUM_WARPS = 8
_NUM_STAGES = 2
_MAX_SPLITS = 8
# Measured on RTX PRO 6000: the smallest split count that fills this share of the
# last wave is within 1-2% of the best one at 1-8 groups; more add merge traffic.
_MIN_WAVE_EFFICIENCY = 0.8
# Triton computes Q, output and partial offsets in 32 bits.
_MAX_OFFSET = 2**31


@triton.jit
def _shared_context_split_kv_kernel(
    q,  # [rows, q_heads, HEAD_DIM], contiguous along HEAD_DIM
    k_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's K cache
    v_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's V cache
    row_indices,  # [groups, BEAM_WIDTH] int32, each beam's row of q and output
    context_slots,  # [>= total prompt tokens] int32 KV slots, groups back to back
    context_cu_seqlens,  # [groups + 1] int32, each group's start in context_slots
    decode_slots,  # [groups, capacity * BEAM_WIDTH] int32 KV slots, step-major
    decode_lens,  # [groups] int32, decode steps each beam attends to
    output,  # like q; written here only when WRITE_PARTIALS is false
    partial_output,  # fp32 [num_splits, groups * BEAM_WIDTH, q_heads, HEAD_DIM]
    partial_lse,  # fp32 [num_splits, groups * BEAM_WIDTH, q_heads], log2 softmax sums
    softmax_scale,
    num_splits,  # prompt splits per tile; the grid's axis 0 is groups * num_splits
    stride_q_row,
    stride_q_head,
    stride_row_group,
    stride_row_beam,
    stride_k_slot,
    stride_k_head,
    stride_v_slot,
    stride_v_head,
    stride_decode_group,  # stride(1) must be 1
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
    WRITE_PARTIALS: tl.constexpr,  # num_splits > 1; if false, partial_* are unused
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    WIDE_KV_OFFSETS: tl.constexpr,  # K/V offsets past 2^31: slot * stride needs 64 bits
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
    blocks_per_split = tl.cdiv(tl.cdiv(context_length, BLOCK_N), num_splits)
    split_start_token = split * blocks_per_split * BLOCK_N
    split_end_token = tl.minimum(
        split_start_token + blocks_per_split * BLOCK_N, context_length
    )

    for block_start_token in tl.range(split_start_token, split_end_token, BLOCK_N):
        tokens = block_start_token + context_offsets
        token_mask = tokens < split_end_token
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

    if split == num_splits - 1:
        num_decode_steps = tl.load(decode_lens + group)
        for step in range(0, num_decode_steps):
            slots = tl.load(
                decode_slots + group * stride_decode_group + step * BEAM_WIDTH + beams,
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
    partial_output,  # fp32 [num_splits, groups * BEAM_WIDTH, Q_HEADS, HEAD_DIM]
    partial_lse,  # fp32 [num_splits, groups * BEAM_WIDTH, Q_HEADS], -inf if empty
    row_indices,  # [groups, BEAM_WIDTH] int32, each beam's row of output
    output,  # [rows, Q_HEADS, HEAD_DIM]; only the rows in row_indices are written
    num_splits,  # splits the split-KV kernel wrote, more than 1
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


def resident_ctas_per_sm(kernel) -> int:
    """CTAs of a compiled Triton kernel that one SM holds at once (1 on RTX PRO 6000)."""
    from cuda.bindings import driver

    kernel._init_handles()
    error, count = driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(
        kernel.function, kernel.metadata.num_warps * 32, kernel.metadata.shared
    )
    if error != driver.CUresult.CUDA_SUCCESS or count < 1:
        raise RuntimeError(f"{kernel.name} fits no CTA on an SM: {error}, {count}")
    return count


_WAVE_CTAS: dict[tuple, int] = {}


def wave_ctas(
    *,
    key: tuple,
    compile_kernel: Callable,
) -> int:
    wave = _WAVE_CTAS.get(key)
    if wave is None:
        wave = current_arch().num_sms * resident_ctas_per_sm(compile_kernel())
        _WAVE_CTAS[key] = wave
    return wave


def launch_plan_key(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    row_indices: torch.Tensor,
    context_cu_seqlens: torch.Tensor,
    decode_slots: torch.Tensor,
    decode_lens: torch.Tensor,
    output: torch.Tensor,
) -> tuple:
    return (
        q.shape,
        q.stride(),
        q.dtype,
        q.device.index,
        k_buffer.shape,
        k_buffer.stride(),
        k_buffer.dtype,
        v_buffer.shape,
        v_buffer.stride(),
        row_indices.shape,
        row_indices.stride(),
        context_cu_seqlens.shape,
        decode_slots.shape,
        decode_slots.stride(),
        decode_lens.shape,
        output.shape,
        output.stride(),
    )


# Arbitrary. The decode capacity is part of the key, so each step of a long
# generation adds a plan; a full cache starts over.
_MAX_PLANS = 256


class _KeptPartials(threading.local):
    def __init__(self):
        self.by_device: dict[int, tuple[int, torch.Tensor, torch.Tensor]] = {}


_KEPT_PARTIALS = _KeptPartials()


def partial_buffers(
    device: torch.device,
    output_numel: int,
    lse_numel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat fp32 partial workspaces, kept per thread and device between eager launches.

    Launches on one stream run in order, so they can share one pair; the pair is
    replaced when the current stream changes.
    """
    if torch.cuda.is_current_stream_capturing():
        # A graph replays the pointers it captured, so it must own its buffers.
        return (
            torch.empty(output_numel, dtype=torch.float32, device=device),
            torch.empty(lse_numel, dtype=torch.float32, device=device),
        )
    stream = triton.runtime.driver.active.get_current_stream(device.index)
    kept = _KEPT_PARTIALS.by_device.get(device.index)
    if kept is not None and kept[0] == stream:
        if kept[1].numel() >= output_numel and kept[2].numel() >= lse_numel:
            return kept[1], kept[2]
        output_numel = max(output_numel, kept[1].numel())
        lse_numel = max(lse_numel, kept[2].numel())
    # Release the old pair first. Its queued launches stay safe: PyTorch gives a
    # freed block only to the stream it was allocated on.
    kept = None
    _KEPT_PARTIALS.by_device.pop(device.index, None)
    kept = _KEPT_PARTIALS.by_device[device.index] = (
        stream,
        torch.empty(output_numel, dtype=torch.float32, device=device),
        torch.empty(lse_numel, dtype=torch.float32, device=device),
    )
    return kept[1], kept[2]


def pick_num_splits(
    num_tiles: int,
    wave: int,
) -> int:
    """Static shapes only, so a graph keeps the split count."""
    best_splits, best_efficiency = 1, 0.0
    for splits in range(1, _MAX_SPLITS + 1):
        ctas = num_tiles * splits
        efficiency = ctas / (triton.cdiv(ctas, wave) * wave)
        if efficiency >= _MIN_WAVE_EFFICIENCY:
            return splits
        if efficiency > best_efficiency:
            best_splits, best_efficiency = splits, efficiency
    return best_splits


class _Plan(msgspec.Struct, frozen=True):
    grid: tuple[int, int, int]
    args: tuple
    constexprs: dict
    write_partials: bool  # False with one split
    partial_numel: int
    lse_numel: int
    merge_grid: tuple[int]
    merge_args: tuple
    merge_constexprs: dict


_PLANS: dict[tuple, _Plan] = {}


def _make_plan(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    row_indices: torch.Tensor,
    context_slots: torch.Tensor,
    context_cu_seqlens: torch.Tensor,
    decode_slots: torch.Tensor,
    decode_lens: torch.Tensor,
    output: torch.Tensor,
    softmax_scale: float,
    num_splits: Optional[int],
) -> _Plan:
    check_shared_context_inputs(
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

    rows = num_groups * beam_width
    beam_blocks = triton.cdiv(beam_width, _BLOCK_M)
    block_d = triton.next_power_of_2(head_dim)
    constexprs = dict(
        BEAM_WIDTH=beam_width,
        Q_PER_KV=q_heads // kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=block_d,
        WIDE_KV_OFFSETS=max(map(max_element_offset, (k_buffer, v_buffer)))
        >= _MAX_OFFSET,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    strides = (
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
    partial_strides = (rows * q_heads * head_dim, q_heads * head_dim, head_dim) + (
        rows * q_heads,
        q_heads,
    )

    if num_splits is None:
        # Any split count above 1 compiles to the same kernel.
        wave = wave_ctas(
            key=(q.device.index, q.dtype, k_buffer.dtype)
            + tuple(sorted(constexprs.items())),
            compile_kernel=lambda: _shared_context_split_kv_kernel.warmup(
                q,
                k_buffer,
                v_buffer,
                row_indices,
                context_slots,
                context_cu_seqlens,
                decode_slots,
                decode_lens,
                output,
                torch.float32,
                torch.float32,
                softmax_scale,
                2,
                *strides,
                *partial_strides,
                WRITE_PARTIALS=True,
                grid=(1,),
                **constexprs,
            ),
        )
        num_splits = pick_num_splits(num_groups * q_heads * beam_blocks, wave)
    if (
        max(
            max_element_offset(q),
            max_element_offset(output),
            num_splits * rows * q_heads * head_dim,
        )
        >= _MAX_OFFSET
    ):
        raise ValueError(
            "Q, the output, and the split partials must have fewer than 2^31 elements"
        )
    write_partials = num_splits > 1
    if not write_partials:
        # Unread with one split; Triton compiles strides of 1 in as constants.
        partial_strides = (1, 1, 1, 1, 1)
    return _Plan(
        grid=(num_groups * num_splits, q_heads, beam_blocks),
        args=(num_splits, *strides, *partial_strides),
        constexprs=dict(WRITE_PARTIALS=write_partials, **constexprs),
        write_partials=write_partials,
        partial_numel=num_splits * rows * q_heads * head_dim,
        lse_numel=num_splits * rows * q_heads,
        merge_grid=(rows,),
        merge_args=(
            num_splits,
            *partial_strides,
            row_indices.stride(0),
            row_indices.stride(1),
            output.stride(0),
            output.stride(1),
        ),
        merge_constexprs=dict(
            BEAM_WIDTH=beam_width,
            Q_HEADS=q_heads,
            HEAD_DIM=head_dim,
            BLOCK_H=triton.next_power_of_2(q_heads),
            BLOCK_D=block_d,
            num_warps=4,
        ),
    )


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
    num_splits: Optional[int] = None,  # None: from the GPU's wave size
) -> torch.Tensor:
    """`shared_context_attention` with the prompt split across programs; same contract."""
    if output is None:
        output = torch.empty_like(q)
    # Cached: every layer of a decode step repeats the checks and the plan.
    key = launch_plan_key(
        q,
        k_buffer,
        v_buffer,
        row_indices,
        context_cu_seqlens,
        decode_slots,
        decode_lens,
        output,
    ) + (num_splits,)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _make_plan(
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
            num_splits,
        )
        if len(_PLANS) >= _MAX_PLANS:
            _PLANS.clear()
        _PLANS[key] = plan
    if plan.write_partials:
        partial_output, partial_lse = partial_buffers(
            q.device, plan.partial_numel, plan.lse_numel
        )
    else:
        partial_output = partial_lse = output

    _shared_context_split_kv_kernel[plan.grid](
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
        *plan.args,
        **plan.constexprs,
    )
    if plan.write_partials:
        _merge_splits_kernel[plan.merge_grid](
            partial_output,
            partial_lse,
            row_indices,
            output,
            *plan.merge_args,
            **plan.merge_constexprs,
        )
    return output
