# SPDX-License-Identifier: MIT
# Copyright 2018-2020 Philippe Tillet
# Copyright 2020-2022 OpenAI
# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Dense FAv3 forward kernel specialized for AMD gfx1250.

Inputs and output use SGLang diffusion's ``[batch, sequence, heads, dim]``
layout. The production path supports non-causal BF16/FP16 MHA with head
dimensions 64, 128, and 256. BF16/D128 retains the specialized guarded and
persistent schedules. Unsupported contracts are handled by the diffusion backend
before this module is imported.

"""

from __future__ import annotations

import math

import torch
import triton.experimental.gluon.language as gl
from triton.experimental import gluon

from sglang.kernels.ops.attention.gluon.fa3.gfx1250.config import _AttentionConfig
from sglang.kernels.ops.attention.gluon.fa3.gfx1250.policy import (
    GuardFlag,
    LaunchConfig,
    NumericalMode,
    resolve_execution_policy,
    select_launch_config,
)

# Keep flag values usable by both host tests and Gluon without reading GPU data
# on the host. NORM_UNSAFE covers Q/K norms and the V product/sum bounds.
_NORM_UNSAFE = gl.constexpr(GuardFlag.NORM_UNSAFE.value)
_FUSED_EXP_UNSAFE = gl.constexpr(GuardFlag.FUSED_EXP_UNSAFE.value)
_ROUTE_NORM = gl.constexpr(0)
_ROUTE_FUSED_EXP = gl.constexpr(1)
_ROUTE_STABLE = gl.constexpr(2)
# The two bounded algorithms share an entry point; stable fallback keeps its
# own register allocation. These are launch routes, not guard flag values.
_ROUTE_NORM_OR_FUSED_EXP = gl.constexpr(3)


@gluon.jit
def _round_norm_probabilities(probabilities):
    # Pack eight values with the same RTNE instruction used by the BF16 cast.
    # Return both the BF16 values for PV and their exact FP32 expansion for the
    # denominator, avoiding redundant 16-bit moves in the generic conversion.
    # Outputs 0:4 are packed BF16 words, outputs 4:12 are FP32, inputs 12:20
    # are FP32. Early-clobber outputs preserve inputs until all four conversions
    # have consumed them; expanded outputs can then reuse input registers.
    return gl.inline_asm_elementwise(
        """
        v_cvt_pk_bf16_f32 $0, $12, $13
        v_cvt_pk_bf16_f32 $1, $14, $15
        v_cvt_pk_bf16_f32 $2, $16, $17
        v_cvt_pk_bf16_f32 $3, $18, $19
        v_lshlrev_b32 $4, 16, $0
        v_and_b32 $5, 0xffff0000, $0
        v_lshlrev_b32 $6, 16, $1
        v_and_b32 $7, 0xffff0000, $1
        v_lshlrev_b32 $8, 16, $2
        v_and_b32 $9, 0xffff0000, $2
        v_lshlrev_b32 $10, 16, $3
        v_and_b32 $11, 0xffff0000, $3
        """,
        constraints="=&v,=&v,=&v,=&v,=v,=v,=v,=v,=v,=v,=v,=v,v,v,v,v,v,v,v,v",
        args=[probabilities],
        dtype=(gl.bfloat16, gl.float32),
        is_pure=True,
        pack=8,
    )


@gluon.aggregate
class _AttentionProgram:
    cfg: _AttentionConfig
    q: gl.tensor
    k_desc: gl.amd.cdna5.tdm.tensor_descriptor
    k_buffer: gl.shared_memory_descriptor
    v_desc: gl.amd.cdna5.tdm.tensor_descriptor
    v_buffer: gl.shared_memory_descriptor
    output_ptr: gl.tensor
    output_offsets: gl.tensor
    output_mask: gl.tensor
    output_uses_buffer: gl.constexpr
    softmax_scale_log2: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        cfg,
        q,
        k_desc,
        k_buffer,
        v_desc,
        v_buffer,
        output_ptr,
        output_offsets,
        output_mask,
        softmax_scale,
        output_uses_buffer,
    ):
        self.cfg = cfg
        self.q = q
        self.k_desc = k_desc
        self.k_buffer = k_buffer
        self.v_desc = v_desc
        self.v_buffer = v_buffer
        self.output_ptr = output_ptr
        self.output_offsets = output_offsets
        self.output_mask = output_mask
        self.output_uses_buffer = gl.constexpr(output_uses_buffer)
        self.softmax_scale_log2 = gl.constexpr(softmax_scale * 1.4426950408889634)

    @gluon.jit
    def create(
        cfg,
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        stride_qb,
        stride_qh,
        stride_qs,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vs,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_os,
        stride_od,
        softmax_scale: gl.constexpr,
        use_query_buffer: gl.constexpr,
        use_output_buffer: gl.constexpr,
    ):
        batch_id = gl.program_id(2 - cfg.query_axis)
        head_id = gl.program_id(1)
        query_start = gl.program_id(cfg.query_axis) * cfg.BLOCK_M
        batch_id_i64 = batch_id.to(gl.int64)
        head_id_i64 = head_id.to(gl.int64)

        query_local_rows = gl.arange(
            0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout)
        )
        query_rows = query_start + query_local_rows
        query_dims = gl.arange(0, cfg.HEAD_SIZE, layout=gl.SliceLayout(0, cfg.q_layout))
        query_mask = query_rows[:, None] < cfg.SEQLEN_Q
        # Buffer offsets are 32-bit. Put the sequence tile offset in the
        # 64-bit base so long videos do not wrap at the buffer-address limit.
        query_base = (
            q_ptr
            + stride_qb * batch_id_i64
            + stride_qh * head_id_i64
            + stride_qs * query_start.to(gl.int64)
        )
        if use_query_buffer:
            query_offsets = (
                stride_qs * query_local_rows[:, None] + stride_qd * query_dims[None, :]
            )
            q = gl.amd.cdna5.buffer_load(
                query_base, query_offsets, mask=query_mask, other=0.0
            )
        else:
            # A sparse view can span the buffer range inside one query tile.
            query_offsets = stride_qs * query_local_rows[:, None].to(
                gl.int64
            ) + stride_qd * query_dims[None, :].to(gl.int64)
            q = gl.load(query_base + query_offsets, mask=query_mask, other=0.0)

        k_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=k_ptr + stride_kb * batch_id_i64 + stride_kh * head_id_i64,
            shape=(cfg.SEQLEN_K, cfg.HEAD_SIZE),
            strides=(stride_ks, stride_kd),
            block_shape=(cfg.BLOCK_N, cfg.HEAD_SIZE),
            layout=cfg.k_smem_layout,
        )
        k_buffer = gl.allocate_shared_memory(
            k_desc.dtype,
            shape=[cfg.NUM_BUFFERS] + k_desc.block_shape,
            layout=k_desc.layout,
        )
        v_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=v_ptr + stride_vb * batch_id_i64 + stride_vh * head_id_i64,
            shape=(cfg.SEQLEN_K, cfg.HEAD_SIZE),
            strides=(stride_vs, stride_vd),
            block_shape=(cfg.BLOCK_N, cfg.HEAD_SIZE),
            layout=cfg.v_smem_layout,
        )
        v_buffer = gl.allocate_shared_memory(
            v_desc.dtype,
            shape=[cfg.NUM_BUFFERS] + v_desc.block_shape,
            layout=v_desc.layout,
        )

        output_local_rows = gl.arange(
            0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout)
        )
        output_rows = query_start + output_local_rows
        output_dims = gl.arange(
            0, cfg.HEAD_SIZE, layout=gl.SliceLayout(0, cfg.pv_layout)
        )
        if use_output_buffer:
            output_offsets = (
                stride_os * output_local_rows[:, None]
                + stride_od * output_dims[None, :]
            )
        else:
            output_offsets = stride_os * output_local_rows[:, None].to(
                gl.int64
            ) + stride_od * output_dims[None, :].to(gl.int64)
        output_mask = output_rows[:, None] < cfg.SEQLEN_Q
        output_base = (
            output_ptr
            + stride_ob * batch_id_i64
            + stride_oh * head_id_i64
            + stride_os * query_start.to(gl.int64)
        )

        return _AttentionProgram(
            cfg,
            q,
            k_desc,
            k_buffer,
            v_desc,
            v_buffer,
            output_base,
            output_offsets,
            output_mask,
            softmax_scale,
            use_output_buffer,
        )

    @gluon.jit
    def load_k(self, buffer_index, wait_count):
        gl.amd.cdna5.tdm.async_wait(0 if self.cfg.FULL_TDM_WAIT else wait_count)
        return (
            self.k_buffer.index(buffer_index)
            .permute([1, 0])
            .load(layout=self.cfg.k_layout)
        )

    @gluon.jit
    def load_v(self, buffer_index, wait_count):
        gl.amd.cdna5.tdm.async_wait(0 if self.cfg.FULL_TDM_WAIT else wait_count)
        return self.v_buffer.index(buffer_index).load(layout=self.cfg.v_layout)

    @gluon.jit
    def prefetch_k(self, sequence_start, buffer_index):
        gl.amd.cdna5.tdm.async_load(
            self.k_desc,
            [sequence_start, 0],
            self.k_buffer.index(buffer_index),
            warp_used_hint=0x0F,
            cache_modifier=".cg",
        )

    @gluon.jit
    def prefetch_v(self, sequence_start, buffer_index):
        gl.amd.cdna5.tdm.async_load(
            self.v_desc,
            [sequence_start, 0],
            self.v_buffer.index(buffer_index),
            warp_used_hint=0x0F,
            cache_modifier=".cg",
        )

    @gluon.jit
    def qk(self, k, sequence_start):
        scores = gl.zeros(
            [self.cfg.BLOCK_M, self.cfg.BLOCK_N],
            dtype=gl.float32,
            layout=self.cfg.qk_layout,
        )
        scores = gl.amd.cdna5.wmma(self.q, k, scores)
        key_cols = sequence_start + gl.arange(
            0, self.cfg.BLOCK_N, layout=gl.SliceLayout(0, self.cfg.qk_layout)
        )
        return gl.where(key_cols[None, :] < self.cfg.SEQLEN_K, scores, float("-inf"))

    @gluon.jit
    def qk_full(self, k):
        scores = gl.zeros(
            [self.cfg.BLOCK_M, self.cfg.BLOCK_N],
            dtype=gl.float32,
            layout=self.cfg.qk_layout,
        )
        return gl.amd.cdna5.wmma(self.q, k, scores)

    @gluon.jit
    def softmax_part0(self, scores, row_max):
        if self.cfg.NORM_EXP:
            # The norm guard bounds scaled logits by 32, so this fixed shift
            # keeps every exponent in [-64, 0] without online rescaling.
            probabilities = gl.exp2(scores * self.softmax_scale_log2 - 32.0)
            alpha = gl.full(
                [self.cfg.BLOCK_M],
                1.0,
                gl.float32,
                layout=gl.SliceLayout(1, self.cfg.qk_layout),
            )
            new_row_max = gl.full(
                [self.cfg.BLOCK_M],
                0.0,
                gl.float32,
                layout=gl.SliceLayout(1, self.cfg.qk_layout),
            )
        else:
            new_row_max = gl.maximum(row_max, gl.max(scores, 1))
            # The fused expression is only enabled after the GPU range check.
            # Unbounded inputs subtract before scaling so equal large logits have
            # a zero exponent even when the compiler enables FP fusion.
            if self.cfg.FAST_EXP:
                shifted = gl.fma(
                    scores,
                    self.softmax_scale_log2,
                    -new_row_max[:, None] * self.softmax_scale_log2,
                )
            else:
                shifted = (scores - new_row_max[:, None]) * self.softmax_scale_log2
            probabilities = gl.exp2(shifted)
            max_delta = (row_max - new_row_max) * self.softmax_scale_log2
            alpha = gl.exp2(max_delta)
        return probabilities, alpha, new_row_max

    @gluon.jit
    def softmax_prepare(self, scores, row_max):
        new_row_max = gl.maximum(row_max, gl.max(scores, 1))
        max_delta = (row_max - new_row_max) * self.softmax_scale_log2
        return max_delta, new_row_max

    @gluon.jit
    def softmax_finish(self, scores, new_row_max, max_delta):
        if self.cfg.FAST_EXP:
            shifted = gl.fma(
                scores,
                self.softmax_scale_log2,
                -new_row_max[:, None] * self.softmax_scale_log2,
            )
        else:
            shifted = (scores - new_row_max[:, None]) * self.softmax_scale_log2
        return gl.exp2(shifted), gl.exp2(max_delta)

    @gluon.jit
    def softmax_part1(self, probabilities, row_sum, accumulator, alpha):
        if self.cfg.NORM_EXP:
            # Use the same rounded probabilities for PV and the denominator.
            probabilities, rounded = _round_norm_probabilities(probabilities)
            tile_sum = gl.sum(rounded, 1)
            row_sum = row_sum + tile_sum
        else:
            tile_sum = gl.sum(probabilities, 1)
            accumulator = accumulator * alpha[:, None]
            # Round to nearest to avoid biasing every nonnegative probability down.
            probabilities = probabilities.to(self.q.dtype, fp_downcast_rounding="rtne")
            row_sum = row_sum * alpha + tile_sum
        return probabilities, row_sum, accumulator

    @gluon.jit
    def pv(self, probabilities, v, accumulator):
        probabilities = gl.convert_layout(probabilities, self.cfg.p_layout)
        return gl.amd.cdna5.wmma(probabilities, v, accumulator)

    @gluon.jit
    def store_output(self, output):
        output = output.to(self.output_ptr.dtype.element_ty)
        if self.output_uses_buffer:
            gl.amd.cdna5.buffer_store(
                output,
                self.output_ptr,
                self.output_offsets,
                mask=self.output_mask,
            )
        else:
            gl.store(
                self.output_ptr + self.output_offsets,
                output,
                mask=self.output_mask,
            )

    @gluon.jit
    def store_output_tdm(self, output, stride_os, stride_od):
        layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        query_start = gl.program_id(self.cfg.query_axis) * self.cfg.BLOCK_M
        output_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=self.output_ptr,
            shape=(self.cfg.SEQLEN_Q - query_start, self.cfg.HEAD_SIZE),
            strides=(stride_os, stride_od),
            block_shape=(self.cfg.BLOCK_M, self.cfg.HEAD_SIZE),
            layout=layout,
        )
        output_buffer = gl.allocate_shared_memory(
            self.output_ptr.dtype.element_ty,
            shape=(self.cfg.BLOCK_M, self.cfg.HEAD_SIZE),
            layout=layout,
        )
        output_buffer.store(output.to(self.output_ptr.dtype.element_ty))
        gl.amd.cdna5.tdm.async_store(
            output_desc,
            [0, 0],
            output_buffer,
            cache_modifier=".cs",
        )
        gl.amd.cdna5.tdm.async_wait(0)
        output_buffer._keep_alive()


@gluon.jit
def _attention_forward_pipeline(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    SOFTMAX_SCALE: gl.constexpr,
    SEQLEN_Q: gl.constexpr,
    SEQLEN_K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    TDM_OUTPUT: gl.constexpr = False,
    USE_QUERY_BUFFER: gl.constexpr = True,
    USE_OUTPUT_BUFFER: gl.constexpr = True,
):
    NUM_BUFFERS: gl.constexpr = 2
    NUM_WARPS: gl.constexpr = 4
    cfg = _AttentionConfig(
        SEQLEN_Q,
        SEQLEN_K,
        HEAD_SIZE,
        BLOCK_M,
        BLOCK_N,
        NUM_BUFFERS,
        NUM_WARPS,
        # Partial TDM waits intermittently corrupt FP16 storage-offset inputs,
        # including D64 with default scheduling. Drain pending loads before
        # reading K/V until the underlying synchronization issue is understood.
        full_tdm_wait=q_ptr.dtype.element_ty == gl.float16,
    )
    program = _AttentionProgram.create(
        cfg,
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        stride_qb,
        stride_qh,
        stride_qs,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vs,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_os,
        stride_od,
        SOFTMAX_SCALE,
        USE_QUERY_BUFFER,
        USE_OUTPUT_BUFFER,
    )

    peeled_iterations: gl.constexpr = 3
    loop_blocks = max((SEQLEN_K + BLOCK_N - 1) // BLOCK_N - peeled_iterations, 1)
    has_remainder: gl.constexpr = SEQLEN_K < peeled_iterations * BLOCK_N
    if has_remainder:
        loop_blocks -= 1

    row_max = gl.full(
        [BLOCK_M],
        float("-inf"),
        dtype=gl.float32,
        layout=gl.SliceLayout(1, cfg.pv_layout),
    )
    row_sum = gl.full(
        [BLOCK_M],
        1.0,
        dtype=gl.float32,
        layout=gl.SliceLayout(1, cfg.pv_layout),
    )
    accumulator = gl.zeros([BLOCK_M, HEAD_SIZE], dtype=gl.float32, layout=cfg.pv_layout)

    program.prefetch_k(0, 0)
    program.prefetch_k(BLOCK_N, 1)
    program.prefetch_v(0, 0)
    k = program.load_k(0, 2)
    scores = program.qk(k, 0)
    probabilities, alpha, row_max = program.softmax_part0(scores, row_max)
    program.prefetch_k(2 * BLOCK_N, 0)
    program.prefetch_v(BLOCK_N, 1)
    k = program.load_k(1, 3)

    iteration = 0
    for block_start in range(0, loop_blocks * BLOCK_N, BLOCK_N):
        next_v_start = block_start + 2 * BLOCK_N
        next_k_start = block_start + 3 * BLOCK_N
        scores = program.qk_full(k)
        probabilities, row_sum, accumulator = program.softmax_part1(
            probabilities, row_sum, accumulator, alpha
        )
        v = program.load_v(iteration % NUM_BUFFERS, 2)
        program.prefetch_k(next_k_start, (iteration + 1) % NUM_BUFFERS)
        accumulator = program.pv(probabilities, v, accumulator)
        probabilities, alpha, row_max = program.softmax_part0(scores, row_max)
        k = program.load_k(iteration % NUM_BUFFERS, 2)
        program.prefetch_v(next_v_start, iteration % NUM_BUFFERS)
        iteration += 1

    if has_remainder:
        current_start = iteration * BLOCK_N + BLOCK_N
        next_v_start = iteration * BLOCK_N + 2 * BLOCK_N
        next_k_start = iteration * BLOCK_N + 3 * BLOCK_N
        scores = program.qk(k, current_start)
        probabilities, row_sum, accumulator = program.softmax_part1(
            probabilities, row_sum, accumulator, alpha
        )
        v = program.load_v(iteration % NUM_BUFFERS, 2)
        program.prefetch_k(next_k_start, (iteration + 1) % NUM_BUFFERS)
        accumulator = program.pv(probabilities, v, accumulator)
        probabilities, alpha, row_max = program.softmax_part0(scores, row_max)
        k = program.load_k(iteration % NUM_BUFFERS, 2)
        program.prefetch_v(next_v_start, iteration % NUM_BUFFERS)
        iteration += 1

    epilogue_base = (iteration - 1) * BLOCK_N
    second_last_start = epilogue_base + 2 * BLOCK_N
    last_start = epilogue_base + 3 * BLOCK_N

    probabilities, row_sum, accumulator = program.softmax_part1(
        probabilities, row_sum, accumulator, alpha
    )
    v = program.load_v(iteration % NUM_BUFFERS, 2)
    accumulator = program.pv(probabilities, v, accumulator)

    scores = program.qk(k, second_last_start)
    probabilities, alpha, row_max = program.softmax_part0(scores, row_max)
    k = program.load_k(iteration % NUM_BUFFERS, 1)
    program.prefetch_v(last_start, iteration % NUM_BUFFERS)

    scores = program.qk(k, last_start)
    probabilities, row_sum, accumulator = program.softmax_part1(
        probabilities, row_sum, accumulator, alpha
    )
    v = program.load_v((iteration + 1) % NUM_BUFFERS, 1)
    accumulator = program.pv(probabilities, v, accumulator)
    probabilities, alpha, row_max = program.softmax_part0(scores, row_max)
    probabilities, row_sum, accumulator = program.softmax_part1(
        probabilities, row_sum, accumulator, alpha
    )
    v = program.load_v(iteration % NUM_BUFFERS, 0)
    accumulator = program.pv(probabilities, v, accumulator)

    program.store_output(accumulator * (1.0 / row_sum)[:, None])


@gluon.jit
def _attention_forward_pingpong(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    stride_qb,
    stride_qh,
    stride_qs,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_os,
    stride_od,
    SOFTMAX_SCALE: gl.constexpr,
    SEQLEN_Q: gl.constexpr,
    SEQLEN_K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    TDM_OUTPUT: gl.constexpr = False,
    USE_QUERY_BUFFER: gl.constexpr = True,
    USE_OUTPUT_BUFFER: gl.constexpr = True,
    FAST_EXP: gl.constexpr = False,
    NORM_EXP: gl.constexpr = False,
):
    NUM_BUFFERS: gl.constexpr = 2
    NUM_WARPS: gl.constexpr = 8
    REBALANCE_SOFTMAX: gl.constexpr = (
        HEAD_SIZE == 128 and BLOCK_M == 256 and BLOCK_N == 64 and not NORM_EXP
    )
    # Move half of the score shift into stage1 to balance the two scalar
    # stages without extending the lifetime of the probability fragments.
    EARLY_SHIFT: gl.constexpr = REBALANCE_SOFTMAX and SEQLEN_K >= 16_384
    USE_TDM_OUTPUT: gl.constexpr = (
        TDM_OUTPUT and HEAD_SIZE == 128 and BLOCK_M == 256 and BLOCK_N == 64
    )
    cfg = _AttentionConfig(
        SEQLEN_Q,
        SEQLEN_K,
        HEAD_SIZE,
        BLOCK_M,
        BLOCK_N,
        NUM_BUFFERS,
        NUM_WARPS,
        FAST_EXP,
        NORM_EXP,
    )
    program = _AttentionProgram.create(
        cfg,
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        stride_qb,
        stride_qh,
        stride_qs,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vs,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_os,
        stride_od,
        SOFTMAX_SCALE,
        USE_QUERY_BUFFER,
        USE_OUTPUT_BUFFER,
    )

    peeled_iterations: gl.constexpr = 3
    loop_blocks = max((SEQLEN_K + BLOCK_N - 1) // BLOCK_N - peeled_iterations, 1)
    # A single steady-state iteration is not safe for warp-group pipelining.
    # Route up to four KV tiles through the synchronized remainder path.
    has_remainder: gl.constexpr = SEQLEN_K <= (peeled_iterations + 1) * BLOCK_N
    if has_remainder:
        loop_blocks -= 1

    row_max = gl.full(
        [BLOCK_M],
        float("-inf"),
        dtype=gl.float32,
        layout=gl.SliceLayout(1, cfg.pv_layout),
    )
    row_sum = gl.full(
        [BLOCK_M],
        0.0 if NORM_EXP else 1.0,
        dtype=gl.float32,
        layout=gl.SliceLayout(1, cfg.pv_layout),
    )
    accumulator = gl.zeros([BLOCK_M, HEAD_SIZE], dtype=gl.float32, layout=cfg.pv_layout)

    program.prefetch_k(0, 0)
    program.prefetch_k(BLOCK_N, 1)
    program.prefetch_v(0, 0)
    k = program.load_k(0, 2)
    scores = program.qk(k, 0)
    probabilities, alpha, row_max = program.softmax_part0(scores, row_max)
    program.prefetch_k(2 * BLOCK_N, 0)
    program.prefetch_v(BLOCK_N, 1)
    k = program.load_k(1, 3)

    iteration = 0
    for block_start in range(0, loop_blocks * BLOCK_N, BLOCK_N):
        with gl.amd.warp_pipeline_stage("stage0"):
            next_v_start = block_start + 2 * BLOCK_N
            next_k_start = block_start + 3 * BLOCK_N
            scores = program.qk_full(k)

        gl.amd.cdna5.tdm.async_wait(2)
        with gl.amd.warp_pipeline_stage("stage1"):
            probabilities, row_sum, accumulator = program.softmax_part1(
                probabilities, row_sum, accumulator, alpha
            )
            v = program.v_buffer.index(iteration % NUM_BUFFERS).load(
                layout=program.cfg.v_layout
            )
            program.prefetch_k(next_k_start, (iteration + 1) % NUM_BUFFERS)
            # Balance scalar work while keeping exponentiation in stage3.
            if NORM_EXP:
                norm_left_scores = gl.amd.slice(scores, [BLOCK_M, BLOCK_N // 2], [0, 0])
                norm_shifted_left = gl.fma(
                    norm_left_scores, program.softmax_scale_log2, -32.0
                )
            if REBALANCE_SOFTMAX:
                max_delta, new_row_max = program.softmax_prepare(scores, row_max)
                if EARLY_SHIFT:
                    scores_left = gl.amd.slice(scores, [BLOCK_M, BLOCK_N // 2], [0, 0])
                    if FAST_EXP:
                        shifted_left = gl.fma(
                            scores_left,
                            program.softmax_scale_log2,
                            -new_row_max[:, None] * program.softmax_scale_log2,
                        )
                    else:
                        shifted_left = (
                            scores_left - new_row_max[:, None]
                        ) * program.softmax_scale_log2

        with gl.amd.warp_pipeline_stage("stage2"):
            accumulator = program.pv(probabilities, v, accumulator)

        gl.amd.cdna5.tdm.async_wait(2)
        with gl.amd.warp_pipeline_stage("stage3"):
            if NORM_EXP:
                norm_left = gl.exp2(norm_shifted_left)
                norm_right_scores = gl.amd.slice(
                    scores, [BLOCK_M, BLOCK_N // 2], [0, BLOCK_N // 2]
                )
                norm_right = gl.exp2(
                    gl.fma(norm_right_scores, program.softmax_scale_log2, -32.0)
                )
                probabilities = (
                    gl.join(norm_left, norm_right)
                    .permute(0, 2, 1)
                    .reshape([BLOCK_M, BLOCK_N])
                )
                probabilities = gl.convert_layout(
                    probabilities, cfg.qk_layout, assert_trivial=True
                )
            elif REBALANCE_SOFTMAX:
                if EARLY_SHIFT:
                    scores_right = gl.amd.slice(
                        scores, [BLOCK_M, BLOCK_N // 2], [0, BLOCK_N // 2]
                    )
                    if FAST_EXP:
                        shifted_right = gl.fma(
                            scores_right,
                            program.softmax_scale_log2,
                            -new_row_max[:, None] * program.softmax_scale_log2,
                        )
                    else:
                        shifted_right = (
                            scores_right - new_row_max[:, None]
                        ) * program.softmax_scale_log2
                    probabilities = (
                        gl.join(gl.exp2(shifted_left), gl.exp2(shifted_right))
                        .permute(0, 2, 1)
                        .reshape([BLOCK_M, BLOCK_N])
                    )
                    probabilities = gl.convert_layout(
                        probabilities, cfg.qk_layout, assert_trivial=True
                    )
                    alpha = gl.exp2(max_delta)
                else:
                    probabilities, alpha = program.softmax_finish(
                        scores, new_row_max, max_delta
                    )
                row_max = new_row_max
            else:
                probabilities, alpha, row_max = program.softmax_part0(scores, row_max)
            k = (
                program.k_buffer.index(iteration % NUM_BUFFERS)
                .permute([1, 0])
                .load(layout=program.cfg.k_layout)
            )
            program.prefetch_v(next_v_start, iteration % NUM_BUFFERS)
            iteration += 1

    if has_remainder:
        current_start = iteration * BLOCK_N + BLOCK_N
        next_v_start = iteration * BLOCK_N + 2 * BLOCK_N
        next_k_start = iteration * BLOCK_N + 3 * BLOCK_N
        scores = program.qk(k, current_start)
        probabilities, row_sum, accumulator = program.softmax_part1(
            probabilities, row_sum, accumulator, alpha
        )
        v = program.load_v(iteration % NUM_BUFFERS, 2)
        program.prefetch_k(next_k_start, (iteration + 1) % NUM_BUFFERS)
        accumulator = program.pv(probabilities, v, accumulator)
        probabilities, alpha, row_max = program.softmax_part0(scores, row_max)
        k = program.load_k(iteration % NUM_BUFFERS, 2)
        program.prefetch_v(next_v_start, iteration % NUM_BUFFERS)
        iteration += 1

    epilogue_base = (iteration - 1) * BLOCK_N
    second_last_start = epilogue_base + 2 * BLOCK_N
    last_start = epilogue_base + 3 * BLOCK_N

    probabilities, row_sum, accumulator = program.softmax_part1(
        probabilities, row_sum, accumulator, alpha
    )
    v = program.load_v(iteration % NUM_BUFFERS, 2)
    accumulator = program.pv(probabilities, v, accumulator)

    scores = program.qk(k, second_last_start)
    probabilities, alpha, row_max = program.softmax_part0(scores, row_max)
    k = program.load_k(iteration % NUM_BUFFERS, 1)
    program.prefetch_v(last_start, iteration % NUM_BUFFERS)

    scores = program.qk(k, last_start)
    probabilities, row_sum, accumulator = program.softmax_part1(
        probabilities, row_sum, accumulator, alpha
    )
    v = program.load_v((iteration + 1) % NUM_BUFFERS, 1)
    accumulator = program.pv(probabilities, v, accumulator)
    probabilities, alpha, row_max = program.softmax_part0(scores, row_max)
    probabilities, row_sum, accumulator = program.softmax_part1(
        probabilities, row_sum, accumulator, alpha
    )
    v = program.load_v(iteration % NUM_BUFFERS, 0)
    accumulator = program.pv(probabilities, v, accumulator)

    output = accumulator * (1.0 / row_sum)[:, None]
    if USE_TDM_OUTPUT:
        # Let the allocator reuse the dead K/V storage for output staging.
        program.k_buffer._keep_alive()
        program.v_buffer._keep_alive()
        program.store_output_tdm(output, stride_os, stride_od)
    else:
        program.store_output(output)


# Specialize tensor strides as in TokenSpeed so TDM address arithmetic folds
# at compile time. Shape specialization already distinguishes these launches.
@gluon.jit
def _attention_forward_norm_selected(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    invalid_ptr,
    stride_qb: gl.constexpr,
    stride_qh: gl.constexpr,
    stride_qs: gl.constexpr,
    stride_qd: gl.constexpr,
    stride_kb: gl.constexpr,
    stride_kh: gl.constexpr,
    stride_ks: gl.constexpr,
    stride_kd: gl.constexpr,
    stride_vb: gl.constexpr,
    stride_vh: gl.constexpr,
    stride_vs: gl.constexpr,
    stride_vd: gl.constexpr,
    stride_ob: gl.constexpr,
    stride_oh: gl.constexpr,
    stride_os: gl.constexpr,
    stride_od: gl.constexpr,
    SOFTMAX_SCALE: gl.constexpr,
    SEQLEN_Q: gl.constexpr,
    SEQLEN_K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    ROUTE: gl.constexpr,
    TDM_OUTPUT: gl.constexpr = False,
    USE_QUERY_BUFFER: gl.constexpr = True,
    USE_OUTPUT_BUFFER: gl.constexpr = True,
):
    bits = gl.load(invalid_ptr)
    forward_args = (
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        stride_qb,
        stride_qh,
        stride_qs,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vs,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_os,
        stride_od,
    )
    if ROUTE == _ROUTE_NORM_OR_FUSED_EXP:
        if (bits & _NORM_UNSAFE) == 0:
            _attention_forward_pingpong(
                *forward_args,
                SOFTMAX_SCALE,
                SEQLEN_Q,
                SEQLEN_K,
                BLOCK_M,
                BLOCK_N,
                HEAD_SIZE,
                TDM_OUTPUT,
                USE_QUERY_BUFFER,
                USE_OUTPUT_BUFFER,
                FAST_EXP=False,
                NORM_EXP=True,
            )
        elif bits == _NORM_UNSAFE:
            _attention_forward_pingpong(
                *forward_args,
                SOFTMAX_SCALE,
                SEQLEN_Q,
                SEQLEN_K,
                BLOCK_M,
                BLOCK_N,
                HEAD_SIZE,
                TDM_OUTPUT,
                USE_QUERY_BUFFER,
                USE_OUTPUT_BUFFER,
                FAST_EXP=True,
                NORM_EXP=False,
            )
    else:
        selected = (
            ((ROUTE == _ROUTE_NORM) & ((bits & _NORM_UNSAFE) == 0))
            | ((ROUTE == _ROUTE_FUSED_EXP) & (bits == _NORM_UNSAFE))
            | ((ROUTE == _ROUTE_STABLE) & (bits == (_NORM_UNSAFE | _FUSED_EXP_UNSAFE)))
        )
        if selected:
            _attention_forward_pingpong(
                *forward_args,
                SOFTMAX_SCALE,
                SEQLEN_Q,
                SEQLEN_K,
                BLOCK_M,
                BLOCK_N,
                HEAD_SIZE,
                TDM_OUTPUT,
                USE_QUERY_BUFFER,
                USE_OUTPUT_BUFFER,
                FAST_EXP=ROUTE == _ROUTE_FUSED_EXP,
                NORM_EXP=ROUTE == _ROUTE_NORM,
            )


@gluon.jit
def _clear_invalid(invalid):
    gl.store(invalid, 0)


def _guard_bound(scale):
    # Bound D*max(|Q|)*max(|K|)*scale_log2. Reserve FP32 accumulation margin.
    # At magnitude <=1024, rounding the scaled maximum introduces at most
    # 2**-14 of exponent error, keeping its fused exponent near zero.
    return math.sqrt(1024 / (128 * scale * 1.4426950408889634)) * (1 - 1e-5)


def _norm_guard_bound(scale):
    # Cauchy-Schwarz bounds |QK| by the product of the row norms. The 1e-4
    # margin covers FP32 norm reduction and dot-product accumulation rounding;
    # the fixed shift then keeps exp2 arguments in [-64, 0].
    return (32.0 / (scale * 1.4426950408889634)) * (1.0 - 1e-4)


@gluon.jit
def _norm_range_atomic(
    x_ptr,
    invalid_ptr,
    B: gl.constexpr,
    S: gl.constexpr,
    H: gl.constexpr,
    SB: gl.constexpr,
    SS: gl.constexpr,
    SH: gl.constexpr,
    SD: gl.constexpr,
    BOUND: gl.constexpr,
    FMA_BOUND: gl.constexpr,
    CONTIGUOUS: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    rows = gl.program_id(0).to(gl.int64) * 16 + gl.arange(
        0, 16, layout=gl.SliceLayout(1, layout)
    ).to(gl.int64)
    dims = gl.arange(0, 128, layout=gl.SliceLayout(0, layout)).to(gl.int64)
    if CONTIGUOUS:
        offsets = rows[:, None] * 128 + dims[None, :]
    else:
        batch = rows // (S * H)
        seq = (rows // H) % S
        head = rows % H
        offsets = (
            batch[:, None] * SB
            + seq[:, None] * SS
            + head[:, None] * SH
            + dims[None, :] * SD
        )
    x = gl.load(x_ptr + offsets, mask=rows[:, None] < B * S * H, other=0.0).to(
        gl.float32
    )
    norm = gl.sum(x * x, 1)
    invalid_norm = gl.sum(((norm > BOUND) | (norm != norm)).to(gl.int32), 0) != 0
    invalid_fma = (
        gl.sum(gl.sum(((gl.abs(x) > FMA_BOUND) | (x != x)).to(gl.int32), 1), 0) != 0
    )
    bits = (invalid_norm.to(gl.int32) * _NORM_UNSAFE) | (
        invalid_fma.to(gl.int32) * _FUSED_EXP_UNSAFE
    )
    if bits != 0:
        gl.atomic_or(invalid_ptr, bits, sem="relaxed")


@gluon.jit
def _value_range_atomic(
    x_ptr,
    invalid_ptr,
    B: gl.constexpr,
    S: gl.constexpr,
    H: gl.constexpr,
    SB: gl.constexpr,
    SS: gl.constexpr,
    SH: gl.constexpr,
    SD: gl.constexpr,
    CONTIGUOUS: gl.constexpr,
    BLOCK: gl.constexpr,
):
    i = gl.program_id(0).to(gl.int64) * BLOCK + gl.arange(
        0, BLOCK, layout=gl.BlockedLayout([1], [32], [4], [0])
    ).to(gl.int64)
    if CONTIGUOUS:
        offsets = i
    else:
        rows = i // 128
        offsets = (
            (rows // (S * H)) * SB
            + ((rows // H) % S) * SS
            + (rows % H) * SH
            + (i % 128) * SD
        )
    x = gl.abs(
        gl.load(x_ptr + offsets, mask=i < B * S * H * 128, other=0.0).to(gl.float32)
    )
    # With P in [2^-64, 1], this range keeps nonzero P*V products normal
    # in FP32 and sums finite for the guarded sequence-length limit.
    invalid = (
        (x != x)
        | (x > 18446744073709551616.0)
        | ((x != 0.0) & (x < 2.3283064365386963e-10))
    )
    if gl.sum(invalid.to(gl.int32), 0) != 0:
        gl.atomic_or(invalid_ptr, _NORM_UNSAFE, sem="relaxed")


def _check_norm_bounded_inputs(q, k, v, invalid, scale):
    _clear_invalid[(1,)](invalid, num_warps=4)
    for x in (q, k):
        b, s, h, d = x.shape
        _norm_range_atomic[(math.ceil(b * s * h / 16),)](
            x,
            invalid,
            b,
            s,
            h,
            *x.stride(),
            _norm_guard_bound(scale),
            _guard_bound(scale),
            x.is_contiguous(),
            num_warps=4,
        )
    b, s, h, d = v.shape
    _value_range_atomic[(math.ceil(v.numel() / 8192),)](
        v, invalid, b, s, h, *v.stride(), v.is_contiguous(), 8192, num_warps=4
    )


def _launch_norm_guarded_attention(query, key, value, output, invalid, softmax_scale):
    # These ordered kernels are captured together. Every replay clears and
    # recomputes the flag, so mutated inputs cannot reuse an earlier decision.
    _check_norm_bounded_inputs(query, key, value, invalid, softmax_scale)
    batch, seqlen_q, num_heads, _ = query.shape
    seqlen_k = key.shape[1]
    query_tiles = math.ceil(seqlen_q / 256)
    grid = (
        (query_tiles, num_heads, batch)
        if seqlen_k >= 16384
        else (batch, num_heads, query_tiles)
    )
    strides = tuple(
        s
        for tensor in (query, key, value, output)
        for s in (
            tensor.stride(0),
            tensor.stride(2),
            tensor.stride(1),
            tensor.stride(3),
        )
    )
    query_tile_bytes = (
        255 * query.stride(1) + 127 * query.stride(3) + 1
    ) * query.element_size()
    output_tile_bytes = (
        255 * output.stride(1) + 127 * output.stride(3) + 1
    ) * output.element_size()
    for route in (_ROUTE_NORM_OR_FUSED_EXP.value, _ROUTE_STABLE.value):
        _attention_forward_norm_selected[grid](
            query,
            key,
            value,
            output,
            invalid,
            *strides,
            softmax_scale,
            seqlen_q,
            seqlen_k,
            256,
            64,
            128,
            route,
            TDM_OUTPUT=True,
            USE_QUERY_BUFFER=query_tile_bytes <= 2**31,
            USE_OUTPUT_BUFFER=output_tile_bytes <= 2**31,
            num_warps=8,
            waves_per_eu=2,
            llvm_fn_attrs="amdgpu-sched-strategy=max-ilp",
            enable_fp_fusion=True,
        )


def flash_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    launch_config: LaunchConfig | None = None,
) -> torch.Tensor:
    """Run non-causal BF16/FP16 MHA on gfx1250.

    Q/K/V use BSHD layout. Query and key sequence lengths may differ.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Gluon FAv3 expects rank-4 BSHD query, key, and value")
    if (
        query.dtype not in (torch.bfloat16, torch.float16)
        or key.dtype != query.dtype
        or value.dtype != query.dtype
    ):
        raise TypeError(
            "Gluon FAv3 requires matching BF16 or FP16 query, key, and value"
        )
    if not query.is_cuda or key.device != query.device or value.device != query.device:
        raise ValueError("Gluon FAv3 requires Q/K/V on the same ROCm device")

    batch, seqlen_q, num_heads, head_size = query.shape
    key_batch, seqlen_k, num_kv_heads, key_head_size = key.shape
    if value.shape != key.shape:
        raise ValueError("Gluon FAv3 requires key and value to have the same shape")
    if (key_batch, num_kv_heads, key_head_size) != (
        batch,
        num_heads,
        head_size,
    ):
        raise ValueError("Gluon FAv3 requires full MHA with matching Q/K/V heads")
    if head_size not in (64, 128, 256):
        raise ValueError(
            f"Gluon FAv3 requires head_size in (64, 128, 256), got {head_size}"
        )
    if min(batch, num_heads, seqlen_q, seqlen_k) <= 0:
        raise ValueError("Gluon FAv3 requires non-empty batches, heads, and sequences")
    if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("Gluon FAv3 requires contiguous head dimensions")

    arch = getattr(torch.cuda.get_device_properties(query.device), "gcnArchName", "")
    if "gfx1250" not in arch:
        raise RuntimeError(
            f"Gluon FAv3 requires gfx1250, found {arch or 'unknown GPU'}"
        )

    policy = resolve_execution_policy(
        query, key, value, softmax_scale=softmax_scale, launch_config=launch_config
    )
    config = policy.launch_config
    softmax_scale = policy.softmax_scale
    if config.schedule == "persistent":
        from sglang.kernels.ops.attention.gluon.fa3.gfx1250.persistent import (
            persistent_attention,
        )

        return persistent_attention(query, key, value, softmax_scale=softmax_scale)
    kernel = (
        _attention_forward_pingpong
        if config.schedule == "pingpong"
        else _attention_forward_pipeline
    )
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    # Keep every tile-local byte offset in the signed 32-bit buffer range.
    # The constexpr flags preserve the usual buffer path without runtime branches.
    query_tile_bytes = (
        (config.block_m - 1) * query.stride(1) + (head_size - 1) * query.stride(3) + 1
    ) * query.element_size()
    output_tile_bytes = (
        (config.block_m - 1) * output.stride(1) + (head_size - 1) * output.stride(3) + 1
    ) * output.element_size()
    query_tiles = math.ceil(seqlen_q / config.block_m)
    grid = (
        (query_tiles, num_heads, batch)
        if config.block_m == 256 and seqlen_k >= 16_384
        else (batch, num_heads, query_tiles)
    )
    with torch.cuda.device(query.device):
        if policy.numerical_mode == NumericalMode.GUARDED:
            invalid = torch.empty((), dtype=torch.int32, device=query.device)
            _launch_norm_guarded_attention(
                query, key, value, output, invalid, softmax_scale
            )
            return output
        kernel[grid](
            query,
            key,
            value,
            output,
            query.stride(0),
            query.stride(2),
            query.stride(1),
            query.stride(3),
            key.stride(0),
            key.stride(2),
            key.stride(1),
            key.stride(3),
            value.stride(0),
            value.stride(2),
            value.stride(1),
            value.stride(3),
            output.stride(0),
            output.stride(2),
            output.stride(1),
            output.stride(3),
            softmax_scale,
            seqlen_q,
            seqlen_k,
            config.block_m,
            config.block_n,
            head_size,
            TDM_OUTPUT=config.tdm_output,
            USE_QUERY_BUFFER=query_tile_bytes <= 2**31,
            USE_OUTPUT_BUFFER=output_tile_bytes <= 2**31,
            num_warps=config.num_warps,
            waves_per_eu=2 if config.schedule == "pingpong" else 1,
            llvm_fn_attrs=config.llvm_fn_attrs,
            enable_fp_fusion=True,
        )
    return output


__all__ = [
    "LaunchConfig",
    "flash_attention",
    "resolve_execution_policy",
    "select_launch_config",
]
