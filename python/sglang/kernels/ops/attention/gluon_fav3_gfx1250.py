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
layout. The production path is intentionally narrow: non-causal BF16 MHA with
head dimension 128. Unsupported contracts are handled by the diffusion backend
before this module is imported.

This is adapted from Triton's ``f16_fa_cdna5.py`` example:
https://github.com/triton-lang/triton/blob/b63e34c521dfc55a5f9e419cfeaa64df8263891f/third_party/amd/python/examples/gluon/f16_fa_cdna5.py

It carries the gfx1250 launch, scheduler, softmax-stage, and TDM-output tuning
validated by the TokenSpeed/Triton optimization work.
"""

from __future__ import annotations

import functools
import math
import os
from typing import NamedTuple

import torch
import triton.experimental.gluon.language as gl
from triton.experimental import gluon
from triton.runtime import driver


class FAv3LaunchConfig(NamedTuple):
    schedule: str
    block_m: int
    block_n: int
    num_warps: int
    tdm_output: bool
    llvm_fn_attrs: str


@gluon.aggregate
class _AttentionConfig:
    SEQLEN_Q: gl.constexpr
    SEQLEN_K: gl.constexpr
    HEAD_SIZE: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    NUM_BUFFERS: gl.constexpr

    qk_layout: gl.constexpr
    pv_layout: gl.constexpr
    k_smem_layout: gl.constexpr
    v_smem_layout: gl.constexpr
    q_layout: gl.constexpr
    k_layout: gl.constexpr
    v_layout: gl.constexpr
    p_layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(
        self,
        seqlen_q,
        seqlen_k,
        head_size,
        block_m,
        block_n,
        num_buffers,
        num_warps,
    ):
        self.SEQLEN_Q = gl.constexpr(seqlen_q)
        self.SEQLEN_K = gl.constexpr(seqlen_k)
        self.HEAD_SIZE = gl.constexpr(head_size)
        self.BLOCK_M = gl.constexpr(block_m)
        self.BLOCK_N = gl.constexpr(block_n)
        self.NUM_BUFFERS = gl.constexpr(num_buffers)

        assert num_warps == 4 or num_warps == 8
        warp_bases = [[1, 0], [2, 0]]
        if num_warps == 8:
            warp_bases.append([4, 0])

        self.qk_layout = gl.constexpr(
            gl.amd.AMDWMMALayout(
                3,
                transposed=True,
                warp_bases=warp_bases,
                instr_shape=[16, 16, 32],
            )
        )
        self.pv_layout = gl.constexpr(
            gl.amd.AMDWMMALayout(
                3,
                transposed=True,
                warp_bases=warp_bases,
                instr_shape=[16, 16, 32],
            )
        )
        self.k_smem_layout = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for(
                [[head_size, 8]], [block_n, head_size], [1, 0]
            )
        )
        self.v_smem_layout = gl.constexpr(
            gl.PaddedSharedLayout.with_identity_for(
                [[head_size, 16]], [block_n, head_size], [1, 0]
            )
        )
        self.q_layout = gl.constexpr(gl.DotOperandLayout(0, self.qk_layout, 8))
        self.k_layout = gl.constexpr(gl.DotOperandLayout(1, self.qk_layout, 8))
        self.v_layout = gl.constexpr(gl.DotOperandLayout(1, self.pv_layout, 8))
        self.p_layout = gl.constexpr(gl.DotOperandLayout(0, self.pv_layout, 8))


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
    ):
        batch_id = gl.program_id(0)
        head_id = gl.program_id(1)
        query_start = gl.program_id(2) * cfg.BLOCK_M

        query_rows = query_start + gl.arange(
            0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.q_layout)
        )
        query_dims = gl.arange(0, cfg.HEAD_SIZE, layout=gl.SliceLayout(0, cfg.q_layout))
        query_offsets = (
            stride_qs * query_rows[:, None] + stride_qd * query_dims[None, :]
        )
        query_mask = query_rows[:, None] < cfg.SEQLEN_Q
        query_base = q_ptr + stride_qb * batch_id + stride_qh * head_id
        q = gl.amd.cdna5.buffer_load(
            query_base, query_offsets, mask=query_mask, other=0.0
        )

        k_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=k_ptr + stride_kb * batch_id + stride_kh * head_id,
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
            base=v_ptr + stride_vb * batch_id + stride_vh * head_id,
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

        output_rows = query_start + gl.arange(
            0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout)
        )
        output_dims = gl.arange(
            0, cfg.HEAD_SIZE, layout=gl.SliceLayout(0, cfg.pv_layout)
        )
        output_offsets = (
            stride_os * output_rows[:, None] + stride_od * output_dims[None, :]
        )
        output_mask = output_rows[:, None] < cfg.SEQLEN_Q
        output_base = output_ptr + stride_ob * batch_id + stride_oh * head_id

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
        )

    @gluon.jit
    def load_k(self, buffer_index, wait_count):
        gl.amd.cdna5.tdm.async_wait(wait_count)
        return (
            self.k_buffer.index(buffer_index)
            .permute([1, 0])
            .load(layout=self.cfg.k_layout)
        )

    @gluon.jit
    def load_v(self, buffer_index, wait_count):
        gl.amd.cdna5.tdm.async_wait(wait_count)
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
        new_row_max = gl.maximum(row_max, gl.max(scores, 1))
        new_row_max_scaled = new_row_max * self.softmax_scale_log2
        shifted = self.softmax_scale_log2 * scores - new_row_max_scaled[:, None]
        probabilities = gl.exp2(shifted)
        max_delta = self.softmax_scale_log2 * row_max - new_row_max_scaled
        alpha = gl.exp2(max_delta)
        return probabilities, alpha, new_row_max

    @gluon.jit
    def softmax_prepare(self, scores, row_max):
        new_row_max = gl.maximum(row_max, gl.max(scores, 1))
        new_row_max_scaled = new_row_max * self.softmax_scale_log2
        max_delta = self.softmax_scale_log2 * row_max - new_row_max_scaled
        return new_row_max_scaled, max_delta, new_row_max

    @gluon.jit
    def softmax_prepare_fixed(self, row_max):
        row_max_scaled = row_max * self.softmax_scale_log2
        max_delta = gl.zeros(row_max.shape, gl.float32, row_max.type.layout)
        return row_max_scaled, max_delta, row_max

    @gluon.jit
    def softmax_finish(self, scores, new_row_max_scaled, max_delta):
        shifted = self.softmax_scale_log2 * scores - new_row_max_scaled[:, None]
        return gl.exp2(shifted), gl.exp2(max_delta)

    @gluon.jit
    def softmax_finish_fixed(self, scores, row_max_scaled, row_max):
        shifted = self.softmax_scale_log2 * scores - row_max_scaled[:, None]
        probabilities = gl.exp2(shifted)
        alpha = gl.full(row_max.shape, 1.0, gl.float32, row_max.type.layout)
        return probabilities, alpha

    @gluon.jit
    def softmax_part0_fixed(self, scores, row_max):
        row_max_scaled = row_max * self.softmax_scale_log2
        shifted = self.softmax_scale_log2 * scores - row_max_scaled[:, None]
        probabilities = gl.exp2(shifted)
        alpha = gl.full(row_max.shape, 1.0, gl.float32, row_max.type.layout)
        return probabilities, alpha, row_max

    @gluon.jit
    def softmax_part0_mode(self, scores, row_max, fixed_shift: gl.constexpr):
        if fixed_shift:
            return self.softmax_part0_fixed(scores, row_max)
        return self.softmax_part0(scores, row_max)

    @gluon.jit
    def softmax_prepare_mode(self, scores, row_max, fixed_shift: gl.constexpr):
        if fixed_shift:
            return self.softmax_prepare_fixed(row_max)
        return self.softmax_prepare(scores, row_max)

    @gluon.jit
    def softmax_finish_mode(
        self,
        scores,
        row_max_scaled,
        max_delta,
        row_max,
        new_row_max,
        fixed_shift: gl.constexpr,
    ):
        if fixed_shift:
            probabilities, alpha = self.softmax_finish_fixed(
                scores, row_max_scaled, row_max
            )
            return probabilities, alpha, row_max
        probabilities, alpha = self.softmax_finish(scores, row_max_scaled, max_delta)
        return probabilities, alpha, new_row_max

    @gluon.jit
    def softmax_part1(self, probabilities, row_sum, accumulator, alpha):
        tile_sum = gl.sum(probabilities, 1)
        accumulator = accumulator * alpha[:, None]
        probabilities = probabilities.to(gl.bfloat16, fp_downcast_rounding="rtz")
        row_sum = row_sum * alpha + tile_sum
        return probabilities, row_sum, accumulator

    @gluon.jit
    def softmax_part1_fixed(self, probabilities, row_sum, accumulator):
        tile_sum = gl.sum(probabilities, 1)
        probabilities = probabilities.to(gl.bfloat16, fp_downcast_rounding="rtz")
        return probabilities, row_sum + tile_sum, accumulator

    @gluon.jit
    def softmax_part1_mode(
        self,
        probabilities,
        row_sum,
        accumulator,
        alpha,
        fixed_shift: gl.constexpr,
    ):
        if fixed_shift:
            return self.softmax_part1_fixed(probabilities, row_sum, accumulator)
        return self.softmax_part1(probabilities, row_sum, accumulator, alpha)

    @gluon.jit
    def pv(self, probabilities, v, accumulator):
        probabilities = gl.convert_layout(probabilities, self.cfg.p_layout)
        return gl.amd.cdna5.wmma(probabilities, v, accumulator)

    @gluon.jit
    def store_output(self, output):
        output = output.to(self.output_ptr.dtype.element_ty)
        gl.amd.cdna5.buffer_store(
            output,
            self.output_ptr,
            self.output_offsets,
            mask=self.output_mask,
        )

    @gluon.jit
    def store_output_tdm(self, output, stride_os, stride_od):
        layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        query_start = gl.program_id(2) * self.cfg.BLOCK_M
        output_desc = gl.amd.cdna5.tdm.make_tensor_descriptor(
            base=self.output_ptr,
            shape=(self.cfg.SEQLEN_Q, self.cfg.HEAD_SIZE),
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
            [query_start, 0],
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
    FIXED_SHIFT: gl.constexpr = False,
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
    FIXED_SHIFT: gl.constexpr = False,
):
    NUM_BUFFERS: gl.constexpr = 2
    NUM_WARPS: gl.constexpr = 8
    REBALANCE_SOFTMAX: gl.constexpr = (
        HEAD_SIZE == 128 and BLOCK_M == 256 and BLOCK_N == 64
    )
    USE_TDM_OUTPUT: gl.constexpr = TDM_OUTPUT and REBALANCE_SOFTMAX
    cfg = _AttentionConfig(
        SEQLEN_Q,
        SEQLEN_K,
        HEAD_SIZE,
        BLOCK_M,
        BLOCK_N,
        NUM_BUFFERS,
        NUM_WARPS,
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
    if FIXED_SHIFT:
        row_sum = gl.full(
            [BLOCK_M],
            0.0,
            dtype=gl.float32,
            layout=gl.SliceLayout(1, cfg.pv_layout),
        )
    else:
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
        with gl.amd.warp_pipeline_stage("stage0"):
            next_v_start = block_start + 2 * BLOCK_N
            next_k_start = block_start + 3 * BLOCK_N
            scores = program.qk_full(k)

        gl.amd.cdna5.tdm.async_wait(2)
        with gl.amd.warp_pipeline_stage("stage1"):
            probabilities, row_sum, accumulator = program.softmax_part1_mode(
                probabilities, row_sum, accumulator, alpha, FIXED_SHIFT
            )
            v = program.v_buffer.index(iteration % NUM_BUFFERS).load(
                layout=program.cfg.v_layout
            )
            program.prefetch_k(next_k_start, (iteration + 1) % NUM_BUFFERS)
            if REBALANCE_SOFTMAX:
                (
                    new_row_max_scaled,
                    max_delta,
                    new_row_max,
                ) = program.softmax_prepare_mode(scores, row_max, FIXED_SHIFT)

        with gl.amd.warp_pipeline_stage("stage2"):
            accumulator = program.pv(probabilities, v, accumulator)

        gl.amd.cdna5.tdm.async_wait(2)
        with gl.amd.warp_pipeline_stage("stage3"):
            if REBALANCE_SOFTMAX:
                probabilities, alpha, row_max = program.softmax_finish_mode(
                    scores,
                    new_row_max_scaled,
                    max_delta,
                    row_max,
                    new_row_max,
                    FIXED_SHIFT,
                )
            else:
                probabilities, alpha, row_max = program.softmax_part0_mode(
                    scores, row_max, FIXED_SHIFT
                )
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
        probabilities, row_sum, accumulator = program.softmax_part1_mode(
            probabilities, row_sum, accumulator, alpha, FIXED_SHIFT
        )
        v = program.load_v(iteration % NUM_BUFFERS, 2)
        program.prefetch_k(next_k_start, (iteration + 1) % NUM_BUFFERS)
        accumulator = program.pv(probabilities, v, accumulator)
        probabilities, alpha, row_max = program.softmax_part0_mode(
            scores, row_max, FIXED_SHIFT
        )
        k = program.load_k(iteration % NUM_BUFFERS, 2)
        program.prefetch_v(next_v_start, iteration % NUM_BUFFERS)
        iteration += 1

    epilogue_base = (iteration - 1) * BLOCK_N
    second_last_start = epilogue_base + 2 * BLOCK_N
    last_start = epilogue_base + 3 * BLOCK_N

    probabilities, row_sum, accumulator = program.softmax_part1_mode(
        probabilities, row_sum, accumulator, alpha, FIXED_SHIFT
    )
    v = program.load_v(iteration % NUM_BUFFERS, 2)
    accumulator = program.pv(probabilities, v, accumulator)

    scores = program.qk(k, second_last_start)
    probabilities, alpha, row_max = program.softmax_part0_mode(
        scores, row_max, FIXED_SHIFT
    )
    k = program.load_k(iteration % NUM_BUFFERS, 1)
    program.prefetch_v(last_start, iteration % NUM_BUFFERS)

    scores = program.qk(k, last_start)
    probabilities, row_sum, accumulator = program.softmax_part1_mode(
        probabilities, row_sum, accumulator, alpha, FIXED_SHIFT
    )
    v = program.load_v((iteration + 1) % NUM_BUFFERS, 1)
    accumulator = program.pv(probabilities, v, accumulator)
    probabilities, alpha, row_max = program.softmax_part0_mode(
        scores, row_max, FIXED_SHIFT
    )
    probabilities, row_sum, accumulator = program.softmax_part1_mode(
        probabilities, row_sum, accumulator, alpha, FIXED_SHIFT
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


@functools.lru_cache(maxsize=None)
def _get_num_cus(device_index: int) -> int:
    properties = driver.active.utils.get_device_properties(device_index)
    return int(properties["multiprocessor_count"])


def get_num_cus(device: torch.device | int | None = None) -> int:
    """Return the CU count for a specific device, cached per device index."""
    if device is None:
        device_index = driver.active.get_device_interface().current_device()
    elif isinstance(device, torch.device):
        device_index = device.index
        if device_index is None:
            device_index = driver.active.get_device_interface().current_device()
    else:
        device_index = int(device)
    return _get_num_cus(device_index)


def select_fav3_launch_config(
    batch: int,
    num_heads: int,
    seqlen_q: int,
    *,
    seqlen_k: int | None = None,
    num_cus: int | None = None,
) -> FAv3LaunchConfig:
    """Select the measured gfx1250 D128 launch policy."""
    if min(batch, num_heads, seqlen_q) <= 0:
        raise ValueError("batch, num_heads, and seqlen_q must be positive")
    if batch == 2 and num_heads == 40 and seqlen_q == 176_400 and seqlen_k == 512:
        return FAv3LaunchConfig(
            "wan_cross",
            128,
            256,
            4,
            True,
            "amdgpu-sched-strategy=iterative-ilp",
        )
    if num_cus is None:
        num_cus = get_num_cus()

    wide_block_m = 256
    workgroups = batch * num_heads * math.ceil(seqlen_q / wide_block_m)
    if workgroups >= num_cus:
        # TDM output is profitable at and above the same full-device boundary
        # that admits the wide ping-pong tile.
        scheduler = (
            "amdgpu-sched-strategy=max-ilp"
            if seqlen_k is not None and seqlen_k <= 512
            else ""
        )
        use_fixed_shift = (
            batch == 2
            and num_heads == 40
            and seqlen_q == 176_400
            and seqlen_k == seqlen_q
            and os.environ.get("SGLANG_GLUON_FAV3_WAN_FIXED_SHIFT", "0") == "1"
        )
        schedule = "wan_self" if use_fixed_shift else "pingpong"
        return FAv3LaunchConfig(schedule, 256, 64, 8, True, scheduler)
    return FAv3LaunchConfig(
        "pipeline",
        128,
        64,
        4,
        False,
        "amdgpu-sched-strategy=max-ilp",
    )


def gluon_fav3_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    softmax_scale: float | None = None,
    launch_config: FAv3LaunchConfig | None = None,
) -> torch.Tensor:
    """Run non-causal BF16 MHA on gfx1250.

    Q/K/V use BSHD layout. Query and key sequence lengths may differ.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Gluon FAv3 expects rank-4 BSHD query, key, and value")
    if (
        query.dtype != torch.bfloat16
        or key.dtype != query.dtype
        or value.dtype != query.dtype
    ):
        raise TypeError("Gluon FAv3 requires matching BF16 query, key, and value")
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
    if head_size != 128:
        raise ValueError(f"Gluon FAv3 requires head_size=128, got {head_size}")
    if seqlen_q <= 0 or seqlen_k <= 0:
        raise ValueError("Gluon FAv3 requires non-empty query and key sequences")
    if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError("Gluon FAv3 requires contiguous head dimensions")

    if softmax_scale is None:
        softmax_scale = head_size**-0.5
    softmax_scale = float(softmax_scale)
    if not math.isfinite(softmax_scale) or softmax_scale <= 0.0:
        raise ValueError("softmax_scale must be finite and positive")

    arch = getattr(torch.cuda.get_device_properties(query.device), "gcnArchName", "")
    if "gfx1250" not in arch:
        raise RuntimeError(
            f"Gluon FAv3 requires gfx1250, found {arch or 'unknown GPU'}"
        )

    config = launch_config or select_fav3_launch_config(
        batch,
        num_heads,
        seqlen_q,
        seqlen_k=seqlen_k,
        num_cus=get_num_cus(query.device),
    )
    wan_self_contract = (
        batch == 2 and num_heads == 40 and seqlen_q == 176_400 and seqlen_k == seqlen_q
    )
    fixed_shift = config.schedule == "wan_self"
    if fixed_shift and (
        not wan_self_contract
        or os.environ.get("SGLANG_GLUON_FAV3_WAN_FIXED_SHIFT", "0") != "1"
    ):
        raise ValueError(
            "wan_self FAv3 requires B2/S176400/H40/D128 and "
            "SGLANG_GLUON_FAV3_WAN_FIXED_SHIFT=1"
        )
    if config.schedule == "wan_cross":
        expected_geometry = (128, 256, 4)
        actual_geometry = (config.block_m, config.block_n, config.num_warps)
        if actual_geometry != expected_geometry:
            raise ValueError(
                "wan_cross requires (block_m, block_n, num_warps)="
                f"{expected_geometry}, got {actual_geometry}"
            )
        if not config.tdm_output:
            raise ValueError("wan_cross requires TDM output and generic online softmax")
        if query.is_contiguous() and key.is_contiguous() and value.is_contiguous():
            from sglang.kernels.ops.attention.gluon_fav3_wan_cross_gfx1250 import (
                gluon_fav3_wan_cross_attention,
            )

            return gluon_fav3_wan_cross_attention(
                query,
                key,
                value,
                softmax_scale=softmax_scale,
            )
        config = FAv3LaunchConfig(
            "pingpong",
            256,
            64,
            8,
            True,
            "amdgpu-sched-strategy=max-ilp",
        )
    if config.schedule not in ("pipeline", "pingpong", "wan_self"):
        raise ValueError(f"unsupported FAv3 schedule: {config.schedule}")
    expected_geometry = (128, 64, 4) if config.schedule == "pipeline" else (256, 64, 8)
    actual_geometry = (config.block_m, config.block_n, config.num_warps)
    if actual_geometry != expected_geometry:
        raise ValueError(
            f"{config.schedule} requires (block_m, block_n, num_warps)="
            f"{expected_geometry}, got {actual_geometry}"
        )
    kernel = (
        _attention_forward_pingpong
        if config.schedule in ("pingpong", "wan_self")
        else _attention_forward_pipeline
    )
    output = torch.empty_like(query, memory_format=torch.contiguous_format)
    grid = (batch, num_heads, math.ceil(seqlen_q / config.block_m))
    with torch.cuda.device(query.device):
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
            FIXED_SHIFT=fixed_shift,
            num_warps=config.num_warps,
            waves_per_eu=2 if config.schedule in ("pingpong", "wan_self") else 1,
            llvm_fn_attrs=config.llvm_fn_attrs,
        )
    return output


__all__ = [
    "FAv3LaunchConfig",
    "gluon_fav3_attention",
    "select_fav3_launch_config",
]
