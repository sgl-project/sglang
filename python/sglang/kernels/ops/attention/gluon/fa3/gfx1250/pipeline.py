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

"""Short and pipelined tiles for dense gfx1250 attention."""

import triton.experimental.gluon.language as gl
from triton.experimental import gluon
from triton.language import PropagateNan

_PROPAGATE_NAN = gl.constexpr(PropagateNan.ALL)


@gluon.jit
def _maximum(left, right):
    return gl.maximum(left, right, propagate_nan=_PROPAGATE_NAN)


@gluon.jit
def _max(input, axis):
    return gl.reduce(input, axis, _maximum)


@gluon.jit
def _local_probability_sums(probabilities, cfg):
    if cfg.HEAD_SIZE == 128 and cfg.BLOCK_M >= 64:
        linear: gl.constexpr = gl.to_linear_layout(
            cfg.qk_layout, [cfg.BLOCK_M, cfg.BLOCK_N]
        )
        terms = gl.convert_layout(probabilities, linear)
        terms = gl.reshape(terms, [cfg.BLOCK_M, cfg.BLOCK_N // 16, 2, 8])
        return gl.sum(gl.sum(terms, axis=3), axis=1)
    else:
        return probabilities


@gluon.jit
def _probabilities(program, scores, row_max, FIXED_SHIFT: gl.constexpr):
    if FIXED_SHIFT:
        probabilities = gl.exp2(
            gl.fma(
                scores,
                program.softmax_scale_log2,
                -row_max[:, None] * program.softmax_scale_log2,
            )
        )
        alpha = gl.full(
            [program.cfg.BLOCK_M],
            1.0,
            gl.float32,
            gl.SliceLayout(1, program.cfg.pv_layout),
        )
        return probabilities, alpha, row_max
    else:
        return program.softmax_part0(scores, row_max)


@gluon.jit
def _accumulate_probabilities(
    program, FIXED_SHIFT: gl.constexpr, probabilities, row_sum, accumulator, alpha
):
    if FIXED_SHIFT:
        row_sum += _local_probability_sums(probabilities, program.cfg)
        probabilities = gl.convert_layout(
            probabilities.to(program.q.dtype), program.cfg.p_layout
        )
        return probabilities, row_sum, accumulator
    else:
        probabilities, row_sum, accumulator = program.softmax_part1(
            probabilities, row_sum, accumulator, alpha
        )
        return (
            gl.convert_layout(probabilities, program.cfg.p_layout),
            row_sum,
            accumulator,
        )


@gluon.jit
def prefetch_kv(cfg, k_desc, k_buffer, v_desc, v_buffer, sequence_start, buffer_index):
    k_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
        k_desc, add_offsets=[sequence_start, 0], pred=True, clamp_bounds=True
    )
    v_desc = gl.amd.cdna5.tdm.update_tensor_descriptor(
        v_desc, add_offsets=[sequence_start, 0], pred=True, clamp_bounds=True
    )
    k_waves: gl.constexpr = 0b01 if cfg.NUM_WARPS == 2 else 0b0011
    v_waves: gl.constexpr = 0b10 if cfg.NUM_WARPS == 2 else 0b1100
    gl.amd.cdna5.tdm.async_load_fused(
        [
            (k_desc, k_buffer.index(buffer_index), k_waves),
            (v_desc, v_buffer.index(buffer_index), v_waves),
        ],
        cache_modifier=".cg",
    )


@gluon.jit
def process_short(program, tiles: gl.constexpr):
    """Complete one or two resident key tiles without a pipeline prologue."""
    cfg = program.cfg
    for tile in gl.static_range(tiles):
        prefetch_kv(
            cfg,
            program.k_desc,
            program.k_buffer,
            program.v_desc,
            program.v_buffer,
            tile * cfg.BLOCK_N,
            tile,
        )
    gl.amd.cdna5.tdm.async_wait(0)
    row_max = gl.full(
        [cfg.BLOCK_M],
        float("-inf"),
        gl.float32,
        gl.SliceLayout(1, cfg.pv_layout),
    )
    row_sum = gl.zeros([cfg.BLOCK_M], gl.float32, gl.SliceLayout(1, cfg.pv_layout))
    accumulator = gl.zeros([cfg.BLOCK_M, cfg.HEAD_SIZE], gl.float32, cfg.pv_layout)
    for tile in gl.static_range(tiles):
        k = program.k_buffer.index(tile).permute([1, 0]).load(cfg.k_layout)
        scores = program.qk(k, tile * cfg.BLOCK_N)
        probabilities, alpha, row_max = program.softmax_part0(scores, row_max)
        probabilities, row_sum, accumulator = program.softmax_part1(
            probabilities, row_sum, accumulator, alpha
        )
        v = program.v_buffer.index(tile).load(cfg.v_layout)
        accumulator = program.pv(probabilities, v, accumulator)
    program.store_output(accumulator * (1.0 / row_sum)[:, None])


@gluon.jit
def process_online(program):
    output, _, _ = _process_pipeline(program, False)
    program.store_output(output)


@gluon.jit(noinline=True)
def _retry_online(program):
    process_online(program)


@gluon.jit
def process_fixed_shift(program):
    """Retry the complete workgroup when the first-tile shift is unsafe."""
    output, row_max, row_sum = _process_pipeline(program, True)
    output_limit: gl.constexpr = (
        65504.0
        if program.output_ptr.dtype.element_ty == gl.float16
        else 3.3895313892515355e38
    )
    finite_output = _max(gl.abs(output), axis=1) <= output_limit
    bounded_shift = gl.abs(row_max * program.softmax_scale_log2) <= 64.0
    valid = finite_output & bounded_shift & (row_sum > 0.0) & (row_sum < 1.0e30)
    cfg = program.cfg
    rows = gl.arange(0, cfg.BLOCK_M, layout=gl.SliceLayout(1, cfg.pv_layout))
    query_start = gl.program_id(cfg.query_axis) * cfg.BLOCK_M
    valid_rows = query_start + rows < cfg.SEQLEN_Q
    bad = gl.sum((~valid & valid_rows).to(gl.int32), axis=0) > 0
    program.store_output(output)
    if bad:
        _retry_online(program)


@gluon.jit
def _process_pipeline(program, FIXED_SHIFT: gl.constexpr):
    cfg = program.cfg
    BLOCK_M: gl.constexpr = cfg.BLOCK_M
    BLOCK_N: gl.constexpr = cfg.BLOCK_N
    HEAD_SIZE: gl.constexpr = cfg.HEAD_SIZE
    NUM_BUFFERS: gl.constexpr = cfg.NUM_BUFFERS
    # One and two tiles have their own entry; peel exactly three here.
    loop_blocks = gl.cdiv(cfg.SEQLEN_K, BLOCK_N) - 3

    row_max = gl.full(
        [BLOCK_M],
        float("-inf"),
        dtype=gl.float32,
        layout=gl.SliceLayout(1, cfg.pv_layout),
    )
    if FIXED_SHIFT:
        row_sum = _local_probability_sums(
            gl.zeros([BLOCK_M, BLOCK_N], gl.float32, cfg.qk_layout), cfg
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
    # Every tile except the last is full in this three-or-more-tile path.
    scores = program.qk_full(k)
    probabilities, alpha, row_max = program.softmax_part0(scores, row_max)
    program.prefetch_k(2 * BLOCK_N, 0)
    program.prefetch_v(BLOCK_N, 1)
    k = program.load_k(1, 3)

    iteration = 0
    for block_start in range(0, loop_blocks * BLOCK_N, BLOCK_N):
        next_v_start = block_start + 2 * BLOCK_N
        next_k_start = block_start + 3 * BLOCK_N
        probabilities, row_sum, accumulator = _accumulate_probabilities(
            program, FIXED_SHIFT, probabilities, row_sum, accumulator, alpha
        )
        v = program.load_v(iteration % NUM_BUFFERS, 2)
        scores = program.qk_full(k)
        program.prefetch_k(next_k_start, (iteration + 1) % NUM_BUFFERS)
        accumulator = program.pv(probabilities, v, accumulator)
        probabilities, alpha, row_max = _probabilities(
            program, scores, row_max, FIXED_SHIFT
        )
        k = program.load_k(iteration % NUM_BUFFERS, 2)
        program.prefetch_v(next_v_start, iteration % NUM_BUFFERS)
        iteration += 1

    epilogue_base = (iteration - 1) * BLOCK_N
    last_start = epilogue_base + 3 * BLOCK_N

    probabilities, row_sum, accumulator = _accumulate_probabilities(
        program, FIXED_SHIFT, probabilities, row_sum, accumulator, alpha
    )
    v = program.load_v(iteration % NUM_BUFFERS, 2)
    accumulator = program.pv(probabilities, v, accumulator)

    scores = program.qk_full(k)
    probabilities, alpha, row_max = _probabilities(
        program, scores, row_max, FIXED_SHIFT
    )
    k = program.load_k(iteration % NUM_BUFFERS, 1)
    program.prefetch_v(last_start, iteration % NUM_BUFFERS)

    probabilities, row_sum, accumulator = _accumulate_probabilities(
        program, FIXED_SHIFT, probabilities, row_sum, accumulator, alpha
    )
    v = program.load_v((iteration + 1) % NUM_BUFFERS, 1)
    scores = program.qk(k, last_start)
    accumulator = program.pv(probabilities, v, accumulator)
    probabilities, alpha, row_max = _probabilities(
        program, scores, row_max, FIXED_SHIFT
    )
    probabilities, row_sum, accumulator = _accumulate_probabilities(
        program, FIXED_SHIFT, probabilities, row_sum, accumulator, alpha
    )
    v = program.load_v(iteration % NUM_BUFFERS, 0)
    accumulator = program.pv(probabilities, v, accumulator)

    if FIXED_SHIFT:
        row_sum = gl.sum(row_sum, axis=1)
        row_sum = gl.convert_layout(row_sum, gl.SliceLayout(1, cfg.pv_layout))
    output = accumulator * (1.0 / row_sum)[:, None]
    return output, row_max, row_sum
