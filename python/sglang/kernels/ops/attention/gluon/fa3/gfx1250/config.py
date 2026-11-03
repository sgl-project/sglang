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

"""Tile layouts and sequence geometry shared by gfx1250 FA3 kernels."""

from __future__ import annotations

import triton.experimental.gluon.language as gl
from triton.experimental import gluon


@gluon.aggregate
class _AttentionConfig:
    SEQLEN_Q: gl.tensor | gl.constexpr
    SEQLEN_K: gl.tensor | gl.constexpr
    HEAD_SIZE: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr
    NUM_BUFFERS: gl.constexpr
    NUM_WARPS: gl.constexpr
    FAST_EXP: gl.constexpr
    NORM_EXP: gl.constexpr
    LOCAL_SHIFT: gl.constexpr
    FULL_TDM_WAIT: gl.constexpr
    query_axis: gl.constexpr

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
        fast_exp=False,
        norm_exp=False,
        full_tdm_wait=False,
        pipeline=False,
        local_shift=False,
    ):
        self.SEQLEN_Q = (
            seqlen_q if isinstance(seqlen_q, gl.tensor) else gl.constexpr(seqlen_q)
        )
        self.SEQLEN_K = (
            seqlen_k if isinstance(seqlen_k, gl.tensor) else gl.constexpr(seqlen_k)
        )
        self.HEAD_SIZE = gl.constexpr(head_size)
        self.BLOCK_M = gl.constexpr(block_m)
        self.BLOCK_N = gl.constexpr(block_n)
        self.NUM_BUFFERS = gl.constexpr(num_buffers)
        self.NUM_WARPS = gl.constexpr(num_warps)
        self.FAST_EXP = gl.constexpr(fast_exp)
        self.NORM_EXP = gl.constexpr(norm_exp)
        self.LOCAL_SHIFT = gl.constexpr(local_shift)
        self.FULL_TDM_WAIT = gl.constexpr(full_tdm_wait)
        # Keep nearby long-sequence query tiles on the same head to improve
        # K/V cache reuse across workgroups.
        self.query_axis = gl.constexpr(
            0 if block_m == 256 and seqlen_k >= 16_384 else 2
        )

        assert num_warps in (2, 4, 8)
        if num_warps == 2:
            assert block_m == 32
            warp_bases = [[1, 0]]
        else:
            warp_bases = [[1, 0], [2, 0]]
        if num_warps == 8:
            warp_bases.append([4, 0])

        # Alternate row fragments to shorten four-wave operand live ranges.
        register_bases = (
            [[4, 0]]
            if pipeline
            and head_size == 128
            and block_m == 128
            and block_n == 128
            and num_warps == 4
            else None
        )
        self.qk_layout = gl.constexpr(
            gl.amd.AMDWMMALayout(
                3,
                transposed=True,
                warp_bases=warp_bases,
                instr_shape=[16, 16, 32],
                reg_bases=register_bases,
            )
        )
        self.pv_layout = gl.constexpr(
            gl.amd.AMDWMMALayout(
                3,
                transposed=True,
                warp_bases=warp_bases,
                instr_shape=[16, 16, 32],
                reg_bases=register_bases,
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
