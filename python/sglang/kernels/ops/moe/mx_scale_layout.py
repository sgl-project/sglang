# SPDX-License-Identifier: Apache-2.0
"""MXFP8 (e8m0 per-32 group) scale helpers matching aiter's fused MoE sort layout."""

import triton
import triton.language as tl


@triton.jit
def mx_e8m0(x2):
    # per-32 group e8m0 scale, rounded up as aiter does: (exponent byte, scale)
    amax = tl.maximum(tl.max(tl.abs(x2), axis=1), 1e-10)
    bits = (amax * (1.0 / 448.0)).to(tl.int32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    exp = tl.where((bits & 0x7FFFFF) != 0, exp + 1, exp)
    return exp, (exp << 23).to(tl.float32, bitcast=True)


@triton.jit
def mx_scale_row(dest, SCALEN_PAD: tl.constexpr):
    # aiter mx_scale_shuffle_idx, row part
    return (dest // 32) * (SCALEN_PAD * 32) + (dest % 16) * 4 + (dest % 32) // 16


@triton.jit
def mx_scale_col(y):
    # aiter mx_scale_shuffle_idx, column (group) part
    return (y // 8) * 256 + (y % 4) * 64 + ((y % 8) // 4) * 2
