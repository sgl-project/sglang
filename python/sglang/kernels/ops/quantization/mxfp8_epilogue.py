"""MXFP8 quantization written as an epilogue of the kernel that produces the row.

At the speculative BS=1 shapes (<=8 rows) the standalone FlashInfer
``mxfp8_quantize`` launch costs about as much as the norm it follows, even
though it only re-reads what that norm just wrote.  These kernels do the norm
and the quantization in one pass; the quantized values come from the same BF16
rounding the standalone pair produces, so both outputs are bitwise identical to
``rmsnorm`` followed by ``mxfp8_quantize(..., is_sf_swizzled_layout=True)``.

The scale factors use FlashInfer's 128x4 swizzle:
``off = (r // 128) * ceil(G / 4) * 512 + (g // 4) * 512 + ((r % 32) * 4 + ((r // 32) % 4)) * 4 + (g % 4)``
over a row count padded to a multiple of 128, with the UE8M0 conversion
(positive rounding, subnormals included) that ``mxfp8_quantize`` uses.
"""

from __future__ import annotations

import triton
import triton.language as tl


# FlashInfer's positive-rounding UE8M0 conversion of a per-group amax, subnormals
# included: the scale byte and the multiplier that maps the group into e4m3 range.
@triton.jit
def ue8m0_scale(amax):
    normalized = amax * (1.0 / 448.0)
    bits = normalized.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    bump = (mantissa != 0) & ~((exponent == 0) & (mantissa <= 0x400000))
    sf = tl.where(normalized <= 0, 0, tl.minimum(exponent + bump.to(tl.int32), 254))
    inv = tl.where(sf == 0, 0, ((254 - sf) << 23)).to(tl.float32, bitcast=True)
    return sf, inv


@triton.jit
def mxfp8_epilogue(
    y, row, Q, S, K: tl.constexpr, BLOCK: tl.constexpr, GROUPS: tl.constexpr, g_lo, g_hi
):
    # Stores only groups in [g_lo, g_hi) so a row can be split across CTAs.
    GP: tl.constexpr = BLOCK // 32
    g = tl.arange(0, GP)
    gmask = (g < GROUPS) & (g >= g_lo) & (g < g_hi)
    e = tl.arange(0, 32)
    idx = g[:, None] * 32 + e[None, :]
    v = tl.reshape(y.to(tl.float32), (GP, 32))
    amax = tl.max(tl.abs(v), 1)
    sf, scale = ue8m0_scale(amax)
    q = tl.minimum(tl.maximum(v * scale[:, None], -448.0), 448.0).to(tl.float8e4nv)
    tl.store(Q + row * K + idx, q, gmask[:, None])
    tile = (row // 128) * (((GROUPS + 3) // 4) * 512)
    off = tile + (g // 4) * 512 + ((row % 32) * 4 + ((row // 32) % 4)) * 4 + (g % 4)
    tl.store(S + off, sf.to(tl.uint8), gmask)
