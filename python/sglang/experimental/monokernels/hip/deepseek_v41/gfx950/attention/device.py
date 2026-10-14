# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device primitives of K1's attention front, adapted from ATOM's mono decode framework."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from sglang.experimental.monokernels.hip.common.epoch import _gptr
from sglang.experimental.monokernels.hip.common.ops import bf16_pair, rsrc
from sglang.experimental.monokernels.hip.common.sync import _spin

# The execution model: one CTA a CU of an MI355X, all resident.
BLOCKS = 256
THREADS = 512
WAVES = THREADS // 64

# gfx95x cache policy bits: SC0 = 1, SC1 = 16 (device scope), NT = 2 (read once)
CM_DEV = 16

FP8_MAX = 448.0
FP8, FP4 = 0, 4  # f8f6f4 MFMA operand formats
FLT_MIN = 1.1754943508222875e-38

POLL_MAX = 12  # mailbox pairs polled a batch


# ----------------------------------------------------------------------------- memory


def ld_i32(ptr, i, cm=0):
    return fx.Int32(
        bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.i32, cache_modifier=cm)
    )


# ----------------------------------------------------------------------------- math


def tile_map(tid, width):
    """Return (tid // width, tid % width) without a division."""
    if width & (width - 1) == 0:
        sh = width.bit_length() - 1
        return tid >> sh, tid & (width - 1)
    m = -(-(1 << 20) // width)
    assert all((t * m) >> 20 == t // width for t in range(THREADS)), width
    row = (tid * m) >> 20
    return row, tid - row * width


def fma(a, b, c):
    return fx.Float32(fmath.fma(fx.Float32(a), fx.Float32(b), fx.Float32(c)))


def div_rn(x, d, r):
    """``x / d`` rounded to nearest (IEEE), given ``r = RN(1 / d)``."""
    q = x * r
    return fma(fma(-q, d, x), r, q)


# ----------------------------------------------------------------------------- packing


def ceil_exp(v):
    """Biased exponent of the smallest power of two >= v (v > 0 finite, normal)."""
    bits = v.bitcast(fx.Int32)
    return ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).select(
        fx.Int32(1), fx.Int32(0)
    )


def mx_code(amax):
    """vLLM's ROCm MXFP8 scale code: clamp(ceil(log2(max(amax, FLT_MIN) / 448)) + 127, 0, 254)."""
    a = fx.max(amax, fx.Float32(FLT_MIN))
    q = div_rn(a, fx.Float32(FP8_MAX), fx.Float32(1.0 / FP8_MAX))
    return fx.min(fx.max(ceil_exp(q), fx.Int32(0)), fx.Int32(254))


def mx_mul(code):
    """2^(127 - code) as fp32, for code in [0, 254] (the quant multiplier)."""
    return ((254 - code) << 23).bitcast(fx.Float32)


# ----------------------------------------------------------------------------- MFMA


def mfma_scaled(a, b, c, sa, sb, a_fmt=FP8, b_fmt=FP8):
    """One 16x16x128 scaled MFMA (gfx950 f8f6f4) on hand-laid operands."""
    return fx.Vector(
        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
            T.vec(4, T.f32), [a, b, c, a_fmt, b_fmt, 0, sa, 0, sb]
        )
    )


# ----------------------------------------------------------------------------- mailbox


class Mailbox:
    """Poll tagged CTA hand-offs; the caller supplies the launch tag."""

    def __init__(self, tag):
        self.tag = tag

    def put(self, base, i, v):
        bits = v.bitcast(fx.Int32) if isinstance(v, fx.Float32) else fx.Int32(v)
        bo.buffer_store(
            fx.Vector.from_elements([bits, self.tag], fx.Int32),
            rsrc(base),
            i * 2,
            cache_modifier=CM_DEV,
        )

    def put_bf(self, base, i, vs):
        """Elements i .. i + len(vs) (2 or 4, i aligned) as bf16 pairs, one store."""
        words = []
        for j in range_constexpr(len(vs) // 2):
            words += [bf16_pair(vs[2 * j], vs[2 * j + 1]).bitcast(fx.Int32), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(words, fx.Int32),
            rsrc(base),
            i,
            cache_modifier=CM_DEV,
        )

    def put_words(self, base, i, words):
        """Raw 32-bit words (1 or 2) at pairs i.. (fp8 words must come this way)."""
        vals = []
        for w in words:
            vals += [fx.Int32(w), self.tag]
        bo.buffer_store(
            fx.Vector.from_elements(vals, fx.Int32),
            rsrc(base),
            i * 2,
            cache_modifier=CM_DEV,
        )

    def poll(self, specs, batch=POLL_MAX):
        """Poll the pairs in ``specs`` until every tag is current; return the value words."""
        if const_expr(len(specs) == 0):
            return []
        if const_expr(len(specs) > batch):
            return self.poll(specs[:batch], batch) + self.poll(specs[batch:], batch)
        tag = self.tag

        def load_all():
            words = []
            for b, i, n in specs:
                w = fx.Vector(
                    bo.buffer_load(
                        rsrc(b),
                        fx.Int32(i) * 2,
                        vec_width=2 * n,
                        dtype=T.i32,
                        cache_modifier=CM_DEV,
                    )
                )
                words += [w[e] for e in range(2 * n)]
            return fx.Vector.from_elements(words, fx.Int32)

        nw = sum(2 * n for _, _, n in specs)

        def pending(v):
            bad = v[1] != tag
            for e in range_constexpr(3, nw, 2):
                bad = bad | (v[e] != tag)
            return bad

        v = _spin(load_all, pending)
        outs, e = [], 0
        for _, _, n in specs:
            outs.append([v[e + 2 * q] for q in range(n)])
            e += 2 * n
        return outs


# ----------------------------------------------------------------------------- K2


def gload(addr, words=4, nt=False):
    """``words`` i32 (1, 2 or 4) at a 64-bit global address: a cache past 4 GiB is
    reachable only this way (a buffer offset is 32 bits)."""
    i32 = ir.IntegerType.get_signless(32)
    ty = i32 if words == 1 else ir.VectorType.get([words], i32)
    v = llvm.LoadOp(ty, _gptr(addr), alignment=4 * words, nontemporal=nt or None).result
    return fx.Int32(v) if words == 1 else fx.Vector(v)


def pow2(code):
    """The fp32 power of two a biased exponent (1..254) stands for."""
    return (code << 23).bitcast(fx.Float32)


def fp8x8_bf16(w0, w1, scale):
    """Eight E4M3 of two dwords times an f32 ``scale`` -> eight bf16 as a 4-dword
    Vector (exact for a power-of-two scale in range)."""
    out = []
    for w in (w0, w1):
        for hi in (False, True):
            p = rocdl.cvt_scalef32_pk_bf16_fp8(
                T.bf16x2, as_ir_value(fx.Int32(w)), as_ir_value(fx.Float32(scale)), hi
            )
            out.append(fx.Vector(p).bitcast(fx.Int32)[0])
    return fx.Vector.from_elements(out, fx.Int32)


def stamp(c, point):
    """The attention back's stage marks; this build records no timeline."""
    return
