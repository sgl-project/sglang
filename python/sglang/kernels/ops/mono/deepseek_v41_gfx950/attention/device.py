# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Vendored from vllm-project/vllm @ 193922d6ff: vllm/models/deepseek_v41/amd/mono/attention/device.py
"""Device primitives of the DeepSeek-V4.1 attention mega kernel.

Wave reductions and shuffles, hardware math, bf16 / fp8 / MX packing, the
scaled MFMA, buffer and 64-bit global accesses, and the tagged-pair mailbox.
Adapted from ATOM's mono decode framework (``atom/mono/device``), itself ported
from the FlyDSL GLM-5 layer kernel.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import rocdl
from flydsl.expr.typing import T, as_ir_value

from sglang.kernels.ops.mono.common_hip.device.mx import FP8_MAX
from sglang.kernels.ops.mono.common_hip.device.ops import (
    THREADS,
    div_rn,
    lane_gather,
    memrealtime,
    rsrc,
    traced,
)
from sglang.kernels.ops.mono.common_hip.device.sync import _spin

# The execution model: one CTA a CU of an MI355X, all resident.

# gfx95x cache policy bits (LLVM CPol): SC0 = 1, SC1 = 16, NT = 2. SC1 alone is
# device scope (past the per-XCD L2s); NT streams data read once.

FP8, FP4 = 0, 4  # f8f6f4 MFMA operand formats
FLT_MIN = 1.1754943508222875e-38


def kernel_symbol(stem, **params):
    """Kernel symbol that profilers demangle to ``aiter::<stem>_<params>``."""
    name = f"{stem}_{kernel_signature(**params)}"
    return f"_ZN5aiter{len(name)}{name}E"


# ----------------------------------------------------------------------------- memory


def ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


def ld_bf(ptr, i):
    return fx.Float32(
        fx.BFloat16(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16))
    )


def _gptr(addr):
    return llvm.IntToPtrOp(
        ir.Type.parse("!llvm.ptr<1>"), as_ir_value(fx.Int64(addr))
    ).result


def gload(addr, words=4, nt=False):
    """``words`` i32 (1, 2 or 4) at a 64-bit global address: a cache past 4 GiB is
    reachable only this way (a buffer offset is 32 bits)."""
    i32 = ir.IntegerType.get_signless(32)
    ty = i32 if words == 1 else ir.VectorType.get([words], i32)
    v = llvm.LoadOp(ty, _gptr(addr), alignment=4 * words, nontemporal=nt or None).result
    return fx.Int32(v) if words == 1 else fx.Vector(v)


def gstore(addr, value, words=4):
    """Store ``words`` i32 (an Int32 or a Vector) at a 64-bit global address."""
    v = as_ir_value(fx.Int32(value) if words == 1 else value)
    llvm.StoreOp(v, _gptr(addr), alignment=4 * words)


# ----------------------------------------------------------------------------- math


def tile_map(tid, width):
    """(tid // width, tid % width) for 0 <= tid < THREADS without a division:
    shifts for a power of two, else a multiply-shift exact on that range."""
    if width & (width - 1) == 0:
        sh = width.bit_length() - 1
        return tid >> sh, tid & (width - 1)
    m = -(-(1 << 20) // width)
    assert all((t * m) >> 20 == t // width for t in range(THREADS)), width
    row = (tid * m) >> 20
    return row, tid - row * width


def fma(a, b, c):
    return fx.Float32(fmath.fma(fx.Float32(a), fx.Float32(b), fx.Float32(c)))


def lane_f32(v, src_lane):
    return lane_gather(v.bitcast(fx.Int32), src_lane).bitcast(fx.Float32)


# ----------------------------------------------------------------------------- packing


def fp8x8_bf16(w0, w1, scale):
    """Eight E4M3 of two dwords times an f32 ``scale`` -> eight bf16 as a
    4-dword Vector (exact for a power-of-two scale in range)."""
    out = []
    for w in (w0, w1):
        for hi in (False, True):
            p = rocdl.cvt_scalef32_pk_bf16_fp8(
                T.bf16x2, as_ir_value(fx.Int32(w)), as_ir_value(fx.Float32(scale)), hi
            )
            out.append(fx.Vector(p).bitcast(fx.Int32)[0])
    return fx.Vector.from_elements(out, fx.Int32)


def ceil_exp(v):
    """Biased exponent of the smallest power of two >= v (v > 0 finite, normal)."""
    bits = v.bitcast(fx.Int32)
    return ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).select(
        fx.Int32(1), fx.Int32(0)
    )


def mx_code(amax):
    """VLLM's ROCm MXFP8 scale code of a 32-group: clamp(ceil(log2(amax / 448))
    + 127, 0, 254), amax floored at FLT_MIN."""
    a = fx.max(amax, fx.Float32(FLT_MIN))
    q = div_rn(a, fx.Float32(FP8_MAX), fx.Float32(1.0 / FP8_MAX))
    return fx.min(fx.max(ceil_exp(q), fx.Int32(0)), fx.Int32(254))


def mx_mul(code):
    """2^(127 - code) as fp32, for code in [0, 254] (the quant multiplier)."""
    return ((254 - code) << 23).bitcast(fx.Float32)


# ----------------------------------------------------------------------------- MFMA


# ----------------------------------------------------------------------------- mailbox


def spin_until(load, pending):
    """``load()`` again (s_sleep between) while ``pending(value)``; the value."""
    return _spin(load, pending)


def barrier():
    gpu.barrier()


# ----------------------------------------------------------------------------- timeline


@traced
def stamp(c, point):
    """A timeline build's stamp: thread 0 of the CTA stores the clock at
    ``point`` (``c["tl"]`` [BLOCKS, points] i64; None: not a timeline build)."""
    if const_expr(c.get("tl") is None):
        return
    if c["tid"] == 0:
        t = memrealtime()
        w = fx.Vector.from_elements([fx.Int32(t), fx.Int32(t >> 32)], fx.Int32)
        bo.buffer_store(w, rsrc(c["tl"]), (c["bid"] * c["tl_points"] + point) * 2)


@traced
def publish_flag(c, region, i):
    """This CTA's stores visible device-wide, then the flag pair (region, i)."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if c["tid"] == 0:
        c["mb"].put(region, i, fx.Int32(1))
