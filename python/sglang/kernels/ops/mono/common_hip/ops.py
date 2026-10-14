# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM) at 97359d6, MIT License,
# Copyright (C) 2026, Advanced Micro Devices, Inc.:
# atom/mono/device/ops.py
# atom/mono/device/ranks.py
# atom/mono/device/stamps.py
"""Device primitives of the mono kernels: wave ops, hardware math, packing and buffer access."""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.dpp_utils import update_dpp_i32
from flydsl._mlir.dialects import llvm
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from sglang.kernels.ops.mono.common_hip.plan import THREADS, WAVES

# gfx95x cache policy bits: SC1 = device scope, SC0 | SC1 = system scope (peers over XGMI)
CM_DEV = 16


CM_SYS = 17


# NT = 2: nontemporal, for weights a layer reads once
CM_NT = 2


def rsrc(addr, nbytes=None):
    return bo.create_buffer_resource_from_addr(addr, num_records_bytes=nbytes)


def permlane_swap(off, x, y):
    """v_permlane{32,16}_swap on registers (x, y); returns (x', y')."""
    swap = rocdl.permlane32_swap if off == 32 else rocdl.permlane16_swap
    pr = swap(
        llvm.StructType.get_literal([T.i32, T.i32]),
        as_ir_value(fx.Int32(x)),
        as_ir_value(fx.Int32(y)),
        False,
        False,
    )
    return tuple(fx.Int32(llvm.extractvalue(T.i32, pr, [j])) for j in range(2))


def lane_gather(v, src_lane):
    """Lane ``src_lane``'s value of v, a source lane a lane (ds_bpermute)."""
    return fx.Int32(
        rocdl.ds_bpermute(T.i32, fx.Int32(src_lane) * 4, fx.Int32(v).ir_value())
    )


def fresh(v):
    """Pass ``v`` through an opaque move, so LLVM does not hoist lane math into long-lived registers."""
    return fx.Int32(
        llvm.inline_asm(
            T.i32,
            [fx.Int32(v).ir_value()],
            "v_mov_b32 $0, $1",
            "=v,v",
            has_side_effects=True,
        )
    )


def uniform(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(v).ir_value()))


def _hw_f32(name, x):
    return fx.Float32(
        llvm.call_intrinsic(T.f32, name, [fx.Float32(x).ir_value()], [], [])
    )


def hw_rsq(x):
    """v_rsq_f32 (~1 ulp, no libm range fixups)."""
    return _hw_f32("llvm.amdgcn.rsq.f32", x)


def hw_rcp(x):
    return _hw_f32("llvm.amdgcn.rcp.f32", x)


def hw_exp2(x):
    return _hw_f32("llvm.amdgcn.exp2.f32", x)


def div_rn(x, d, r):
    """Return ``x / d`` rounded to nearest, bit-identical to IEEE division, given ``r = 1.0 / d``."""
    q = x * r
    return fx.Float32(fmath.fma(fx.Float32(fmath.fma(-q, d, x)), r, q))


def xshfl(v, off):
    """Return the value of lane ``lane ^ off`` (power-of-two offsets only)."""
    assert off in (1, 2, 4, 8, 16, 32), off
    if off >= 16:
        return v.shuffle_xor(off, 64)
    is_f = isinstance(v, fx.Float32)
    x = v.bitcast(fx.Int32) if is_f else fx.Int32(v)
    if off == 8:
        y = fx.Int32(update_dpp_i32(x, x, 0x118, 0xF, 0xC, False))
        y = fx.Int32(update_dpp_i32(y, x, 0x108, 0xF, 0x3, False))
    elif off == 4:
        y = fx.Int32(update_dpp_i32(x, x, 0x114, 0xF, 0xA, False))
        y = fx.Int32(update_dpp_i32(y, x, 0x104, 0xF, 0x5, False))
    elif off == 2:
        y = fx.Int32(update_dpp_i32(x, x, 0x4E, 0xF, 0xF, False))
    else:
        y = fx.Int32(update_dpp_i32(x, x, 0xB1, 0xF, 0xF, False))
    return y.bitcast(fx.Float32) if is_f else y


def xred(v, off, op):
    """op(v, lane ^ off) for a symmetric op; 32 / 16 take both halves of one swap."""
    if off < 16:
        return op(v, xshfl(v, off))
    is_f = isinstance(v, fx.Float32)
    x = v.bitcast(fx.Int32) if is_f else fx.Int32(v)
    a, b = permlane_swap(off, x, x)
    if is_f:
        return op(a.bitcast(fx.Float32), b.bitcast(fx.Float32))
    return op(type(v)(a), type(v)(b))


def _add(a, b):
    return a + b


def butterfly(v, offsets, op=_add):
    """Butterfly reduction over the xor offsets in the given order (the order sets the result bits)."""
    for off in offsets:
        v = xred(v, off, op)
    return v


def traced(fn):
    """Apply FlyDSL's AST rewrite to a helper, so its control flow on traced values lowers to scf."""
    return ASTRewriter.transform(fn)


LOAD_BATCH = 4  # rounds of a strided loop whose loads go out together


def first_rounds(tid, n, load, batch=LOAD_BATCH):
    """``batched_rounds``' first batch of loads, issued ahead of a wait they do not
    depend on (pass them as its ``first``)."""
    rounds = -(-n // THREADS)
    return [load(fx.min(tid + THREADS * i, n - 1)) for i in range(min(batch, rounds))]


@traced
def batched_rounds(tid, n, load, use, batch=LOAD_BATCH, first=None):
    """Strided CTA loop over v < n: ``batch`` rounds of ``load(v)`` issued before each
    ``use``. ``first``: the first batch's loads, already issued (``first_rounds``)."""
    rounds = -(-n // THREADS)
    assert first is None or len(first) == min(batch, rounds), (len(first), n)
    for b0 in range_constexpr(0, rounds, batch):
        idx = range(b0, min(b0 + batch, rounds))
        if const_expr(b0 == 0 and first is not None):
            got = first
        else:
            got = [load(fx.min(tid + THREADS * i, n - 1)) for i in idx]
        for i, g in zip(idx, got):
            v = tid + THREADS * i
            if v < n:
                use(v, g)


def mfma_bf16(a, b, c):
    return fx.Vector(rocdl.mfma_f32_16x16x32_bf16(T.vec(4, T.f32), [a, b, c]))


def row_sum(red, r, t, waves=None):
    """Return row r, token t of the waves' MFMA accumulators, summed in wave order."""
    tot = fx.Float32(0.0)
    for w in waves if waves is not None else range(WAVES):
        tot = tot + fx.ptr_load(red + ((w * 64 + 16 * (r // 4) + t) * 4 + r % 4))
    return tot


def ballot(pred):
    """The wave's lanes where ``pred`` holds, a bit a lane."""
    return fx.Int64(rocdl.ballot(T.i64, as_ir_value(pred)))


def popcount(x):
    return fx.Int32(fx.ctpop(x))


def lanes_below(bal, lane):
    """How many of the lanes in ballot ``bal`` are below ``lane``."""
    return popcount(bal & ((fx.Int64(1) << fx.Int64(lane)) - 1))


def load_ptr64(table, i):
    """Load entry ``i`` of an i64 table, wave-uniform."""
    pw = fx.Vector(bo.buffer_load(rsrc(table), i * 2, vec_width=2, dtype=T.i32))
    return (fx.Int64(uniform(pw[1])) << 32) | fx.Int64(fx.Uint32(uniform(pw[0])))


def bf16_round(a):
    return fx.Float32(fx.Float32(a).to(fx.BFloat16))


def bf16_pair(a, b):
    """Two f32 -> one f32-typed word holding (bf16(a), bf16(b))."""
    return (
        fx.Vector.from_elements([a, b], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Float32)[0]
    )


def bf_lo(w):
    """The f32 of a packed bf16 pair word's low half."""
    return (w << 16).bitcast(fx.Float32)


def bf_hi(w):
    """The f32 of a packed bf16 pair word's high half."""
    return (w & fx.Int32(-65536)).bitcast(fx.Float32)


def bf2_f32(w):
    """Packed bf16 pair word -> (f32 low, f32 high)."""
    return bf_lo(w), bf_hi(w)


def fp8_pack4(a, b, c, d):
    """Four f32 (already in range) -> one dword of four E4M3 bytes, RNE."""
    lo = fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, a, b, fx.Int32(0), False))
    return fx.Int32(rocdl.cvt_pk_fp8_f32(T.i32, c, d, lo, True))


# ---------------------------------------------------------------- TP ranks
def peer_bases(peers, tp):
    """Load every rank's peer buffer base from the ``peers`` table once."""
    return [load_ptr64(peers, p) for p in range(tp)]


def sum_partials(poll, own, pair_of, tp):
    """Poll the ``tp`` ranks' bf16 pair partials and sum them in fp32 in rank order."""
    ws = poll([(own, pair_of(src), 1) for src in range(tp)])
    acc0, acc1 = bf2_f32(ws[0][0])
    for src in range_constexpr(1, tp):
        lo, hi = bf2_f32(ws[src][0])
        acc0 = acc0 + lo
        acc1 = acc1 + hi
    return acc0, acc1
