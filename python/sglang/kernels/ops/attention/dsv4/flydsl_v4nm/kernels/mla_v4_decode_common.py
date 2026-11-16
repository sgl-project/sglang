# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Vendored from ROCm/aiter#5882 (https://github.com/ROCm/aiter/pull/5882),
# commit 03359c43df (03359c43df40abf91ac38a2040c9b8e284d92297),
# file aiter/ops/flydsl/kernels/mla_v4_decode_common.py.
# Unchanged except for import paths and sglang formatting (isort, ruff format).
# TODO(aiter#5882): remove once ROCm/aiter#5882 is in the aiter pin

"""Shared helpers of the FlyDSL v4 nm MLA decode (mla_v4_decode_gfx950.py): math
intrinsics, cross-lane butterflies, the global atomics / loads / fences of the
cross-split merge protocol, and the mxfp8 (ceil-e8m0) block scale."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import communication_ops_utils as cu
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl.expr import arith as _arith
from flydsl.expr.typing import T

FP8_MAX = 448.0
EX_FLOOR = 94  # smallest e8m0 block scale emitted: 2^(94 - 127)


def exp2(x):
    return fx.Float32(fx.rocdl.exp2(T.f32, fx.Float32(x).ir_value()))


def rcp(x):
    return fx.Float32(fx.rocdl.rcp(T.f32, fx.Float32(x).ir_value()))


def ldexp(x, e):
    return fx.Float32(
        _llvm_d.call_intrinsic(
            T.f32,
            "llvm.ldexp.f32.i32",
            [fx.Float32(x).ir_value(), fx.Int32(e).ir_value()],
            [],
            [],
        )
    )


def fabs_f32(x):
    return fx.Float32(
        _llvm_d.call_intrinsic(
            T.f32, "llvm.fabs.f32", [fx.Float32(x).ir_value()], [], []
        )
    )


def maxnum_f32(a, b):
    return fx.Float32(
        _llvm_d.call_intrinsic(
            T.f32,
            "llvm.maxnum.f32",
            [fx.Float32(a).ir_value(), fx.Float32(b).ir_value()],
            [],
            [],
        )
    )


def fabs_bits(x):
    return fabs_f32(x).bitcast(fx.Int32)


def udiv(a, b):
    return fx.Int32(_arith.divui(fx.Int32(a).ir_value(), fx.Int32(b).ir_value()))


def urem(a, b):
    return fx.Int32(_arith.remui(fx.Int32(a).ir_value(), fx.Int32(b).ir_value()))


def smax_i32(a, b):
    return (a > b).select(a, b)


def smin_i32(a, b):
    return (a < b).select(a, b)


def dpp_i32(x, ctrl):
    return fx.Int32(
        fx.rocdl.update_dpp(
            T.i32, fx.Int32(0).ir_value(), fx.Int32(x).ir_value(), ctrl, 0xF, 0xF, True
        )
    )


def plswap(x, width, op2):
    st = _llvm_d.StructType.get_literal([T.i32, T.i32])
    op = fx.rocdl.permlane16_swap if width == 16 else fx.rocdl.permlane32_swap
    r = op(st, fx.Int32(x).ir_value(), fx.Int32(x).ir_value(), False, False)
    a = fx.Int32(_llvm_d.extractvalue(T.i32, r, [0]))
    b = fx.Int32(_llvm_d.extractvalue(T.i32, r, [1]))
    return op2(a, b)


_DPP_XOR = {1: 0xB1, 2: 0x4E, 4: 0x141, 8: 0x128}


def butterfly_i32(x, nlanes, op2, first=1):
    k = first
    while k < nlanes * first:
        if k <= 8:
            x = op2(x, dpp_i32(x, _DPP_XOR[k]))
        else:
            x = plswap(x, k, op2)
        k *= 2
    return x


def mxfp8_exp(amb):
    # e8m0 of ceil(amax / 448) from amax's bits: 448 = 1.75 * 2^8, so a mantissa
    # above 0.75 (0x600000) rounds the exponent up
    ea = amb >> fx.Int32(23)
    ex = (
        ea
        - fx.Int32(8)
        + ((amb & fx.Int32(0x7FFFFF)) > fx.Int32(0x600000)).select(
            fx.Int32(1), fx.Int32(0)
        )
    )
    ex = (ex < fx.Int32(EX_FLOOR)).select(fx.Int32(EX_FLOOR), ex)
    return (ea >= fx.Int32(255)).select(fx.Int32(254), ex)


def fp8_scaled(y, nexp):
    v = ldexp(y, nexp).maximumf(fx.Float32(-FP8_MAX))
    return (v > fx.Float32(FP8_MAX)).select(fx.Float32(FP8_MAX), v)


def xcc_id():
    return fx.Int32(
        _llvm_d.InlineAsmOp(
            T.i32,
            [],
            "s_getreg_b32 $0, hwreg(HW_REG_XCC_ID)",
            "=s",
            has_side_effects=True,
        ).res
    )


def pin_v64(x):
    return fx.Int64(
        _llvm_d.InlineAsmOp(
            T.i64, [fx.Int64(x).ir_value()], "; pin $0", "=v,0", has_side_effects=True
        ).res
    )


def gload(addr_i64, n):
    return fx.generic_load(cu._to_ptr_global(addr_i64, fx.Float32, 4 * n), count=n)


def inv_l1():
    _llvm_d.InlineAsmOp(None, [], "buffer_inv sc0", "", has_side_effects=True)


def rmw_add64_agent(addr_i64, val):
    return fx.atomic_add(
        cu._to_ptr_global(addr_i64, fx.Int64, 8), fx.Int64(val), syncscope="agent"
    )


def ld64_agent(addr_i64, volatile=True):
    return fx.generic_load(
        cu._to_ptr_global(addr_i64, fx.Int64, 8),
        volatile=volatile,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    )


def st_raw(addr_i64, value, align):
    fx.generic_store(cu._to_ptr_global(addr_i64, alignment=align), value)


def st_agent(addr_i64, value, align):
    fx.generic_store(
        cu._to_ptr_global(addr_i64, alignment=align),
        value,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    )


def st_agent_vec(addr_i64, value, nbytes):
    # sc1 store: written through to the device coherence point (no L2 write-back
    # needed); s_nop 1 covers the store-data VGPR hazard the backend does not see
    if nbytes == 4:
        v = fx.Float32(value).bitcast(fx.Int32)
        op = "global_store_dword $0, $1, off sc1\n\ts_nop 1"
    else:
        assert nbytes == 16, nbytes
        v = fx.Vector(value).bitcast(fx.Int32)
        op = "global_store_dwordx4 $0, $1, off sc1\n\ts_nop 1"
    _llvm_d.InlineAsmOp(
        None,
        [fx.Int64(addr_i64).ir_value(), _arith.unwrap(v)],
        op,
        "v,v,~{memory}",
        has_side_effects=True,
    )


def ld_agent_f32(addr_i64, n):
    if n == 1:
        return ld_agent_i32(addr_i64, volatile=False).bitcast(fx.Float32)
    parts = [
        ld64_agent(addr_i64 + fx.Int64(8 * i), volatile=False) for i in range(n // 2)
    ]
    return fx.Vector.from_elements(parts, fx.Int64).bitcast(fx.Float32)


@flyc.jit
def spin_all_arrived(addr_i64, full, bit):
    cur = ld64_agent(addr_i64)
    while ((cur & full) != full) & ((cur & bit) != fx.Int64(0)):
        cur = ld64_agent(addr_i64)
    return cur


def ld_agent_i32(addr_i64, volatile=True):
    return fx.generic_load(
        cu._to_ptr_global(addr_i64),
        volatile=volatile,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    )


def tickets_state(v, want):
    z = fx.Int64(fx.rocdl.ballot(T.i64, (v == fx.Int32(0)).ir_value()))
    o = fx.Int64(fx.rocdl.ballot(T.i64, (v != want).ir_value()))
    return (z == fx.Int64(0)).select(fx.Int32(1), fx.Int32(0)) | (
        (o == fx.Int64(0)).select(fx.Int32(2), fx.Int32(0))
    )


@flyc.jit
def spin_tickets_state(addr_lane_i64, v, max_polls, want):
    r = tickets_state(v, want)
    k = fx.Int32(0)
    while ((r & fx.Int32(1)) == fx.Int32(0)) & (k < fx.Int32(max_polls)):
        r = tickets_state(ld_agent_i32(addr_lane_i64), want)
        k = k + fx.Int32(1)
    return r
