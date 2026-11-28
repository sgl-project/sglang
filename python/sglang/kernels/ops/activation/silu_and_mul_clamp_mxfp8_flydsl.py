"""gfx950 FlyDSL clamped SwiGLU + MXFP8 quant for prefill-sized M, bit-equal to the Triton
``_silu_and_mul_clamp_kernel`` EMIT_FP8 route.

The [M, I] output is cut into 8-column chunks laid out flat across rows (I / 8 per row), one chunk
per lane per step, so no program carries a masked tail. Each lane reads 16 B of gate and 16 B of up;
a 32-column MX group is four adjacent lanes (I % 32 == 0 keeps groups inside a row and quads
aligned), so the group amax is two quad DPP swaps. Rows past M fall outside the buffer descriptors:
their loads return 0 and their stores are dropped.
"""

from __future__ import annotations

import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import llvm as _llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from sglang.kernels.ops.quantization.mxfp8_amd_gfx95 import Mxfp8Activation

THREADS = 256
COLS_PER_LANE = 8


def _f32_bits(x: float) -> int:
    return struct.unpack("<i", struct.pack("<f", x))[0]


# fp32 constants exactly as the Triton kernel rounds them
_LOG2E = _f32_bits(1.4426950408889634)
_INV_448 = _f32_bits(1.0 / 448.0)
_AMAX_FLOOR = _f32_bits(1e-10)


def _f32(bits: int):
    return fx.Int32(bits).bitcast(fx.Float32)


def _bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def _maxnum(a, b):
    return fx.Float32(_arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _minnum(a, b):
    return fx.Float32(_arith.MinNumFOp(fx.as_ir_value(a), fx.as_ir_value(b)).result)


def _dpp_max(a, ctrl):
    o = _llvm.mlir_undef(T.i32)
    b = _rocdl.update_dpp(T.i32, o, fx.as_ir_value(_bits(a)), ctrl, 0xF, 0xF, True)
    return _maxnum(a, fx.Int32(b).bitcast(fx.Float32))


def _unpack_bf16(words):
    out = []
    for w in words:
        w = fx.Int32(w)
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


def _round_bf16(a, b):
    """fp32 pair -> RNE bf16 -> fp32 pair."""
    w = fx.Int32(
        fx.rocdl.cvt_pk_bf16_f32(
            fx.as_ir_value(fx.Float32(a)), fx.as_ir_value(fx.Float32(b))
        )
    )
    return (w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)


def _cvt_pk_fp8(old, a, b, hi: bool):
    v2i16 = _ir.VectorType.get([2], _ir.IntegerType.get_signless(16))
    res = _llvm.call_intrinsic(
        v2i16,
        "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
        [
            _llvm.bitcast(v2i16, fx.as_ir_value(old)),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(fx.Float32(1.0)),
            fx.Boolean(hi).ir_value(),
        ],
        [],
        [],
    )
    return fx.Int32(_llvm.bitcast(T.i32, res))


def _build(inter: int, rows_per_lane: int):
    chunks_per_row = inter // COLS_PER_LANE
    step = THREADS * rows_per_lane
    name = f"silu_and_mul_clamp_mxfp8_i{inter}_r{rows_per_lane}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        i32_sx: fx.Int32,  # gate_up row stride, elements (even)
        arg_x: fx.Int64,  # [m, 2 * inter] bf16
        arg_q: fx.Int64,  # [m, inter] e4m3
        arg_s: fx.Int64,  # [m, inter / 32] u8
        i32_limit: fx.Int32,  # fp32 bits
    ):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        limit = i32_limit.bitcast(fx.Float32)
        xr = buffer_ops.create_buffer_resource_from_addr(
            arg_x,
            num_records_bytes=(fx.Int64(i32_m) - 1) * fx.Int64(i32_sx * 2) + 4 * inter,
        )
        qr = buffer_ops.create_buffer_resource_from_addr(
            arg_q, num_records_bytes=fx.Int64(i32_m) * inter
        )
        sr = buffer_ops.create_buffer_resource_from_addr(
            arg_s, num_records_bytes=fx.Int64(i32_m) * (inter // 32)
        )
        half_sx = i32_sx // 2
        loaded = []
        for r in range_constexpr(rows_per_lane):
            c = bid * step + r * THREADS + tx
            row = c // chunks_per_row
            col = c - row * chunks_per_row
            off = row * half_sx + col * (COLS_PER_LANE // 2)
            gw = Vec(buffer_ops.buffer_load(xr, off, vec_width=4, dtype=fx.Int32))
            uw = Vec(
                buffer_ops.buffer_load(
                    xr, off + inter // 2, vec_width=4, dtype=fx.Int32
                )
            )
            loaded.append((c, gw, uw))
        for c, gw, uw in loaded:
            g = [_minnum(v, limit) for v in _unpack_bf16([gw[i] for i in range(4)])]
            u = [
                _minnum(_maxnum(v, -limit), limit)
                for v in _unpack_bf16([uw[i] for i in range(4)])
            ]
            y = []
            for i in range_constexpr(COLS_PER_LANE // 2):
                pair = []
                for j in (2 * i, 2 * i + 1):
                    e = fx.Float32(
                        _rocdl.exp2(T.f32, fx.as_ir_value(-g[j] * _f32(_LOG2E)))
                    )
                    pair.append(g[j] / (fx.Float32(1.0) + e) * u[j])
                y += list(_round_bf16(*pair))
            amax = fx.math.absf(y[0])
            for v in y[1:]:
                amax = _maxnum(amax, fx.math.absf(v))
            amax = _dpp_max(amax, 0xB1)
            amax = _dpp_max(amax, 0x4E)
            amax = _maxnum(amax, _f32(_AMAX_FLOOR))
            # smallest power of two >= amax / 448 on the IEEE bits, as fp8_grid_quant
            raw = _bits(amax * _f32(_INV_448))
            ex = ((raw >> 23) & 0xFF) + ((raw & 0x7FFFFF) != 0).select(
                fx.Int32(1), fx.Int32(0)
            )
            ex = (ex < 1).select(fx.Int32(1), ex)
            ex = (ex > 254).select(fx.Int32(254), ex)
            # |y| <= limit^2 keeps ex well below 254, so 2^(127 - ex) is a normal fp32
            inv_scale = ((254 - ex) << 23).bitcast(fx.Float32)
            q = [
                _minnum(_maxnum(v * inv_scale, fx.Float32(-448.0)), fx.Float32(448.0))
                for v in y
            ]
            w0 = _cvt_pk_fp8(
                _cvt_pk_fp8(fx.Int32(0), q[0], q[1], False), q[2], q[3], True
            )
            w1 = _cvt_pk_fp8(
                _cvt_pk_fp8(fx.Int32(0), q[4], q[5], False), q[6], q[7], True
            )
            buffer_ops.buffer_store(Vec.from_elements([w0, w1], fx.Int32), qr, c * 2)
            buffer_ops.buffer_store(
                ex.to(fx.Int8), sr, c // 4, mask=(tx % 4) == 0, offset_is_bytes=True
            )

    @flyc.jit
    def launch(
        arg_x: fx.Int64,
        arg_q: fx.Int64,
        arg_s: fx.Int64,
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
        i32_limit: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(i32_m, i32_sx, arg_x, arg_q, arg_s, i32_limit).launch(
            grid=(fx.Int64((i32_m * chunks_per_row + step - 1) // step), 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    return launch


_KERNELS = {}


def silu_and_mul_clamp_mxfp8_supported(inter_size: int) -> bool:
    return inter_size % 32 == 0


def silu_and_mul_clamp_mxfp8_flydsl(
    gate_up: torch.Tensor, swiglu_limit: float, rows_per_lane: int = 1
) -> Mxfp8Activation:
    """gate_up bf16 [M, 2 * I] (unit column stride, even row stride) -> fp8 e4m3 [M, I] + ue8m0
    [M, I / 32] of silu(min(g, lim)) * clamp(u, -lim, lim), bit-equal to
    silu_and_mul_clamp_triton(..., emit_fp8=True)."""
    assert gate_up.dtype == torch.bfloat16 and gate_up.dim() == 2, gate_up.shape
    assert gate_up.stride(1) == 1 and gate_up.stride(0) % 2 == 0, gate_up.stride()
    M, N = gate_up.shape
    inter = N // 2
    assert silu_and_mul_clamp_mxfp8_supported(inter), inter
    q = torch.empty((M, inter), dtype=torch.float8_e4m3fn, device=gate_up.device)
    s = torch.empty((M, inter // 32), dtype=torch.uint8, device=gate_up.device)
    if M:
        args = (
            gate_up.data_ptr(),
            q.data_ptr(),
            s.data_ptr(),
            M,
            gate_up.stride(0),
            _f32_bits(float(swiglu_limit)),
            fx.Stream(torch.cuda.current_stream()),
        )
        key = (inter, rows_per_lane)
        if key not in _KERNELS:
            _KERNELS[key] = flyc.compile(_build(*key), *args)  # compiling runs it once
        else:
            _KERNELS[key](*args)
    return Mxfp8Activation(q, s)
