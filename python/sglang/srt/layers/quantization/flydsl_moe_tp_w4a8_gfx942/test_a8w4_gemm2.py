# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Standalone A8W4 MoE stage2 (down-proj GEMM + routed atomic scatter) FlyDSL kernel
for gfx942.

Extracted from the a16w4 port (``moe_2stage_a16wmix/gemm2.py``, ``a_dtype="fp8r"``)
with only the configuration the tuned A8W4 build actually runs
(``flydsl_moe2_afp8r_wfp4_bf16_t{BM}x{TN}x{TK}_atomic[_bnt*][_xcd*][_ftz]``):

  * A  : FP8-FNUZ row-quantized stage1 intermediate ``[sorted_size, inter_dim]``
         (by SORTED position) + f32 per-sorted-row scale
  * W2 : mxfp4 (E2M1 + per-1x32 E8M0) ``[E, model_dim, inter_dim/2]``, standard
         N-major preshuffle
  * per-output-channel max E8M0 exponent derived in-kernel from the W2 scales
  * MFMA(16,16,32,fp8) with fp4 -> fp8 decode through a per-32K-group byte LUT
    (+ a sign-mask LUT, so FNUZ -0 never needs a per-fragment fix-up)
  * K (= inter_dim) loop unrolled, XOR-swizzled A-LDS, 4 waves split TILE_N
  * ``dyn_rows``: per-WG runtime skip of all-padding 16-row sub-blocks
  * epilogue: acc * a_scale[row] * 2**(exp-132) * route_weight -> bf16 LDS tile
    (half the port's f32 staging) -> packed bf16x2 global atomic add into
    ``[tokens, model_dim]``

Run:
    pytest op_tests/test_a8w4_gemm2.py -q
    python op_tests/test_a8w4_gemm2.py -t 512 -s 4096,256,256,8
"""

import argparse
import functools
import itertools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import pandas as pd
import pytest
import torch
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

import aiter
from aiter import QuantType, dtypes
from aiter.fused_moe import fused_topk, moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix import flydsl_a16w4_gemm2
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.a8w4 import quantize_rows
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.utils import (
    _buffer_i32_scalar_read,
    _global_i32_at,
    _udiv,
    _umod,
)
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import (
    _gep1,
    _global_base_ptr1,
    _global_i32_buffer_tiles,
    _global_i32_buffer_view,
    global_typed_ptr,
    lds_typed_ptr,
    lds_vec_load,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility import fp4_utils

SUPPORTED_GFX = ["gfx942"]  # FP8-FNUZ MFMA + global bf16x2 atomics
# LDS epilogue: drain the sorted-id loads before the first atomic (see
# _atomic_bf16_epilog); tagged into the kernel name ("_dr") so the JIT key changes
EPI_DRAIN = os.environ.get("EPI_DRAIN", "1") == "1"
# LDS epilogue bf16 rounding: RNE by integer add, no NaN canonicalization (2 VALU
# per value vs 5 + s_nop); partials are finite. Tagged "_bf".
BF16_FAST = os.environ.get("BF16_FAST", "0") == "1"
# fp4 -> fp8 decode without the even/odd interleave v_perms ("_wp", 2 of 6 perms
# per dword). Correct with the STANDARD preshuffled weight: the single-hot probe
# (experiments/fused_ar_opt_ab/wperm_probe.py) pairs A[k] with W[k] either way; the reason
# was not pinned down. No measurable speedup (doc sec 11), so off by default.
WPERM = os.environ.get("WPERM", "0") == "1"
# persistent-lite experiment: launch PERSIST * #CU compute CTAs that loop over the
# tiles (grid stride) instead of one CTA per tile; 0 = off. Tagged "_p{k}".
PERSIST = int(os.environ.get("PERSIST", "0"))
TOL_ERR_RATIO = 0.05
ROUTINGS = ["uniform", "hot"]
# "hot" routing: 1/32 of the experts get this score bias, so they land in most
# tokens' top-k (a few experts own a large share of the routed rows).
HOT_EXPERT_DIV = 32
HOT_EXPERT_BIAS = 2.0


# =============================================================================
# Device helpers: fp4 -> fp8 decode, channel factor, MFMA (same as test_a8w4_gemm1)
# =============================================================================
#
# W stays E2M1/E8M0. Output channel n is normalized by 2**(max_exp[n]-127-5)
# (``_channel_factor``, applied in the epilogue); per 32-K group the remaining
# 2**(5-delta), delta = max_exp - group_exp, is folded into an 8-entry
# E2M1-magnitude -> FP8-FNUZ byte table (``_fp8_lut``) so each nibble decodes with
# one v_perm.


def _perm_u32(hi, lo, sel):
    return fx.Uint32(
        rocdl.perm_b32(
            fx.Uint32(hi).ir_value(), fx.Uint32(lo).ir_value(), fx.Uint32(sel).ir_value()
        )
    )


def _fp8_lut(group_code, max_code, flush_tiny):
    """8-entry ``|E2M1| * 2**(5-delta)`` -> FP8-FNUZ byte table.

    Returns ``(lo, hi, sign_lo, sign_hi)``: the magnitude table plus a same-shaped
    mask holding 0x80 for every non-zero entry (FNUZ has no -0; 0x80 is NaN).
    """
    difference = max_code - group_code
    delta = fx.min(difference, 11).to(fx.Uint32) * fx.Uint32(0x08080808)
    lo = (fx.Uint32(0x6C686080) - delta) & fx.Uint32(0xFFFFFF00)
    hi = fx.Uint32(0x7C787470) - delta
    if const_expr(flush_tiny):
        # Approximation: drop groups < 2**-11 of this channel's largest scale.
        lo = (difference > 11).select(fx.Uint32(0), lo)
        hi = (difference > 11).select(fx.Uint32(0), hi)
    else:
        for diff, low, high in (
            (12, 0x0C080400, 0x1C181410),
            (13, 0x06040200, 0x14100C08),
            (14, 0x03020100, 0x0C080604),
            (15, 0x02010000, 0x06040302),
            (16, 0x01000000, 0x03020201),
            (17, 0, 0x02010100),
            (18, 0, 0x01000000),
        ):
            lo = (difference == diff).select(fx.Uint32(low), lo)
            hi = (difference == diff).select(fx.Uint32(high), hi)
        lo = (difference >= 19).select(fx.Uint32(0), lo)
        hi = (difference >= 19).select(fx.Uint32(0), hi)
    # Entries are <= 0x7C, so +0x7F sets bit 7 exactly for non-zero bytes (no carry).
    sign_lo = (lo + fx.Uint32(0x7F7F7F7F)) & fx.Uint32(0x80808080)
    sign_hi = (hi + fx.Uint32(0x7F7F7F7F)) & fx.Uint32(0x80808080)
    return lo, hi, sign_lo, sign_hi


def _fp4x8_to_fp8x8(raw, lut):
    """Eight consecutive FP4 nibbles (one i32) -> v8 FP8-FNUZ MFMA operand."""
    raw = fx.Uint32(raw)
    lo, hi, sign_lo, sign_hi = lut
    sel_even = raw & 0x07070707  # magnitudes of nibbles 0,2,4,6
    sel_odd = (raw >> 4) & 0x07070707  # magnitudes of nibbles 1,3,5,7
    # byte = |value| | (nibble sign & 0x80-if-non-zero)
    even = (_perm_u32(sign_hi, sign_lo, sel_even) & (raw << 4)) | _perm_u32(
        hi, lo, sel_even
    )
    odd = (_perm_u32(sign_hi, sign_lo, sel_odd) & raw) | _perm_u32(hi, lo, sel_odd)
    if WPERM:  # even = k0..k3, odd = k4..k7 already
        lo32, hi32 = even.bitcast(fx.Int32), odd.bitcast(fx.Int32)
    else:
        lo32 = _perm_u32(odd, even, 0x05010400).bitcast(fx.Int32)
        hi32 = _perm_u32(odd, even, 0x07030602).bitcast(fx.Int32)
    return fx.Vector.from_elements([lo32, hi32], fx.Int32).bitcast(fx.Float8E4M3FNUZ)


# ---- fast decode (``fast_dec``, same scheme as test_a8w4_gemm1.py) ---------------
# W pre-transformed by ``prep_w2_fast``: byte j of a dword = (k_j, k_{j+4}) and no -0,
# so even/odd nibbles decode straight to K 0..3 / 4..7 and the sign is a bit move.
# fast_dec=2 also stores min(max-code, 11)*8 per group in the scale bytes (flushed
# groups zeroed in W) and the per-channel max codes (int32) right after the scales.


def _fp8_lut_fast(group_code, max_code):
    difference = max_code - group_code
    delta = fx.min(difference, 11).to(fx.Uint32) * fx.Uint32(0x08080808)
    lo = (fx.Uint32(0x6C686080) - delta) & fx.Uint32(0xFFFFFF00)
    hi = fx.Uint32(0x7C787470) - delta
    flushed = difference > 11
    lo = flushed.select(fx.Uint32(0), lo)
    hi = flushed.select(fx.Uint32(0), hi)
    smask = flushed.select(fx.Uint32(0), fx.Uint32(0x80808080))
    return lo, hi, smask


def _fp8_lut_predelta(delta):
    delta = fx.Uint32(delta)
    lo = (fx.Uint32(0x6C686080) - delta) & fx.Uint32(0xFFFFFF00)
    hi = fx.Uint32(0x7C787470) - delta
    return lo, hi, fx.Uint32(0x80808080)


def _fp4x8_to_fp8x8_fast(raw, lut):
    lo, hi, smask = lut
    raw = fx.Uint32(raw)
    even = ((raw << 4) & smask) | _perm_u32(hi, lo, raw & 0x07070707)
    odd = (raw & smask) | _perm_u32(hi, lo, (raw >> 4) & 0x07070707)
    return fx.Vector.from_elements(
        [even.bitcast(fx.Int32), odd.bitcast(fx.Int32)], fx.Int32
    ).bitcast(fx.Float8E4M3FNUZ)


def _channel_factor(exponent):
    """2**(exponent-127-5) as f32 (subnormal-safe)."""
    normal = ((exponent - 5).to(fx.Uint32) << 23).bitcast(fx.Float32)
    subnormal = (fx.Uint32(1) << (exponent + 17).to(fx.Uint32)).bitcast(fx.Float32)
    return (exponent >= 6).select(normal, subnormal)


def _fp8_frag8(v8):
    t = fx.make_rmem_tensor(fx.make_layout(8, 1), fx.Float8E4M3FNUZ)
    t.store(v8)
    return t


def _mma_fp8(mma_atom, acc, a8, b8):
    """acc += A(16x32) * B(32x16), one MFMA(16,16,32,fp8 FNUZ)."""
    fx.gemm(mma_atom, acc, _fp8_frag8(a8), _fp8_frag8(b8), acc)


# =============================================================================
# Epilogue: routing-weighted bf16x2 global atomic scatter (gfx942)
# =============================================================================


def _flat_buffer(arg, elem_ty, align):
    ptr = global_typed_ptr(arg, elem_ty, align=align)
    view = fx.Tensor(fx.make_view(ptr, fx.make_layout((1, 1), (1, 1))))
    return fx.rocdl.make_buffer_tensor(view, max_size=True)


def _atomic_add_bf16x2(out_base, elem_off, lo, hi):
    """``out[elem_off : elem_off+2] += (lo, hi)`` as one global_atomic_pk_add_bf16."""
    pk = Vec.from_elements([lo, hi], fx.Float32).to(fx.BFloat16)
    llvm.AtomicRMWOp(
        llvm.AtomicBinOp.fadd,
        _gep1(out_base, elem_off * fx.Int32(2)),
        _raw(pk),
        llvm.AtomicOrdering.monotonic,
        syncscope="agent",
        alignment=4,
    )


# @flyc.jit is LOAD-BEARING (both epilogues): it AST-rewrites ``if token < i32_M``
# into an scf.if. As a plain function the guard would run as a Python if (dropped at
# trace), so the atomic adds would fire on padding rows.
@flyc.jit
def _direct_atomic_epilog(
    accm, arg_out, arg_stids, arg_sweights, m_row, col0, lane, i32_M, N_OUT
):
    """LDS-free scatter straight from the MFMA accumulator layout.

    MFMA 16x16 acc: lane L holds column L%16 of rows (L//16)*4 + v, v = 0..3. Lane
    pair (L, L^1) owns two adjacent columns, so one ``shuffle_xor(1)`` per value
    lets the even lane emit rows v=0,1 and the odd lane rows v=2,3 as bf16x2
    (even col, odd col) atomics: all lanes busy, no LDS staging, no barrier.
    ``accm[mi][ni]`` is already dequantized; ``col0`` is this lane's column for
    ni = 0 (adjacent N blocks are 16 columns apart).
    """
    lane_div_16 = lane // fx.Int32(16)
    odd = lane & fx.Int32(1)
    is_odd = odd == fx.Int32(1)
    stids = _flat_buffer(arg_stids, T.i32, 4)
    sweights = _flat_buffer(arg_sweights, T.f32, 4)
    load_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    load_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    out_base = _global_base_ptr1(arg_out)
    col_even = col0 - odd

    def load_scalar(atom, src, index, elem_ty):
        frag = fx.make_rmem_tensor(1, elem_ty)
        fx.copy(atom, src[None, index], frag)
        return Vec(frag.load())[0]

    for mi in range_constexpr(len(accm)):
        # This lane's two output rows of the 16-row block: even lane v=0,1, odd v=2,3.
        rows = [
            m_row + fx.Int32(mi * 16 + k) + lane_div_16 * fx.Int32(4) + odd * fx.Int32(2)
            for k in range_constexpr(2)
        ]
        toks = [
            load_scalar(load_i32, stids, r, fx.Int32) & fx.Int32(0x00FFFFFF) for r in rows
        ]
        wts = [load_scalar(load_f32, sweights, r, fx.Float32) for r in rows]
        for ni in range_constexpr(len(accm[mi])):
            v = Vec(accm[mi][ni])
            pairs = []
            for k in range_constexpr(2):
                mine = is_odd.select(v[2 + k], v[k])
                # partner needs my value of its rows: even partner wants v[k] (rows
                # 0,1), odd partner wants v[2+k] (rows 2,3).
                theirs = is_odd.select(v[k], v[2 + k]).shuffle_xor(
                    fx.Int32(1), fx.Int32(64)
                )
                lo = is_odd.select(theirs, mine)  # even column
                hi = is_odd.select(mine, theirs)  # odd column
                pairs.append((lo, hi))
            for k in range_constexpr(2):
                if toks[k] < i32_M:  # padding rows carry a sentinel token id
                    _atomic_add_bf16x2(
                        out_base,
                        toks[k] * fx.Int32(N_OUT) + col_even + fx.Int32(ni * 16),
                        pairs[k][0] * wts[k],
                        pairs[k][1] * wts[k],
                    )


@flyc.jit
def _atomic_bf16_epilog(
    lds_base_i32, accm, arg_out, arg_stids, arg_sweights, m_row, n_block, wave, lane,
    i32_M, BM, N_OUT, BN,
):
    """Coalesced scatter through LDS: apply the routing weight in the MFMA layout,
    stage the tile as bf16 [BM, BN] (same bf16 rounding point as the port's f32
    staging, half the LDS), re-read it as (8 rows x 32 lanes x 2 cols) and
    ``global_atomic_pk_add_bf16`` into ``out[token, n]`` (128 B per row per op)."""
    # Row count comes from the accumulator list, not BM: a nested @flyc.jit call
    # does not re-specialize on Python int arguments (dyn_rows calls this with
    # several row variants in one kernel).
    M_CHUNKS = len(accm)
    M_REPS = M_CHUNKS * 2  # 256 threads = 8 rows x 32 lanes per pass
    n_per_wave = BN // 4
    num_acc_n = n_per_wave // 16
    s_count = BN // 64  # each pass covers 64 cols (32 lanes x 2)
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_bf16 = lds_typed_ptr(lds_base_i32, T.bf16)

    tx = fx.Int32(gpu.thread_id("x"))
    m_lane = tx // fx.Int32(32)
    col_start = (tx % fx.Int32(32)) * fx.Int32(2)

    stids = _flat_buffer(arg_stids, T.i32, 4)
    sweights = _flat_buffer(arg_sweights, T.f32, 4)
    load_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    load_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    # gfx942 has no buffer_atomic_pk_add_bf16 -> global packed atomic.
    out_base = _global_base_ptr1(arg_out)

    def load_scalar(atom, src, index, elem_ty):
        frag = fx.make_rmem_tensor(1, elem_ty)
        fx.copy(atom, src[None, index], frag)
        return Vec(frag.load())[0]

    packed = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + fx.Int32(mr * 8) + m_lane
        packed.append(load_scalar(load_i32, stids, sorted_pos, fx.Int32))

    # MFMA acc layout (lane: col L%16, rows (L//16)*4 + v) -> weighted bf16 in LDS.
    for i in range_constexpr(M_CHUNKS):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        w = [
            load_scalar(load_f32, sweights, m_row + row_base + fx.Int32(v), fx.Float32)
            for v in range_constexpr(4)
        ]
        for J in range_constexpr(num_acc_n):
            col = wave * fx.Int32(n_per_wave) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            prod = Vec.from_elements(
                [vec[v] * w[v] for v in range_constexpr(4)], fx.Float32
            )
            if const_expr(BF16_FAST):
                # RNE: high half of bits + 0x7FFF + lsb(bits >> 16) (finite inputs)
                u = Vec(prod.bitcast(fx.Int32))
                r = [
                    fx.Int32(u[v]) + fx.Int32(0x7FFF)
                    + ((fx.Int32(u[v]) >> fx.Int32(16)) & fx.Int32(1))
                    for v in range_constexpr(4)
                ]
                hi = Vec(Vec.from_elements(r, fx.Int32).bitcast(fx.BFloat16))
                bf4 = [hi[2 * v + 1] for v in range_constexpr(4)]
            else:
                bf4 = prod.to(fx.BFloat16)
            for v in range_constexpr(4):
                lds_bf16[(row_base + fx.Int32(v)) * fx.Int32(BN) + col] = bf4[v]

    gpu.barrier()
    if const_expr(EPI_DRAIN):
        # Drain the (long-landed) sorted-id loads before the first atomic: gfx9 has
        # one in-order vmcnt for loads and stores, so a token id consumed after
        # atomics were issued makes LLVM wait for those atomics (ATT: ~0.57 us/wave
        # of mid-epilogue vmcnt stalls serializing the atomic groups).
        rocdl.s_waitcnt(vmcnt=0)

    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:  # padding rows carry a sentinel token id
            row_base_addr = (
                token_id * fx.Int32(N_OUT) + n_block * fx.Int32(BN) + col_start
            )
            for s in range_constexpr(s_count):
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                pk = lds_vec_load(
                    lds_base_i32,
                    idx0 * fx.Int32(2),
                    Vec.make_type(2, fx.BFloat16),
                    fx.BFloat16,
                    align=4,
                )
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    _gep1(out_base, (row_base_addr + fx.Int32(s * 64)) * fx.Int32(2)),
                    _raw(pk),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )


def _stage_bf16_tile(lds_base_i32, accm, wave, lane, BN):
    """NTILE epilogue, part 1: dequantized + route-weighted f32 acc (MFMA layout:
    lane L, value v of block i = row i*16 + (L//16)*4 + v, column L%16 of the
    wave's 16-col block) -> bf16 [rows, BN] tile in LDS."""
    n_per_wave = BN // 4
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_bf16 = lds_typed_ptr(lds_base_i32, T.bf16)
    for i in range_constexpr(len(accm)):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(len(accm[i])):
            col = wave * fx.Int32(n_per_wave) + fx.Int32(J * 16) + lane_mod_16
            if const_expr(BF16_FAST):
                # RNE by integer add (finite inputs), 2 VALU per value
                u = Vec(Vec(accm[i][J]).bitcast(fx.Int32))
                r = [
                    fx.Int32(u[v]) + fx.Int32(0x7FFF)
                    + ((fx.Int32(u[v]) >> fx.Int32(16)) & fx.Int32(1))
                    for v in range_constexpr(4)
                ]
                hi = Vec(Vec.from_elements(r, fx.Int32).bitcast(fx.BFloat16))
                bf4 = [hi[2 * v + 1] for v in range_constexpr(4)]
            else:
                bf4 = Vec(accm[i][J]).to(fx.BFloat16)
            for v in range_constexpr(4):
                lds_bf16[(row_base + fx.Int32(v)) * fx.Int32(BN) + col] = bf4[v]


@flyc.jit
def _scatter_bf16_tile(lds_base_i32, ids, arg_out, n_block, i32_M, N_OUT, BN):
    """NTILE epilogue, part 2 (after a barrier): re-read the staged tile as
    (8 rows x 32 lanes x 2 cols) passes and bf16x2-atomic-add 128 B per row into
    ``out[token, n]``. ``ids[mr]``: sorted id of row mr*8 + tx//32 (preloaded)."""
    tx = fx.Int32(gpu.thread_id("x"))
    m_lane = tx // fx.Int32(32)
    col_start = (tx % fx.Int32(32)) * fx.Int32(2)
    out_base = _global_base_ptr1(arg_out)
    for mr in range_constexpr(len(ids)):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = fx.Int32(ids[mr]) & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:  # padding rows carry a sentinel token id
            row_base_addr = token_id * fx.Int32(N_OUT) + n_block * fx.Int32(BN) + col_start
            for s in range_constexpr(BN // 64):
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                pk = lds_vec_load(
                    lds_base_i32,
                    idx0 * fx.Int32(2),
                    Vec.make_type(2, fx.BFloat16),
                    fx.BFloat16,
                    align=4,
                )
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    _gep1(out_base, (row_base_addr + fx.Int32(s * 64)) * fx.Int32(2)),
                    _raw(pk),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )


@flyc.jit
def _direct_scatter_tile(accm, ids, arg_out, col0, lane, i32_M, N_OUT):
    """NTILE ``direct`` epilogue: ``_direct_atomic_epilog`` on an already weighted
    acc with preloaded sorted ids (``ids[mi][k]``: this lane's rows k of block mi)."""
    odd = lane & fx.Int32(1)
    is_odd = odd == fx.Int32(1)
    out_base = _global_base_ptr1(arg_out)
    col_even = col0 - odd
    for mi in range_constexpr(len(accm)):
        toks = [fx.Int32(ids[mi][k]) & fx.Int32(0x00FFFFFF) for k in range(2)]
        for ni in range_constexpr(len(accm[mi])):
            v = Vec(accm[mi][ni])
            pairs = []
            for k in range_constexpr(2):
                mine = is_odd.select(v[2 + k], v[k])
                theirs = is_odd.select(v[k], v[2 + k]).shuffle_xor(
                    fx.Int32(1), fx.Int32(64)
                )
                pairs.append((is_odd.select(theirs, mine), is_odd.select(mine, theirs)))
            for k in range_constexpr(2):
                if toks[k] < i32_M:
                    _atomic_add_bf16x2(
                        out_base,
                        toks[k] * fx.Int32(N_OUT) + col_even + fx.Int32(ni * 16),
                        pairs[k][0],
                        pairs[k][1],
                    )


@flyc.jit
def _direct64_scatter_tile(accm, ids, arg_out, col_w, lane, i32_M, N_OUT):
    """LDS-free scatter with 64 B row segments (NTILE ``direct64`` epilogue).

    ``accm[mi][ni]``: the wave's two 16-col blocks (ni = 0, 1; already weighted),
    MFMA layout lane L = col L%16 of rows (L//16)*4 + v. Per row v one
    shuffle_xor(1): the even lane of a pair keeps block 0 (cols c, c+1), the odd
    lane block 1 (cols c-1, c), so the 16 lanes of a row group cover the wave's
    32 contiguous columns -> 16 x bf16x2 = 64 B per row per atomic instruction
    (the 32 B segments of ``direct`` halve the atomic throughput, >= 64 B is full
    rate: experiments/atomic_bw). ``ids[mi][v]``: sorted id of row mi*16 +
    (L//16)*4 + v; ``col_w``: first column of the wave."""
    odd = lane & fx.Int32(1)
    is_odd = odd == fx.Int32(1)
    col = col_w + odd * fx.Int32(16) + (lane % fx.Int32(16)) - odd
    out_base = _global_base_ptr1(arg_out)
    for mi in range_constexpr(len(accm)):
        v0 = Vec(accm[mi][0])
        v1 = Vec(accm[mi][1])
        for k in range_constexpr(4):
            recv = is_odd.select(v0[k], v1[k]).shuffle_xor(fx.Int32(1), fx.Int32(64))
            lo = is_odd.select(recv, v0[k])
            hi = is_odd.select(v1[k], recv)
            tok = fx.Int32(ids[mi][k]) & fx.Int32(0x00FFFFFF)
            if tok < i32_M:
                _atomic_add_bf16x2(out_base, tok * fx.Int32(N_OUT) + col, lo, hi)


@flyc.jit
def _route_bf16_epilog(
    lds_base_i32, accm, arg_out, arg_stids, arg_sweights, m_row, n_block, wave, lane,
    i32_M, i32_topk, BM, N_OUT, BN,
):
    """Route-row variant of ``_atomic_bf16_epilog`` (fused all-reduce experiment,
    topk-sum in the comm phase): same weighted bf16 LDS staging, but each routed
    row is STORED (no atomics) to its own row ``out[token * topk + slot, n]``,
    write-through (sc0|sc1) so comm WGs on other XCDs read it past their L2.
    ``out`` is [tokens * topk, N_OUT]; no zeroing needed (every row is written)."""
    M_CHUNKS = len(accm)
    M_REPS = M_CHUNKS * 2  # 256 threads = 8 rows x 32 lanes per pass
    n_per_wave = BN // 4
    num_acc_n = n_per_wave // 16
    s_count = BN // 64  # each pass covers 64 cols (32 lanes x 2)
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_bf16 = lds_typed_ptr(lds_base_i32, T.bf16)

    tx = fx.Int32(gpu.thread_id("x"))
    m_lane = tx // fx.Int32(32)
    col_start = (tx % fx.Int32(32)) * fx.Int32(2)

    stids = _flat_buffer(arg_stids, T.i32, 4)
    sweights = _flat_buffer(arg_sweights, T.f32, 4)
    load_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    load_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
    route = buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(arg_out)),
        num_records_bytes=_raw(fx.Int64(i32_M * i32_topk) * fx.Int64(N_OUT * 2)),
    )

    def load_scalar(atom, src, index, elem_ty):
        frag = fx.make_rmem_tensor(1, elem_ty)
        fx.copy(atom, src[None, index], frag)
        return Vec(frag.load())[0]

    packed = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + fx.Int32(mr * 8) + m_lane
        packed.append(load_scalar(load_i32, stids, sorted_pos, fx.Int32))

    for i in range_constexpr(M_CHUNKS):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        w = [
            load_scalar(load_f32, sweights, m_row + row_base + fx.Int32(v), fx.Float32)
            for v in range_constexpr(4)
        ]
        for J in range_constexpr(num_acc_n):
            col = wave * fx.Int32(n_per_wave) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            bf4 = Vec.from_elements(
                [vec[v] * w[v] for v in range_constexpr(4)], fx.Float32
            ).to(fx.BFloat16)
            for v in range_constexpr(4):
                lds_bf16[(row_base + fx.Int32(v)) * fx.Int32(BN) + col] = bf4[v]

    gpu.barrier()

    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:  # padding rows carry a sentinel token id
            route_row = token_id * i32_topk + packed[mr].shrui(fx.Int32(24))
            row_base_addr = route_row * fx.Int32(N_OUT) + n_block * fx.Int32(BN) + col_start
            for s in range_constexpr(s_count):
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                pk = lds_vec_load(
                    lds_base_i32,
                    idx0 * fx.Int32(2),
                    Vec.make_type(2, fx.BFloat16),
                    fx.BFloat16,
                    align=4,
                )
                buffer_ops.buffer_store(
                    fx.Int32(Vec(pk).bitcast(fx.Int32)[0]),  # scalar: v1i32 store crashes isel
                    route,
                    (row_base_addr + fx.Int32(s * 64)) // fx.Int32(2),
                    cache_modifier=17,
                )


# =============================================================================
# Kernel
# =============================================================================


@functools.cache
def compile_a8w4_gemm2(
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    BM=64,
    TILE_N=128,
    TILE_K=256,
    b_cache_mod=2,
    xcd_swizzle=0,
    waves_per_eu=None,
    flush_tiny=True,
    epilog="lds",
    dyn_rows=False,
    A_GROUP=0,
    TILE_ORDER="m",
    PROBE=False,
    CHUNK_N=512,
    AR=None,
    AR_SEP=False,
    fast_dec=0,
    NTILE=0,
    STAGE_ROWS=0,
    ROW_VGPR=False,
):
    """Build the A8W4 stage2 launcher (N = model_dim, K = inter_dim).

    ``NTILE`` (0 = the original one-tile-per-CTA body): each CTA owns one m-block
    and NTILE adjacent n-blocks (grid / NTILE). A is staged to LDS once and reused;
    the row-static data (ratio s0/s1 of the two A-scale groups, s1 * route weight,
    sorted id) is loaded once and kept in LDS (+12 B/row), so one accumulator set
    is rescaled at the group boundary instead of two; all A loads are issued
    before any LDS write; W / scales of n-tile t+1 are prefetched before tile t
    computes, so they land under tile t's MFMA + epilogue. NTILE=1 keeps the grid
    but gets the load hoisting. ``lds`` epilogue with NTILE>1 needs its own staging
    region (LDS = A + staging); ``direct`` keeps LDS = A tile. ``STAGE_ROWS``
    (NTILE>1, lds; 0 = BM): stage + scatter STAGE_ROWS rows at a time through a
    smaller staging slot (16 -> LDS = A + 4 KB at tile_n 128; 2 barriers per chunk).

    ``TILE_ORDER`` (experiment, fused all-reduce idea): ``"m"`` (default) = m-block
    major tiles (all N tiles of an m-block adjacent); ``"n"`` = N-major (the grid
    sweeps output columns left to right, so a column chunk is final early), with
    the per-XCD contiguous remap; ``"nx"`` = N-major in plain dispatch order;
    ``"g"`` = column groups of ``CHUNK_N`` swept in order, each group m-major with
    the per-XCD contiguous remap inside the group (locality of "m", chunks final
    one after another).
    ``PROBE``: thread 0 of every WG writes (start, end) ``s_memrealtime`` (end after
    its atomics completed) to ``dbg`` int64 [m_blocks * n_blocks, 2] (m-major index).
    ``AR``: fused all-reduce (``gemm2_ar_fused.ar_config`` tuple; needs "g"): the
    first ``cwg`` WGs of the grid run the chunked two-shot all-reduce of ``out``
    over the TP peers (``tbl``) while the rest compute; every compute WG bumps its
    column chunk's ready counter once its atomics have landed.
    ``AR_SEP``: no comm WGs in this kernel (only the ready counters); the comm
    role runs as its own kernel concurrently on another stream
    (``gemm2_ar_fused.compile_ar_kernel``), with its own register allocation.

    ``A_GROUP``: 0 = one A scale per sorted row (``a_scale`` f32 [sorted_size]);
    128 = one per (sorted row, 128-K group) (``a_scale`` f32 [sorted_size, K/128],
    as written by gemm1's fused-quant ``q8s`` epilogue). Each group accumulates
    into its own accumulators, combined at dequant as sum_g acc_g * s_g (a single
    accumulator rescaled by s_g/s_{g+1} at group boundaries needed more VGPRs:
    the compiler hoists the scale loads + divides; 89 vs 77 at bm32).

    ``epilog``: ``"lds"`` (default) stages a weighted bf16 tile in LDS
    (BM*TILE_N*2, half the port's f32 staging) for 128 B contiguous atomics;
    ``"direct"`` scatters straight from the MFMA accumulator (lane-pair shuffle, no
    LDS staging/barrier, LDS = A tile only) but issues 32 B-contiguous atomics.

    ``dyn_rows``: each WG computes only the leading 16-row sub-blocks of its sorted
    block that hold valid rows (valid rows are contiguous from the block start),
    so A loads / MFMA / dequant / scatter skip all-padding sub-blocks. All row
    variants (16, 32, ..., BM) are compiled in; W traffic is unchanged.

    Launch args (raw device pointers): ``a`` fp8 [sorted_size, K], ``a_scale`` f32
    [sorted_size], ``w`` mxfp4 [E, N, K/2] (standard preshuffle), ``w_scale`` e8m0
    (shuffled), ``sorted_expert_ids``, ``num_valid_ids``, ``sorted_ids``,
    ``sorted_weights``, ``out`` bf16 [tokens, N] (pre-zeroed, atomically
    accumulated), then ``tokens``, ``grid`` and the stream.
    """
    K = D_INTER
    N_OUT = D_HIDDEN
    NUM_WAVES = 4
    assert K % TILE_K == 0 and TILE_K % 128 == 0, f"bad K={K} / TILE_K={TILE_K}"
    assert N_OUT % TILE_N == 0, f"model_dim must be a multiple of TILE_N={TILE_N}"
    assert TILE_N % (NUM_WAVES * 16) == 0, "TILE_N must be a multiple of 64"
    assert BM % 16 == 0, f"BM must be a multiple of 16, got {BM}"
    assert (BM * TILE_K) % (256 * 16) == 0, "A tile must be whole 16 B per thread"
    assert epilog in ("direct", "lds", "direct64"), f"bad epilog {epilog!r}"
    assert epilog != "direct64" or (NTILE and TILE_N // NUM_WAVES == 32), (
        "direct64: NTILE path, 32 columns per wave")
    assert fast_dec in (0, 1, 2) and (flush_tiny or not fast_dec), fast_dec
    assert A_GROUP in (0, 128), f"A_GROUP must be 0 or 128, got {A_GROUP}"
    assert TILE_ORDER in ("m", "n", "nx", "g"), TILE_ORDER
    assert N_OUT % CHUNK_N == 0 and CHUNK_N % TILE_N == 0
    CG = CHUNK_N // TILE_N  # n-blocks per column chunk / group
    if AR is not None:
        assert TILE_ORDER == "g" or (len(AR) > 7 and AR[7]), (
            "fused all-reduce needs column-group order (unless nodep)"
        )
        from gemm2_ar_fused import (  # noqa: PLC0415
            AR_TAG, ar_layout, build_ar_role, cnt_slot, tbl_entry,
        )

        AR_CWG = 0 if AR_SEP else AR[1]
        assert AR_CWG % 8 == 0, "comm WGs must keep compute WGs XCD-aligned"
        AR_L = ar_layout(*AR, D=N_OUT, chunk=CHUNK_N)
        # experiment knob for the per-WG ready signal (str: part of the JIT key)
        AR_HOOK = os.environ.get("AR_HOOK", "wave")  # wave|full|novm|nocnt|none
        ar_role = (
            None if AR_SEP
            else build_ar_role(*AR, D=N_OUT, chunk=CHUNK_N, cg=CG, hook=AR_HOOK)
        )
    else:
        AR_CWG = 0
        AR_HOOK = "none"
        AR_TAG = ""
    HOOK_MOD = 64 if AR_HOOK == "wave" else 1 << 20  # counting thread: lane 0 / tx 0
    HOOK_TX = -1 if AR_HOOK == "nocnt" else 0
    ROUTE = AR[6] if AR is not None and len(AR) > 6 else 0
    if ROUTE:
        assert epilog == "lds", "route rows use the LDS epilogue"
    A_GROUPS = K // 128 if A_GROUP else 1  # A scale groups (= accumulator sets)

    NUM_N_BLOCKS = N_OUT // TILE_N
    K_TILES = K // TILE_K
    N_REPEAT = TILE_N // NUM_WAVES // 16  # 16-col MFMA blocks per wave
    K_STEPS = TILE_K // 32  # MFMA K32 steps per tile
    K_GROUPS = TILE_K // 128  # 128-K groups: one W dwordx4 + one scale byte each
    # LDS epilogue: the A tile and the bf16 staging tile never coexist (barrier
    # between) -> max, not sum. Direct epilogue: A tile only.
    LDS_BYTES = (
        BM * TILE_K if epilog in ("direct", "direct64") else max(BM * TILE_K, BM * TILE_N * 2)
    )
    STAGE_OFF = 0  # byte offset of the bf16 staging tile
    SR = STAGE_ROWS or BM
    if NTILE:
        assert NUM_N_BLOCKS % NTILE == 0, f"NTILE={NTILE} must divide {NUM_N_BLOCKS}"
        assert K_TILES == 1, "NTILE keeps the whole A (one K tile) in LDS"
        assert TILE_ORDER == "m" and AR is None and not PROBE and not PERSIST
        assert SR % 16 == 0 and BM % SR == 0, f"STAGE_ROWS={STAGE_ROWS}"
        if epilog == "lds" and NTILE > 1:  # A stays live under every epilogue
            STAGE_OFF = BM * TILE_K
            LDS_BYTES = BM * TILE_K + SR * TILE_N * 2
    ROW_OFF = LDS_BYTES  # NTILE row data: f32 ratio[BM], f32 fin[BM], i32 id[BM]
    if NTILE:
        assert K // 128 <= 2, "NTILE rescales one accumulator across <= 2 A groups"
        if not ROW_VGPR:  # ROW_VGPR: row data per lane in VGPRs, no LDS
            LDS_BYTES += BM * 12
    NBG = NUM_N_BLOCKS // max(NTILE, 1)  # CTAs per m-block

    # W (mxfp4) N-major preshuffle in bytes: (N/16, K/128, klane=4, nlane=16, kpack=16).
    W_K0 = K // 128
    W_BYTES = NE * N_OUT * K // 2
    # E8M0 scale preshuffle (dwords): (N/32, Kpad/256, klane=4, nlane=16); each dword
    # packs [kpack=2] x [npack=2] bytes.
    SCALE_K_PAD = ((K + 255) // 256) * 256
    SC_STRIDE_N0 = (SCALE_K_PAD // 256) * 64
    WS_BYTES = NE * N_OUT * (SCALE_K_PAD // 32)

    ftz_tag = "_ftz" if flush_tiny else ""
    wpe_tag = f"_w{waves_per_eu}" if waves_per_eu else ""
    kname = (
        f"a8w4_gemm2_h{N_OUT}_i{K}_ne{NE}_bm{BM}_tn{TILE_N}_tk{TILE_K}"
        f"_bcm{b_cache_mod}_xcd{xcd_swizzle}{wpe_tag}{ftz_tag}_{epilog}"
        f"{'_dr' if EPI_DRAIN and epilog == 'lds' else ''}"
        f"{'_bf' if BF16_FAST and epilog == 'lds' else ''}{'_wp' if WPERM else ''}"
        f"{f'_fd{fast_dec}' if fast_dec else ''}"
        f"{f'_p{PERSIST}' if PERSIST else ''}"
        f"{'_dyn' if dyn_rows else ''}{f'_ag{A_GROUP}' if A_GROUP else ''}"
        f"{'' if TILE_ORDER == 'm' else '_o' + TILE_ORDER}{'_probe' if PROBE else ''}"
        f"{f'_c{CHUNK_N}' if TILE_ORDER == 'g' else ''}"
        f"{'' if AR is None else '_ar' + '_'.join(str(x).replace('-', 'm') for x in AR) + '_' + AR_HOOK + '_' + AR_TAG}"
        f"{'_sep' if AR_SEP else ''}{f'_nt{NTILE}' if NTILE else ''}"
        f"{f'_sr{STAGE_ROWS}' if NTILE and STAGE_ROWS else ''}"
        f"{'_bf' if BF16_FAST and NTILE else ''}{'_rv' if NTILE and ROW_VGPR else ''}"
    )

    def _body(
        lds_ptr,
        arg_a,
        arg_a_scale,
        arg_w,
        arg_w_scale,
        arg_eids,
        arg_stids,
        arg_sweights,
        arg_out,
        i32_M,
        tile,
        lane,
        wave,
    ):
        lane_div_16 = lane // fx.Int32(16)
        lane_mod_16 = lane % fx.Int32(16)

        # ---- tile decode: m-block (one expert) x n-block (model_dim tile) -------
        m_block = tile // fx.Int32(NBG)
        n_block = tile % fx.Int32(NBG)  # NTILE: n-group, first n-block n_block*NTILE
        e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block)))
        m_row = m_block * fx.Int32(BM)  # first sorted row of this m-block
        by_n = n_block * fx.Int32(TILE_N)

        # ---- W operand resources ----------------------------------------------
        expert_off = e * fx.Int32(N_OUT)
        w_tiles = _global_i32_buffer_tiles(arg_w, min(W_BYTES, 0xFFFFFFFF), 4)
        w_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(b_cache_mod), fx.Int32)
        ws_tiles = _global_i32_buffer_tiles(arg_w_scale, min(WS_BYTES, 0xFFFFFFFF), 1)
        ws_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), fx.Int32)

        # ---- A operand: BM sorted rows -> VGPRs -> LDS (single slot) ----------
        # Rows < cumsum0 always exist in the [sorted_size, K] buffer; the grid never
        # visits rows beyond it, so an unclamped resource is fine.
        x_tiles = fx.logical_divide(
            _global_i32_buffer_view(arg_a, fx.Int64(0xFFFFFFFF)), fx.make_layout(4, 1)
        )
        x_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(b_cache_mod), fx.Int32)
        lds_flat = fx.make_view(
            fx.recast_iter(fx.Int32, lds_ptr), fx.make_layout(LDS_BYTES // 4, 1)
        )
        lds_tiles16 = fx.logical_divide(lds_flat, fx.make_layout(4, 1))
        lds_tiles8 = fx.logical_divide(lds_flat, fx.make_layout(2, 1))
        lds_write_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        lds_read_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)

        def _swizzle(row, col_bytes):
            # A-LDS bank-conflict XOR swizzle; applied to the GMEM source column on
            # the write (LDS destination linear) and to the LDS column on the read.
            return col_bytes ^ ((row & fx.Int32(TILE_K // 16 - 1)) * fx.Int32(16))

        tx = fx.Int32(gpu.thread_id("x"))

        # ---- W columns: wave owns TILE_N/4 output channels ----------------------
        # Addresses are K-invariant per lane; a K tile only adds a static offset.
        # Two 16-col blocks of one 32-row scale block share a scale dword (npack
        # halves) -> load it once.
        share_scale = (TILE_N // NUM_WAVES) % 32 == 0
        n_wave_base = wave * fx.Int32(TILE_N // NUM_WAVES)

        def w_cols(by_n):
            w_base, sc_base, sc_key, sc_npack = [], {}, [], []
            for ni in range_constexpr(N_REPEAT):
                n0 = expert_off + by_n + n_wave_base + fx.Int32(ni * 16)
                w_base.append(
                    (n0 // fx.Int32(16)) * fx.Int32(W_K0 * 64)
                    + lane_div_16 * fx.Int32(16)
                    + lane_mod_16
                )
                key = ni // 2 if share_scale else ni
                sc_key.append(key)
                sc_npack.append((n0 // fx.Int32(16)) % fx.Int32(2))
                if key not in sc_base:
                    sc_base[key] = (
                        (n0 // fx.Int32(32)) * fx.Int32(SC_STRIDE_N0)
                        + lane_div_16 * fx.Int32(16)
                        + lane_mod_16
                    )
            return w_base, sc_base, sc_key, sc_npack

        # ---- e8m0 codes for the whole K (short: K = inter_dim) ------------------
        # Lane L holds 32-K group 4*G + L//16 of every 128-K group G; the dword of
        # group G is (G//2) along K, byte (G%2)*2 + npack.
        def scale_issue(sc_base):
            return {
                (key, G2): _buffer_i32_scalar_read(
                    ws_tiles, sc_base[key] + fx.Int32(G2 * 64), ws_atom
                )
                for key in sc_base
                for G2 in range((K // 128 + 1) // 2)
            }

        def exp_issue(by_n, ni):
            # fast_dec=2: per-channel max codes stored after the scale bytes
            n_lane = expert_off + by_n + n_wave_base + fx.Int32(ni * 16) + lane_mod_16
            return fx.Int32(
                _global_i32_at(fx.Int64(arg_w_scale) + fx.Int64(WS_BYTES), n_lane)
            )

        def scale_codes(packed, sc_key, sc_npack, by_n, exp_pre=None):
            code = {}  # code[ni, G]: this lane's e8m0 byte of 128-K group G
            for ni in range_constexpr(N_REPEAT):
                for G in range_constexpr(K // 128):
                    if const_expr(fast_dec == 2):
                        # byte (G%2)*2 + npack = delta*8, broadcast by one v_perm
                        p = packed[sc_key[ni], G // 2]
                        sel = (fx.Int32((G % 2) * 2) + sc_npack[ni]) * fx.Int32(
                            0x01010101
                        )
                        code[ni, G] = _perm_u32(p, p, sel)
                        continue
                    shift = fx.Int32((G % 2) * 16) + sc_npack[ni] * fx.Int32(8)
                    code[ni, G] = packed[sc_key[ni], G // 2].shrui(shift) & fx.Int32(
                        0xFF
                    )
            # Per-output-channel max e8m0 code over K: lane max, then across the 4
            # lanes (L//16) that share the column (fast_dec=2: stored after the scales).
            exp_n = []
            for ni in range_constexpr(N_REPEAT):
                if const_expr(fast_dec == 2):
                    exp_n.append(exp_issue(by_n, ni) if exp_pre is None else exp_pre[ni])
                    continue
                mx = code[ni, 0]
                for G in range_constexpr(1, K // 128):
                    mx = (code[ni, G] > mx).select(code[ni, G], mx)
                for off in (16, 32):
                    peer = mx.shuffle_xor(fx.Int32(off), fx.Int32(64))
                    mx = (peer > mx).select(peer, mx)
                exp_n.append(mx)
            return code, exp_n

        if const_expr(not NTILE):
            w_base, sc_base, sc_key, sc_npack = w_cols(by_n)
            code, exp_n = scale_codes(scale_issue(sc_base), sc_key, sc_npack, by_n)

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))

        def run_rows(rows):
            """A load, K loop, dequant and scatter for the first ``rows`` rows."""
            M_REPEAT = rows // 16  # 16-row MFMA blocks
            A_LOADS = rows * TILE_K // (256 * 16)  # 16 B A copies per thread

            a_src_tile, a_dst_tile = [], []
            for i in range_constexpr(A_LOADS):
                idx_dw = tx * fx.Int32(4) + fx.Int32(i * 256 * 4)
                row = idx_dw // fx.Int32(TILE_K // 4)
                col_bytes = (idx_dw % fx.Int32(TILE_K // 4)) * fx.Int32(4)
                row_bytes = (m_row + row) * fx.Int32(K)  # A is indexed by sorted row
                a_src_tile.append((row_bytes + _swizzle(row, col_bytes)) // fx.Int32(16))
                a_dst_tile.append((row * fx.Int32(TILE_K) + col_bytes) // fx.Int32(16))

            def store_a_tile(base_k):
                k16 = fx.Int32(base_k // 16)
                for i in range_constexpr(A_LOADS):
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(x_atom, fx.slice(x_tiles, (None, a_src_tile[i] + k16)), r)
                    fx.copy(lds_write_atom, r, fx.slice(lds_tiles16, (None, a_dst_tile[i])))

            def load_a(mi, ku):
                # Lane L of K-step ku: row L%16, K = (ku//4)*128 + (L//16)*32 + (ku%4)*8.
                row = lane_mod_16 + fx.Int32(mi * 16)
                col = lane_div_16 * fx.Int32(32) + fx.Int32((ku // 4) * 128 + (ku % 4) * 8)
                off = (row * fx.Int32(TILE_K) + _swizzle(row, col)) // fx.Int32(8)
                r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
                fx.copy_atom_call(lds_read_atom, fx.slice(lds_tiles8, (None, off)), r)
                return fx.Vector(fx.memref_load_vec(r)).bitcast(fx.Float8E4M3FNUZ)

            # ---- accumulators (one set per A scale group) ---------------------------
            acc_layout = fx.make_layout(4, 1)
            accg = [
                [
                    [fx.make_rmem_tensor(acc_layout, fx.Float32) for _ in range(N_REPEAT)]
                    for _ in range(M_REPEAT)
                ]
                for _ in range(A_GROUPS)
            ]
            accm = accg[0]
            zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
            for acc in accg:
                for mi in range_constexpr(M_REPEAT):
                    for ni in range_constexpr(N_REPEAT):
                        acc[mi][ni].store(zero4)

            as_tiles = _global_i32_buffer_tiles(arg_a_scale, fx.Int64(0xFFFFFFFF), 1)
            as_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            # grouped: the A_GROUPS scales of a row are contiguous -> one load
            asg_tiles = _global_i32_buffer_tiles(
                arg_a_scale, fx.Int64(0xFFFFFFFF), A_GROUPS
            )
            asg_atom = (
                fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
                if A_GROUPS == 2
                else fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
                if A_GROUPS == 4
                else None
            )

            def a_scales_row(mi, ii):
                """[A_GROUPS] f32 scales of this lane's row ii of block mi."""
                row = m_row + fx.Int32(mi * 16 + ii) + lane_div_16 * fx.Int32(4)
                if const_expr(asg_atom is None):
                    return [
                        _buffer_i32_scalar_read(
                            as_tiles, row * fx.Int32(A_GROUPS) + fx.Int32(g), as_atom
                        ).bitcast(fx.Float32)
                        for g in range(A_GROUPS)
                    ]
                r = fx.make_rmem_tensor(fx.make_layout(A_GROUPS, 1), fx.Int32)
                fx.copy(asg_atom, fx.slice(asg_tiles, (None, row)), r)
                v = fx.Vector(fx.memref_load_vec(r))
                return [fx.Int32(v[g]).bitcast(fx.Float32) for g in range(A_GROUPS)]

            # ---- K loop (static: K = inter_dim is a few tiles) ----------------------
            for kt in range_constexpr(K_TILES):
                base_k = kt * TILE_K
                store_a_tile(base_k)
                raw = []  # raw[ni][g][j]: i32 with 8 fp4 for K-step g*4 + j
                for ni in range_constexpr(N_REPEAT):
                    per_g = []
                    for g in range_constexpr(K_GROUPS):
                        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        idx = w_base[ni] + fx.Int32((base_k // 128 + g) * 64)
                        fx.copy(w_atom, fx.slice(w_tiles, (None, idx)), r)
                        v4 = fx.Vector(fx.memref_load_vec(r))
                        per_g.append([fx.Int32(v4[j]) for j in range(4)])
                    raw.append(per_g)
                rocdl.s_waitcnt(lgkmcnt=0)
                gpu.barrier()  # A(kt) visible in LDS
                for ni in range_constexpr(N_REPEAT):
                    luts = [
                        _fp8_lut_predelta(code[ni, base_k // 128 + g])
                        if fast_dec == 2
                        else _fp8_lut_fast(code[ni, base_k // 128 + g], exp_n[ni])
                        if fast_dec
                        else _fp8_lut(code[ni, base_k // 128 + g], exp_n[ni], flush_tiny)
                        for g in range_constexpr(K_GROUPS)
                    ]
                    dec = _fp4x8_to_fp8x8_fast if fast_dec else _fp4x8_to_fp8x8
                    for ku in range_constexpr(K_STEPS):
                        b8 = dec(raw[ni][ku // 4][ku % 4], luts[ku // 4])
                        gg = base_k // 128 + ku // 4  # 128-K group of this K step
                        acc = accg[gg if A_GROUP else 0]
                        for mi in range_constexpr(M_REPEAT):
                            _mma_fp8(mma_atom, acc[mi][ni], load_a(mi, ku), b8)
                gpu.barrier()  # A reads done before the slot (or the epilogue) reuses LDS

            # ---- dequant: sum_g acc_g * a_scale[row, g] * 2**(exp-132) --------------
            f_n = [_channel_factor(x) for x in exp_n]
            for mi in range_constexpr(M_REPEAT):
                per_row = [a_scales_row(mi, ii) for ii in range_constexpr(4)]
                rs = [
                    [per_row[ii][g] for ii in range_constexpr(4)]
                    for g in range_constexpr(A_GROUPS)
                ]
                for ni in range_constexpr(N_REPEAT):
                    if const_expr(A_GROUP):
                        vs = [Vec(accg[g][mi][ni].load()) for g in range(A_GROUPS)]
                        out4 = []
                        for ii in range_constexpr(4):
                            t = vs[0][ii] * rs[0][ii]
                            for g in range_constexpr(1, A_GROUPS):
                                t = t + vs[g][ii] * rs[g][ii]
                            out4.append(t * f_n[ni])
                    else:
                        v = Vec(accm[mi][ni].load())
                        out4 = [v[ii] * (rs[0][ii] * f_n[ni]) for ii in range_constexpr(4)]
                    accm[mi][ni].store(Vec.from_elements(out4, fx.Float32))

            # ---- epilogue: routing weight + bf16x2 atomic scatter -------------------
            accm_v = [
                [accm[mi][ni].load().ir_value() for ni in range(N_REPEAT)]
                for mi in range(M_REPEAT)
            ]
            if const_expr(epilog == "direct"):
                _direct_atomic_epilog(
                    accm_v,
                    arg_out,
                    arg_stids,
                    arg_sweights,
                    m_row,
                    by_n + n_wave_base + lane_mod_16,
                    lane,
                    i32_M,
                    N_OUT,
                )
            elif const_expr(ROUTE):
                _route_bf16_epilog(
                    fx.Int32(fx.ptrtoint(lds_ptr)),
                    accm_v,
                    arg_out,
                    arg_stids,
                    arg_sweights,
                    m_row,
                    n_block,
                    wave,
                    lane,
                    i32_M,
                    fx.Int32(ROUTE),
                    BM,
                    N_OUT,
                    TILE_N,
                )
            else:
                _atomic_bf16_epilog(
                    fx.Int32(fx.ptrtoint(lds_ptr)),
                    accm_v,
                    arg_out,
                    arg_stids,
                    arg_sweights,
                    m_row,
                    n_block,
                    wave,
                    lane,
                    i32_M,
                    BM,
                    N_OUT,
                    TILE_N,
                )


        # ---- NTILE body: one m-block x NTILE adjacent n-blocks ----------------------
        def _ld1(atom, src, index, elem_ty):
            frag = fx.make_rmem_tensor(1, elem_ty)
            fx.copy(atom, src[None, index], frag)
            return Vec(frag.load())[0]

        def run_rows_nt(rows):
            M_REPEAT = rows // 16
            A_LOADS = rows * TILE_K // (256 * 16)

            # A: every global load first (LLVM does not hoist a buffer load above an
            # earlier ds_write); the LDS writes go after the W / row loads below.
            a_regs = []
            for i in range_constexpr(A_LOADS):
                idx_dw = tx * fx.Int32(4) + fx.Int32(i * 256 * 4)
                row = idx_dw // fx.Int32(TILE_K // 4)
                col_bytes = (idx_dw % fx.Int32(TILE_K // 4)) * fx.Int32(4)
                src = ((m_row + row) * fx.Int32(K) + _swizzle(row, col_bytes)) // fx.Int32(16)
                r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy(x_atom, fx.slice(x_tiles, (None, src)), r)
                a_regs.append((r, (row * fx.Int32(TILE_K) + col_bytes) // fx.Int32(16)))

            def w_issue(t):
                """Loads of n-tile t: W (one K tile), scale dwords, max codes."""
                nb = n_block * fx.Int32(NTILE) + fx.Int32(t)
                by_n_t = nb * fx.Int32(TILE_N)
                wb, scb, sck, scn = w_cols(by_n_t)
                packed_t = scale_issue(scb)
                exp_t = (
                    [exp_issue(by_n_t, ni) for ni in range(N_REPEAT)]
                    if fast_dec == 2
                    else None
                )
                raw = []
                for ni in range_constexpr(N_REPEAT):
                    per_g = []
                    for g in range_constexpr(K_GROUPS):
                        r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        fx.copy(w_atom, fx.slice(w_tiles, (None, wb[ni] + fx.Int32(g * 64))), r)
                        v4 = fx.Vector(fx.memref_load_vec(r))
                        per_g.append([fx.Int32(v4[j]) for j in range(4)])
                    raw.append(per_g)
                return dict(nb=nb, by_n=by_n_t, packed=packed_t, sck=sck, scn=scn,
                            exp=exp_t, raw=raw)

            cur = w_issue(0)

            # Row-static data, once per CTA: per sorted row ratio = s0/s1 (A-scale
            # groups 0/1; the single accumulator is rescaled by it at the group
            # boundary), fin = s_last * route weight, and the sorted id.
            # ROW_VGPR: each lane loads its own rows into VGPRs (vector loads, no
            # LDS; LDS = A tile only). Else: cooperatively into LDS (+12 B/row),
            # 4 threads per row writing the same value.
            odd = lane & fx.Int32(1)
            if const_expr(ROW_VGPR):
                # lane-distributed: lane L of every wave holds rows h*64 + L
                # (h < BM/64); a consumer fetches its row with one ds_bpermute
                # (LDS crossbar, no LDS space, ~3 VGPRs per 64 rows)
                rs_as = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_a_scale)))
                rs_w = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_sweights)))
                rs_id = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_stids)))
                RH = -(-BM // 64)
                raw_s, raw_w, rid_h = [], [], []
                for h in range_constexpr(RH):
                    grow_h = m_row + fx.Int32(h * 64) + lane
                    if const_expr(A_GROUPS > 1):
                        v = Vec(buffer_ops.buffer_load(
                            rs_as, grow_h * fx.Int32(A_GROUPS), vec_width=A_GROUPS, dtype=T.i32))
                        raw_s.append([fx.Int32(v[g]) for g in range(A_GROUPS)])
                    else:
                        raw_s.append([fx.Int32(buffer_ops.buffer_load(
                            rs_as, grow_h, vec_width=1, dtype=T.i32))])
                    raw_w.append(fx.Int32(buffer_ops.buffer_load(rs_w, grow_h, vec_width=1, dtype=T.i32)))
                    rid_h.append(fx.Int32(buffer_ops.buffer_load(rs_id, grow_h, vec_width=1, dtype=T.i32)))
            else:
                rrow = tx % fx.Int32(BM)
                grow = m_row + rrow
                if const_expr(A_GROUP):
                    r = fx.make_rmem_tensor(fx.make_layout(A_GROUPS, 1), fx.Int32)
                    fx.copy(asg_atom_nt, fx.slice(asg_tiles_nt, (None, grow)), r)
                    v = fx.Vector(fx.memref_load_vec(r))
                    a_s = [fx.Int32(v[g]).bitcast(fx.Float32) for g in range(A_GROUPS)]
                else:
                    a_s = [_ld1(ld_f32, as_buf_nt, grow, fx.Float32)]
                r_w = _ld1(ld_f32, sweights_buf, grow, fx.Float32)
                r_id = _ld1(ld_i32, stids_buf, grow, fx.Int32)

            for r, dst in a_regs:
                fx.copy(lds_write_atom, r, fx.slice(lds_tiles16, (None, dst)))
            lds_i32 = fx.Int32(fx.ptrtoint(lds_ptr))
            if const_expr(not ROW_VGPR):
                row_f = lds_typed_ptr(lds_i32 + fx.Int32(ROW_OFF), T.f32)
                row_i = lds_typed_ptr(lds_i32 + fx.Int32(ROW_OFF), T.i32)
                s_last = a_s[-1]
                if const_expr(A_GROUP):
                    # an all-zero group quantizes to scale 0 (its acc part is 0 too)
                    s_last = (s_last == fx.Float32(0.0)).select(fx.Float32(1.0), s_last)
                    row_f[rrow] = a_s[0] / s_last
                row_f[fx.Int32(BM) + rrow] = s_last * r_w
                row_i[fx.Int32(2 * BM) + rrow] = r_id
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()  # A (+ LDS row data) visible (stay for every n-tile)

            if const_expr(ROW_VGPR):
                ratio_h, fin_h = [], []  # i32 bit patterns (ds_bpermute moves i32)
                for h in range_constexpr(RH):
                    w_h = raw_w[h].bitcast(fx.Float32)
                    s_l = raw_s[h][-1].bitcast(fx.Float32)
                    if const_expr(A_GROUP):
                        s_l = (s_l == fx.Float32(0.0)).select(fx.Float32(1.0), s_l)
                        ratio_h.append((raw_s[h][0].bitcast(fx.Float32) / s_l).bitcast(fx.Int32))
                    fin_h.append((s_l * w_h).bitcast(fx.Int32))

                def _fetch(q, row_static, row_lane):
                    """q[row // 64] at block row ``row_static + row_lane`` (row_lane <
                    64 - row_static % 64, so the half is static)."""
                    src = fx.Int32(row_static % 64) + row_lane
                    return fx.Int32(rocdl.ds_bpermute(T.i32, src * fx.Int32(4), q[row_static // 64]))

            def row4(slot, mi):
                """f32 row data of this lane's 4 acc rows of block mi."""
                if const_expr(ROW_VGPR):
                    q = ratio_h if slot == 0 else fin_h
                    return [
                        _fetch(q, mi * 16 + v, lane_div_16 * fx.Int32(4)).bitcast(fx.Float32)
                        for v in range(4)
                    ]
                byte = fx.Int32(ROW_OFF + slot * BM * 4 + mi * 64) + lane_div_16 * fx.Int32(16)
                return Vec(lds_vec_load(lds_i32, byte, Vec.make_type(4, fx.Float32), fx.Float32, align=16))

            def row_id(row):
                return lds_vec_load(
                    lds_i32, fx.Int32(ROW_OFF + 2 * BM * 4) + row * fx.Int32(4),
                    T.i32, fx.Int32, align=4,
                )

            def id_pass(mr):
                """lds scatter: sorted id of row mr*8 + tx//32."""
                if const_expr(ROW_VGPR):  # tx//32 = 2*wave + lane//32
                    return _raw(_fetch(rid_h, mr * 8, fx.Int32(2) * fx.Int32(wave) + lane // fx.Int32(32)))
                return _raw(row_id(fx.Int32(mr * 8) + tx // fx.Int32(32)))

            def id_direct(mi, k):
                """direct scatter: this lane's row k of block mi."""
                if const_expr(ROW_VGPR):
                    extra = fx.Int32(0) if epilog == "direct64" else odd * fx.Int32(2)
                    return _raw(_fetch(rid_h, mi * 16 + k, lane_div_16 * fx.Int32(4) + extra))
                if const_expr(epilog == "direct64"):
                    return _raw(row_id(fx.Int32(mi * 16 + k) + lane_div_16 * fx.Int32(4)))
                return _raw(row_id(
                    fx.Int32(mi * 16 + k) + lane_div_16 * fx.Int32(4) + odd * fx.Int32(2)
                ))

            def load_a(mi, ku):
                row = lane_mod_16 + fx.Int32(mi * 16)
                col = lane_div_16 * fx.Int32(32) + fx.Int32((ku // 4) * 128 + (ku % 4) * 8)
                off = (row * fx.Int32(TILE_K) + _swizzle(row, col)) // fx.Int32(8)
                r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
                fx.copy_atom_call(lds_read_atom, fx.slice(lds_tiles8, (None, off)), r)
                return fx.Vector(fx.memref_load_vec(r)).bitcast(fx.Float8E4M3FNUZ)

            zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
            for t in range_constexpr(NTILE):
                # keep the scheduler from interleaving tile t with t+1 (without
                # the lds epilogue's barriers it hoists the next tile's work and
                # the VGPR count doubles: bm128 direct64 nt4 378 -> 1 wave/SIMD)
                rocdl.sched_barrier(0)
                nxt = w_issue(t + 1) if t + 1 < NTILE else None  # prefetch
                code_t, exp_t = scale_codes(
                    cur["packed"], cur["sck"], cur["scn"], cur["by_n"], cur["exp"]
                )
                acc = [
                    [fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32) for _ in range(N_REPEAT)]
                    for _ in range(M_REPEAT)
                ]
                for mi in range_constexpr(M_REPEAT):
                    for ni in range_constexpr(N_REPEAT):
                        acc[mi][ni].store(zero4)
                luts = [
                    [
                        _fp8_lut_predelta(code_t[ni, g])
                        if fast_dec == 2
                        else _fp8_lut_fast(code_t[ni, g], exp_t[ni])
                        if fast_dec
                        else _fp8_lut(code_t[ni, g], exp_t[ni], flush_tiny)
                        for g in range_constexpr(K_GROUPS)
                    ]
                    for ni in range_constexpr(N_REPEAT)
                ]
                dec = _fp4x8_to_fp8x8_fast if fast_dec else _fp4x8_to_fp8x8
                # ku outer: each A fragment is read from LDS once and feeds every
                # ni (half the A LDS traffic of the ni-outer order)
                for ku in range_constexpr(K_STEPS):
                    if const_expr(A_GROUP and ku == 4):
                        # group 0 -> group 1 boundary: acc = acc0 * s0 / s1
                        for mi in range_constexpr(M_REPEAT):
                            rt = row4(0, mi)
                            for ni in range_constexpr(N_REPEAT):
                                a4 = Vec(acc[mi][ni].load())
                                acc[mi][ni].store(
                                    Vec.from_elements([a4[v] * rt[v] for v in range(4)], fx.Float32)
                                )
                    b8s = [
                        dec(cur["raw"][ni][ku // 4][ku % 4], luts[ni][ku // 4])
                        for ni in range_constexpr(N_REPEAT)
                    ]
                    for mi in range_constexpr(M_REPEAT):
                        a8 = load_a(mi, ku)
                        for ni in range_constexpr(N_REPEAT):
                            _mma_fp8(mma_atom, acc[mi][ni], a8, b8s[ni])

                f_n = [_channel_factor(x) for x in exp_t]
                accm_v = []
                for mi in range_constexpr(M_REPEAT):
                    fin = row4(1, mi)
                    row_v = []
                    for ni in range_constexpr(N_REPEAT):
                        a4 = Vec(acc[mi][ni].load())
                        row_v.append(
                            Vec.from_elements(
                                [a4[v] * fin[v] * f_n[ni] for v in range(4)], fx.Float32
                            ).ir_value()
                        )
                    accm_v.append(row_v)

                if const_expr(epilog == "lds"):
                    stage = lds_i32 + fx.Int32(STAGE_OFF)
                    per = SR // 16 if NTILE > 1 else M_REPEAT  # 16-row blocks/chunk
                    chunks = -(-M_REPEAT // per)
                    for c in range_constexpr(chunks):
                        if const_expr(NTILE == 1 or t > 0 or c > 0):
                            # NTILE=1: staging reuses the A slot (A reads done);
                            # else every wave finished scattering the last chunk
                            gpu.barrier()
                        _stage_bf16_tile(
                            stage, accm_v[c * per : (c + 1) * per], wave, lane, TILE_N
                        )
                        gpu.barrier()
                        ids = [id_pass(mr) for mr in range(c * per * 2, (c + 1) * per * 2)]
                        _scatter_bf16_tile(
                            stage, ids, arg_out, cur["nb"], i32_M, N_OUT, TILE_N
                        )
                elif const_expr(epilog == "direct64"):
                    _direct64_scatter_tile(
                        accm_v,
                        [[id_direct(mi, k) for k in range(4)] for mi in range(M_REPEAT)],
                        arg_out,
                        cur["by_n"] + n_wave_base,
                        lane,
                        i32_M,
                        N_OUT,
                    )
                else:
                    ids = [[id_direct(mi, k) for k in range(2)] for mi in range(M_REPEAT)]
                    _direct_scatter_tile(
                        accm_v,
                        ids,
                        arg_out,
                        cur["by_n"] + n_wave_base + lane_mod_16,
                        lane,
                        i32_M,
                        N_OUT,
                    )
                cur = nxt

        if const_expr(NTILE):
            stids_buf = _flat_buffer(arg_stids, T.i32, 4)
            sweights_buf = _flat_buffer(arg_sweights, T.f32, 4)
            ld_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
            ld_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)
            as_buf_nt = _flat_buffer(arg_a_scale, T.f32, 4)
            asg_tiles_nt = _global_i32_buffer_tiles(
                arg_a_scale, fx.Int64(0xFFFFFFFF), A_GROUPS
            )
            asg_atom_nt = (
                fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.Int32)
                if A_GROUPS == 2
                else fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
                if A_GROUPS == 4
                else None
            )
        run = run_rows_nt if NTILE else run_rows

        # ---- row-variant dispatch ------------------------------------------------
        if const_expr(dyn_rows):
            # Valid rows of a sorted block are contiguous from its start, so the
            # first 16-row sub-block whose first row is padding bounds the work.
            pad = {
                i: (
                    fx.Int32(
                        rocdl.readfirstlane(
                            T.i32,
                            _raw(_global_i32_at(arg_stids, m_row + fx.Int32(16 * i))),
                        )
                    )
                    & fx.Int32(0x00FFFFFF)
                )
                >= i32_M
                for i in range(1, BM // 16)
            }

            def _dispatch_from(i):
                # sub-blocks 0..i-1 hold valid rows; decide on sub-block i.
                if i == BM // 16:
                    run(BM)
                    return
                cond = pad[i]

                @flyc.jit
                def _dispatch():
                    if cond:
                        run(16 * i)
                    else:
                        _dispatch_from(i + 1)

                _dispatch()

            _dispatch_from(1)
        else:
            run(BM)

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.jit
    def _entry(
        arg_a,
        arg_a_scale,
        arg_w,
        arg_w_scale,
        arg_eids,
        arg_cumsum,
        arg_stids,
        arg_sweights,
        arg_out,
        i32_M,
        arg_dbg,
        arg_tbl,
        i32_rank,
        i32_epoch,
    ):
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x")) - fx.Int32(AR_CWG)  # compute WG index
        lane = tx % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx // fx.Int32(64))
        cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(NBG)

        # Bijective XCD round-robin over valid tiles (always on: stage2 is HBM-bound);
        # xcd_swizzle>0 also groups xcd_swizzle m-blocks per XCD for L2 locality.
        NXCD = 8
        xq = _udiv(bound, NXCD)
        xr = _umod(bound, NXCD)

        def _xcd(pid):
            xc = _umod(pid, NXCD)
            wgid = xc * xq + fx.min(xc, xr) + _udiv(pid, NXCD)
            if const_expr(xcd_swizzle <= 0):
                return wgid
            ng = fx.Int32(xcd_swizzle * NBG)
            first_m = (wgid // ng) * fx.Int32(xcd_swizzle)
            group_m = fx.min(total_m_blocks - first_m, fx.Int32(xcd_swizzle))
            wig = wgid % ng
            return (first_m + wig % group_m) * fx.Int32(NBG) + wig // group_m

        def _tile_work(t):
            if const_expr(TILE_ORDER == "m"):
                tile = _xcd(t)
            elif const_expr(TILE_ORDER == "g"):
                # group g = column chunk g; inside it m-major + per-XCD contiguous
                gs = total_m_blocks * fx.Int32(CG)
                grp = t // gs
                loc = t - grp * gs
                gq = _udiv(gs, NXCD)
                gr = _umod(gs, NXCD)
                gxc = _umod(loc, NXCD)
                wg = gxc * gq + fx.min(gxc, gr) + _udiv(loc, NXCD)
                tile = (wg // fx.Int32(CG)) * fx.Int32(NUM_N_BLOCKS) + grp * fx.Int32(
                    CG
                ) + wg % fx.Int32(CG)
            else:
                tt = _xcd(t) if const_expr(TILE_ORDER == "n") else t
                # N-major t -> (m_block, n_block) -> m-major tile id used by _body
                tile = (tt % total_m_blocks) * fx.Int32(NUM_N_BLOCKS) + tt // total_m_blocks
            if const_expr(PROBE):
                t_start = fx.Int64(
                    llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], [])
                )
            if const_expr(AR is not None and AR_HOOK != "none"):
                # ready-counter base, fetched before the tile so it is long in SGPRs
                # by the hook (ATT: a dependent vector load here cost ~0.3 us/WG)
                own = tbl_entry(arg_tbl, AR[0], i32_rank)
            _body(
                lds_ptr,
                arg_a,
                arg_a_scale,
                arg_w,
                arg_w_scale,
                arg_eids,
                arg_stids,
                arg_sweights,
                arg_out,
                i32_M,
                tile,
                lane,
                wave,
            )
            if const_expr(PROBE):
                rocdl.s_waitcnt(vmcnt=0)  # this WG's atomics performed
                gpu.barrier()
                if tx == fx.Int32(0):
                    t_end = fx.Int64(
                        llvm.call_intrinsic(
                            T.i64, "llvm.amdgcn.s.memrealtime", [], [], []
                        )
                    )
                    dbg = global_typed_ptr(arg_dbg, T.i64, align=8)
                    dbg[tile * fx.Int32(2)] = t_start
                    dbg[tile * fx.Int32(2) + fx.Int32(1)] = t_end
            if const_expr(AR is not None and AR_HOOK != "none"):
                # chunk ready counter: this WG's atomics landed (vmcnt covers
                # non-returning atomics on gfx9) -> count it for its column chunk
                if const_expr(AR_HOOK != "novm"):
                    rocdl.s_waitcnt(vmcnt=0)
                if const_expr(AR_HOOK != "wave"):
                    gpu.barrier()
                # "wave": every wave counts itself (target x4), no WG barrier
                if tx % fx.Int32(HOOK_MOD) == fx.Int32(HOOK_TX):  # -1: never
                    chunk = (tile % fx.Int32(NUM_N_BLOCKS)) // fx.Int32(CG)
                    slot = (
                        (i32_epoch & fx.Int32(1)) * fx.Int32(N_OUT // CHUNK_N) + chunk
                    ) * fx.Int32(AR_L["nslot"]) + cnt_slot(bx)
                    llvm.AtomicRMWOp(
                        llvm.AtomicBinOp.add,
                        _gep1(
                            _global_base_ptr1(own + fx.Int64(AR_L["cnt"])),
                            slot * fx.Int32(256),
                        ),
                        _raw(fx.Int32(1)),
                        llvm.AtomicOrdering.monotonic,
                        syncscope="agent",
                        alignment=4,
                    )

        if (bx >= fx.Int32(0)) & (bx < bound):
            if const_expr(PERSIST):
                # persistent-lite: a fixed set of compute CTAs strides over the
                # tiles (CTA count % 8 == 0 keeps every tile on its XCD)
                n_cta = fx.Int32(gpu.grid_dim.x) - fx.Int32(AR_CWG)
                for t in range(bx, bound, n_cta):
                    _tile_work(t)
                    gpu.barrier()  # LDS reuse by this CTA's next tile
            else:
                _tile_work(bx)
        if const_expr(AR is not None and not AR_SEP):
            if bx < fx.Int32(0):
                ar_role(
                    arg_tbl,
                    arg_out,
                    i32_M,
                    i32_rank,
                    i32_epoch,
                    bx + fx.Int32(AR_CWG),
                    total_m_blocks,
                )

    @flyc.kernel(name=kname, known_block_size=[256, 1, 1])
    def a8w4_gemm2_kernel(
        arg_a: fx.Int64,
        arg_a_scale: fx.Int64,
        arg_w: fx.Int64,
        arg_w_scale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        arg_out: fx.Int64,
        i32_M: fx.Int32,
        arg_dbg: fx.Int64,
        arg_tbl: fx.Int64,
        i32_rank: fx.Int32,
        i32_epoch: fx.Int32,
    ):
        _entry(
            arg_a,
            arg_a_scale,
            arg_w,
            arg_w_scale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            arg_out,
            i32_M,
            arg_dbg,
            arg_tbl,
            i32_rank,
            i32_epoch,
        )

    @flyc.jit
    def launch(
        arg_a: fx.Int64,
        arg_a_scale: fx.Int64,
        arg_w: fx.Int64,
        arg_w_scale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        arg_out: fx.Int64,
        i32_M: fx.Int32,
        arg_dbg: fx.Int64,
        arg_tbl: fx.Int64,
        i32_rank: fx.Int32,
        i32_epoch: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        a8w4_gemm2_kernel(
            arg_a,
            arg_a_scale,
            arg_w,
            arg_w_scale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            arg_out,
            i32_M,
            arg_dbg,
            arg_tbl,
            i32_rank,
            i32_epoch,
            value_attrs=(
                {"rocdl.waves_per_eu": waves_per_eu} if waves_per_eu else None
            ),
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(256, 1, 1), stream=stream)

    return launch


def a8w4_gemm2(
    a_fp8,
    a_scale,
    w2,
    w2_scale,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    out,
    *,
    num_experts,
    tile_m=64,
    tile_n=128,
    tile_k=256,
    b_cache_mod=2,
    xcd_swizzle=0,
    flush_tiny=True,
    waves_per_eu=None,
    epilog="lds",
    dyn_rows=False,
    a_group=0,
    tile_order="m",
    dbg=None,
    chunk_n=512,
    ar=None,
    tbl=None,
    rank=0,
    epoch=0,
    ar_sep=False,
    fast_dec=0,
    n_tiles=0,
    stage_rows=0,
    row_vgpr=False,
):
    """Host launcher. ``out`` is bf16 ``[tokens, model_dim]``; it is accumulated
    into (atomics), so the caller zeroes it (moe_sorting's moe_buf already is).
    ``a_group=128``: ``a_scale`` is f32 [sorted_size, inter_dim/128].
    ``dbg``: int64 [m_blocks * n_blocks * 2] -> per-WG timestamp probe.
    ``ar``/``tbl``/``rank``/``epoch``: fused all-reduce (see gemm2_ar_fused).
    ``fast_dec``: 1/2 need ``w2`` / ``w2_scale`` from ``prep_w2_fast``.
    ``n_tiles``: n-blocks per CTA (``NTILE`` of ``compile_a8w4_gemm2``; 0 = off)."""
    tokens, model_dim = out.shape
    if ar is not None and len(ar) > 6 and ar[6]:
        tokens //= ar[6]  # route mode: out is [tokens * topk, model_dim]
    inter_dim = a_fp8.shape[1]
    launch = compile_a8w4_gemm2(
        D_HIDDEN=model_dim,
        D_INTER=inter_dim,
        NE=num_experts,
        BM=tile_m,
        TILE_N=tile_n,
        TILE_K=tile_k,
        b_cache_mod=b_cache_mod,
        xcd_swizzle=xcd_swizzle,
        flush_tiny=flush_tiny,
        waves_per_eu=waves_per_eu,
        epilog=epilog,
        dyn_rows=dyn_rows,
        A_GROUP=a_group,
        TILE_ORDER=tile_order,
        PROBE=dbg is not None,
        CHUNK_N=chunk_n,
        AR=ar,
        AR_SEP=ar_sep,
        fast_dec=fast_dec,
        NTILE=n_tiles,
        STAGE_ROWS=stage_rows,
        ROW_VGPR=row_vgpr,
    )
    grid = sorted_expert_ids.numel() * (model_dim // tile_n) // max(n_tiles, 1)
    if PERSIST:  # k compute CTAs per CU (multiple of 8: XCD-aligned)
        grid = min(grid, PERSIST * torch.cuda.get_device_properties(out.device).multi_processor_count)
    grid += ar[1] if ar and not ar_sep else 0
    _run_compiled(
        launch,
        a_fp8.data_ptr(),
        a_scale.data_ptr(),
        w2.data_ptr(),
        w2_scale.data_ptr(),
        sorted_expert_ids.data_ptr(),
        num_valid_ids.data_ptr(),
        sorted_ids.data_ptr(),
        sorted_weights.data_ptr(),
        out.data_ptr(),
        int(tokens),
        dbg.data_ptr() if dbg is not None else 0,
        tbl.data_ptr() if tbl is not None else 0,
        int(rank),
        int(epoch),
        int(grid),
        torch.cuda.current_stream(),
    )
    return out


def prep_w2_fast(w2_u8, w2_scale_u8, fast_dec):
    """Standard-preshuffled W2 [E, N, K/2] / e8m0 scale bytes -> the ``fast_dec``
    (1/2) form (weight-static, once per weight). fast_dec=2 returns the scale
    buffer with the per-channel max codes (int32 [E*N]) appended after the
    rewritten scale bytes (the kernel reads them at offset WS_BYTES)."""
    from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.a8w4 import channel_exponents

    from .test_a8w4_gemm1 import prep_w1_fast

    w2_scale_u8 = w2_scale_u8.contiguous().view(-1)
    exp = channel_exponents(w2_u8, w2_scale_u8)
    w, s = prep_w1_fast(w2_u8, w2_scale_u8, exp, fast_dec)
    if fast_dec == 2:
        s = torch.cat([s, exp.contiguous().view(-1).view(torch.uint8)])
    return w, s


# =============================================================================
# Reference + test
# =============================================================================


def run_torch(
    a_fp8,
    a_scale,
    w2_qt,
    w2_scale,
    sorted_ids,
    sorted_weights,
    sorted_expert_ids,
    num_valid_ids,
    tokens,
    tile_m,
):
    """fp32 reference: out[t] = sum over t's sorted rows r of
    w_r * (a_deq[r] @ W2[e_r]^T), using the same FP8 A (dequantized with its row
    scale) and exact mxfp4 W2. Returns bf16 [tokens, model_dim]."""
    E, model_dim, _ = w2_qt.shape
    inter_dim = a_fp8.shape[1]
    nv = int(num_valid_ids[0].item())
    ids = sorted_ids[:nv].to(torch.int64)
    token = ids & 0x00FFFFFF
    row_expert = sorted_expert_ids.to(torch.int64).repeat_interleave(tile_m)[:nv]
    valid = token < tokens

    a = a_fp8[:nv].to(torch.float32)
    if a_scale.dim() == 2:  # per (row, 128-K group)
        a = (a.view(nv, -1, 128) * a_scale[:nv].view(nv, -1, 1)).view(nv, inter_dim)
    else:
        a = a * a_scale[:nv].view(-1, 1)
    w_scale = fp4_utils.e8m0_to_f32(w2_scale).view(E, model_dim, inter_dim // 32, 1)
    out = torch.zeros((tokens, model_dim), dtype=torch.float32, device=a.device)
    for e in row_expert[valid].unique().tolist():
        rows = torch.nonzero(valid & (row_expert == e)).squeeze(1)
        w = fp4_utils.mxfp4_to_f32(w2_qt[e]).view(model_dim, inter_dim // 32, 32)
        w = (w * w_scale[e]).view(model_dim, inter_dim)
        y = (a[rows] @ w.t()) * sorted_weights[rows].view(-1, 1)
        out.index_add_(0, token[rows], y)
    return out.to(dtypes.bf16)


def _cos_diff(x, y):
    x, y = x.double(), y.double()
    return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())


def quantize_groups(x, group=128):
    """fp8 FNUZ quant with one f32 scale per (row, ``group`` columns); same formula
    as quantize_rows / gemm1's q8 epilogues. Returns (fp8 [R, C], f32 [R, C/group])."""
    rows, cols = x.shape
    xg = x.float().view(rows, cols // group, group)
    amax = xg.abs().amax(-1, keepdim=True)
    s = torch.where(amax == 0, torch.ones_like(amax), amax / 240.0)
    q = (xg * (1.0 / s)).clamp(-240, 240).to(torch.float8_e4m3fnuz)
    return q.view(rows, cols), s.view(rows, cols // group).contiguous()


@benchmark()
def bench_a8w4_gemm2(
    token,
    model_dim,
    inter_dim,
    expert,
    topk,
    routing="uniform",
    tile_m=64,
    tile_n=128,
    b_cache_mod=2,
    xcd_swizzle=0,
    flush_tiny=True,
    waves_per_eu=None,
    epilog="lds",
    dyn_rows=False,
    a_group=0,
    tile_order="m",
    n_tiles=0,
    stage_rows=0,
):
    tile_k = 256 if inter_dim % 256 == 0 else 128
    device = "cuda"
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # ---- routing: same construction as test_a8w4_gemm1 --------------------------
    inp = torch.randn((token, model_dim), dtype=dtypes.bf16, device=device)
    score = torch.randn((token, expert), dtype=dtypes.bf16, device=device)
    if routing == "hot":
        hot = torch.randperm(expert, device=device)[: max(1, expert // HOT_EXPERT_DIV)]
        score[:, hot] += HOT_EXPERT_BIAS
    else:
        assert routing == "uniform", f"unknown routing {routing!r}"
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    w2 = torch.randn((expert, model_dim, inter_dim), dtype=dtypes.bf16, device=device)
    w2_qt, w2_scale = aiter.get_torch_quant(QuantType.per_1x32)(
        w2, quant_dtype=dtypes.fp4x2
    )
    del w2
    w2_qt = w2_qt.view(expert, model_dim, inter_dim // 2)
    w2_u8 = shuffle_weight_a16w4(w2_qt, 16, False).view(torch.uint8)
    w2_scale_u8 = (
        shuffle_scale_a16w4(w2_scale, expert, False)
        .view(torch.uint8)
        .contiguous()
        .view(-1)
    )

    # ---- per-call preprocessing (not timed): routing sort, A quant ---------------
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, expert, model_dim, dtypes.bf16, tile_m
    )[:5]
    sorted_size = sorted_expert_ids.numel() * tile_m
    # Stage1 output stand-in: bf16 intermediate by sorted position, row-quantized
    # like the production path (only the num_valid_ids rows).
    inter = torch.randn((sorted_size, inter_dim), dtype=dtypes.bf16, device=device)
    if a_group:
        a_fp8, a_scale = quantize_groups(inter, a_group)
    else:
        a_fp8, a_scale = quantize_rows(inter, num_valid_ids)

    ref = run_torch(
        a_fp8,
        a_scale,
        w2_qt,
        w2_scale,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        token,
        tile_m,
    )

    def _out():
        return torch.zeros((token, model_dim), dtype=dtypes.bf16, device=device)

    out_new, out_port, out_fd1, out_fd2 = _out(), _out(), _out(), _out()
    w2_fd = {fd: prep_w2_fast(w2_u8, w2_scale_u8, fd) for fd in (1, 2)}

    def _standalone(out, w, ws, fd):
        return lambda: a8w4_gemm2(
            a_fp8,
            a_scale,
            w,
            ws,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            out,
            num_experts=expert,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            b_cache_mod=b_cache_mod,
            xcd_swizzle=xcd_swizzle,
            flush_tiny=flush_tiny,
            waves_per_eu=waves_per_eu,
            epilog=epilog,
            dyn_rows=dyn_rows,
            a_group=a_group,
            tile_order=tile_order,
            fast_dec=fd,
            n_tiles=n_tiles,
            stage_rows=stage_rows,
        )

    candidates = {
        # this file's standalone kernel
        "a8w4_gemm2": (out_new, _standalone(out_new, w2_u8, w2_scale_u8, 0)),
        # production a16w4 port built for fp8r (baseline)
        "port": (
            out_port,
            lambda: flydsl_a16w4_gemm2(
                inter_sorted_bf16=a_fp8,
                w2_u8=w2_u8,
                w2_scale_u8=w2_scale_u8,
                sorted_expert_ids=sorted_expert_ids,
                cumsum_tensor=num_valid_ids,
                sorted_token_ids=sorted_ids,
                sorted_weights=sorted_weights,
                flat_out=out_port.view(-1),
                M_logical=token,
                max_sorted=sorted_size,
                NE=expert,
                D_HIDDEN=model_dim,
                D_INTER=inter_dim,
                topk=topk,
                tile_m=tile_m,
                tile_n=tile_n,
                tile_k=tile_k,
                b_nt=b_cache_mod,
                xcd_swizzle=xcd_swizzle,
                epilog="atomic",
                a_dtype="fp8r",
                a_scale=a_scale,
                flush_tiny=flush_tiny,
            ),
        ),
        # fast decode on prep_w2_fast weights (1: nibble/sign, 2: + scale deltas)
        "a8w4_fd1": (out_fd1, _standalone(out_fd1, *w2_fd[1], 1)),
        "a8w4_fd2": (out_fd2, _standalone(out_fd2, *w2_fd[2], 2)),
    }
    if a_group:  # the port only takes per-row A scales
        del candidates["port"]

    routed = token * topk
    n_experts_hit = int(topk_ids.unique().numel())
    flops = 2 * routed * model_dim * inter_dim
    nbytes = (
        n_experts_hit * model_dim * (inter_dim // 2 + inter_dim // 32)  # W2 + scale
        + routed * inter_dim  # fp8 A
        + token * model_dim * 2  # bf16 out
    )

    ret = {"gfx": get_gfx()}
    tok_per_expert = torch.bincount(topk_ids.flatten(), minlength=expert)
    ret["experts hit"] = n_experts_hit
    ret["max tok/expert"] = int(tok_per_expert.max())
    ret["m-blocks"] = int(num_valid_ids[0]) // tile_m
    ref_v = ref.to(dtypes.fp32)
    for name, (out, fn) in candidates.items():
        # Atomic accumulation: check one launch into a zeroed buffer (this also
        # JIT-compiles outside the timed region), then time repeated launches.
        out.zero_()
        fn()
        torch.cuda.synchronize()
        out_v = out.to(dtypes.fp32)
        # Output is a bf16 atomic sum of topk partials (each add rounds to bf16), so
        # the tolerance is looser than gemm1's: |ref| ~ 5 mean / 33 max here.
        err = checkAllclose(
            ref_v,
            out_v,
            rtol=2e-2,
            atol=1e-1,
            tol_err_ratio=TOL_ERR_RATIO,
            msg=f"{name}: a8w4 gemm2 ",
        )
        _, us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
        ret[f"{name} cos"] = _cos_diff(ref_v, out_v)
    return ret


@pytest.mark.skipif(
    get_gfx() not in SUPPORTED_GFX, reason="A8W4 needs gfx942 FP8-FNUZ MFMA"
)
@pytest.mark.parametrize("model_dim,inter_dim,expert,topk", [(4096, 256, 256, 8)])
@pytest.mark.parametrize("token", [512])
@pytest.mark.parametrize("tile_m", [64, 32])
@pytest.mark.parametrize("dyn_rows", [False, True])
@pytest.mark.parametrize("routing", ROUTINGS)
@pytest.mark.parametrize("a_group", [0, 128])
def test_a8w4_gemm2(
    token, model_dim, inter_dim, expert, topk, tile_m, dyn_rows, routing, a_group
):
    """Mimo v2.6 stage2 shape: standalone kernel and port vs fp32 reference."""
    ret = bench_a8w4_gemm2(
        token,
        model_dim,
        inter_dim,
        expert,
        topk,
        routing=routing,
        tile_m=tile_m,
        dyn_rows=dyn_rows,
        a_group=a_group,
    )
    names = ("a8w4_gemm2", "a8w4_fd1", "a8w4_fd2")
    for name in names if a_group else names + ("port",):
        assert ret[f"{name} err"] <= TOL_ERR_RATIO, f"{name} mismatch {ret}"
        assert ret[f"{name} cos"] < 1e-4, f"{name} cos diff too large {ret}"


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("a8w4 gemm2 unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="A8W4 MoE stage2 (down-proj + atomic scatter) correctness / perf",
    )
    parser.add_argument(
        "-t",
        "--token",
        type=int,
        nargs="*",
        default=[128, 256, 512, 1024, 2048],
        help="token counts",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(4096, 256, 256, 8)],
        help="model_dim,inter_dim,expert,topk  e.g. -s 4096,256,256,8",
    )
    parser.add_argument(
        "--tile",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(64, 128, 1), (32, 128, 1)],
        help="tile_m,tile_n[,dyn_rows]  e.g. --tile 64,128 64,128,1 (dyn_rows: skip\n"
        "all-padding 16-row sub-blocks of each sorted block at runtime)",
    )
    parser.add_argument(
        "-r",
        "--routing",
        type=str,
        nargs="*",
        choices=ROUTINGS,
        default=ROUTINGS,
        help="uniform: random top-k; hot: a few hot experts own most routed rows",
    )
    parser.add_argument(
        "--a_group",
        type=int,
        nargs="*",
        default=[0],
        help="A scale granularity: 0 = per row, 128 = per (row, 128-K group)",
    )
    parser.add_argument(
        "--n_tiles",
        type=int,
        nargs="*",
        default=[0],
        help="n-blocks per CTA (0 = one tile per CTA); needs one K tile, e.g. 2 4",
    )
    parser.add_argument(
        "--stage_rows",
        type=int,
        nargs="*",
        default=[0],
        help="n_tiles>1 lds epilogue: rows staged per chunk (0 = tile_m), e.g. 16 32",
    )
    args = parser.parse_args()

    df = []
    for (model_dim, inter_dim, expert, topk), token, routing, tile, a_group, nt, sr in (
        itertools.product(
            args.shape, args.token, args.routing, args.tile, args.a_group,
            args.n_tiles, args.stage_rows,
        )
    ):
        if nt <= 1 and sr != args.stage_rows[0]:
            continue  # stage_rows only matters for n_tiles > 1
        tile_m, tile_n = tile[:2]
        df.append(
            bench_a8w4_gemm2(
                token,
                model_dim,
                inter_dim,
                expert,
                topk,
                routing=routing,
                tile_m=tile_m,
                tile_n=tile_n,
                dyn_rows=bool(tile[2]) if len(tile) > 2 else False,
                a_group=a_group,
                n_tiles=nt,
                stage_rows=sr if nt > 1 else 0,
            )
        )
    df = pd.DataFrame(df)
    aiter.logger.info("a8w4 gemm2 summary (markdown):\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
