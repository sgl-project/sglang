# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused A8W4 MoE (gemm1 + SiLU + fp8 quant + gemm2 + atomic scatter) in ONE
kernel, one CTA per BM=64 expert block, targeting 2 CTAs per CU. gfx942.

Why: at large token counts gemm2's bf16x2 atomics (~1.5 TB/s ceiling) are no
longer hidden by gemm2's own compute (doc-mimo/experiments/gemm2_bm128). Fusing
both stages per expert block lets the atomics of one CTA overlap the gemm1 of the
other CTA on the same CU, and the intermediate never goes through HBM.

Per CTA (sorted rows m_row .. m_row+63, one expert, 4 waves, 1x4 N split):
  gemm1, twice (inter half nb = 0, 1; each wave 32 gate + 32 up columns):
    K loop over H (TILE_K 256) with single-slot A LDS + register prefetch of
    the next tile (A gathered by token, W1 fd2), acc gate/up.
    dequant (x row scale, channel factors) -> SiLU(g)*u -> bf16 staging in LDS
    -> per-(row, 128 col) fp8 quant (same numerics as gemm1's q8s epilogue) ->
    fp8 rows into an 8 KB LDS slot (XOR-swizzled) + scale -> every wave pulls
    the gemm2 A fragments of this K half into VGPRs.
  gemm2: 32 n-tiles of 128 columns, A (64 x 256 fp8) in VGPRs, W2 fd2 prefetch
    distance 1, single accumulator rescaled by s0/s1 at the K-group boundary,
    epilogue of tile t-1 interleaved into tile t's K steps, LDS-free 64 B-segment
    bf16x2 atomics (as test_a8w4_gemm2_bm128).
LDS: A tile / bf16 staging 16 KB + fp8 half 8 KB + scales 0.5 KB = 24.5 KB.

    python doc-mimo/kernels/test_a8w4_fmoe_bm64.py -t 2048 8192
"""

import argparse
import collections
import functools
import os
import sys
from collections import defaultdict

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from . import test_a8w4_gemm1 as G1  # noqa: E402
from . import test_a8w4_gemm2 as G2  # noqa: E402
from . import test_a8w4_gemm2_bm128 as B  # noqa: E402
from flydsl._mlir.dialects import arith, llvm  # noqa: E402

import aiter  # noqa: E402
from aiter import ActivationType, QuantType, dtypes  # noqa: E402
from aiter.fused_moe import fused_topk, moe_sorting, torch_moe_stage1, torch_moe_stage2  # noqa: E402
from aiter.ops.flydsl.kernels import buffer_ops as bo  # noqa: E402
from aiter.ops.flydsl.kernels.act import silu_mul_batch  # noqa: E402
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.a8w4 import _pack_activation, quantize_rows  # noqa: E402
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.utils import _global_i32_at, _udiv, _umod  # noqa: E402
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import lds_typed_ptr, lds_vec_load  # noqa: E402
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled  # noqa: E402
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw  # noqa: E402
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4  # noqa: E402
from aiter.test_common import run_perftest  # noqa: E402

BM = 64
NUM_WAVES = 4
TILE_N1 = 128  # gemm1 inter columns per half (gate and up each)
TILE_K1 = 256
TILE_N2 = 128
N_REPEAT = 2  # 16-col blocks per wave (32 columns)
K_STEPS = 8  # MFMA K32 steps per 256-K tile
K_GROUPS = 2  # 128-K groups per tile
LDS_A = BM * TILE_K1  # 16 KB: A tile, then the bf16 SiLU staging (BM x 128 x 2)
LDS_H = BM * 128  # 8 KB: one quantized fp8 half
LDS_BYTES = LDS_A + LDS_H + BM * 2 * 4
# ablation knobs (experiment only; use a fresh FLYDSL_RUNTIME_CACHE_DIR per value):
#   FMOE_ABL=nog2   skip gemm2 (gemm1 + quant only, output wrong)
#   FMOE_LDS_PAD=n  add n bytes of LDS (e.g. 40000 -> 1 CTA per CU)
#   G2_NO_ATOMIC=1  (test_a8w4_gemm2_bm128) skip the atomics
ABL = os.environ.get("FMOE_ABL", "")
ABLS = set(ABL.split(",")) - {""}  # also: nodec (gemm1 W as raw fp8), nomma (no gemm1 MFMA)
LDS_PAD = int(os.environ.get("FMOE_LDS_PAD", "0"))
DELAY = int(os.environ.get("FMOE_DELAY", "0"))  # x s_sleep(127) (~8k cycles each)
# gemm1 K loop: 1 = next-tile loads spread over the K steps, W read lazily (default);
# 0 = all loads up front after copying W out (the step-1 version)
G1_SPREAD = os.environ.get("FMOE_G1SPREAD", "1") == "1"
# FMOE_PROBE=1: every CTA writes i64 [start, gemm2 start, end, hw_id | xcc_id << 32]
# (s_memrealtime, 100 MHz) at dbg[m_block * 4 ..] (see --probe)
PROBE = os.environ.get("FMOE_PROBE", "0") == "1"


def _now():
    return fx.Int64(llvm.call_intrinsic(T.i64, "llvm.amdgcn.s.memrealtime", [], [], []))


def _getreg(simm16):
    return fx.Int32(llvm.call_intrinsic(T.i32, "llvm.amdgcn.s.getreg",
                                        [_raw(fx.Int32(simm16))], [], []))


def _rnu_hi(x):
    """f32 -> i32 whose high half is bf16(x), rounded to nearest, ties away from
    zero (one add; differs from RNE only on exact ties)."""
    return fx.Float32(x).bitcast(fx.Int32) + fx.Int32(0x8000)


def _atomics4_saddr(datas, offs, out_i64):
    """bf16x2 atomic adds of ``datas[k]`` at ``out + offs[k]`` bytes (offs >= 0,
    per-tile i32, so the zext stays next to the atomic -> SGPR-base form)."""
    base = G2._global_base_ptr1(out_i64)
    for k in range_constexpr(len(offs)):
        if const_expr(B.NO_ATOMIC):
            continue
        off64 = arith.ExtUIOp(T.i64, offs[k]).result
        pk = Vec.from_elements([fx.Int32(datas[k])], fx.Int32).bitcast(fx.BFloat16)
        llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd, bo.get_element_ptr(base, byte_offset=off64),
                         _raw(pk), llvm.AtomicOrdering.monotonic, syncscope="agent",
                         alignment=4)


@flyc.jit
def _atomics_mi(datas, offs, col, out_i64, full, part):
    """One 16-row block's atomics at ``out + offs[k] + col``. Valid rows are a prefix
    of the expert block, so ``full`` (all 16 rows valid) and ``part`` (some valid)
    are wave-uniform: full blocks need no per-atomic exec masking; padding rows have
    offs < 0."""
    if full:
        _atomics4_saddr(datas, [_raw(fx.Int32(o) + col) for o in offs], out_i64)
    else:
        if part:
            base = G2._global_base_ptr1(out_i64)
            for k in range_constexpr(len(offs)):
                off = fx.Int32(offs[k])
                if off >= fx.Int32(0):
                    off64 = arith.ExtUIOp(T.i64, _raw(off + col)).result
                    pk = Vec.from_elements([fx.Int32(datas[k])], fx.Int32).bitcast(fx.BFloat16)
                    llvm.AtomicRMWOp(llvm.AtomicBinOp.fadd,
                                     bo.get_element_ptr(base, byte_offset=off64), _raw(pk),
                                     llvm.AtomicOrdering.monotonic, syncscope="agent",
                                     alignment=4)


@functools.cache
def compile_a8w4_fmoe_bm64(*, D_HIDDEN, D_INTER, NE, TOPK=8, b_cache_mod=2,
                           waves_per_eu=2, dyn_rows=True, xcd=True, ksplit=1):
    H, INTER = D_HIDDEN, D_INTER
    assert INTER % 256 == 0 and H % TILE_K1 == 0 and H % TILE_N2 == 0
    NH = INTER // 128  # gemm1 column chunks of 128 = gemm2 K groups (one scale each)
    # ksplit > 1: the NH chunks of an expert block are split over ksplit CTAs; each
    # runs gemm1 for its chunks and gemm2 over that K slice (partial sums, atomics)
    assert NH % ksplit == 0
    CH = NH // ksplit  # chunks per CTA
    K2_STEPS = CH * 4  # gemm2 MFMA K steps (A2 fragments per row block)
    LDS_BYTES_L = LDS_A + LDS_H + BM * CH * 4
    N1 = 2 * INTER  # gate | up rows of W1
    K1_TILES = H // TILE_K1
    NT2 = H // TILE_N2
    assert NT2 % 2 == 0
    W1_K0 = H // 128
    SC1_STRIDE = (H // 256) * 64
    W2_K0 = INTER // 128
    SC2_STRIDE = (INTER // 256) * 64
    WS2_BYTES = NE * H * (INTER // 32)
    kname = (f"a8w4_fmoe_bm64_h{H}_i{INTER}_ne{NE}_k{TOPK}_bcm{b_cache_mod}"
             f"_w{waves_per_eu}{'_dyn' if dyn_rows else ''}{'_xcd' if xcd else ''}"
             f"{f'_ks{ksplit}' if ksplit > 1 else ''}"
             f"{'_' + ABL.replace(',', '_') if ABL else ''}{f'_lp{LDS_PAD}' if LDS_PAD else ''}"
             f"{'_noat' if B.NO_ATOMIC else ''}{f'_dl{DELAY}' if DELAY else ''}"
             f"{'' if G1_SPREAD else '_g1up'}{'_probe' if PROBE else ''}")

    def _body(lds_ptr, arg_x, arg_xs, arg_w1, arg_ws1, arg_exp1, arg_w2, arg_ws2,
              arg_eids, arg_stids, arg_sw, arg_out, i32_M, m_block, lane, wave, arg_dbg, c0):
        tx = fx.Int32(gpu.thread_id("x"))
        lane_div_16 = lane // fx.Int32(16)
        lane_mod_16 = lane % fx.Int32(16)
        e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block)))
        m_row = m_block * fx.Int32(BM)
        n_wave_base = fx.Int32(wave) * fx.Int32(32)
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))
        lds_i32 = fx.Int32(fx.ptrtoint(lds_ptr))

        rs_x = bo.create_buffer_resource_from_addr(
            _raw(fx.Int64(arg_x)), num_records_bytes=_raw(fx.Int64(i32_M) * fx.Int64(H)))
        rs_xs = bo.create_buffer_resource_from_addr(
            _raw(fx.Int64(arg_xs)), num_records_bytes=_raw(fx.Int64(i32_M) * fx.Int64(4)))
        rs_w1, rs_ws1, rs_e1 = B._rsrc(arg_w1), B._rsrc(arg_ws1), B._rsrc(arg_exp1)
        rs_w2, rs_ws2 = B._rsrc(arg_w2), B._rsrc(arg_ws2)
        rs_e2 = bo.create_buffer_resource_from_addr(
            _raw(fx.Int64(arg_ws2) + fx.Int64(WS2_BYTES)))
        rs_sw, rs_id = B._rsrc(arg_sw), B._rsrc(arg_stids)

        def ld(rs, off_i32, n=1):
            v = bo.buffer_load(rs, off_i32, vec_width=n, dtype=T.i32)
            return fx.Int32(v) if n == 1 else Vec(v)

        def run(rows):
            MR = rows // 16
            A_LOADS = rows * TILE_K1 // (256 * 16)

            # ---- A (x rows gathered by token) copy slots, K-invariant --------------
            # XOR-swizzled LDS columns (as test_a8w4_gemm1): source column swizzled,
            # destination linear; reads swizzle the column again.
            def swz(row, col_bytes):
                return col_bytes ^ ((row & fx.Int32(TILE_K1 // 16 - 1)) * fx.Int32(16))

            a_src, a_dst = [], []
            for i in range_constexpr(A_LOADS):
                idx_dw = tx * fx.Int32(4) + fx.Int32(i * 256 * 4)
                row = idx_dw // fx.Int32(TILE_K1 // 4)
                col_b = (idx_dw % fx.Int32(TILE_K1 // 4)) * fx.Int32(4)
                tok = ld(rs_id, m_row + row) & fx.Int32(0x00FFFFFF)
                a_src.append((tok * fx.Int32(H) + swz(row, col_b)) // fx.Int32(4))
                a_dst.append((row * fx.Int32(TILE_K1) + col_b) // fx.Int32(16))
            a_regs = [fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32) for _ in range(A_LOADS)]
            lds_flat = fx.make_view(fx.recast_iter(fx.Int32, lds_ptr),
                                    fx.make_layout(LDS_BYTES_L // 4, 1))
            lds16 = fx.logical_divide(lds_flat, fx.make_layout(4, 1))
            lds8 = fx.logical_divide(lds_flat, fx.make_layout(2, 1))
            wr_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
            rd_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)

            def load_a_global(base_k):
                for i in range_constexpr(A_LOADS):
                    load_a_one(i, base_k)

            def load_a_one(i, base_k):
                a_regs[i].store(Vec(bo.buffer_load(
                    rs_x, a_src[i] + base_k // fx.Int32(4), vec_width=4, dtype=T.i32)))

            def store_a_lds():
                for i in range_constexpr(A_LOADS):
                    fx.copy(wr_atom, a_regs[i], fx.slice(lds16, (None, a_dst[i])))

            def load_a(mi, ku):
                row = lane_mod_16 + fx.Int32(mi * 16)
                col = lane_div_16 * fx.Int32(32) + fx.Int32((ku // 4) * 128 + (ku % 4) * 8)
                off = (row * fx.Int32(TILE_K1) + swz(row, col)) // fx.Int32(8)
                r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
                fx.copy_atom_call(rd_atom, fx.slice(lds8, (None, off)), r)
                return fx.Vector(fx.memref_load_vec(r)).bitcast(fx.Float8E4M3FNUZ)

            # x row scale per MFMA-layout row (token; padding rows read 0)
            xs = []
            for mi in range_constexpr(MR):
                for ii in range_constexpr(4):
                    tok = ld(rs_id, m_row + fx.Int32(mi * 16 + ii) + lane_div_16 * fx.Int32(4)) \
                        & fx.Int32(0x00FFFFFF)
                    xs.append(ld(rs_xs, tok).bitcast(fx.Float32))

            a2 = {}  # gemm2 A fragments: (mi, ku) -> Vec2 i32
            for jc in range_constexpr(CH):
                nb = c0 + fx.Int32(jc)  # absolute 128-column chunk
                # ---- gemm1 W1 columns (gate rows n, up rows INTER + n) -------------
                w1_base, sc1_base = {}, {}
                n_cols = []
                for ni in range_constexpr(N_REPEAT):
                    n_local = nb * fx.Int32(TILE_N1) + fx.Int32(ni * 16) + n_wave_base
                    n_cols.append(n_local + lane_mod_16)
                    for gu in range_constexpr(2):
                        n0 = e * fx.Int32(N1) + n_local + fx.Int32(gu * INTER)
                        w1_base[gu, ni] = (n0 // fx.Int32(16)) * fx.Int32(W1_K0 * 64) \
                            + lane_div_16 * fx.Int32(16) + lane_mod_16
                        if ni == 0:
                            sc1_base[gu] = (n0 // fx.Int32(32)) * fx.Int32(SC1_STRIDE) \
                                + lane_div_16 * fx.Int32(16) + lane_mod_16
                w_regs = {(gu, ni, g): fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                          for gu in range(2) for ni in range(N_REPEAT) for g in range(K_GROUPS)}
                s_regs = {gu: fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32) for gu in range(2)}

                def load_w_global(base_k):
                    load_s(base_k)
                    for (gu, ni, g) in w_regs:
                        load_w_one(gu, ni, g, base_k)

                def load_s(base_k):
                    for gu in range_constexpr(2):
                        s_regs[gu].store(Vec.from_elements([ld(
                            rs_ws1, sc1_base[gu] + (base_k // fx.Int32(256)) * fx.Int32(64))],
                            fx.Int32))

                def load_w_one(gu, ni, g, base_k):
                    idx = w1_base[gu, ni] + (base_k // fx.Int32(128) + fx.Int32(g)) * fx.Int32(64)
                    w_regs[gu, ni, g].store(Vec(bo.buffer_load(
                        rs_w1, idx * fx.Int32(4), vec_width=4, dtype=T.i32,
                        cache_modifier=b_cache_mod)))

                def read_w():
                    raw = {k: [fx.Int32(Vec(r.load())[j]) for j in range(4)]
                           for k, r in w_regs.items()}
                    luts = {}
                    for gu in range_constexpr(2):
                        p = fx.Int32(Vec(s_regs[gu].load())[0])
                        for ni in range_constexpr(N_REPEAT):
                            for g in range_constexpr(K_GROUPS):
                                luts[gu, ni, g] = G1._fp8_lut_predelta(
                                    G1._perm_u32(p, p, ((g % 2) * 2 + ni) * 0x01010101))
                    return raw, luts

                acc = {(gu, mi, ni): fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
                       for gu in range(2) for mi in range(MR) for ni in range(N_REPEAT)}
                zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
                for t_ in acc.values():
                    t_.store(zero4)

                def compute_tile(w_tile, nxt=None):
                    """One 256-K tile. ``w_tile`` None: read W lazily from w_regs and,
                    if ``nxt`` is given, issue tile nxt's loads spread over the K steps
                    (A at K steps 0.., W group 0 once consumed at 4..7, group 1 last)."""
                    if const_expr(w_tile is None):
                        luts = {}
                        for gu in range_constexpr(2):
                            p = fx.Int32(Vec(s_regs[gu].load())[0])
                            for ni in range_constexpr(N_REPEAT):
                                for g in range_constexpr(K_GROUPS):
                                    luts[gu, ni, g] = G1._fp8_lut_predelta(
                                        G1._perm_u32(p, p, ((g % 2) * 2 + ni) * 0x01010101))
                        if const_expr(nxt is not None):
                            load_s(nxt)
                        raw = None
                    else:
                        raw, luts = w_tile
                    w_items = [(gu, ni) for gu in range(2) for ni in range(N_REPEAT)]

                    def raw_at(gu, ni, g, j):
                        if const_expr(raw is None):
                            return fx.Int32(Vec(w_regs[gu, ni, g].load())[j])
                        return raw[gu, ni, g][j]

                    for ku in range_constexpr(K_STEPS):
                        g, j = ku // 4, ku % 4
                        if const_expr("nodec" in ABLS):
                            bs = {(gu, ni): Vec.from_elements(
                                [raw_at(gu, ni, g, j)] * 2, fx.Int32).bitcast(
                                    fx.Float8E4M3FNUZ)
                                for gu in range(2) for ni in range(N_REPEAT)}
                        else:
                            bs = {(gu, ni): G1._fp4x8_to_fp8x8_fast(raw_at(gu, ni, g, j),
                                                                     luts[gu, ni, g])
                                  for gu in range(2) for ni in range(N_REPEAT)}
                        for mi in range_constexpr(MR if "nomma" not in ABLS else 0):
                            a8 = load_a(mi, ku)
                            for ni in range_constexpr(N_REPEAT):
                                for gu in range_constexpr(2):
                                    G1._mma_fp8(mma_atom, acc[gu, mi, ni], a8, bs[gu, ni])
                        if const_expr(nxt is not None):
                            if const_expr(ku < A_LOADS):
                                load_a_one(ku, nxt)
                            if const_expr(ku >= K_STEPS - len(w_items)):
                                gu_, ni_ = w_items[ku - (K_STEPS - len(w_items))]
                                load_w_one(gu_, ni_, 0, nxt)
                    if const_expr(nxt is not None):
                        for gu_, ni_ in w_items:
                            load_w_one(gu_, ni_, 1, nxt)

                # ---- K loop: single A slot, registers prefetch tile kt+1 ----------
                load_a_global(fx.Int32(0))
                load_w_global(fx.Int32(0))
                store_a_lds()

                @flyc.jit
                def _k_loop(k_first):
                    for kt in range(K1_TILES - 1):
                        nxt = (k_first + fx.Int32(kt) + fx.Int32(1)) * fx.Int32(TILE_K1)
                        rocdl.s_waitcnt(lgkmcnt=0)
                        gpu.barrier()  # A(kt) visible
                        if const_expr(G1_SPREAD):
                            compute_tile(None, nxt)
                        else:
                            w_tile = read_w()
                            load_a_global(nxt)
                            load_w_global(nxt)
                            rocdl.sched_barrier(0)
                            compute_tile(w_tile)
                        gpu.barrier()  # A(kt) reads done
                        store_a_lds()

                _k_loop(fx.Int32(0))
                rocdl.s_waitcnt(lgkmcnt=0)
                gpu.barrier()
                compute_tile(None if G1_SPREAD else read_w())
                gpu.barrier()  # A reads done: the A slot becomes the bf16 staging

                # ---- dequant -> SiLU -> bf16 staging [rows, 128] ---------------------
                f_g = [G1._channel_factor(ld(rs_e1, e * fx.Int32(N1) + c)) for c in n_cols]
                f_u = [G1._channel_factor(ld(rs_e1, e * fx.Int32(N1) + fx.Int32(INTER) + c))
                       for c in n_cols]
                stg = lds_typed_ptr(lds_i32, T.bf16)
                for mi in range_constexpr(MR):
                    ag = [Vec(acc[0, mi, ni].load()) for ni in range(N_REPEAT)]
                    au = [Vec(acc[1, mi, ni].load()) for ni in range(N_REPEAT)]
                    for ii in range_constexpr(4):
                        rs = xs[mi * 4 + ii]
                        row = fx.Int32(mi * 16 + ii) + lane_div_16 * fx.Int32(4)
                        for ni in range_constexpr(N_REPEAT):
                            y = silu_mul_batch([ag[ni][ii] * (rs * f_g[ni])],
                                               [au[ni][ii] * (rs * f_u[ni])])[0]
                            col = n_wave_base + fx.Int32(ni * 16) + lane_mod_16
                            stg[row * fx.Int32(TILE_N1) + col] = y.to(fx.BFloat16)
                gpu.barrier()

                # ---- per-(row, 128) fp8 quant -> LDS half slot (swizzled) + scale -----
                r_in = tx // fx.Int32(8)
                sub = tx % fx.Int32(8)
                sc_f = lds_typed_ptr(lds_i32 + fx.Int32(LDS_A + LDS_H), T.f32)
                for p in range_constexpr(rows // 32):
                    row = fx.Int32(p * 32) + r_in
                    vals = []
                    for h in range_constexpr(2):
                        v = Vec(lds_vec_load(
                            lds_i32, (row * fx.Int32(TILE_N1) + sub * fx.Int32(16)) * fx.Int32(2)
                            + fx.Int32(h * 16), Vec.make_type(8, fx.BFloat16), fx.BFloat16,
                            align=16)).to(fx.Float32)
                        vals.extend(v[k] for k in range(8))
                    amax = fx.Float32(0.0)
                    for k in range_constexpr(16):
                        amax = fx.max(amax, fx.max(vals[k], -vals[k]))
                    for bit in range_constexpr(3):
                        amax = fx.max(amax, amax.shuffle_xor(1 << bit, 64))
                    s = (amax == 0.0).select(fx.Float32(1.0), amax / 240.0)
                    inv = 1.0 / s
                    packed = Vec.from_elements(
                        [_pack_activation(*[vals[4 * q + k] * inv for k in range(4)])
                         for q in range(4)], fx.Int32)
                    dst = (row * fx.Int32(128) + ((sub ^ (row & fx.Int32(7))) * fx.Int32(16))) \
                        // fx.Int32(16)
                    r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    r.store(packed)
                    fx.copy(wr_atom, r, fx.slice(lds16, (None, fx.Int32(LDS_A // 16) + dst)))
                    sc_f[row * fx.Int32(CH) + fx.Int32(jc)] = s  # 8 lanes, same value
                rocdl.s_waitcnt(lgkmcnt=0)
                gpu.barrier()

                # ---- gemm2 A fragments of K half nb -> VGPRs -----------------------
                for mi in range_constexpr(MR):
                    row = lane_mod_16 + fx.Int32(mi * 16)
                    for jj in range_constexpr(4):
                        col = lane_div_16 * fx.Int32(32) + fx.Int32(jj * 8)
                        byte = row * fx.Int32(128) + (((col // fx.Int32(16)) ^ (row & fx.Int32(7)))
                                                      * fx.Int32(16)) + col % fx.Int32(16)
                        r = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
                        fx.copy_atom_call(rd_atom, fx.slice(
                            lds8, (None, (fx.Int32(LDS_A) + byte) // fx.Int32(8))), r)
                        a2[mi, jc * 4 + jj] = Vec(fx.memref_load_vec(r))

            # ---- row data (lane L = row L): ratios s_g/s_{g+1}, fin s_last * w, id ---
            # gemm2 keeps ONE accumulator per output: at K group boundary g -> g+1 it
            # is rescaled by s_g / s_{g+1}; the epilogue multiplies by s_last * w.
            sc_f = lds_typed_ptr(lds_i32 + fx.Int32(LDS_A + LDS_H), T.f32)
            sg = [sc_f[lane * fx.Int32(CH) + fx.Int32(g)] for g in range(CH)]
            sg = [sg[0]] + [(s == fx.Float32(0.0)).select(fx.Float32(1.0), s) for s in sg[1:]]
            w_r = ld(rs_sw, m_row + lane).bitcast(fx.Float32)
            ratio_l = [(sg[g] / sg[g + 1]).bitcast(fx.Int32) for g in range(CH - 1)]
            id_l = ld(rs_id, m_row + lane)
            # padding rows (a suffix of the block): fin 0, offset -1 (skipped)
            valid_l = (id_l & fx.Int32(0x00FFFFFF)) < i32_M
            fin_l = valid_l.select(sg[CH - 1] * w_r, fx.Float32(0.0)).bitcast(fx.Int32)

            def row_valid(r):  # wave-uniform
                t = fx.Int32(rocdl.readlane(T.i32, _raw(id_l), _raw(fx.Int32(r))))
                return (t & fx.Int32(0x00FFFFFF)) < i32_M

            def fetch(q, row_static, row_lane):
                src = fx.Int32(row_static) + row_lane
                return fx.Int32(rocdl.ds_bpermute(T.i32, src * fx.Int32(4), q))

            odd = lane & fx.Int32(1)
            is_odd = odd == fx.Int32(1)
            col_b = (odd * fx.Int32(16) + lane_mod_16 - odd) * fx.Int32(2)
            sel_pair = is_odd.select(fx.Int32(0x03020706), fx.Int32(0x05040100))
            ratio4, fin4, offs4 = [[] for _ in range(CH - 1)], [], []
            for mi in range_constexpr(MR):
                rr, ff, oo = [[] for _ in range(CH - 1)], [], []
                for v in range_constexpr(4):
                    for g in range_constexpr(CH - 1):
                        rr[g].append(fetch(ratio_l[g], mi * 16 + v,
                                           lane_div_16 * fx.Int32(4)).bitcast(fx.Float32))
                    ff.append(fetch(fin_l, mi * 16 + v, lane_div_16 * fx.Int32(4)).bitcast(fx.Float32))
                    tok = fetch(id_l, mi * 16 + v, lane_div_16 * fx.Int32(4)) & fx.Int32(0x00FFFFFF)
                    oo.append((tok < i32_M).select(tok * fx.Int32(H * 2) + col_b, fx.Int32(-1)))
                for g in range_constexpr(CH - 1):
                    ratio4[g].append(rr[g])
                fin4.append(ff)
                offs4.append(oo)
            mi_full = [row_valid(mi * 16 + 15) for mi in range(MR)]
            mi_part = [row_valid(mi * 16) for mi in range(MR)]
            if const_expr(PROBE):
                G1.global_typed_ptr(arg_dbg, T.i64, align=8)[
                    m_block * fx.Int32(4) + fx.Int32(1)] = _now()
            if const_expr("nog2" in ABLS):
                return

            def a_frag(mi, ku):
                return a2[mi, ku].bitcast(fx.Float8E4M3FNUZ)

            # ---- gemm2 over NT2 n-tiles: ping-pong acc sets, W2 prefetch 1 ----------
            wb = [{
                "raw": {(ni, g): fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        for ni in range(N_REPEAT) for g in range(CH)},
                "packed": fx.make_rmem_tensor(fx.make_layout(CH, 1), fx.Int32),
                "exp": fx.make_rmem_tensor(fx.make_layout(N_REPEAT, 1), fx.Int32),
            } for _ in range(2)]
            acc2 = [[[fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
                      for _ in range(N_REPEAT)] for _ in range(MR)] for _ in range(2)]
            fn2 = [fx.make_rmem_tensor(fx.make_layout(N_REPEAT, 1), fx.Float32) for _ in range(2)]
            zero4 = fx.Vector.filled(4, 0.0, fx.Float32)

            def issue_w2(t, b):
                n0 = e * fx.Int32(H) + t * fx.Int32(TILE_N2) + n_wave_base
                for ni in range_constexpr(N_REPEAT):
                    wbase = ((n0 + fx.Int32(ni * 16)) // fx.Int32(16)) * fx.Int32(W2_K0 * 64) \
                        + lane_div_16 * fx.Int32(16) + lane_mod_16
                    for g in range_constexpr(CH):
                        wb[b]["raw"][ni, g].store(Vec(bo.buffer_load(
                            rs_w2, (wbase + (c0 + fx.Int32(g)) * fx.Int32(64)) * fx.Int32(4),
                            vec_width=4, dtype=T.i32, cache_modifier=b_cache_mod)))
                scb = (n0 // fx.Int32(32)) * fx.Int32(SC2_STRIDE) + lane_div_16 * fx.Int32(16) \
                    + lane_mod_16
                # one scale dword per 256 K (4 bytes: (group % 2, ni)); one per local group
                wb[b]["packed"].store(Vec.from_elements(
                    [ld(rs_ws2, scb + ((c0 + fx.Int32(g)) // fx.Int32(2)) * fx.Int32(64))
                     for g in range(CH)], fx.Int32))
                wb[b]["exp"].store(Vec.from_elements(
                    [ld(rs_e2, n0 + fx.Int32(ni * 16) + lane_mod_16) for ni in range(N_REPEAT)],
                    fx.Int32))

            def epilogue_mi(s, t_prev, mi, ks=(0, 1, 2, 3)):
                fn = Vec(fn2[s].load())
                datas = []
                for k in ks:
                    r0 = _rnu_hi(Vec(acc2[s][mi][0].load())[k] * (fin4[mi][k] * fn[0]))
                    r1 = _rnu_hi(Vec(acc2[s][mi][1].load())[k] * (fin4[mi][k] * fn[1]))
                    pk = G1._perm_u32(r1, r0, 0x07060302)
                    nbr = fx.Int32(rocdl.update_dpp(T.i32, _raw(pk.bitcast(fx.Int32)),
                                                    _raw(pk.bitcast(fx.Int32)),
                                                    B.DPP_QUAD_SWAP, 0xF, 0xF, True))
                    datas.append(_raw(G1._perm_u32(nbr, pk, sel_pair).bitcast(fx.Int32)))
                col = (t_prev * fx.Int32(TILE_N2) + n_wave_base) * fx.Int32(2)
                _atomics_mi(datas, [_raw(offs4[mi][k]) for k in ks], col, arg_out,
                            mi_full[mi], mi_part[mi])

            def tile(t, s, b, prev, has_next):
                """n-tile t into acc set s from W buffer b; prev = (set, t_prev) whose
                epilogue rides along (row block ku at K step ku)."""
                rocdl.sched_barrier(0)
                if const_expr(has_next):
                    issue_w2(t + fx.Int32(1), 1 - b)
                raw = {k: [fx.Int32(Vec(r.load())[j]) for j in range(4)]
                       for k, r in wb[b]["raw"].items()}
                pk_s = Vec(wb[b]["packed"].load())
                ex = Vec(wb[b]["exp"].load())
                fn2[s].store(Vec.from_elements(
                    [G1._channel_factor(fx.Int32(ex[ni])) for ni in range(N_REPEAT)], fx.Float32))
                luts = []
                for ni in range_constexpr(N_REPEAT):
                    row_l = []
                    for g in range_constexpr(CH):
                        p = fx.Int32(pk_s[g])
                        sel = (((c0 + fx.Int32(g)) % fx.Int32(2)) * fx.Int32(2) + fx.Int32(ni)) \
                            * fx.Int32(0x01010101)
                        row_l.append(G1._fp8_lut_predelta(G1._perm_u32(p, p, sel)))
                    luts.append(row_l)
                for mi in range_constexpr(MR):
                    for ni in range_constexpr(N_REPEAT):
                        acc2[s][mi][ni].store(zero4)
                # previous tile's epilogue: its MR*4 row atomics spread over the K steps
                units = MR * 4
                epi_at = [[u for u in range(units) if u * K2_STEPS // units == ku]
                          for ku in range(K2_STEPS)]
                for ku in range_constexpr(K2_STEPS):
                    if const_expr(ku > 0 and ku % 4 == 0):  # K group boundary: rescale
                        g_prev = ku // 4 - 1
                        for mi in range_constexpr(MR):
                            for ni in range_constexpr(N_REPEAT):
                                a4 = Vec(acc2[s][mi][ni].load())
                                acc2[s][mi][ni].store(Vec.from_elements(
                                    [a4[v] * ratio4[g_prev][mi][v] for v in range(4)],
                                    fx.Float32))
                    if const_expr("g2nodec" in ABLS):
                        b8s = [Vec.from_elements([raw[ni, ku // 4][ku % 4]] * 2, fx.Int32)
                               .bitcast(fx.Float8E4M3FNUZ) for ni in range(N_REPEAT)]
                    else:
                        b8s = [G1._fp4x8_to_fp8x8_fast(raw[ni, ku // 4][ku % 4],
                                                        luts[ni][ku // 4])
                               for ni in range(N_REPEAT)]
                    for mi in range_constexpr(MR):
                        a8 = a_frag(mi, ku)
                        for ni in range_constexpr(N_REPEAT):
                            G1._mma_fp8(mma_atom, acc2[s][mi][ni], a8, b8s[ni])
                    if const_expr(prev is not None):
                        us = epi_at[ku]
                        for mi_ in range_constexpr(MR):
                            ks = tuple(u % 4 for u in us if u // 4 == mi_)
                            if const_expr(len(ks) > 0):
                                epilogue_mi(prev[0], prev[1], mi_, ks)

            issue_w2(fx.Int32(0), 0)
            tile(fx.Int32(0), 0, 0, None, True)

            @flyc.jit
            def _g2_loop(p_first):
                for p in range((NT2 - 2) // 2):
                    t1 = (p_first + fx.Int32(p)) * fx.Int32(2) + fx.Int32(1)
                    tile(t1, 1, 1, (0, t1 - fx.Int32(1)), True)
                    tile(t1 + fx.Int32(1), 0, 0, (1, t1), True)

            _g2_loop(fx.Int32(0))
            t_last = fx.Int32(NT2 - 1)
            tile(t_last, 1, 1, (0, t_last - fx.Int32(1)), False)
            for mi in range_constexpr(MR):
                epilogue_mi(1, t_last, mi)

        if const_expr(dyn_rows):
            tok32 = fx.Int32(rocdl.readfirstlane(
                T.i32, _raw(_global_i32_at(arg_stids, m_row + fx.Int32(32))))) \
                & fx.Int32(0x00FFFFFF)
            cond = tok32 >= i32_M  # rows 32.. are padding

            @flyc.jit
            def _dispatch():
                if cond:
                    run(32)
                else:
                    run(BM)

            _dispatch()
        else:
            run(BM)

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, LDS_BYTES_L + LDS_PAD, 16]

    @flyc.jit
    def _entry(arg_x, arg_xs, arg_w1, arg_ws1, arg_exp1, arg_w2, arg_ws2, arg_eids,
               arg_cumsum, arg_stids, arg_sw, arg_out, i32_M, arg_dbg):
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx // fx.Int32(64))
        total_m = _global_i32_at(arg_cumsum, fx.Int32(0)) // fx.Int32(BM)
        if const_expr(DELAY):
            # stagger experiment: CTAs 304..607 (presumably each CU's 2nd slot) start late
            if bx >= fx.Int32(304):
                if bx < fx.Int32(608):
                    for _ in range_constexpr(DELAY):
                        rocdl.s_sleep(127)
        if bx < total_m * fx.Int32(ksplit):
            lid = bx
            if const_expr(xcd):
                # WGs go round-robin over the 8 XCDs: give XCD x a contiguous range of
                # (m-block, split) work items so same-expert blocks share its L2.
                tot = total_m * fx.Int32(ksplit)
                xc = _umod(bx, 8)
                lid = (xc * _udiv(tot, 8) + fx.min(xc, _umod(tot, 8)) + _udiv(bx, 8))
            m_block = lid // fx.Int32(ksplit)
            c0 = (lid % fx.Int32(ksplit)) * fx.Int32(CH)
            if const_expr(PROBE):
                dbg = G1.global_typed_ptr(arg_dbg, T.i64, align=8)
                dbg[m_block * fx.Int32(4)] = _now()
                hw = _getreg(0xF804)  # HW_REG_HW_ID
                xc_id = _getreg(0xF814)  # HW_REG_XCC_ID
                dbg[m_block * fx.Int32(4) + fx.Int32(3)] = (
                    fx.Int64(arith.ExtUIOp(T.i64, _raw(hw)).result)
                    + fx.Int64(arith.ExtUIOp(T.i64, _raw(xc_id)).result) * fx.Int64(1 << 32))
            _body(lds_ptr, arg_x, arg_xs, arg_w1, arg_ws1, arg_exp1, arg_w2, arg_ws2,
                  arg_eids, arg_stids, arg_sw, arg_out, i32_M, m_block, lane, wave, arg_dbg, c0)
            if const_expr(PROBE):
                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                G1.global_typed_ptr(arg_dbg, T.i64, align=8)[
                    m_block * fx.Int32(4) + fx.Int32(2)] = _now()

    @flyc.kernel(name=kname, known_block_size=[256, 1, 1])
    def kernel(arg_x: fx.Int64, arg_xs: fx.Int64, arg_w1: fx.Int64, arg_ws1: fx.Int64,
               arg_exp1: fx.Int64, arg_w2: fx.Int64, arg_ws2: fx.Int64, arg_eids: fx.Int64,
               arg_cumsum: fx.Int64, arg_stids: fx.Int64, arg_sw: fx.Int64,
               arg_out: fx.Int64, i32_M: fx.Int32, arg_dbg: fx.Int64):
        _entry(arg_x, arg_xs, arg_w1, arg_ws1, arg_exp1, arg_w2, arg_ws2, arg_eids,
               arg_cumsum, arg_stids, arg_sw, arg_out, i32_M, arg_dbg)

    @flyc.jit
    def launch(arg_x: fx.Int64, arg_xs: fx.Int64, arg_w1: fx.Int64, arg_ws1: fx.Int64,
               arg_exp1: fx.Int64, arg_w2: fx.Int64, arg_ws2: fx.Int64, arg_eids: fx.Int64,
               arg_cumsum: fx.Int64, arg_stids: fx.Int64, arg_sw: fx.Int64,
               arg_out: fx.Int64, i32_M: fx.Int32, arg_dbg: fx.Int64, i32_grid: fx.Int32,
               stream: fx.Stream):
        kernel(arg_x, arg_xs, arg_w1, arg_ws1, arg_exp1, arg_w2, arg_ws2, arg_eids,
               arg_cumsum, arg_stids, arg_sw, arg_out, i32_M, arg_dbg,
               value_attrs={"rocdl.waves_per_eu": waves_per_eu},
               ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(256, 1, 1), stream=stream)

    return launch


def a8w4_fmoe_bm64(x_fp8, x_scale, weights, sorted_ids, sorted_weights,
                   sorted_expert_ids, num_valid_ids, out, *, topk=8, b_cache_mod=1,
                   waves_per_eu=2, dyn_rows=True, xcd=True, ksplit=1):
    """Fused stage1 + stage2. ``weights``: ``a8w4_prepare_weights(..., fast_dec=2)``
    (doc-mimo/kernels/test_flydsl_moe_a16wfp4.py); sorting block_m 64; ``out`` bf16
    [tokens, model_dim] zeroed (moe_sorting's moe_buf). ``ksplit``: CTAs per expert
    block (each takes inter_dim / ksplit columns; more CTAs for small batches)."""
    assert weights["w1_fast_dec"] == 2 and weights["w2_fast_dec"] == 2
    tokens, model_dim = out.shape
    E, _, _ = weights["w2"].shape
    launch = compile_a8w4_fmoe_bm64(
        D_HIDDEN=model_dim, D_INTER=weights["w1"].shape[1] // 2, NE=E, TOPK=topk,
        b_cache_mod=b_cache_mod, waves_per_eu=waves_per_eu, dyn_rows=dyn_rows, xcd=xcd,
        ksplit=ksplit)
    _run_compiled(
        launch, x_fp8.data_ptr(), x_scale.data_ptr(), weights["w1"].data_ptr(),
        weights["w1_scale"].data_ptr(), weights["w1_exp"].data_ptr(),
        weights["w2"].data_ptr(), weights["w2_scale"].data_ptr(),
        sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), sorted_ids.data_ptr(),
        sorted_weights.data_ptr(), out.data_ptr(), int(tokens),
        _probe_buf(sorted_expert_ids.numel()).data_ptr() if PROBE else 0,
        int(sorted_expert_ids.numel()) * ksplit, torch.cuda.current_stream())
    return out


_PROBE = {}


def _probe_buf(n):
    if _PROBE.get("buf") is None or _PROBE["buf"].numel() < n * 4:
        _PROBE["buf"] = torch.zeros(n * 4, dtype=torch.int64, device="cuda")
    return _PROBE["buf"]


def probe_report():
    """Phase timeline from the last FMOE_PROBE launch (s_memrealtime = 100 MHz)."""
    d = _PROBE["buf"].view(-1, 4).cpu()
    d = d[d[:, 2] != 0]
    n_blocks = d.shape[0]
    t0 = int(d[:, 0].min())
    st, g2, en = [(d[:, i] - t0).double() / 100.0 for i in range(3)]  # us
    span = float(en.max())
    g1d, g2d = (g2 - st), (en - g2)
    print(f"   probe: span {span:.1f} us; per CTA gemm1 {g1d.median():.1f} us, "
          f"gemm2 {g2d.median():.1f} us (p10/p90 {g2d.quantile(0.1):.1f}/{g2d.quantile(0.9):.1f})")
    nb = 20
    in_g1, in_g2 = [], []
    for k in range(nb):
        t = span * (k + 0.5) / nb
        in_g1.append(int(((st <= t) & (t < g2)).sum()))
        in_g2.append(int(((g2 <= t) & (t < en)).sum()))
    print(f"   CTAs in gemm1 over time: {in_g1}")
    print(f"   CTAs in gemm2 over time: {in_g2}")
    # CU pairing: CTAs whose lifetimes overlap on the same (xcc, se, sh, cu)
    hw = d[:, 3]
    cu = ((hw >> 32) * 4096 + ((hw >> 8) & 0xF) + ((hw >> 12) & 1) * 16 + ((hw >> 13) & 7) * 32)
    first = st < span * 0.02
    per_cu = collections.Counter(cu[first].tolist())
    print(f"   CUs used at start {len(per_cu)}, CTAs/CU at start: "
          f"{collections.Counter(per_cu.values())}")
    # phase overlap: fraction of a CTA's gemm2 time during which its CU partner is in gemm1
    by_cu = collections.defaultdict(list)
    for i in range(n_blocks):
        by_cu[int(cu[i])].append(i)
    ov = tot = 0.0
    for ids in by_cu.values():
        for i in ids:
            for j in ids:
                if i == j:
                    continue
                a, b = float(g2[i]), float(en[i])
                c, e_ = float(st[j]), float(g2[j])
                ov += max(0.0, min(b, e_) - max(a, c))
            tot += float(en[i] - g2[i])
    print(f"   gemm2 time overlapped by a same-CU partner's gemm1: {100 * ov / tot:.1f}%")


def fused_moe_a8w4_bm64(x, weights, topk_weights, topk_ids, **kw):
    """sort -> fp8 row-quant(x) -> fused kernel (the whole MoE)."""
    E, model_dim, _ = weights["w2"].shape
    sids, sw, seids, nv, moe_buf = moe_sorting(topk_ids, topk_weights, E, model_dim,
                                               dtypes.bf16, BM)[:5]
    x8, xs = quantize_rows(x)
    return a8w4_fmoe_bm64(x8, xs, weights, sids, sw, seids, nv, moe_buf,
                          topk=topk_ids.shape[1], **kw)


# =============================================================================
# Bench: fused vs the 2-kernel A8W4 pipeline (sort + quant + gemm1 q8s + gemm2)
# =============================================================================


def _stage(name):
    if "moe_sorting" in name:
        return "sort"
    if "fmoe" in name:
        return "fused"
    if "gemm1" in name:
        return "gemm1"
    if "gemm2" in name:
        return "gemm2"
    return "quant" if name == "kernel_0" else "other"


def _profile(fn, iters=30, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    t = defaultdict(float)
    for ev in p.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            t[_stage(ev.name)] += ev.device_time / iters
    t["total"] = sum(t.values())
    return dict(t)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-t", type=int, nargs="*", default=[512, 2048, 4096, 8192, 16384])
    p.add_argument("--wpe", type=int, nargs="*", default=[2])
    p.add_argument("--xcd", type=int, nargs="*", default=[1])
    p.add_argument("--bcm", type=int, nargs="*", default=[1], help="W1/W2 load cache modifier")
    p.add_argument("--ref", action="store_true", help="also check vs the torch reference")
    p.add_argument("--reps", type=int, default=1, help="fused timing repeats (median)")
    args = p.parse_args()

    import test_flydsl_moe_a16wfp4 as M  # 2-kernel pipeline + weight prep

    E, MD, INTER, TOPK = 256, 4096, 256, 8
    torch.manual_seed(0)
    w1 = torch.randn((E, INTER * 2, MD), dtype=dtypes.bf16, device="cuda")
    w2 = torch.randn((E, MD, INTER), dtype=dtypes.bf16, device="cuda")
    tq = aiter.get_torch_quant(QuantType.per_1x32)
    w1q, w1s = tq(w1, quant_dtype=dtypes.fp4x2)
    w2q, w2s = tq(w2, quant_dtype=dtypes.fp4x2)
    del w1, w2
    w1q = w1q.view(E, INTER * 2, MD // 2)
    w2q = w2q.view(E, MD, INTER // 2)
    W = M.a8w4_prepare_weights(
        shuffle_weight_a16w4(w1q, 16, False), shuffle_weight_a16w4(w2q, 16, False),
        shuffle_scale_a16w4(w1s, E, False), shuffle_scale_a16w4(w2s, E, False), fast_dec=2)

    for tok in args.t:
        x = torch.randn((tok, MD), dtype=dtypes.bf16, device="cuda")
        tw, ti = fused_topk(x, torch.randn((tok, E), dtype=dtypes.bf16, device="cuda"), TOPK, True)
        two = lambda: M.a8w4_fused_moe(x, W, tw, ti, block_m=64, fuse_quant=True)  # noqa: E731
        o2 = two().float().clone()
        t2 = _profile(two)
        line = [f"tok {tok:6d}: 2-kernel {t2['total']:7.1f} us (sort {t2.get('sort', 0):.1f} "
                f"quant {t2.get('quant', 0):.1f} g1 {t2.get('gemm1', 0):.1f} g2 {t2.get('gemm2', 0):.1f})"]
        if args.ref:
            o1 = torch_moe_stage1(x, w1q.view(dtypes.fp4x2), w2q.view(dtypes.fp4x2), tw, ti,
                                  dtype=dtypes.bf16, activation=ActivationType.Silu,
                                  quant_type=QuantType.per_1x32, a1_scale=None,
                                  w1_scale=w1s.view(E, INTER * 2, MD // 32), doweight=False)
            ref = torch_moe_stage2(o1.view(tok, TOPK, INTER), w1q.view(dtypes.fp4x2),
                                   w2q.view(dtypes.fp4x2), tw, ti, dtype=dtypes.bf16,
                                   quant_type=QuantType.per_1x32,
                                   w2_scale=w2s.view(E, MD, INTER // 32), a2_scale=None,
                                   doweight=True).float()
            line[0] += f" ref cos {G1._cos_diff(ref, o2):.1e}"
        for wpe, xcd, bcm in [(w, c, m) for w in args.wpe for c in args.xcd for m in args.bcm]:
            fn = lambda: fused_moe_a8w4_bm64(x, W, tw, ti, waves_per_eu=wpe, xcd=bool(xcd),  # noqa: E731
                                             b_cache_mod=bcm)
            try:
                o = fn().float().clone()
                runs = sorted((_profile(fn) for _ in range(args.reps)), key=lambda r: r["total"])
                tf = runs[len(runs) // 2]
                c = G1._cos_diff(o2, o)
                bad = "" if c < 1e-4 else " BAD"
                extra = f" ref cos {G1._cos_diff(ref, o):.1e}" if args.ref else ""
                line.append(f"fused w{wpe} xcd{xcd} bcm{bcm} {tf['total']:7.1f} us (kernel {tf.get('fused', 0):.1f}) "
                            f"{t2['total'] / tf['total']:.2f}x  cos vs 2-kernel {c:.1e}{bad}{extra}")
                if PROBE:
                    for _ in range(10):
                        fn()
                    _PROBE["buf"].zero_()
                    fn()
                    torch.cuda.synchronize()
                    print("\n   ".join(line), flush=True)
                    line = []
                    probe_report()
            except Exception as ex:  # noqa: BLE001
                line.append(f"fused w{wpe} FAIL {type(ex).__name__}: {str(ex)[:300]}")
        if line:
            print("\n   ".join(line), flush=True)


if __name__ == "__main__":
    main()
