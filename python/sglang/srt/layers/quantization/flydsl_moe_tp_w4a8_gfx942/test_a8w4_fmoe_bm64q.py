# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Persistent work-queue fused A8W4 MoE (doc-mimo/experiments/fmoe_bm64, step 3).

Same per-block math as test_a8w4_fmoe_bm64 (gemm1 -> SiLU -> fp8 quant -> gemm2 ->
bf16x2 atomics, BM=64, 4 waves, 2 CTAs per CU), but gemm1 and gemm2 of a block
are two TASKS taken from a per-XCD queue by persistent CTAs:

    queue of XCD x (its contiguous range of n blocks, D = CTAs per XCD):
        g1(0) .. g1(D-1), g2(0), g1(D), g2(1), g1(D+1), ..., g2(n-1)

so after the first round about half of the CTAs run gemm2 (atomics, decode VALU)
while the other half run gemm1 (MFMA, W1 loads) -- in test_a8w4_fmoe_bm64 all
CTAs start in gemm1 and stay phase-locked, so the atomics come in bursts at the
~1.5 TB/s ceiling (removing them makes that kernel 20-27% faster).

The quantized gemm1 output (A2: fp8 [64, 256] + f32 scale [64, 2] per block,
~16.5 KB) goes through global memory; producer and consumer of a block are on the
same XCD (queue chosen by HW_REG_XCC_ID), so it is an L2 round trip. Ready flag
per block = launch epoch (no reset); queue counters are double-buffered by epoch
parity and the other parity is zeroed for the next launch.

    python doc-mimo/kernels/test_a8w4_fmoe_bm64q.py -t 2048 4096 8192 16384
"""

import argparse
import functools
import os
import sys

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from . import test_a8w4_fmoe_bm64 as F  # noqa: E402
from . import test_a8w4_gemm1 as G1  # noqa: E402
from . import test_a8w4_gemm2_bm128 as B  # noqa: E402

from aiter import dtypes  # noqa: E402
from aiter.fused_moe import moe_sorting  # noqa: E402
from aiter.ops.flydsl.kernels import buffer_ops as bo  # noqa: E402
from aiter.ops.flydsl.kernels.act import silu_mul_batch  # noqa: E402
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.a8w4 import _pack_activation, quantize_rows  # noqa: E402
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.utils import _global_i32_at, _udiv, _umod  # noqa: E402
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import lds_typed_ptr, lds_vec_load  # noqa: E402
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled  # noqa: E402
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw  # noqa: E402

BM = 64
TILE_N1, TILE_K1, TILE_N2 = 128, 256, 128
N_REPEAT, K_STEPS, K_GROUPS = 2, 8, 2
LDS_A = BM * TILE_K1  # A tile, then bf16 SiLU staging (BM x 128 x 2)
LDS_BYTES = LDS_A + 16  # + task id broadcast
A2_BYTES = BM * 256  # fp8 [64, 256] per block
NXCD = 8
CTL_FLAGS = 64  # i32 index of the per-block flags in the control buffer
CM_SC1 = 16  # gfx942 buffer cache policy sc1 (sc0=1, nt=2)
DBG = os.environ.get("FMOEQ_DBG", "")  # skipg1 / skipg2 / trace (debug only, wrong output)
TRACE_N = 4096


@functools.cache
def compile_a8w4_fmoe_bm64q(*, D_HIDDEN, D_INTER, NE, TOPK=8, b_cache_mod=1, n_ctas=608):
    H, INTER = D_HIDDEN, D_INTER
    N1 = 2 * INTER
    K1_TILES = H // TILE_K1
    NT2 = H // TILE_N2
    assert INTER == 256 and H % TILE_K1 == 0 and NT2 % 2 == 0 and n_ctas % NXCD == 0
    CPX = n_ctas // NXCD  # CTAs per XCD = queue lag D
    W1_K0 = H // 128
    SC1_STRIDE = (H // 256) * 64
    W2_K0 = INTER // 128
    SC2_STRIDE = (INTER // 256) * 64
    WS2_BYTES = NE * H * (INTER // 32)
    kname = (f"a8w4_fmoe_bm64q_h{H}_i{INTER}_ne{NE}_k{TOPK}_bcm{b_cache_mod}_c{n_ctas}"
             f"{'_noat' if B.NO_ATOMIC else ''}{'_' + DBG if DBG else ''}")

    def _cta(lds_ptr, arg_x, arg_xs, arg_w1, arg_ws1, arg_exp1, arg_w2, arg_ws2, arg_eids,
             arg_cumsum, arg_stids, arg_sw, arg_out, i32_M, arg_a2, arg_sc, arg_ctl, i32_epoch):
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx // fx.Int32(64))
        lane_div_16 = lane // fx.Int32(16)
        lane_mod_16 = lane % fx.Int32(16)
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
        rs_a2, rs_sc, rs_ctl = B._rsrc(arg_a2), B._rsrc(arg_sc), B._rsrc(arg_ctl)

        def ld(rs, off_i32, n=1, cm=0):
            v = bo.buffer_load(rs, off_i32, vec_width=n, dtype=T.i32, cache_modifier=cm)
            return fx.Int32(v) if n == 1 else Vec(v)

        lds_flat = fx.make_view(fx.recast_iter(fx.Int32, lds_ptr), fx.make_layout(LDS_BYTES // 4, 1))
        lds16 = fx.logical_divide(lds_flat, fx.make_layout(4, 1))
        lds8 = fx.logical_divide(lds_flat, fx.make_layout(2, 1))
        wr_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        rd_atom = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)
        zero4 = fx.Vector.filled(4, 0.0, fx.Float32)

        def swz(row, col_bytes):
            return col_bytes ^ ((row & fx.Int32(TILE_K1 // 16 - 1)) * fx.Int32(16))

        # ---- this CTA's queue: the XCD it actually runs on ---------------------------
        xc = rfl(F._getreg(0xF814) & fx.Int32(NXCD - 1))  # HW_REG_XCC_ID
        # uniform (readfirstlane): the task loop contains barriers, its exit must
        # not look divergent to the structurizer
        total_m = rfl(_global_i32_at(arg_cumsum, fx.Int32(0)) // fx.Int32(BM))
        bq, br = _udiv(total_m, NXCD), _umod(total_m, NXCD)
        blk0 = xc * bq + fx.min(xc, br)
        nblk = bq + (xc < br).select(fx.Int32(1), fx.Int32(0))
        n_tasks = nblk * fx.Int32(2)
        lag = fx.min(fx.Int32(CPX), nblk)
        par = i32_epoch & fx.Int32(1)
        ctr_idx = par * fx.Int32(NXCD) + xc

        # =============================== gemm1 task ===================================
        def g1_task(m_block, rows):
            MR = rows // 16
            A_LOADS = rows * TILE_K1 // (256 * 16)
            e = rfl(_global_i32_at(arg_eids, m_block))
            m_row = m_block * fx.Int32(BM)
            a_src, a_dst = [], []
            for i in range_constexpr(A_LOADS):
                idx_dw = tx * fx.Int32(4) + fx.Int32(i * 256 * 4)
                row = idx_dw // fx.Int32(TILE_K1 // 4)
                cb = (idx_dw % fx.Int32(TILE_K1 // 4)) * fx.Int32(4)
                tok = ld(rs_id, m_row + row) & fx.Int32(0x00FFFFFF)
                a_src.append((tok * fx.Int32(H) + swz(row, cb)) // fx.Int32(4))
                a_dst.append((row * fx.Int32(TILE_K1) + cb) // fx.Int32(16))
            a_regs = [fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32) for _ in range(A_LOADS)]

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

            xs = []
            for mi in range_constexpr(MR):
                for ii in range_constexpr(4):
                    tok = ld(rs_id, m_row + fx.Int32(mi * 16 + ii) + lane_div_16 * fx.Int32(4)) \
                        & fx.Int32(0x00FFFFFF)
                    xs.append(ld(rs_xs, tok).bitcast(fx.Float32))

            for nb in range_constexpr(2):
                w1_base, sc1_base, n_cols = {}, {}, []
                for ni in range_constexpr(N_REPEAT):
                    n_local = fx.Int32(nb * TILE_N1 + ni * 16) + n_wave_base
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

                acc = {(gu, mi, ni): fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
                       for gu in range(2) for mi in range(MR) for ni in range(N_REPEAT)}
                for t_ in acc.values():
                    t_.store(zero4)
                w_items = [(gu, ni) for gu in range(2) for ni in range(N_REPEAT)]

                def compute_tile(nxt):
                    """One 256-K tile, W read lazily from w_regs; with ``nxt`` tile nxt's
                    loads are spread over the K steps."""
                    luts = {}
                    for gu in range_constexpr(2):
                        p = fx.Int32(Vec(s_regs[gu].load())[0])
                        for ni in range_constexpr(N_REPEAT):
                            for g in range_constexpr(K_GROUPS):
                                luts[gu, ni, g] = G1._fp8_lut_predelta(
                                    G1._perm_u32(p, p, ((g % 2) * 2 + ni) * 0x01010101))
                    if const_expr(nxt is not None):
                        load_s(nxt)
                    for ku in range_constexpr(K_STEPS):
                        g, j = ku // 4, ku % 4
                        bs = {(gu, ni): G1._fp4x8_to_fp8x8_fast(
                            fx.Int32(Vec(w_regs[gu, ni, g].load())[j]), luts[gu, ni, g])
                            for gu in range(2) for ni in range(N_REPEAT)}
                        for mi in range_constexpr(MR):
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

                for i in range_constexpr(A_LOADS):
                    load_a_one(i, fx.Int32(0))
                load_s(fx.Int32(0))
                for (gu, ni, g) in w_regs:
                    load_w_one(gu, ni, g, fx.Int32(0))
                store_a_lds()

                @flyc.jit
                def _k_loop(k_first):
                    for kt in range(K1_TILES - 1):
                        nxt = (k_first + fx.Int32(kt) + fx.Int32(1)) * fx.Int32(TILE_K1)
                        rocdl.s_waitcnt(lgkmcnt=0)
                        gpu.barrier()  # A(kt) visible
                        compute_tile(nxt)
                        gpu.barrier()  # A(kt) reads done
                        store_a_lds()

                _k_loop(fx.Int32(0))
                rocdl.s_waitcnt(lgkmcnt=0)
                gpu.barrier()
                compute_tile(None)
                gpu.barrier()  # A reads done: the A slot becomes the bf16 staging

                # ---- dequant -> SiLU -> bf16 staging [rows, 128] ------------------------
                f_g = [G1._channel_factor(ld(rs_e1, e * fx.Int32(N1) + c)) for c in n_cols]
                f_u = [G1._channel_factor(ld(rs_e1, e * fx.Int32(N1) + fx.Int32(INTER) + c))
                       for c in n_cols]
                stg = lds_typed_ptr(lds_i32, T.bf16)
                for mi in range_constexpr(MR):
                    ag = [Vec(acc[0, mi, ni].load()) for ni in range(N_REPEAT)]
                    au = [Vec(acc[1, mi, ni].load()) for ni in range(N_REPEAT)]
                    for ii in range_constexpr(4):
                        rsc = xs[mi * 4 + ii]
                        row = fx.Int32(mi * 16 + ii) + lane_div_16 * fx.Int32(4)
                        for ni in range_constexpr(N_REPEAT):
                            y = silu_mul_batch([ag[ni][ii] * (rsc * f_g[ni])],
                                               [au[ni][ii] * (rsc * f_u[ni])])[0]
                            col = n_wave_base + fx.Int32(ni * 16) + lane_mod_16
                            stg[row * fx.Int32(TILE_N1) + col] = y.to(fx.BFloat16)
                gpu.barrier()

                # ---- per-(row, 128) fp8 quant -> A2 / scale in global (L2) ----------------
                r_in = tx // fx.Int32(8)
                sub = tx % fx.Int32(8)
                for pp in range_constexpr(rows // 32):
                    row = fx.Int32(pp * 32) + r_in
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
                    # row-major [64, 256] fp8: dword offset
                    dst = m_block * fx.Int32(A2_BYTES // 4) + row * fx.Int32(64) \
                        + fx.Int32(nb * 32) + sub * fx.Int32(4)
                    bo.buffer_store(_raw(packed), rs_a2, dst)
                    sc_i = m_block * fx.Int32(BM * 2) + row * fx.Int32(2) + fx.Int32(nb)
                    bo.buffer_store(_raw(s.bitcast(fx.Int32)), rs_sc, sc_i)
                if const_expr(nb == 0):
                    gpu.barrier()  # staging reads done before half 1's A stores

            # ---- publish: all A2 stores landed in L2, then the flag ------------------------
            rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()
            publish(m_block)

        @flyc.jit
        def publish(m_block):
            # producer and consumer share the XCD's L2: the A2 stores were acked by
            # L2 (vmcnt(0) + barrier above), so a plain sc1 store of the flag is
            # enough -- no agent release (buffer_wbl2 would write back the L2)
            if wave == fx.Int32(0):  # whole wave stores the same value
                bo.buffer_store(_raw(i32_epoch), rs_ctl, fx.Int32(CTL_FLAGS) + m_block,
                                cache_modifier=CM_SC1)

        # =============================== gemm2 task ===================================
        def g2_task(m_block, rows):
            MR = rows // 16
            e = rfl(_global_i32_at(arg_eids, m_block))
            m_row = m_block * fx.Int32(BM)
            wait_ready(m_block)
            # A2 fragments: lane (row lane%16 of block mi) reads 32 contiguous bytes per half
            a2 = {}
            for mi in range_constexpr(MR):
                row = lane_mod_16 + fx.Int32(mi * 16)
                for nb in range_constexpr(2):
                    for h in range_constexpr(2):
                        v = ld(rs_a2, m_block * fx.Int32(A2_BYTES // 4) + row * fx.Int32(64)
                               + fx.Int32(nb * 32 + h * 4) + lane_div_16 * fx.Int32(8), n=4, cm=CM_SC1)
                        a2[mi, nb * 4 + h * 2] = Vec.from_elements([v[0], v[1]], fx.Int32)
                        a2[mi, nb * 4 + h * 2 + 1] = Vec.from_elements([v[2], v[3]], fx.Int32)
            # row data (lane L = row L)
            s0 = ld(rs_sc, m_block * fx.Int32(BM * 2) + lane * fx.Int32(2), cm=CM_SC1).bitcast(fx.Float32)
            s1 = ld(rs_sc, m_block * fx.Int32(BM * 2) + lane * fx.Int32(2) + fx.Int32(1),
                    cm=CM_SC1).bitcast(fx.Float32)
            s1 = (s1 == fx.Float32(0.0)).select(fx.Float32(1.0), s1)
            w_r = ld(rs_sw, m_row + lane).bitcast(fx.Float32)
            ratio_l = (s0 / s1).bitcast(fx.Int32)
            id_l = ld(rs_id, m_row + lane)
            valid_l = (id_l & fx.Int32(0x00FFFFFF)) < i32_M
            fin_l = valid_l.select(s1 * w_r, fx.Float32(0.0)).bitcast(fx.Int32)

            def row_valid(r):  # wave-uniform
                t = fx.Int32(rocdl.readlane(T.i32, _raw(id_l), _raw(fx.Int32(r))))
                return (t & fx.Int32(0x00FFFFFF)) < i32_M

            def fetch(q, row_static):
                src = fx.Int32(row_static) + lane_div_16 * fx.Int32(4)
                return fx.Int32(rocdl.ds_bpermute(T.i32, src * fx.Int32(4), q))

            odd = lane & fx.Int32(1)
            is_odd = odd == fx.Int32(1)
            col_b = (odd * fx.Int32(16) + lane_mod_16 - odd) * fx.Int32(2)
            sel_pair = is_odd.select(fx.Int32(0x03020706), fx.Int32(0x05040100))
            ratio4, fin4, offs4 = [], [], []
            for mi in range_constexpr(MR):
                rr, ff, oo = [], [], []
                for v in range_constexpr(4):
                    rr.append(fetch(ratio_l, mi * 16 + v).bitcast(fx.Float32))
                    ff.append(fetch(fin_l, mi * 16 + v).bitcast(fx.Float32))
                    tok = fetch(id_l, mi * 16 + v) & fx.Int32(0x00FFFFFF)
                    oo.append((tok < i32_M).select(tok * fx.Int32(H * 2) + col_b, fx.Int32(-1)))
                ratio4.append(rr)
                fin4.append(ff)
                offs4.append(oo)
            mi_full = [row_valid(mi * 16 + 15) for mi in range(MR)]
            mi_part = [row_valid(mi * 16) for mi in range(MR)]

            wb = [{
                "raw": {(ni, g): fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        for ni in range(N_REPEAT) for g in range(K_GROUPS)},
                "packed": fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32),
                "exp": fx.make_rmem_tensor(fx.make_layout(N_REPEAT, 1), fx.Int32),
            } for _ in range(2)]
            acc2 = [[[fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
                      for _ in range(N_REPEAT)] for _ in range(MR)] for _ in range(2)]
            fn2 = [fx.make_rmem_tensor(fx.make_layout(N_REPEAT, 1), fx.Float32) for _ in range(2)]

            def issue_w2(t, b):
                n0 = e * fx.Int32(H) + t * fx.Int32(TILE_N2) + n_wave_base
                for ni in range_constexpr(N_REPEAT):
                    wbase = ((n0 + fx.Int32(ni * 16)) // fx.Int32(16)) * fx.Int32(W2_K0 * 64) \
                        + lane_div_16 * fx.Int32(16) + lane_mod_16
                    for g in range_constexpr(K_GROUPS):
                        wb[b]["raw"][ni, g].store(Vec(bo.buffer_load(
                            rs_w2, (wbase + fx.Int32(g * 64)) * fx.Int32(4), vec_width=4,
                            dtype=T.i32, cache_modifier=b_cache_mod)))
                scb = (n0 // fx.Int32(32)) * fx.Int32(SC2_STRIDE) + lane_div_16 * fx.Int32(16) \
                    + lane_mod_16
                wb[b]["packed"].store(Vec.from_elements([ld(rs_ws2, scb)], fx.Int32))
                wb[b]["exp"].store(Vec.from_elements(
                    [ld(rs_e2, n0 + fx.Int32(ni * 16) + lane_mod_16) for ni in range(N_REPEAT)],
                    fx.Int32))

            def epilogue_mi(s, t_prev, mi, ks=(0, 1, 2, 3)):
                fn = Vec(fn2[s].load())
                datas = []
                for k in ks:
                    r0 = F._rnu_hi(Vec(acc2[s][mi][0].load())[k] * (fin4[mi][k] * fn[0]))
                    r1 = F._rnu_hi(Vec(acc2[s][mi][1].load())[k] * (fin4[mi][k] * fn[1]))
                    pk = G1._perm_u32(r1, r0, 0x07060302)
                    nbr = fx.Int32(rocdl.update_dpp(T.i32, _raw(pk.bitcast(fx.Int32)),
                                                    _raw(pk.bitcast(fx.Int32)),
                                                    B.DPP_QUAD_SWAP, 0xF, 0xF, True))
                    datas.append(_raw(G1._perm_u32(nbr, pk, sel_pair).bitcast(fx.Int32)))
                col = (t_prev * fx.Int32(TILE_N2) + n_wave_base) * fx.Int32(2)
                F._atomics_mi(datas, [_raw(offs4[mi][k]) for k in ks], col, arg_out,
                              mi_full[mi], mi_part[mi])

            def tile(t, s, b, prev, has_next):
                rocdl.sched_barrier(0)
                if const_expr(has_next):
                    issue_w2(t + fx.Int32(1), 1 - b)
                raw = {k: [fx.Int32(Vec(r.load())[j]) for j in range(4)]
                       for k, r in wb[b]["raw"].items()}
                p = fx.Int32(Vec(wb[b]["packed"].load())[0])
                ex = Vec(wb[b]["exp"].load())
                fn2[s].store(Vec.from_elements(
                    [G1._channel_factor(fx.Int32(ex[ni])) for ni in range(N_REPEAT)], fx.Float32))
                luts = [[G1._fp8_lut_predelta(G1._perm_u32(p, p, ((g % 2) * 2 + ni) * 0x01010101))
                         for g in range(K_GROUPS)] for ni in range(N_REPEAT)]
                for mi in range_constexpr(MR):
                    for ni in range_constexpr(N_REPEAT):
                        acc2[s][mi][ni].store(zero4)
                for ku in range_constexpr(K_STEPS):
                    if const_expr(ku == 4):  # acc = acc0 * s0/s1, then K group 1
                        for mi in range_constexpr(MR):
                            for ni in range_constexpr(N_REPEAT):
                                a4 = Vec(acc2[s][mi][ni].load())
                                acc2[s][mi][ni].store(Vec.from_elements(
                                    [a4[v] * ratio4[mi][v] for v in range(4)], fx.Float32))
                    b8s = [G1._fp4x8_to_fp8x8_fast(raw[ni, ku // 4][ku % 4], luts[ni][ku // 4])
                           for ni in range(N_REPEAT)]
                    for mi in range_constexpr(MR):
                        a8 = a2[mi, ku].bitcast(fx.Float8E4M3FNUZ)
                        for ni in range_constexpr(N_REPEAT):
                            G1._mma_fp8(mma_atom, acc2[s][mi][ni], a8, b8s[ni])
                    if const_expr(prev is not None):
                        per = MR * 4 // K_STEPS
                        u0 = ku * per
                        epilogue_mi(prev[0], prev[1], u0 // 4,
                                    tuple(u % 4 for u in range(u0, u0 + per)))

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

        @flyc.jit
        def wait_ready(m_block):
            # sc1 loads miss the CU's L1; no agent acquire (buffer_inv sc1 would drop
            # the XCD's L2, i.e. the cached expert weights)
            f = rfl(ld(rs_ctl, fx.Int32(CTL_FLAGS) + m_block, cm=CM_SC1))
            while f != i32_epoch:
                rocdl.s_sleep(2)
                f = rfl(ld(rs_ctl, fx.Int32(CTL_FLAGS) + m_block, cm=CM_SC1))

        # ================================ scheduler ===================================
        def rows_of(m_block):
            tok32 = rfl(_global_i32_at(arg_stids, m_block * fx.Int32(BM) + fx.Int32(32))) \
                & fx.Int32(0x00FFFFFF)
            return tok32 >= i32_M  # rows 32.. are padding

        @flyc.jit
        def run_task(j):
            r = j - lag
            mid = nblk - lag
            head = j < lag
            in_mid = r < mid * fx.Int32(2)
            r_odd = (r & fx.Int32(1)) == fx.Int32(1)
            is_g1 = head | (in_mid & r_odd)
            i_mid = r_odd.select(lag + r // fx.Int32(2), r // fx.Int32(2))
            i = head.select(j, in_mid.select(i_mid, r - mid))
            m_block = blk0 + i
            if const_expr(DBG == "trace"):
                trace(m_block, is_g1)
            else:
                run_block(m_block, is_g1)

        @flyc.jit
        def trace(m_block, is_g1):
            if wave == fx.Int32(0):
                idx = fx.max(fx.min(m_block, fx.Int32(TRACE_N - 1)), fx.Int32(0))
                inc = (lane == fx.Int32(0)).select(is_g1.select(fx.Int32(1), fx.Int32(1000)),
                                                   fx.Int32(0))
                ptr = bo.get_element_ptr(F.G2._global_base_ptr1(arg_ctl),
                                         byte_offset=_raw((fx.Int32(CTL_FLAGS) + idx) * fx.Int32(4)))
                llvm.AtomicRMWOp(llvm.AtomicBinOp.add, ptr, _raw(inc),
                                 llvm.AtomicOrdering.monotonic, syncscope="agent")

        @flyc.jit
        def run_block(m_block, is_g1):
            short = rows_of(m_block)
            if is_g1:
                if const_expr(DBG == "skipg1"):
                    publish(m_block)
                elif short:
                    g1_task(m_block, 32)
                else:
                    g1_task(m_block, BM)
            else:
                if const_expr(DBG == "skipg2"):
                    wait_ready(m_block)
                elif short:
                    g2_task(m_block, 32)
                else:
                    g2_task(m_block, BM)

        tid_slot = lds_typed_ptr(lds_i32 + fx.Int32(LDS_A), T.i32)

        @flyc.jit
        def grab():
            # no lane-divergent branches around the barriers (they confuse the CFG
            # structurizer inside the task loop): all of wave 0 adds, lane 0 adds 1
            gpu.barrier()  # previous task done with LDS / the slot
            if wave == fx.Int32(0):
                ctr = bo.get_element_ptr(F.G2._global_base_ptr1(arg_ctl),
                                         byte_offset=_raw(ctr_idx * fx.Int32(4)))
                inc = (lane == fx.Int32(0)).select(fx.Int32(1), fx.Int32(0))
                old = fx.Int32(llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.add, ctr, _raw(inc),
                    llvm.AtomicOrdering.monotonic, syncscope="agent").res)
                tid_slot[0] = rfl(old)
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()

        @flyc.jit
        def reset_next():
            # zero the other parity's counters for the next launch
            if bx == fx.Int32(0):
                if tx < fx.Int32(NXCD):
                    bo.buffer_store(_raw(fx.Int32(0)), rs_ctl, (fx.Int32(1) - par) * fx.Int32(NXCD) + tx)

        @flyc.jit
        def loop():
            grab()
            j = rfl(tid_slot[0])
            while j < n_tasks:
                run_task(j)
                grab()
                j = rfl(tid_slot[0])

        reset_next()
        loop()

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.kernel(name=kname, known_block_size=[256, 1, 1])
    def kernel(arg_x: fx.Int64, arg_xs: fx.Int64, arg_w1: fx.Int64, arg_ws1: fx.Int64,
               arg_exp1: fx.Int64, arg_w2: fx.Int64, arg_ws2: fx.Int64, arg_eids: fx.Int64,
               arg_cumsum: fx.Int64, arg_stids: fx.Int64, arg_sw: fx.Int64,
               arg_out: fx.Int64, i32_M: fx.Int32, arg_a2: fx.Int64, arg_sc: fx.Int64,
               arg_ctl: fx.Int64, i32_epoch: fx.Int32):
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        _cta(lds_ptr, arg_x, arg_xs, arg_w1, arg_ws1, arg_exp1, arg_w2, arg_ws2, arg_eids,
             arg_cumsum, arg_stids, arg_sw, arg_out, i32_M, arg_a2, arg_sc, arg_ctl, i32_epoch)

    @flyc.jit
    def launch(arg_x: fx.Int64, arg_xs: fx.Int64, arg_w1: fx.Int64, arg_ws1: fx.Int64,
               arg_exp1: fx.Int64, arg_w2: fx.Int64, arg_ws2: fx.Int64, arg_eids: fx.Int64,
               arg_cumsum: fx.Int64, arg_stids: fx.Int64, arg_sw: fx.Int64,
               arg_out: fx.Int64, i32_M: fx.Int32, arg_a2: fx.Int64, arg_sc: fx.Int64,
               arg_ctl: fx.Int64, i32_epoch: fx.Int32, stream: fx.Stream):
        kernel(arg_x, arg_xs, arg_w1, arg_ws1, arg_exp1, arg_w2, arg_ws2, arg_eids,
               arg_cumsum, arg_stids, arg_sw, arg_out, i32_M, arg_a2, arg_sc, arg_ctl, i32_epoch,
               value_attrs={"rocdl.waves_per_eu": 2},
               ).launch(grid=(n_ctas, 1, 1), block=(256, 1, 1), stream=stream)

    return launch


def rfl(v):
    return fx.Int32(rocdl.readfirstlane(T.i32, _raw(fx.Int32(v))))


_WS = {}


def _workspace(n_blocks, device):
    """A2 [n_blocks, 64, 256] fp8, scales [n_blocks, 64, 2] f32, control i32
    [CTL_FLAGS + n_blocks] (2 x NXCD queue counters, then per-block ready flags)."""
    ws = _WS.get(device)
    if ws is None or ws["n"] < n_blocks:
        ws = {
            "n": n_blocks,
            "a2": torch.empty(n_blocks * A2_BYTES, dtype=torch.uint8, device=device),
            "sc": torch.empty(n_blocks * BM * 2, dtype=torch.float32, device=device),
            "ctl": torch.zeros(CTL_FLAGS + n_blocks, dtype=torch.int32, device=device),
            "epoch": 0,
        }
        _WS[device] = ws
    ws["epoch"] += 1
    return ws


def a8w4_fmoe_bm64q(x_fp8, x_scale, weights, sorted_ids, sorted_weights,
                    sorted_expert_ids, num_valid_ids, out, *, topk=8, b_cache_mod=1,
                    n_ctas=None):
    """Persistent work-queue fused stage1 + stage2 (inputs as
    test_a8w4_fmoe_bm64.a8w4_fmoe_bm64)."""
    assert weights["w1_fast_dec"] == 2 and weights["w2_fast_dec"] == 2
    tokens, model_dim = out.shape
    E = weights["w2"].shape[0]
    if n_ctas is None:
        n_ctas = 2 * torch.cuda.get_device_properties(out.device).multi_processor_count
    ws = _workspace(sorted_expert_ids.numel(), out.device)
    launch = compile_a8w4_fmoe_bm64q(
        D_HIDDEN=model_dim, D_INTER=weights["w1"].shape[1] // 2, NE=E, TOPK=topk,
        b_cache_mod=b_cache_mod, n_ctas=n_ctas)
    _run_compiled(
        launch, x_fp8.data_ptr(), x_scale.data_ptr(), weights["w1"].data_ptr(),
        weights["w1_scale"].data_ptr(), weights["w1_exp"].data_ptr(),
        weights["w2"].data_ptr(), weights["w2_scale"].data_ptr(),
        sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), sorted_ids.data_ptr(),
        sorted_weights.data_ptr(), out.data_ptr(), int(tokens), ws["a2"].data_ptr(),
        ws["sc"].data_ptr(), ws["ctl"].data_ptr(), ws["epoch"], torch.cuda.current_stream())
    return out


def fused_moe_a8w4_bm64q(x, weights, topk_weights, topk_ids, **kw):
    E, model_dim, _ = weights["w2"].shape
    sids, sw, seids, nv, moe_buf = moe_sorting(topk_ids, topk_weights, E, model_dim,
                                               dtypes.bf16, BM)[:5]
    x8, xs = quantize_rows(x)
    return a8w4_fmoe_bm64q(x8, xs, weights, sids, sw, seids, nv, moe_buf,
                           topk=topk_ids.shape[1], **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-t", type=int, nargs="*", default=[2048, 4096, 8192, 16384])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--no_bm64", action="store_true", help="skip the non-persistent kernel")
    args = p.parse_args()

    import test_flydsl_moe_a16wfp4 as M
    from aiter import QuantType, get_torch_quant
    from aiter.fused_moe import fused_topk
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    E, MD, INTER, TOPK = 256, 4096, 256, 8
    torch.manual_seed(0)
    w1 = torch.randn((E, INTER * 2, MD), dtype=dtypes.bf16, device="cuda")
    w2 = torch.randn((E, MD, INTER), dtype=dtypes.bf16, device="cuda")
    tq = get_torch_quant(QuantType.per_1x32)
    w1q, w1s = tq(w1, quant_dtype=dtypes.fp4x2)
    w2q, w2s = tq(w2, quant_dtype=dtypes.fp4x2)
    del w1, w2
    W = M.a8w4_prepare_weights(
        shuffle_weight_a16w4(w1q.view(E, INTER * 2, MD // 2), 16, False),
        shuffle_weight_a16w4(w2q.view(E, MD, INTER // 2), 16, False),
        shuffle_scale_a16w4(w1s, E, False), shuffle_scale_a16w4(w2s, E, False), fast_dec=2)

    def timed(fn):
        runs = sorted((F._profile(fn) for _ in range(args.reps)), key=lambda r: r["total"])
        return runs[len(runs) // 2]

    for tok in args.t:
        x = torch.randn((tok, MD), dtype=dtypes.bf16, device="cuda")
        tw, ti = fused_topk(x, torch.randn((tok, E), dtype=dtypes.bf16, device="cuda"), TOPK, True)
        two = lambda: M.a8w4_fused_moe(x, W, tw, ti, block_m=64, fuse_quant=True)  # noqa: E731
        o2 = two().float().clone()
        t2 = timed(two)
        line = [f"tok {tok:6d}: 2-kernel {t2['total']:7.1f} us"]
        cands = [("bm64q", lambda: fused_moe_a8w4_bm64q(x, W, tw, ti))]
        if not args.no_bm64:
            cands.insert(0, ("bm64 ", lambda: F.fused_moe_a8w4_bm64(x, W, tw, ti)))
        for name, fn in cands:
            try:
                o = fn().float().clone()
                tf = timed(fn)
                c = G1._cos_diff(o2, o)
                line.append(f"{name} {tf['total']:7.1f} us (kernel {tf.get('fused', 0):6.1f}) "
                            f"{t2['total'] / tf['total']:.2f}x  cos {c:.1e}{'' if c < 1e-4 else ' BAD'}")
            except Exception as ex:  # noqa: BLE001
                line.append(f"{name} FAIL {type(ex).__name__}: {str(ex)[:400]}")
        print("\n   ".join(line), flush=True)


if __name__ == "__main__":
    main()
