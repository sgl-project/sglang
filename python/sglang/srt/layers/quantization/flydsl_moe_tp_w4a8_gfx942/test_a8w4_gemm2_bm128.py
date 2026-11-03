# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""A8W4 MoE stage2 (down-proj + routed bf16x2 atomic scatter) for LARGE token
counts: BM=128, low occupancy, software pipelined. gfx942.

Separate from ``test_a8w4_gemm2.py`` (whose kernel hides latency with occupancy):
here a CTA's resources are large (1 wave/SIMD by design, ``waves_per_eu=1`` ->
up to 512 VGPRs), so latency is hidden by instruction-level pipelining instead:

  * A (128 x 256 fp8) goes straight from global into VGPRs in the MFMA operand
    layout (128 VGPRs/lane), once per CTA: no LDS, no barrier, no ds_read in the
    MFMA loop. Every wave holds all 128 rows (1x4 N split: each wave owns 32 of
    the tile's 128 columns), so each decoded W fragment feeds 8 MFMAs (vs 4 at
    BM=64): half the fp4 decode VALU per MFMA.
  * A CTA walks NT adjacent n-tiles of one m-block. W / scale / max-code loads
    run 2 tiles ahead (prefetch distance 2).
  * The epilogue of tile t-1 (dequant, 64 B-segment bf16x2 atomics, no LDS) is
    interleaved with tile t's MFMAs: K step ku of tile t carries row block mi=ku
    of tile t-1 (two accumulator sets live).
  * Row-static data (ratio s0/s1 of the two A-scale groups, s1 * route weight,
    sorted id) is lane-distributed (lane L holds rows L, 64+L) and fetched with
    ds_bpermute. One accumulator set per tile, rescaled by s0/s1 at the 128-K
    group boundary.

``OUT_MODE`` (compile time): ``"atomic"`` = bf16x2 atomics into ``out[token]``
(needs a zeroed ``out``); ``"sorted"`` = plain bf16x2 stores of every (weighted)
row into ``routes[sorted_pos]`` (the block's 128 rows are contiguous rows there,
no token ids / padding predicates in the kernel), then ``invert_sorted_ids``
(loc[token, slot] = sorted position) + ``sorted_sum`` (out[t] = sum over the topk
rows routes[loc[t, k]]); after pyhip's moe_gemm_2stage/moe_reduce.py. ``out``
needs no zeroing.

Fixed configuration: inter_dim (K) = 256, tile_n 128, ``fast_dec=2`` weights
(``prep_w2_fast``), A scales per (row, 128-K group) (gemm1's ``q8s``), sorting
block_m 128. ``dyn_rows``: a block whose rows 64.. are all padding runs a 64-row
variant.

    python doc-mimo/kernels/test_a8w4_gemm2_bm128.py -t 2048 8192 16384
"""

import argparse
import functools
import os
import sys

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl._mlir.dialects import arith, llvm

from . import test_a8w4_gemm2 as G  # noqa: E402  (shared device helpers + reference)

import aiter  # noqa: E402
from aiter import QuantType, dtypes  # noqa: E402
from aiter.fused_moe import fused_topk, moe_sorting  # noqa: E402
from aiter.ops.flydsl.kernels import buffer_ops as bo  # noqa: E402
from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.utils import (  # noqa: E402
    _global_i32_at,
    _udiv,
    _umod,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled  # noqa: E402
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw  # noqa: E402
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4  # noqa: E402
from aiter.test_common import run_perftest  # noqa: E402

BM = 128
TILE_N = 128
NUM_WAVES = 4
N_REPEAT = 2  # 16-col blocks per wave (32 columns)
K_STEPS = 8  # MFMA K32 steps (K = 256)
K_GROUPS = 2  # 128-K groups (one W dwordx4 + one A-scale group each)


def _rsrc(addr_i64):
    return bo.create_buffer_resource_from_addr(_raw(fx.Int64(addr_i64)))


# ablation knob (experiment only): 1 = skip the atomics (compute-only time).
# Tagged into the kernel name; use a fresh FLYDSL_RUNTIME_CACHE_DIR anyway.
NO_ATOMIC = os.environ.get("G2_NO_ATOMIC", "0") == "1"
DPP_QUAD_SWAP = 0xB1  # quad_perm [1, 0, 3, 2]: lane L <- lane L ^ 1


def _rne_hi(x):
    """f32 -> i32 whose high half is bf16(x) (round to nearest even; finite x)."""
    u = fx.Float32(x).bitcast(fx.Int32)
    return u + fx.Int32(0x7FFF) + ((u >> fx.Int32(16)) & fx.Int32(1))


@flyc.jit
def _atomics4(datas, offs, base_i64):
    """bf16x2 atomic add of ``datas[k]`` (i32 = 2 x bf16) at ``base + offs[k]``
    bytes; offs[k] < 0 marks a padding row (skipped). ``base`` is uniform and the
    offset a non-negative i32 (zero-extended) -> SGPR-base global atomics."""
    base = G._global_base_ptr1(base_i64)
    for k in range_constexpr(len(offs)):
        off = fx.Int32(offs[k])
        if off >= fx.Int32(0 if not NO_ATOMIC else 1 << 30):
            off64 = arith.ExtUIOp(T.i64, _raw(off)).result
            pk = Vec.from_elements([fx.Int32(datas[k])], fx.Int32).bitcast(fx.BFloat16)
            llvm.AtomicRMWOp(
                llvm.AtomicBinOp.fadd,
                bo.get_element_ptr(base, byte_offset=off64),
                _raw(pk),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            )


def _stores4(datas, offs, base_i64):
    """Plain bf16x2 stores of ``datas[k]`` at ``base + offs[k]`` (uniform base,
    non-negative i32 offsets, zero-extended -> SGPR-base global stores)."""
    base = G._global_base_ptr1(base_i64)
    for k in range_constexpr(len(offs)):
        off64 = arith.ExtUIOp(T.i64, _raw(fx.Int32(offs[k]))).result
        llvm.StoreOp(_raw(fx.Int32(datas[k])), bo.get_element_ptr(base, byte_offset=off64),
                     alignment=4)


@functools.cache
def compile_a8w4_gemm2_bm128(*, D_HIDDEN, D_INTER, NE, NT=8, b_cache_mod=2,
                             waves_per_eu=1, dyn_rows=True, OUT_MODE="atomic",
                             ROW_PAD=0):
    K, N_OUT = D_INTER, D_HIDDEN
    assert K == 256, "one K tile: inter_dim 256"
    assert OUT_MODE in ("atomic", "sorted"), OUT_MODE
    SORTED = OUT_MODE == "sorted"
    # sorted mode: routes row stride in bytes (padding spreads the TOPK rows that
    # sorted_sum gathers at one column over different HBM channels)
    RS = N_OUT * 2 + ROW_PAD
    assert ROW_PAD % 16 == 0
    NB = N_OUT // TILE_N
    assert NB % NT == 0, f"NT={NT} must divide {NB}"
    NBG = NB // NT  # CTAs per m-block
    W_K0 = K // 128
    SC_STRIDE_N0 = (K // 256) * 64
    WS_BYTES = NE * N_OUT * (K // 32)
    kname = (
        f"a8w4_gemm2bm128_h{N_OUT}_i{K}_ne{NE}_nt{NT}_bcm{b_cache_mod}"
        f"_w{waves_per_eu}{'_dyn' if dyn_rows else ''}{'_noat' if NO_ATOMIC else ''}"
        f"{'_srt' if SORTED else ''}{f'_p{ROW_PAD}' if SORTED and ROW_PAD else ''}"
    )

    def _body(arg_a, arg_a_scale, arg_w, arg_w_scale, arg_eids, arg_stids,
              arg_sweights, arg_out, i32_M, tile, lane, wave):
        lane_div_16 = lane // fx.Int32(16)
        lane_mod_16 = lane % fx.Int32(16)
        m_block = tile // fx.Int32(NBG)
        n_grp = tile % fx.Int32(NBG)
        e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block)))
        m_row = m_block * fx.Int32(BM)
        expert_off = e * fx.Int32(N_OUT)
        n_wave_base = fx.Int32(wave) * fx.Int32(TILE_N // NUM_WAVES)
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))

        rs_a = _rsrc(arg_a)
        rs_w = _rsrc(arg_w)
        rs_ws = _rsrc(arg_w_scale)
        rs_exp = bo.create_buffer_resource_from_addr(
            _raw(fx.Int64(arg_w_scale) + fx.Int64(WS_BYTES)))
        rs_as = _rsrc(arg_a_scale)
        rs_sw = _rsrc(arg_sweights)
        rs_id = _rsrc(arg_stids)

        def w_issue(t):
            """W (2 ni x 2 K groups x 16 B), scale dword, max codes of n-tile t."""
            nb = n_grp * fx.Int32(NT) + fx.Int32(t)
            by_n = nb * fx.Int32(TILE_N)
            n0 = expert_off + by_n + n_wave_base
            raw = []
            for ni in range_constexpr(N_REPEAT):
                wb = ((n0 + fx.Int32(ni * 16)) // fx.Int32(16)) * fx.Int32(W_K0 * 64) \
                    + lane_div_16 * fx.Int32(16) + lane_mod_16
                per_g = []
                for g in range_constexpr(K_GROUPS):
                    v = Vec(bo.buffer_load(rs_w, (wb + fx.Int32(g * 64)) * fx.Int32(4),
                                           vec_width=4, dtype=T.i32,
                                           cache_modifier=b_cache_mod))
                    per_g.append([fx.Int32(v[j]) for j in range(4)])
                raw.append(per_g)
            # 32 columns share one scale dword (npack = ni); fast_dec=2 deltas
            scb = (n0 // fx.Int32(32)) * fx.Int32(SC_STRIDE_N0) \
                + lane_div_16 * fx.Int32(16) + lane_mod_16
            packed = fx.Int32(bo.buffer_load(rs_ws, scb, vec_width=1, dtype=T.i32))
            exps = [
                fx.Int32(bo.buffer_load(rs_exp, n0 + fx.Int32(ni * 16) + lane_mod_16,
                                        vec_width=1, dtype=T.i32))
                for ni in range_constexpr(N_REPEAT)
            ]
            return dict(by_n=by_n, raw=raw, packed=packed, exp=exps)

        def run(rows):
            MR = rows // 16

            # ---- A -> VGPRs in the MFMA layout: lane L, row L%16 of block mi,
            # K bytes g*128 + (L//16)*32 .. +32 (two dwordx4) per 128-K group g
            a_v = {}
            for mi in range_constexpr(MR):
                row_b = (m_row + fx.Int32(mi * 16) + lane_mod_16) * fx.Int32(K)
                for g in range_constexpr(K_GROUPS):
                    for h in range_constexpr(2):
                        off = row_b + fx.Int32(g * 128 + h * 16) + lane_div_16 * fx.Int32(32)
                        a_v[mi, g, h] = Vec(bo.buffer_load(rs_a, off // fx.Int32(4),
                                                           vec_width=4, dtype=T.i32))

            def a_frag(mi, ku):
                g, j = ku // 4, ku % 4
                v = a_v[mi, g, j // 2]
                d = (2 * j) % 4
                return Vec.from_elements(
                    [fx.Int32(v[d]), fx.Int32(v[d + 1])], fx.Int32
                ).bitcast(fx.Float8E4M3FNUZ)

            # ---- W pipeline: 2 tiles ahead ------------------------------------
            W = [None] * NT
            for t in range_constexpr(min(2, NT)):
                W[t] = w_issue(t)

            # ---- row data, lane-distributed (lane L: rows L and 64 + L) --------
            ratio_h, fin_h, rid_h = [], [], []
            for hh in range_constexpr(BM // 64):
                grow = m_row + fx.Int32(hh * 64) + lane
                sv = Vec(bo.buffer_load(rs_as, grow * fx.Int32(2), vec_width=2, dtype=T.i32))
                s0 = fx.Int32(sv[0]).bitcast(fx.Float32)
                s1 = fx.Int32(sv[1]).bitcast(fx.Float32)
                w_r = fx.Int32(bo.buffer_load(rs_sw, grow, vec_width=1, dtype=T.i32)).bitcast(fx.Float32)
                if const_expr(not SORTED):
                    rid_h.append(fx.Int32(bo.buffer_load(rs_id, grow, vec_width=1, dtype=T.i32)))
                # an all-zero group quantizes to scale 0 (its acc part is 0 too)
                s1 = (s1 == fx.Float32(0.0)).select(fx.Float32(1.0), s1)
                ratio_h.append((s0 / s1).bitcast(fx.Int32))
                fin_h.append((s1 * w_r).bitcast(fx.Int32))

            def fetch(q, row_static, row_lane):
                src = fx.Int32(row_static % 64) + row_lane
                return fx.Int32(rocdl.ds_bpermute(T.i32, src * fx.Int32(4), q[row_static // 64]))

            # Per CTA, hoisted out of the tile loop (rows do not change across
            # n-tiles): this lane's 4 acc rows (L//16)*4 + v of every block mi.
            ratio4, fin4, offs4 = [], [], []
            odd = lane & fx.Int32(1)
            is_odd = odd == fx.Int32(1)
            # direct64 pairing: lane keeps block odd (cols c, c+1 of the pair)
            col_b = (odd * fx.Int32(16) + lane_mod_16 - odd) * fx.Int32(2)
            sel_pair = is_odd.select(fx.Int32(0x03020706), fx.Int32(0x05040100))
            for mi in range_constexpr(MR):
                rr, ff, oo = [], [], []
                for v in range_constexpr(4):
                    rr.append(fetch(ratio_h, mi * 16 + v, lane_div_16 * fx.Int32(4)).bitcast(fx.Float32))
                    ff.append(fetch(fin_h, mi * 16 + v, lane_div_16 * fx.Int32(4)).bitcast(fx.Float32))
                    if const_expr(SORTED):  # row of the block in routes[sorted_pos]
                        oo.append((fx.Int32(mi * 16 + v) + lane_div_16 * fx.Int32(4))
                                  * fx.Int32(RS) + col_b)
                    else:
                        tok = fetch(rid_h, mi * 16 + v, lane_div_16 * fx.Int32(4)) & fx.Int32(0x00FFFFFF)
                        oo.append((tok < i32_M).select(tok * fx.Int32(N_OUT * 2) + col_b, fx.Int32(-1)))
                ratio4.append(rr)
                fin4.append(ff)
                offs4.append(oo)

            def epilogue_mi(prev, mi):
                datas = []
                for k in range_constexpr(4):
                    r0 = _rne_hi(Vec(prev["acc"][mi][0].load())[k] * (fin4[mi][k] * prev["f_n"][0]))
                    r1 = _rne_hi(Vec(prev["acc"][mi][1].load())[k] * (fin4[mi][k] * prev["f_n"][1]))
                    pk = G._perm_u32(r1, r0, 0x07060302)  # (bf16 blk0, bf16 blk1)
                    nb = fx.Int32(rocdl.update_dpp(T.i32, _raw(pk.bitcast(fx.Int32)),
                                                   _raw(pk.bitcast(fx.Int32)),
                                                   DPP_QUAD_SWAP, 0xF, 0xF, True))
                    datas.append(_raw(G._perm_u32(nb, pk, sel_pair).bitcast(fx.Int32)))
                if const_expr(SORTED):
                    base = fx.Int64(arg_out) + fx.Int64(m_row) * fx.Int64(RS) \
                        + fx.Int64(prev["by_n"] + n_wave_base) * fx.Int64(2)
                    _stores4(datas, offs4[mi], base)
                else:
                    base = fx.Int64(arg_out) + fx.Int64((prev["by_n"] + n_wave_base) * fx.Int32(2))
                    _atomics4(datas, [_raw(o) for o in offs4[mi]], base)

            zero4 = fx.Vector.filled(4, 0.0, fx.Float32)
            prev = None
            for t in range_constexpr(NT):
                if const_expr(t + 2 < NT):
                    W[t + 2] = w_issue(t + 2)
                cur = W[t]
                p = cur["packed"]
                luts = [[
                    G._fp8_lut_predelta(G._perm_u32(p, p, ((g % 2) * 2 + ni) * 0x01010101))
                    for g in range(K_GROUPS)] for ni in range(N_REPEAT)]
                acc = [[fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
                        for _ in range(N_REPEAT)] for _ in range(MR)]
                for mi in range_constexpr(MR):
                    for ni in range_constexpr(N_REPEAT):
                        acc[mi][ni].store(zero4)
                for ku in range_constexpr(K_STEPS):
                    if const_expr(ku == 4):  # acc = acc0 * s0 / s1, then group 1
                        for mi in range_constexpr(MR):
                            rt = ratio4[mi]
                            for ni in range_constexpr(N_REPEAT):
                                a4 = Vec(acc[mi][ni].load())
                                acc[mi][ni].store(Vec.from_elements(
                                    [a4[v] * rt[v] for v in range(4)], fx.Float32))
                    b8s = [G._fp4x8_to_fp8x8_fast(cur["raw"][ni][ku // 4][ku % 4],
                                                  luts[ni][ku // 4])
                           for ni in range_constexpr(N_REPEAT)]
                    for mi in range_constexpr(MR):
                        a8 = a_frag(mi, ku)
                        for ni in range_constexpr(N_REPEAT):
                            G._mma_fp8(mma_atom, acc[mi][ni], a8, b8s[ni])
                    # previous tile's row block ku rides along with this K step
                    if const_expr(prev is not None and ku < MR):
                        epilogue_mi(prev, ku)
                prev = dict(acc=acc, by_n=cur["by_n"],
                            f_n=[G._channel_factor(x) for x in cur["exp"]])
            for mi in range_constexpr(MR):
                epilogue_mi(prev, mi)

        if const_expr(dyn_rows):
            tok64 = fx.Int32(rocdl.readfirstlane(
                T.i32, _raw(_global_i32_at(arg_stids, m_row + fx.Int32(64))))) \
                & fx.Int32(0x00FFFFFF)
            cond = tok64 >= i32_M  # rows 64.. are all padding

            @flyc.jit
            def _dispatch():
                if cond:
                    run(64)
                else:
                    run(BM)

            _dispatch()
        else:
            run(BM)

    @flyc.jit
    def _entry(arg_a, arg_a_scale, arg_w, arg_w_scale, arg_eids, arg_cumsum,
               arg_stids, arg_sweights, arg_out, i32_M):
        tx = fx.Int32(gpu.thread_id("x"))
        bx = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx // fx.Int32(64))
        total_m = _global_i32_at(arg_cumsum, fx.Int32(0)) // fx.Int32(BM)
        bound = total_m * fx.Int32(NBG)
        # bijective XCD round-robin over the valid tiles (as test_a8w4_gemm2)
        xq = _udiv(bound, 8)
        xr = _umod(bound, 8)
        if bx < bound:
            xc = _umod(bx, 8)
            tile = xc * xq + fx.min(xc, xr) + _udiv(bx, 8)
            _body(arg_a, arg_a_scale, arg_w, arg_w_scale, arg_eids, arg_stids,
                  arg_sweights, arg_out, i32_M, tile, lane, wave)

    @flyc.kernel(name=kname, known_block_size=[256, 1, 1])
    def kernel(arg_a: fx.Int64, arg_a_scale: fx.Int64, arg_w: fx.Int64,
               arg_w_scale: fx.Int64, arg_eids: fx.Int64, arg_cumsum: fx.Int64,
               arg_stids: fx.Int64, arg_sweights: fx.Int64, arg_out: fx.Int64,
               i32_M: fx.Int32):
        _entry(arg_a, arg_a_scale, arg_w, arg_w_scale, arg_eids, arg_cumsum,
               arg_stids, arg_sweights, arg_out, i32_M)

    @flyc.jit
    def launch(arg_a: fx.Int64, arg_a_scale: fx.Int64, arg_w: fx.Int64,
               arg_w_scale: fx.Int64, arg_eids: fx.Int64, arg_cumsum: fx.Int64,
               arg_stids: fx.Int64, arg_sweights: fx.Int64, arg_out: fx.Int64,
               i32_M: fx.Int32, i32_grid: fx.Int32, stream: fx.Stream):
        kernel(arg_a, arg_a_scale, arg_w, arg_w_scale, arg_eids, arg_cumsum,
               arg_stids, arg_sweights, arg_out, i32_M,
               value_attrs={"rocdl.waves_per_eu": waves_per_eu},
               ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(256, 1, 1), stream=stream)

    return launch


# ---- sorted-order reduce: invert_sorted_ids + sorted_sum ------------------------------------
@functools.cache
def compile_invert_sorted_ids(TOPK):
    """loc[token * TOPK + slot] = sorted position, for the valid sorted rows."""

    @flyc.jit
    def body(arg_sids, arg_loc, arg_nv, i32_M):
        slot = fx.Int32(gpu.block_id("x")) * fx.Int32(256) + fx.Int32(gpu.thread_id("x"))
        if slot < _global_i32_at(arg_nv, fx.Int32(0)):
            sid = fx.Int32(_global_i32_at(arg_sids, slot))
            tok = sid & fx.Int32(0x00FFFFFF)
            if tok < i32_M:
                G.global_typed_ptr(arg_loc, T.i32)[tok * fx.Int32(TOPK) + sid.shrui(fx.Int32(24))] = slot

    @flyc.kernel(name=f"a8w4_invert_sorted_ids_k{TOPK}", known_block_size=[256, 1, 1])
    def kernel(arg_sids: fx.Int64, arg_loc: fx.Int64, arg_nv: fx.Int64, i32_M: fx.Int32):
        body(arg_sids, arg_loc, arg_nv, i32_M)

    @flyc.jit
    def launch(arg_sids: fx.Int64, arg_loc: fx.Int64, arg_nv: fx.Int64, i32_M: fx.Int32,
               i32_grid: fx.Int32, stream: fx.Stream):
        kernel(arg_sids, arg_loc, arg_nv, i32_M).launch(
            grid=(fx.Int64(i32_grid), 1, 1), block=(256, 1, 1), stream=stream)

    return launch


@functools.cache
def compile_sorted_sum(TOPK, N, ROW_PAD=0):
    """out[t, :] = sum_k routes[loc[t, k], :] (bf16, f32 sum); 16 B per lane, one
    WG covers 2048 columns of one token (the TOPK row ids are uniform loads)."""
    assert N % 2048 == 0
    WPR = N // 2048  # WGs per token row

    @flyc.jit
    def body(arg_loc, arg_routes, arg_out):
        bx = fx.Int32(gpu.block_id("x"))
        tok = bx // fx.Int32(WPR)
        c = (bx % fx.Int32(WPR)) * fx.Int32(256) + fx.Int32(gpu.thread_id("x"))  # 16 B chunk
        rs_r = _rsrc(arg_routes)
        rs_o = _rsrc(arg_out)
        locs = [fx.Int32(_global_i32_at(arg_loc, tok * fx.Int32(TOPK) + fx.Int32(k)))
                for k in range_constexpr(TOPK)]
        vs = [Vec(bo.buffer_load(rs_r, locs[k] * fx.Int32((N * 2 + ROW_PAD) // 4) + c * fx.Int32(4),
                                 vec_width=4, dtype=T.i32))
              for k in range_constexpr(TOPK)]
        acc = Vec(vs[0].bitcast(fx.BFloat16)).to(fx.Float32)
        for k in range_constexpr(1, TOPK):
            acc = acc + Vec(vs[k].bitcast(fx.BFloat16)).to(fx.Float32)
        bo.buffer_store(_raw(Vec(acc.to(fx.BFloat16)).bitcast(fx.Int32)), rs_o,
                        tok * fx.Int32(N // 2) + c * fx.Int32(4))

    @flyc.kernel(name=f"a8w4_sorted_sum_k{TOPK}_n{N}_p{ROW_PAD}", known_block_size=[256, 1, 1])
    def kernel(arg_loc: fx.Int64, arg_routes: fx.Int64, arg_out: fx.Int64):
        body(arg_loc, arg_routes, arg_out)

    @flyc.jit
    def launch(arg_loc: fx.Int64, arg_routes: fx.Int64, arg_out: fx.Int64,
               i32_grid: fx.Int32, stream: fx.Stream):
        kernel(arg_loc, arg_routes, arg_out).launch(
            grid=(fx.Int64(i32_grid), 1, 1), block=(256, 1, 1), stream=stream)

    return launch, WPR


def a8w4_gemm2_bm128(a_fp8, a_scale, w2, w2_scale, sorted_ids, sorted_weights,
                     sorted_expert_ids, num_valid_ids, out, *, num_experts, n_tiles=8,
                     b_cache_mod=2, waves_per_eu=1, dyn_rows=True, out_mode="atomic",
                     topk=8, routes=None, loc=None, stages=("gemm", "invert", "sum"),
                     row_pad=0):
    """Host launcher. Sorting block_m must be 128; ``w2`` / ``w2_scale`` from
    ``prep_w2_fast(..., 2)``; ``a_scale`` f32 [sorted_size, 2]; ``out`` bf16
    [tokens, model_dim]: zeroed for ``out_mode="atomic"``; ``"sorted"`` writes
    ``routes`` bf16 [sorted_size, model_dim] (allocated if None) and reduces
    into ``out`` through ``loc`` int32 [tokens * topk]. ``stages`` (sorted mode,
    for timing) selects which of the three launches run."""
    tokens, model_dim = out.shape
    stream = torch.cuda.current_stream()
    sorted_size = sorted_expert_ids.numel() * BM
    if out_mode == "sorted":
        if routes is None:
            routes = torch.empty((sorted_size, model_dim + row_pad // 2), dtype=out.dtype,
                                 device=out.device)
        if loc is None:
            loc = torch.empty(tokens * topk, dtype=torch.int32, device=out.device)
    dst = routes if out_mode == "sorted" else out
    if "gemm" in stages:
        launch = compile_a8w4_gemm2_bm128(
            D_HIDDEN=model_dim, D_INTER=a_fp8.shape[1], NE=num_experts, NT=n_tiles,
            b_cache_mod=b_cache_mod, waves_per_eu=waves_per_eu, dyn_rows=dyn_rows,
            OUT_MODE=out_mode, ROW_PAD=row_pad if out_mode == "sorted" else 0)
        grid = sorted_expert_ids.numel() * (model_dim // TILE_N // n_tiles)
        _run_compiled(
            launch, a_fp8.data_ptr(), a_scale.data_ptr(), w2.data_ptr(), w2_scale.data_ptr(),
            sorted_expert_ids.data_ptr(), num_valid_ids.data_ptr(), sorted_ids.data_ptr(),
            sorted_weights.data_ptr(), dst.data_ptr(), int(tokens), int(grid), stream,
        )
    if out_mode == "sorted":
        if "invert" in stages:
            _run_compiled(compile_invert_sorted_ids(topk), sorted_ids.data_ptr(),
                          loc.data_ptr(), num_valid_ids.data_ptr(), int(tokens),
                          int((sorted_size + 255) // 256), stream)
        if "sum" in stages:
            launch_sum, wpr = compile_sorted_sum(topk, model_dim, row_pad)
            _run_compiled(launch_sum, loc.data_ptr(), routes.data_ptr(), out.data_ptr(),
                          int(tokens * wpr), stream)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-t", type=int, nargs="*", default=[2048, 8192, 16384])
    p.add_argument("--nt", type=int, nargs="*", default=[4, 8])
    p.add_argument("--wpe", type=int, nargs="*", default=[1])
    p.add_argument("--mode", nargs="*", default=["atomic", "sorted"])
    p.add_argument("--pad", type=int, nargs="*", default=[0], help="sorted routes row padding (B)")
    args = p.parse_args()

    E, MD, INTER, TOPK = 256, 4096, 256, 8
    torch.manual_seed(0)
    w2 = torch.randn((E, MD, INTER), dtype=dtypes.bf16, device="cuda")
    w2_qt, w2_scale = aiter.get_torch_quant(QuantType.per_1x32)(w2, quant_dtype=dtypes.fp4x2)
    del w2
    w2_qt = w2_qt.view(E, MD, INTER // 2)
    w2_u8 = shuffle_weight_a16w4(w2_qt, 16, False).view(torch.uint8)
    w2_s = shuffle_scale_a16w4(w2_scale, E, False).view(torch.uint8).contiguous().view(-1)
    W, WS = G.prep_w2_fast(w2_u8, w2_s, 2)

    for tok in args.t:
        x = torch.randn((tok, MD), dtype=dtypes.bf16, device="cuda")
        tw, ti = fused_topk(x, torch.randn((tok, E), dtype=dtypes.bf16, device="cuda"), TOPK, True)
        prep = {}
        for bm in (64, 128):
            sids, sw, seids, nv, _ = moe_sorting(ti, tw, E, MD, dtypes.bf16, bm)[:5]
            inter = torch.randn((seids.numel() * bm, INTER), dtype=dtypes.bf16, device="cuda")
            a8, asc = G.quantize_groups(inter, 128)
            ref = G.run_torch(a8, asc, w2_qt, w2_scale, sids, sw, seids, nv, tok, bm).float()
            prep[bm] = (a8, asc, sids, sw, seids, nv, ref)

        def run(fn_of_out, bm):
            out = torch.zeros((tok, MD), dtype=dtypes.bf16, device="cuda")
            fn = lambda: fn_of_out(out)  # noqa: E731
            fn()
            torch.cuda.synchronize()
            cos = G._cos_diff(prep[bm][-1], out.float())
            _, us = run_perftest(fn)
            return us, cos

        def g64(nt, ep):
            a8, asc, sids, sw, seids, nv, _ = prep[64]
            return lambda out: G.a8w4_gemm2(
                a8, asc, W, WS, sids, sw, seids, nv, out, num_experts=E, tile_m=64,
                tile_n=128, epilog=ep, dyn_rows=True, a_group=128, fast_dec=2,
                n_tiles=nt, row_vgpr=bool(nt))

        us0, c0 = run(g64(0, "lds"), 64)
        us1, c1 = run(g64(4, "direct64"), 64)
        line = [f"tok {tok:6d}: gemm2 bm64 {us0:7.1f} us (cos {c0:.1e}) | "
                f"bm64 nt4 direct64 {us1:7.1f}"]
        a8, asc, sids, sw, seids, nv, _ = prep[128]
        loc = torch.empty(tok * TOPK, dtype=torch.int32, device="cuda")
        modes = [(m, pd) for m in args.mode for pd in (args.pad if m == "sorted" else [0])]
        for mode, pad in modes:
            routes = torch.empty((seids.numel() * 128, MD + pad // 2), dtype=dtypes.bf16,
                                 device="cuda")
            for nt in args.nt:
                for wpe in args.wpe:
                    tag = f"bm128 {mode[:3]}{f' p{pad}' if pad else ''} nt{nt} w{wpe}"

                    def call(out, stages=("gemm", "invert", "sum")):
                        return a8w4_gemm2_bm128(
                            a8, asc, W, WS, sids, sw, seids, nv, out, num_experts=E,
                            n_tiles=nt, waves_per_eu=wpe, out_mode=mode, topk=TOPK,
                            routes=routes, loc=loc, stages=stages, row_pad=pad)

                    try:
                        us, c = run(call, 128)
                        bad = "" if c < 1e-4 else " BAD"
                        extra = ""
                        if mode == "sorted":  # per-stage kernel times
                            st = {k: run(lambda out, k=k: call(out, (k,)), 128)[0]
                                  for k in ("gemm", "invert", "sum")}
                            extra = (f" [gemm {st['gemm']:.1f} invert {st['invert']:.1f}"
                                     f" sum {st['sum']:.1f}]")
                        line.append(f"{tag} {us:7.1f} ({us0 / us:.2f}x cos {c:.1e}{bad}){extra}")
                    except Exception as ex:  # noqa: BLE001
                        line.append(f"{tag} FAIL {type(ex).__name__}: {str(ex)[:200]}")
        print("\n   ".join(line), flush=True)


if __name__ == "__main__":
    main()
