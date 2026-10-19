# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Vendored from ROCm/aiter#5882 (https://github.com/ROCm/aiter/pull/5882),
# commit 03359c43df (03359c43df40abf91ac38a2040c9b8e284d92297),
# file aiter/ops/flydsl/kernels/mla_v4_decode_gfx950.py.
# Unchanged except for import paths and sglang formatting (isort, ruff format).
# TODO(aiter#5882): remove once ROCm/aiter#5882 is in the aiter pin

"""DeepSeek-V4 fp8 MLA decode (v4 nm contract) in one FlyDSL launch on gfx950.

Replaces the asm mla_a8w8_qh64_qseqlen1_gqaratio64_nm kernel + its split merge.
q / kv rows: 448 e4m3 + 7 e8m0 scales (each stored twice at bytes 448..461; bytes
462..511 never read), bf16 rope, page size 1, one kv head. Optional per-q-row key
ranges (q_kv_bounds, grouped DSpark verify) and a DeepSeek-V4 epilogue (inverse RoPE +
wo_a mxfp8 quant).

Grid: ceil(B / 8) * 8 * NRG * S CTAs; bid % 8 is the XCD, so all CTAs of a sequence
share one L2 (SPREAD / XR / SK / ADAPT change the map). Split s walks key tiles
[s * tps, (s + 1) * tps); kv lengths are never compile-time values.

Tile loops (V4Cfg.LAYOUT; rows = 16 * RB q rows of one sequence per CTA):
  K   4 waves, 64-key tiles: fp8 QK with the row e8m0 scales as MFMA block scales, V
      dequantized once to bf16 in LDS, lazy online softmax, each wave owns 128 of the
      512 output columns.
  RQ  32-row waves, 32-key tiles (64 or 128-row CTAs): 32x32 fp8 QK, PV over all 512
      columns from AGPR accumulators, K / V tiles DMA'd global -> LDS ahead.
  RD  the RQ QK with PV split over dv; Q16: two wave roles, 16-row QK waves (QK +
      softmax of step s + 1) next to 4 PV waves (PV of step s + dequant), one barrier
      per tile.
A variant's SUB bodies are other K-layout (or, under Q16, 32-row Q16) variants in the
same kernel; each sequence runs the first whose key bound covers its stream.

Cross-split merge (S > 1), co-residency free and graph replayable: per merge group one
SLOT_BYTES slot = arrive | stay mask (i64) + one start ticket per split. Every CTA
stores its ticket at entry; at the arrive wave 0 reads the tickets (bounded re-polls):
all started -> stay and merge its own slice, else leave the slice to the last
arriver. Release: vmcnt(0) if all siblings are on this XCD (or partials were written
through), else an agent-scope L2 write-back; then one returning i64 atomic add. The
last arriver zeroes the slot and merges the slices of every leaver. No CTA ever waits
for one that has not started.
"""

import collections
import functools
import hashlib
import os
import types
from dataclasses import dataclass, replace

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels import communication_ops_utils as cu
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm as _llvm_d
from flydsl._mlir.dialects import rocdl as _rocdl_d
from flydsl.expr import arith as _arith
from flydsl.expr import math as fly_math
from flydsl.expr.typing import T

from .mla_v4_decode_common import (
    butterfly_i32,
    exp2,
    fabs_bits,
    fabs_f32,
    fp8_scaled,
    gload,
    inv_l1,
    ld64_agent,
    ld_agent_f32,
    ld_agent_i32,
    maxnum_f32,
    mxfp8_exp,
    pin_v64,
    plswap,
    rcp,
    rmw_add64_agent,
    smax_i32,
    smin_i32,
    spin_all_arrived,
    spin_tickets_state,
    st_agent,
    st_agent_vec,
    st_raw,
    udiv,
    urem,
    xcc_id,
)

WAVE = 64
NWAVES = 4
NTHREADS = WAVE * NWAVES
NOPE = 448
ROPE = 64
DV = 512
ROWB = 512
KT_K = 64  # keys per K-layout tile (V4Cfg.KT)
VPITCH = DV + 8
PPITCH = KT_K + 8
MAX_S = 32
SCALE_LOG2E = DV**-0.5 * LOG2E
# merge-group slot: arrive | stay mask (i64 @ A_OFF), start tickets (i32 per split @
# TK_OFF, XCC_ID + 1); WS_HDR bytes of slots, zero between launches
SLOT_BYTES = 256
WS_HDR = 1 << 20
MAX_GROUPS = WS_HDR // SLOT_BYTES
A_OFF, TK_OFF = 0, 64
# bounded re-polls of the start tickets before a CTA leaves its slice
START_POLLS = 8
# non-temporal load cache policy
CPOL_NT = 2
# RQ / RD LDS ring slot bytes: fp8 keys, rope keys, V (one 32-key tile each)
RQ_KF = 32 * 32 * 16
RQ_KR = 8 * 32 * 16
RQ_V = 56 * 32 * 16


def rq_rings(rq, rows):
    """(key tiles in flight, fp8-key / rope-key / V / key-index ring slots)."""
    if rq and rows == 64:
        return 2, 2, 3, 1, 3
    return 3, 4, 4, 2, 4


def rq_lds_layout(rd, rows, nkd, rings, npb, nml):
    _, nkf, nkr, nv, nix = rings
    kr_off = nkf * RQ_KF
    v_off = kr_off + nkr * RQ_KR
    ix_off = v_off + nv * RQ_V
    p_off = ix_off + nix * (2 * 256 * 3 if nkd == 3 else 4 * 256 * 2)
    a_off = p_off + npb * rows * 80
    ml_off = a_off + npb * rows * 4
    total = (ml_off + nml * rows * 4) if rd else p_off
    return kr_off, v_off, ix_off, p_off, a_off, ml_off, total


@dataclass(frozen=True)
class V4Cfg:
    """One compiled variant (also the compile-cache key); plans in mla_v4_decode.py.

    H, MSQ: q heads, q rows per sequence (grouped verify: per group). RB: 16-row blocks
    per CTA (NRG = H * MSQ / rows row groups). S: kv splits per (sequence, row group).
    EPI: "bf16" output or "invrope_mxfp8" (inverse RoPE + wo_a mxfp8 quant).
    KC: coalesced K-layout KV gather. LAYOUT: tile loop (module docstring).
    MASK: "bounds" reads q_kv_bounds. ADAPT (K): run-time split count per sequence from
    its tiles and the launch's, for ADAPT CTA slots. MINT: minimum tiles per split.
    Q16 (RD): the two-role loop. SHORT (Q16, SK): streams of <= SHORT tiles run unsplit;
    SHORTM: also capped by the launch's mean stream; SHORTF: <= SHORTF tiles always.
    PF16 (Q16): fp16 split partials, power-of-2 scaled. XR: a sequence's row groups on
    consecutive XCDs. SUB: ((max keys, body), ...) stream-length bodies.
    SPREAD: a sequence's splits over all XCDs (small batches, written-through
    partials). PFN (Q16): after its last tile a CTA prefetches the metadata, q rows and
    first key indices of CTA bid + PFN into L2. OWN (RQ): split j merges column
    quarters j (mod S), its own partial stays in registers. SK (K): grid of SK CTAs,
    SKD splits per sequence, or, when the longest split would set the tail (runs of
    >= SKT tiles, or one of > SKT tiles, 2 runs and 4 shares), the batch's keys cut into
    SK equal runs; SKMEAN: only streams longer than the mean split. PV16R = (lo, hi)
    (unsplit Q16): streams of lo < keys <= hi use 16x16x32 PV MFMAs (less energy per
    key under the power cap). TAIL: units (sequences, XR: row groups) past the last
    complete XCD octet spread over all XCDs, their partials written through.
    IDLE = (x, x2, f, w) (Q16): one CTA per (sequence, row group) on an unsplit body,
    plus CTAs up to x + x2 per XCD taking splits 1 .. S - 1 of the XCD's longest
    streams (ranked on device) of > f tiles and longer than the mean; past x only of
    > w tiles.
    """

    H: int = 16
    MSQ: int = 1
    RB: int = 1
    S: int = 1
    EPI: str = "bf16"
    KC: bool = False
    LAYOUT: str = "K"
    MASK: str = "none"
    ADAPT: int = 0
    MINT: int = 1
    Q16: bool = False
    SHORT: int = 0
    SHORTM: bool = False
    SHORTF: int = 0
    PF16: bool = False
    XR: bool = False
    SUB: tuple = ()
    SPREAD: bool = False
    PFN: int = 0
    OWN: bool = False
    SK: int = 0
    SKD: int = 1
    SKMEAN: bool = False
    SKT: int = 12
    PV16R: tuple = ()
    TAIL: bool = False
    IDLE: tuple = ()

    def __post_init__(self):
        assert self.H * self.MSQ % (16 * self.RB) == 0 and 1 <= self.S <= MAX_S, self
        assert not (self.TAIL and (self.SPREAD or self.ADAPT)), self
        assert not (self.TAIL and (self.SHORTM or self.LAYOUT == "RQ")), self

    @property
    def KT(self):
        return 64 if self.LAYOUT == "K" else 32

    def ws_seq(self):
        if self.S == 1:
            return 0, 0
        sp = (self.S + 3) // 4 * 4
        return self.NRG, self.NRG * self.rows * (self.S * DV * 4 + sp * 4)

    def ws_need(self, B):
        g = b = 0
        for c in (self,) + tuple(sub for _, sub in self.SUB):
            g1, b1 = c.ws_seq()
            g, b = g + g1, b + b1
        bpad = -(-B // 8) * 8
        return bpad * g, bpad * b

    @property
    def cps(self):
        return max([self.NRG * self.S] + [sub.NRG * sub.S for _, sub in self.SUB])

    @property
    def NRG(self):
        return self.H * self.MSQ // (16 * self.RB)

    @property
    def rows(self):
        return 16 * self.RB

    @property
    def nthreads(self):
        if self.LAYOUT == "RQ":
            return WAVE * self.RB // 2
        if self.LAYOUT == "RD" and self.Q16:
            return WAVE * (4 + self.RB)
        if self.LAYOUT in ("K", "RD"):
            return NTHREADS
        return WAVE * self.RB


def plswap_f(x, width, op2):
    def op2f(a, b):
        return op2(a.bitcast(fx.Float32), b.bitcast(fx.Float32))

    return plswap(fx.Float32(x).bitcast(fx.Int32), width, op2f)


def _fmax(a, b):
    return a.maximumf(b)


def _fadd(a, b):
    return a + b


def _div_const(a, d):
    if d & (d - 1) == 0:
        return a >> fx.Int32(d.bit_length() - 1)
    return udiv(a, fx.Int32(d))


def _ult(a, b):
    return fx.Boolean(_arith.cmpi(_arith.CmpIPredicate.ult, a, b))


def _rem_const(a, d):
    if d & (d - 1) == 0:
        return a & fx.Int32(d - 1)
    return urem(a, fx.Int32(d))


def xcd_ranks(indptr_ptr, nseq, x, lane, nmax):
    """Lane l: stream j = 8 l + x, whether it exists, and its rank among those streams
    (longest first, ties by j); at most nmax of them."""
    j = (lane << fx.Int32(3)) | x
    ok = j < nseq
    jc = ok.select(j, fx.Int32(0))
    a0 = fx.Int32(fx.ptr_load(indptr_ptr + fx.Int64(jc)))
    n = fx.Int32(fx.ptr_load(indptr_ptr + fx.Int64(jc) + fx.Int64(1))) - a0
    n = ok.select(n, fx.Int32(-1))
    rk = fx.Int32(0)
    for m in range(nmax):
        nm = fx.Int32(fx.rocdl.readlane(T.i32, n.ir_value(), fx.Int32(m).ir_value()))
        first = (nm > n) | ((nm == n) & (lane > fx.Int32(m)))
        rk = rk + first.select(fx.Int32(1), fx.Int32(0))
    return j, ok, rk


def _sload(ptr, i):
    return fx.Int32(
        buffer_ops.buffer_load(
            buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(fx.ptrtoint(ptr)).ir_value()
            ),
            i,
            vec_width=1,
            is_scalar=True,
        )
    )


# one opaque SGPR use of all xs: loads issued before a branch stay one round trip
def _pin_s(xs):
    vs = [x.ir_value() for x in xs]
    n = len(vs)
    r = _llvm_d.InlineAsmOp(
        _llvm_d.StructType.get_literal([v.type for v in vs]),
        vs,
        "; " + " ".join(f"${i}" for i in range(n)),
        ",".join(["=s"] * n + [str(i) for i in range(n)]),
        has_side_effects=True,
    ).res
    return [
        type(x)(_llvm_d.extractvalue(v.type, r, [i]))
        for i, (x, v) in enumerate(zip(xs, vs))
    ]


def _tail_map(bid, nseq, units, per):
    # V4Cfg.TAIL: -> (unit, CTA of the unit); complete octets of units keep unit % 8 as
    # the XCD, the rest take consecutive CTAs (all XCDs)
    full = (nseq * fx.Int32(units)) & fx.Int32(-8)
    ncol = full * fx.Int32(per)
    bj = bid >> fx.Int32(3)
    k = _div_const(bj, per)
    t = bid - ncol
    kt = _div_const(t, per)
    colo = bid < ncol
    return (
        colo.select((k << fx.Int32(3)) | (bid & fx.Int32(7)), full + kt),
        colo.select(bj - k * fx.Int32(per), t - kt * fx.Int32(per)),
    )


# one merge round's rows of a split slice (see slice_addrs)
_SliceRow = collections.namedtuple(
    "_SliceRow", "act head qpos d0 la pa sa frr cblk psa"
)


class _AccRow:
    def __init__(self, acc, npb, rb, lazy):
        self._acc, self._npb, self._rb = acc, npb, rb
        self._runs = None if lazy else [self._run(n) for n in range(16)]

    def _run(self, n):
        blk = self._acc[(n // 4) * self._npb + self._rb]
        return fx.Vector.from_elements(
            [blk[4 * (n % 4) + i] for i in range(4)], fx.Float32
        )

    def __getitem__(self, n):
        return self._run(n) if self._runs is None else self._runs[n]


def _emitter(cfg: V4Cfg, ws_prev=(0, 0)):
    # emit(): split_range / load_q / protocol_entry, the LAYOUT tile loop (RQ, then
    # RD / Q16, K), then direct_epi (unsplit) or merged_epi (cross-split merge)
    H, MSQ, RB, S = cfg.H, cfg.MSQ, cfg.RB, cfg.S
    HB = H // 16
    NRG = cfg.NRG
    ROWS = cfg.rows
    EPI_Q = cfg.EPI == "invrope_mxfp8"
    KC = cfg.KC
    RQ = cfg.LAYOUT == "RQ"
    RD = cfg.LAYOUT == "RD"
    KL = cfg.LAYOUT == "K"
    OWN = cfg.OWN and S > 1
    RQM = RQ and (RB == 4 or OWN)
    PF16 = cfg.Q16 and cfg.PF16
    PEB = 2 if PF16 else 4
    BOUNDS = cfg.MASK == "bounds"
    NT = cfg.nthreads
    KTL = cfg.KT
    KT_SH = KTL.bit_length() - 1
    NW = NT // WAVE
    RINGS = rq_rings(RQ, cfg.rows)
    DD, NKF, NKR, NVB_RQ, NIX = RINGS
    NROLE = 4 // NW if RQ else 1
    Q16 = cfg.Q16
    # key rows nobody else reads (one row group) stream in as non-temporal loads: they
    # evict no dirty last-level-cache lines that would be written back mid-kernel
    KV_NT = Q16 and NRG == 1
    PV16R = cfg.PV16R if (Q16 and S == 1) else ()
    RD_NQW = cfg.RB if Q16 else cfg.rows // 32
    RD_PVD = 2
    (
        RQ_KR_OFF,
        RQ_V_OFF,
        RQ_IX_OFF,
        RD_P_OFF,
        RD_A_OFF,
        RD_ML_OFF,
        RQD_LDS,
    ) = rq_lds_layout(
        RD,
        cfg.rows,
        3 if (RD and RD_NQW < 4 and not Q16) else 2,
        RINGS,
        2 if Q16 else 1,
        3 if Q16 else 2,
    )
    MINT = cfg.MINT
    ADAPT = cfg.ADAPT if S > 1 else 0
    SHORT = cfg.SHORT if S > 1 and not cfg.SK else 0
    SHORTM = cfg.SHORTM and SHORT > 0
    SHORTF = cfg.SHORTF if SHORTM else 0
    SK = cfg.SK if S > 1 else 0
    DYN = ADAPT or SK
    QROW_EPI = MSQ > 1
    MERGED = S > 1
    Q16_OSC = Q16 and (not MERGED or SHORT > 0)
    WTP = (cfg.SPREAD or cfg.TAIL and not cfg.SK) and MERGED
    SP = (S + 3) // 4 * 4
    NQB = ROWS * 4
    lse_chunks = []
    k = 0
    while k < S:
        n = 4 if S - k >= 4 else (2 if S - k >= 2 else 1)
        lse_chunks.append((k, n))
        k += n
    bpc = (NQB + S - 1) // S
    if DYN:
        bpc = (NQB + 1) // 2
    EPT = 2 if bpc <= 4 else 4
    TPB = 128 // EPT
    BPR = NT // TPB
    NRND = (bpc + BPR - 1) // BPR
    MCH = max(1, min(NRND, 64 // (S * EPT)))
    SKP0 = min(SP, -(-max(4, cfg.SKD) // 4) * 4)
    if SK:
        MCH = max(1, min(NRND, 64 // (SKP0 * EPT), -(-NQB // (cfg.SKD * BPR))))
    MCHUNKS = list(range(0, NRND, MCH))
    FULL = (1 << S) - 1
    FULL32 = FULL - (1 << 32) if FULL >= 1 << 31 else FULL

    if RQ or RD:

        @fx.struct
        class Smem:
            buf: fx.Array[fx.Uint8, RQD_LDS, 16]
            flag: fx.Array[fx.Int32, 4, 16]

    else:

        @fx.struct
        class Smem:
            vlds: fx.Array[fx.BFloat16, KT_K * VPITCH, 16]
            plds: fx.Array[fx.BFloat16, ROWS * PPITCH, 16]
            rmax: fx.Array[fx.Float32, NWAVES * ROWS, 16]
            rsum: fx.Array[fx.Float32, NWAVES * ROWS, 16]
            flag: fx.Array[fx.Int32, 4, 16]

    name = (
        f"mla_v4_decode_h{H}_q{MSQ}_rb{RB}_s{S}_{cfg.EPI}"
        + ("_kc" if KC else "")
        + ("_rq" if RQ else "")
        + ("_rd" if RD else "")
        + ("q16" if Q16 else "")
        + ("_xr" if cfg.XR else "")
        + (f"_sh{cfg.SHORT}" if cfg.SHORT and S > 1 else "")
        + ("m" if cfg.SHORTM and S > 1 else "")
        + (f"f{cfg.SHORTF}" if cfg.SHORTM and cfg.SHORTF and S > 1 else "")
        + ("_pf16" if cfg.PF16 and S > 1 else "")
        + ("_bnd" if BOUNDS else "")
        + (f"_ad{cfg.ADAPT}m{cfg.MINT}" if ADAPT else "")
        + (f"_sk{SK}d{cfg.SKD}t{cfg.SKT}" if SK else "")
        + ("m" if SK and cfg.SKMEAN else "")
        + ("_own" if OWN else "")
        + (f"_pv{PV16R[0]}_{PV16R[1]}" if PV16R else "")
        + ("_tl" if cfg.TAIL else "")
    )

    def map_bid(bid, nseq):
        xcd = bid & fx.Int32(7)
        bj = bid >> fx.Int32(3)
        if fx.const_expr(cfg.TAIL and cfg.XR):
            unit, split = _tail_map(bid, nseq, NRG, S)
            return _div_const(unit, NRG), _rem_const(unit, NRG), split
        if fx.const_expr(cfg.TAIL):
            seq, c = _tail_map(bid, nseq, 1, NRG * S)
            rg = _div_const(c, S)
            return seq, rg, c - rg * fx.Int32(S)
        if fx.const_expr(ADAPT or (SHORT and not cfg.XR)):
            # split-major: the CTAs of short (unsplit) streams dispatch first
            gpx = ((nseq + fx.Int32(7)) >> fx.Int32(3)) * fx.Int32(NRG)
            split = udiv(bj, gpx)
            k2 = bj - split * gpx
        elif fx.const_expr(cfg.XR):
            split = _rem_const(bj, S)
            unit = (_div_const(bj, S) << fx.Int32(3)) | xcd
            return _div_const(unit, NRG), _rem_const(unit, NRG), split
        else:
            split = _rem_const(bj, S)
            k2 = _div_const(bj, S)
        rg = _rem_const(k2, NRG)
        seq = (_div_const(k2, NRG) << fx.Int32(3)) | xcd
        return seq, rg, split

    @flyc.jit
    def emit(
        nseq: fx.Int32,
        num_rows: fx.Int32,
        indptr_ptr: fx.Pointer,
        idx_ptr: fx.Pointer,
        q_ptr: fx.Pointer,
        kv_ptr: fx.Pointer,
        kvr_ptr: fx.Pointer,
        qr_ptr: fx.Pointer,
        qo_ptr: fx.Pointer,
        pos_ptr: fx.Pointer,
        ws_ptr: fx.Pointer,
        sink_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        xs_ptr: fx.Pointer,
        fr_ptr: fx.Pointer,
        out_s0: fx.Int32,
        out_s1: fx.Int32,
        hdr_ptr: fx.Pointer,
        bnd_ptr: fx.Pointer,
        lds,
        seq,
        rg,
        split,
        kvm=None,
        skr=None,
    ):
        v16u8 = fx.Vector.make_type(16, fx.Uint8)
        v4f32 = fx.Vector.make_type(4, fx.Float32)
        v8bf16 = fx.Vector.make_type(8, fx.BFloat16)
        v4bf16 = fx.Vector.make_type(4, fx.BFloat16)
        v2bf16 = fx.Vector.make_type(2, fx.BFloat16)
        zero_i = fx.Int32(0)
        true_ = zero_i == fx.Int32(0)
        false_ = zero_i != fx.Int32(0)
        ninf = fx.Float32(float("-inf"))
        qnan = fx.Float32(float("nan"))
        scale_log2e = fx.Float32(SCALE_LOG2E)

        tid = fx.Int32(fx.thread_idx.x)
        wave = tid // fx.Int32(WAVE)
        lane = tid % fx.Int32(WAVE)
        m = lane % fx.Int32(16)
        g = lane // fx.Int32(16)
        group = seq * fx.Int32(NRG) + rg
        ws_base = fx.Int64(fx.ptrtoint(ws_ptr))
        bpad = fx.Int64(((nseq + fx.Int32(7)) >> fx.Int32(3)) * fx.Int32(8))
        ngroups = bpad * fx.Int64(NRG)
        hdr_a = fx.Int64(fx.ptrtoint(hdr_ptr))
        part_base = ws_base + bpad * fx.Int64(ws_prev[1])
        lse_base = part_base + ngroups * fx.Int64(S * ROWS * DV * PEB)
        psc_base = lse_base + ngroups * fx.Int64(ROWS * SP * 4)

        if fx.const_expr(Q16):
            # two waves per SIMD: pin the AGPR budget to the PV accumulators
            _llvm_d.InlineAsmOp(
                None, [], "; q16 agpr budget", "~{a127}", has_side_effects=True
            )
        if fx.const_expr(RQ or RD):
            p_rq = lds.buf.ptr
        else:
            p_v = lds.vlds.ptr
            p_p = lds.plds.ptr
            p_rmax = lds.rmax.ptr
            p_rsum = lds.rsum.ptr
        p_flag = lds.flag.ptr

        def lds_st(ptr, idx, val):
            fx.ptr_store(val, ptr + idx)

        def lds_ldf(ptr, idx):
            return fx.Float32(fx.ptr_load(ptr + idx))

        seq_c = (seq < nseq).select(seq, zero_i)
        kv_md = None
        kv_all = None
        if fx.const_expr(kvm is not None and len(kvm) == 4):
            q0, qlen = kvm[2], kvm[3]
        elif fx.const_expr(EPI_Q and MSQ == 1):
            q0 = seq_c
            qlen = fx.Int32(1)
        elif fx.const_expr(SK):
            q0, qlen = skr[5]
        else:
            md = [
                _sload(ptr, seq_c + fx.Int32(j))
                for ptr in (qo_ptr, indptr_ptr)
                for j in range(2)
            ]
            if fx.const_expr(SHORTM):
                # the launch's key count in the same round trip
                md += [_sload(indptr_ptr, nseq), _sload(indptr_ptr, zero_i)]
            if fx.const_expr(kvm is None):
                if fx.const_expr(not EPI_Q):
                    # (with the merge slots' kernarg)
                    md = _pin_s(md + [hdr_a] * MERGED)
                    hdr_a = md[-1] if MERGED else hdr_a
                kv_md = (md[2], md[3] - md[2])
            if fx.const_expr(SHORTM):
                kv_all = md[4] - md[5]
            q0 = md[0]
            qlen = md[1] - q0
        slot_base = hdr_a + bpad * fx.Int64(ws_prev[0] * SLOT_BYTES)
        jb0 = rg * fx.Int32(RB)
        active = (seq < nseq) & (_div_const(jb0, HB) < qlen)

        def split_range(sq):
            if fx.const_expr(kvm is not None):
                kv0, kvlen = kvm[0], kvm[1]
            elif fx.const_expr(kv_md is not None):
                kv0, kvlen = kv_md
            else:
                kv0 = fx.Int32(fx.ptr_load(indptr_ptr + fx.Int64(sq)))
                kvlen = (
                    fx.Int32(fx.ptr_load(indptr_ptr + fx.Int64(sq) + fx.Int64(1))) - kv0
                )
            if fx.const_expr(SK):
                t0, t1, ns, lastseg, wthru, _, tps, ntiles = skr
                return kv0, kvlen, ntiles, tps, t0, t1, ns, true_, lastseg, wthru
            ntiles = (kvlen + fx.Int32(KTL - 1)) >> fx.Int32(KT_SH)
            tps = _div_const(ntiles + fx.Int32(S - 1), S)
            if fx.const_expr(MINT > 1):
                tps = (tps < fx.Int32(MINT)).select(fx.Int32(MINT), tps)
            if fx.const_expr(ADAPT):
                kvall = fx.Int32(fx.ptr_load(indptr_ptr + fx.Int64(nseq))) - fx.Int32(
                    fx.ptr_load(indptr_ptr)
                )
                wt = (kvall + nseq * fx.Int32(KTL // 2)) >> fx.Int32(KT_SH)
                nsl = (nseq > fx.Int32(ADAPT)).select(nseq, fx.Int32(ADAPT))
                tgt = udiv(wt + nsl - fx.Int32(1), nsl)
                ns = udiv(ntiles + tgt - fx.Int32(1), tgt)
                if fx.const_expr(MINT > 1):
                    nsf = _div_const(ntiles, MINT)
                    ns = (ns > nsf).select(nsf, ns)
                ns = (ns > fx.Int32(S)).select(fx.Int32(S), ns)
                ns = (ns < fx.Int32(1)).select(fx.Int32(1), ns)
                tps = udiv(ntiles + ns - fx.Int32(1), ns)
                tps1 = (tps < fx.Int32(1)).select(fx.Int32(1), tps)
                ns = (tps > zero_i).select(
                    udiv(ntiles + tps1 - fx.Int32(1), tps1), fx.Int32(1)
                )
                live = split < ns
                t0 = live.select(split * tps, zero_i)
                t1r = t0 + tps
                t1 = live.select((t1r < ntiles).select(t1r, ntiles), zero_i)
            elif fx.const_expr(SHORT):
                short = ntiles <= fx.Int32(SHORT)
                if fx.const_expr(SHORTM):
                    kvall = kv_all
                    if fx.const_expr(kvall is None):
                        kvall = _sload(indptr_ptr, nseq) - _sload(indptr_ptr, zero_i)
                    kts = nseq * fx.Int32(KTL)
                    upto = ntiles <= udiv(kvall + kts - fx.Int32(1), kts)
                    if fx.const_expr(SHORTF):
                        upto = upto | (ntiles <= fx.Int32(SHORTF))
                    short = short & upto
                ns = short.select(fx.Int32(1), fx.Int32(S))
                live = (short == false_) | (split == zero_i)
                t0 = short.select(zero_i, split * tps)
                t1r = t0 + tps
                t1 = short.select(ntiles, (t1r < ntiles).select(t1r, ntiles))
            else:
                ns, live = fx.Int32(S), true_
                t0 = split * tps
                t1r = t0 + tps
                t1 = (t1r < ntiles).select(t1r, ntiles)
            return kv0, kvlen, ntiles, tps, t0, t1, ns, live, true_, false_

        sr_pre = None
        if fx.const_expr(ADAPT or SHORT):
            sr_pre = split_range(seq_c)
            active = active & sr_pre[7]

        if active:

            def load16(ptr, off):
                return fx.ptr_load(ptr + fx.Int64(off), result_type=v16u8).bitcast(
                    fx.Int32
                )

            def join8(lo, hi):
                return fx.Vector.from_elements(
                    [lo[i] for i in fx.range_constexpr(4)]
                    + [hi[i] for i in fx.range_constexpr(4)],
                    fx.Int32,
                )

            zero4 = fx.Vector.filled(4, 0, fx.Int32)

            kv0, kvlen, ntiles, tps, t0, t1, ns, live, sk_last, sk_wt = (
                sr_pre if (ADAPT or SHORT) else split_range(seq)
            )
            ibase = (kvlen > zero_i).select(
                fx.Int64(fx.ptrtoint(idx_ptr)) + fx.Int64(kv0) * fx.Int64(4),
                fx.Int64(fx.ptrtoint(indptr_ptr)) + fx.Int64(seq) * fx.Int64(4),
            )

            def ld_idx(off):
                return fx.Int32(
                    fx.generic_load(
                        cu._to_ptr_global(ibase + fx.Int64(off) * fx.Int64(4))
                    )
                )

            rb_qrow, rb_h0, rb_ok = [], [], []
            NRBW = 1 if RQ else (ROWS // 32 if RD else RB)
            if fx.const_expr(RQ or RD):
                wv = fx.Int32(fx.rocdl.readfirstlane(T.i32, wave.ir_value()))
            for rb in fx.range_constexpr(NRBW):
                if fx.const_expr(RQ):
                    rq_row = (jb0 + wv * fx.Int32(2)) * fx.Int32(16) + (
                        lane & fx.Int32(31)
                    )
                    qpos = rq_row >> fx.Int32(H.bit_length() - 1)
                    h0 = (rq_row & fx.Int32(H - 1)) - (lane & fx.Int32(31))
                else:
                    if fx.const_expr(RD):
                        jb = jb0 + fx.Int32(2 * rb)
                    else:
                        jb = jb0 + fx.Int32(rb)
                    qpos = _div_const(jb, HB)
                    h0 = _rem_const(jb, HB) * fx.Int32(16)
                rb_ok.append(qpos < qlen)
                rb_qrow.append(q0 + (qpos < qlen).select(qpos, qlen - fx.Int32(1)))
                rb_h0.append(h0)

            if fx.const_expr(BOUNDS and not Q16):
                rb_bnd = []
                for rb in fx.range_constexpr(NRBW):
                    bv = fx.ptr_load(
                        bnd_ptr + fx.Int64(rb_qrow[rb]) * fx.Int64(4),
                        result_type=fx.Vector.make_type(4, fx.Int32),
                    )
                    rb_bnd.append([fx.Int32(bv[i]) for i in range(4)])

            def in_bounds(kk, bnd):
                a0, a1, b0, b1 = bnd
                return ((kk >= a0) & (kk < a1)) | ((kk >= b0) & (kk < b1))

            if fx.const_expr(EPI_Q):
                if fx.const_expr(QROW_EPI):
                    rb_pos = [
                        fx.Int64(fx.ptr_load(pos_ptr + fx.Int64(rb_qrow[rb])))
                        for rb in range(NRBW)
                    ]
                else:
                    pos = fx.Int64(fx.ptr_load(pos_ptr + fx.Int64(seq)))
                    rb_pos = [pos] * NRBW

            gsh = g * fx.Int32(8)

            def _scl(word, cc):
                b = (fx.Int32(word) >> gsh) & fx.Int32(0xFF)
                if fx.const_expr(cc == 3):
                    b = (gsh >= fx.Int32(16)).select(fx.Int32(127), b)
                return b

            def row_ok(raw):
                inb = (raw >= zero_i) & (raw < num_rows)
                return inb.select(raw, zero_i)

            def load_q():
                qfr, qscl, qrf = [], [], []
                for rb in fx.range_constexpr(NRBW):
                    qbase = (
                        fx.Int64(rb_qrow[rb]) * fx.Int64(H) + fx.Int64(rb_h0[rb] + m)
                    ) * fx.Int64(ROWB)
                    fr_ = []
                    for cc in fx.range_constexpr(4):
                        lo = load16(
                            q_ptr,
                            qbase + fx.Int64(cc * 128) + fx.Int64(g * fx.Int32(16)),
                        )
                        if fx.const_expr(cc < 3):
                            hi = load16(
                                q_ptr,
                                qbase
                                + fx.Int64(cc * 128 + 64)
                                + fx.Int64(g * fx.Int32(16)),
                            )
                        else:
                            hi = zero4
                        fr_.append(join8(lo, hi))
                    qfr.append(fr_)
                    qsc = load16(q_ptr, qbase + fx.Int64(NOPE))
                    qscl.append([_scl(qsc[cc], cc) for cc in range(4)])
                    qrbase = (
                        fx.Int64(rb_qrow[rb]) * fx.Int64(H) + fx.Int64(rb_h0[rb] + m)
                    ) * fx.Int64(ROPE)
                    qrf.append(
                        [
                            fx.ptr_load(
                                qr_ptr
                                + qrbase
                                + fx.Int64(g * fx.Int32(8) + fx.Int32(32 * j)),
                                result_type=v8bf16,
                            )
                            for j in range(2)
                        ]
                    )
                return qfr, qscl, qrf

            # start ticket (fire and forget): siblings that see every ticket set may wait for
            # this CTA, which is then known to be running
            def protocol_entry():
                slot = slot_base + fx.Int64(group) * fx.Int64(SLOT_BYTES)
                myx = xcc_id() & fx.Int32(7)
                tk_on = tid == zero_i
                if fx.const_expr(DYN or SHORT):
                    tk_on = tk_on & live & (ns > fx.Int32(1))
                if tk_on:
                    st_agent(
                        slot + fx.Int64(TK_OFF) + fx.Int64(split) * fx.Int64(4),
                        myx + fx.Int32(1),
                        4,
                    )
                return slot, myx, ticket_lane(slot, lane)

            def ticket_lane(slot, lane):
                nsp = ns if (DYN or SHORT) else fx.Int32(S)
                return (
                    slot
                    + fx.Int64(TK_OFF)
                    + fx.Int64((lane < nsp).select(lane, nsp - fx.Int32(1)))
                    * fx.Int64(4)
                )

            if fx.const_expr(RQ):
                n32 = lane & fx.Int32(31)
                h2 = lane >> fx.Int32(5)
                r_cta = wv * fx.Int32(32) + n32
                row_st = r_cta < fx.Int32(ROWS)

                def stg(e):
                    return p_rq + e * fx.Int32(2)

                def row_meta(rc):
                    jbr = jb0 + (rc >> fx.Int32(4))
                    qpr = _div_const(jbr, HB)
                    hdr = _rem_const(jbr, HB) * fx.Int32(16) + (rc & fx.Int32(15))
                    okr = (qpr < qlen) & (rc < fx.Int32(ROWS))
                    return q0 + (qpr < qlen).select(qpr, qlen - fx.Int32(1)), hdr, okr

                PE = 264  # staging row pitch (bf16): 256 + 8 against bank conflicts

                def emit_rows(val, wrow, d_lo, nblk, empty_r, pos_r):
                    ob = wv * fx.Int32(32 * PE)
                    NPB = min(nblk, 8) if not EPI_Q else nblk
                    RBY = NPB * 32 * (1 if EPI_Q else 2)
                    LPR = RBY // 16
                    RPI = 64 // LPR

                    def read_back(gaddr_of):
                        for rr in fx.range_constexpr(32 // RPI):
                            r_ = fx.Int32(rr * RPI) + lane // fx.Int32(LPR)
                            cb = (lane % fx.Int32(LPR)) * fx.Int32(16)
                            v_ = fx.ptr_load(
                                stg(ob + r_ * fx.Int32(PE) + (cb >> fx.Int32(1))),
                                result_type=v8bf16,
                            )
                            qrr, hdr, okr = row_meta(wrow * fx.Int32(32) + r_)
                            if okr:
                                st_raw(gaddr_of(qrr, hdr) + fx.Int64(cb), v_, 16)

                    def dvl(k, q_):
                        return fx.Int32(32 * k + 8 * q_) + h2 * fx.Int32(4)

                    qrl, hdl, okl = row_meta(wrow * fx.Int32(32) + n32)
                    xrow = fx.Int64(qrl) * fx.Int64(H) + fx.Int64(hdl)
                    for bk in fx.range_constexpr(nblk // 4):
                        ys = []
                        amb = zero_i
                        blk_ = d_lo // fx.Int32(4) + fx.Int32(bk)
                        is_rope = blk_ == fx.Int32(3)
                        for k in fx.range_constexpr(4 * bk, 4 * bk + 4):
                            for q_ in fx.range_constexpr(4):
                                x = val(k, q_).to(fx.BFloat16).to(fx.Float32)
                                if fx.const_expr(k % 4 >= 2):
                                    dcol = (
                                        (d_lo + fx.Int32(k)) * fx.Int32(32)
                                        + fx.Int32(8 * q_)
                                        + h2 * fx.Int32(4)
                                    )
                                    fcol = (dcol - fx.Int32(NOPE)) & fx.Int32(63)
                                    fr = fx.ptr_load(
                                        fr_ptr
                                        + pos_r * fx.Int64(ROPE)
                                        + fx.Int64(fcol),
                                        result_type=v4f32,
                                    )
                                    yy = []
                                    for pp in fx.range_constexpr(2):
                                        c_ = fx.Float32(fr[2 * pp])
                                        sn = fx.Float32(fr[2 * pp + 1])
                                        xa = fx.Float32(x[2 * pp])
                                        xb = fx.Float32(x[2 * pp + 1])
                                        yy.append(is_rope.select(xa * c_ + xb * sn, xa))
                                        yy.append(is_rope.select(xb * c_ - xa * sn, xb))
                                    x = fx.Vector.from_elements(yy, fx.Float32)
                                y = x.to(fx.BFloat16).to(fx.Float32)
                                ys.append((k, q_, y))
                                for i in fx.range_constexpr(4):
                                    amb = smax_i32(amb, fabs_bits(fx.Float32(y[i])))
                        amb = plswap(amb, 32, smax_i32)
                        ex = mxfp8_exp(amb)
                        ex = empty_r.select(fx.Int32(255), ex)
                        nexp = fx.Int32(127) - ex
                        for k, q_, y in ys:
                            qv = [fp8_scaled(fx.Float32(y[i]), nexp) for i in range(4)]
                            pkq = fx.rocdl.cvt_pk_fp8_f32(
                                T.i32, qv[0], qv[1], fx.Int32(0), False
                            )
                            pkq = fx.rocdl.cvt_pk_fp8_f32(
                                T.i32, qv[2], qv[3], pkq, True
                            )
                            pkq = empty_r.select(fx.Int32(0x7F7F7F7F), fx.Int32(pkq))
                            fx.ptr_store(
                                fx.Vector.from_elements([pkq], fx.Int32).bitcast(
                                    fx.BFloat16
                                ),
                                stg(
                                    ob
                                    + n32 * fx.Int32(PE)
                                    + (dvl(k, q_) >> fx.Int32(1))
                                ),
                            )
                        if okl & (h2 == zero_i):
                            fx.ptr_store(
                                fx.Uint8(ex),
                                xs_ptr + xrow * fx.Int64(4) + fx.Int64(blk_),
                            )
                    xbase = fx.Int64(fx.ptrtoint(out_ptr))
                    read_back(
                        lambda qr_, hd_: (
                            xbase
                            + (fx.Int64(qr_) * fx.Int64(H) + fx.Int64(hd_))
                            * fx.Int64(DV)
                            + fx.Int64(d_lo * fx.Int32(32))
                        )
                    )

                def rq_part_val(acc_f, inv_lp, d, q_):
                    v = fx.Vector.from_elements(
                        [fx.Float32(acc_f[d][4 * q_ + i]) for i in range(4)],
                        fx.Float32,
                    )
                    return v * inv_lp

                def rq_store_quarter(pw, acc_f, inv_lp, vq):
                    for d in fx.range_constexpr(4 * vq, 4 * vq + 4):
                        for q_ in fx.range_constexpr(4):
                            st_raw(
                                pw + fx.Int64((4 * d + q_) * 256 * PEB),
                                rq_part_val(acc_f, inv_lp, d, q_),
                                4 * PEB,
                            )

                def rq_part_addr():
                    return part_base + (
                        (fx.Int64(group) * fx.Int64(S) + fx.Int64(split))
                        * fx.Int64(ROWS * DV)
                        + fx.Int64(wv * fx.Int32(64 * 256) + lane * fx.Int32(4))
                    ) * fx.Int64(PEB)

                def rq_partials(m_f, l_f):
                    has = l_f > fx.Float32(0.0)
                    lf_ = l_f.maximumf(fx.Float32(1e-30))
                    inv_lp = has.select(fx.Float32(1.0) / lf_, fx.Float32(0.0))
                    lse_p = has.select(m_f + fly_math.log2(lf_), ninf)
                    return inv_lp, lse_p

                def rq_store(inv_lp, lse_p, acc_f, leave):
                    pw = rq_part_addr()
                    if t0 < t1:
                        okr = row_meta(r_cta)[2]
                        for vq in fx.range_constexpr(4):
                            other = fx.Int32(vq % S) != (split & fx.Int32(S - 1))
                            if (leave | other) & okr:
                                rq_store_quarter(pw, acc_f, inv_lp, vq)
                        if row_st & (h2 == zero_i):
                            st_raw(
                                lse_base
                                + (
                                    (fx.Int64(group) * fx.Int64(ROWS) + fx.Int64(r_cta))
                                    * fx.Int64(SP)
                                    + fx.Int64(split)
                                )
                                * fx.Int64(4),
                                lse_p,
                                4,
                            )

                def merge_own(j, own):
                    rc = wv * fx.Int32(32) + n32
                    qrl, hdl, okl = row_meta(rc)
                    ipm = fx.Float32(
                        _llvm_d.InlineAsmOp(
                            T.f32,
                            [fx.Float32(own_ip).ir_value()],
                            "; own $0",
                            "=v,0",
                            has_side_effects=True,
                        ).res
                    )
                    if row_meta(wv * fx.Int32(32))[2]:
                        lane_ld = okl.select(lane, h2 * fx.Int32(32))
                        rld = okl.select(rc, wv * fx.Int32(32))
                        la = lse_base + (
                            fx.Int64(group) * fx.Int64(ROWS) + fx.Int64(rld)
                        ) * fx.Int64(SP * 4)
                        lsv = []
                        for k0, n in lse_chunks:
                            vv_ = fx.Vector(gload(la + fx.Int64(4 * k0), n))
                            lsv += [fx.Float32(vv_[jx]) for jx in range(n)]
                        snk = fx.Float32(
                            fx.ptr_load(sink_ptr + fx.Int64(hdl))
                        ) * fx.Float32(LOG2E)
                        pos_r = fx.Int64(fx.ptr_load(pos_ptr + fx.Int64(qrl)))
                        gmax = snk
                        lv, vsl = [], []
                        for k in fx.range_constexpr(S):
                            vs = (tps * fx.Int32(k)) < ntiles
                            vsl.append(vs)
                            lv.append(vs.select(lsv[k], ninf))
                            gmax = gmax.maximumf(lv[k])
                        gl = lv[0]
                        for k in fx.range_constexpr(1, S):
                            gl = gl.maximumf(lv[k])
                        empty_r = gl == ninf
                        den = exp2(snk - gmax)
                        wts = []
                        for k in fx.range_constexpr(S):
                            w_ = exp2(lv[k] - gmax)
                            den = den + w_
                            wts.append(w_)
                        fsc = empty_r.select(qnan, rcp(den))
                        pb = part_base + (
                            fx.Int64(group) * fx.Int64(S * ROWS * DV)
                            + fx.Int64(wv * fx.Int32(64 * 256) + lane_ld * fx.Int32(4))
                        ) * fx.Int64(PEB)
                        zv4 = fx.Vector.filled(4, 0.0, fx.Float32)
                        for vq in fx.range_constexpr(4):
                            if fx.Int32(vq % S) == (j & fx.Int32(S - 1)):
                                accm = [zv4] * 16
                                for k in fx.range_constexpr(S):
                                    mine = own and k == vq % S
                                    pk = pb + fx.Int64(k * ROWS * DV * PEB)
                                    for gg in fx.range_constexpr(16):
                                        d_, q_ = 4 * vq + gg // 4, gg % 4
                                        if fx.const_expr(mine):
                                            v_ = fx.Vector(
                                                rq_part_val(own_acc, ipm, d_, q_)
                                            )
                                        else:
                                            if fx.const_expr(q_ == 0):
                                                pbase = pin_v64(
                                                    pk + fx.Int64(4 * d_ * 256 * PEB)
                                                )
                                            a_ = pbase + fx.Int64(q_ * 256 * PEB)
                                            v_ = fx.Vector(gload(a_, 4))
                                        accm[gg] = accm[gg] + vsl[k].select(
                                            v_ * wts[k], zv4
                                        )

                                def val(k_, q_, accm=accm):
                                    return accm[4 * k_ + q_] * fsc

                                emit_rows(val, wv, fx.Int32(4 * vq), 4, empty_r, pos_r)

            if fx.const_expr(RQ or RD):
                v4i32 = fx.Vector.make_type(4, fx.Int32)
                v2i32 = fx.Vector.make_type(2, fx.Int32)
                v16f32 = fx.Vector.make_type(16, fx.Float32)
                hq = lane // fx.Int32(32)
                r32 = lane % fx.Int32(32)
                _dom = '#llvm.alias_scope_domain<id = distinct[0]<>, description = "rqlds">'
                _dma_scope = _ir.Attribute.parse(
                    f"#llvm.alias_scope<id = distinct[1]<>, domain = {_dom}>"
                )
                DMA_AS = _ir.ArrayAttr.get([_dma_scope])
                KPD = 2
                PVD = 3

                def fence():
                    fx.rocdl.sched_barrier(0)

                def early_tickets():
                    tkv = zero_i
                    if wave == zero_i:
                        tkv = ld_agent_i32(tk_lane, volatile=False)
                    return tkv

                def lds_barrier():
                    _llvm_d.InlineAsmOp(
                        None,
                        [],
                        "s_waitcnt lgkmcnt(0)\n s_barrier",
                        "~{memory}",
                        has_side_effects=True,
                    )

                def anchor(x, cls):
                    return fx.Vector(
                        _llvm_d.InlineAsmOp(
                            x.ir_value().type,
                            [x.ir_value()],
                            "; " + cls + " $0",
                            f"={cls},0",
                        ).res
                    )

                def lds_rd(off, ty, align):
                    return fx.Vector(
                        _llvm_d.LoadOp(
                            ty,
                            fx.to_llvm_ptr(p_rq + off),
                            alignment=align,
                            noalias_scopes=DMA_AS,
                        ).result
                    )

                def lds_rd16(off):
                    return lds_rd(off, v4i32, 16)

                def lds_wr16(off, val):
                    _llvm_d.StoreOp(
                        _arith.unwrap(val),
                        fx.to_llvm_ptr(p_rq + off),
                        alignment=16,
                        noalias_scopes=DMA_AS,
                    )

                def lds_tr(off):
                    return fx.Vector(
                        fx.rocdl.ds_read_tr16_b64(
                            v4bf16, fx.to_llvm_ptr(p_rq + off), noalias_scopes=DMA_AS
                        ).result
                    )

                def dma16(gaddr, off, nt=False):
                    _rocdl_d.global_load_lds(
                        fx.to_llvm_ptr(cu._to_ptr_global(gaddr)),
                        fx.to_llvm_ptr(p_rq + off),
                        16,
                        0,
                        aux=_ir.IntegerAttr.get(T.i32, CPOL_NT) if nt else None,
                        alias_scopes=DMA_AS,
                    )

                def kslot(c, k):
                    return fx.Int32(16) * (
                        fx.Int32(32) * c + (k ^ ((c & fx.Int32(3)) * fx.Int32(4)))
                    )

                def kf_base(t):
                    return _rem_const(t, NKF) * fx.Int32(RQ_KF)

                def kr_base(t):
                    return fx.Int32(RQ_KR_OFF) + _rem_const(t, NKR) * fx.Int32(RQ_KR)

                def v_base(t):
                    return fx.Int32(RQ_V_OFF) + (t & fx.Int32(NVB_RQ - 1)) * fx.Int32(
                        RQ_V
                    )

                def kpx(k):
                    return (k >> fx.Int32(1)) & fx.Int32(7)

                def kfo(c, k):
                    return fx.Int32(16) * (
                        (c >> fx.Int32(3)) * fx.Int32(256)
                        + k * fx.Int32(8)
                        + ((c & fx.Int32(7)) ^ kpx(k))
                    )

                def run_mixed(main, side, every=1):
                    side = list(side) if side is not None else []
                    for k, f in enumerate(main, 1):
                        f()
                        if side and k % every == 0:
                            side.pop(0)()
                        fence()
                    for f in side:
                        f()
                        fence()

                def run_gen(steps):
                    for f in steps:
                        f()
                        fence()

                # inline asm keeps the accumulator in AGPRs and the operands in VGPRs. The backend's
                # hazard recognizer does not see it: `fresh` (operand just written by VALU) needs 2
                # wait states, `last` the MFMA -> VALU read of the accumulator
                def pv_mfma(va, pb, acc, last, fresh=False):
                    tail = "\n s_nop 7\n s_nop 7\n s_nop 3" if last else ""
                    return fx.Vector(
                        _llvm_d.InlineAsmOp(
                            v16f32,
                            [va.ir_value(), pb.ir_value(), acc.ir_value()],
                            ("s_nop 1\n " if fresh else "")
                            + "v_mfma_f32_32x32x16_bf16 $0, $1, $2, $3"
                            + tail,
                            "=a,v,v,0",
                        ).res
                    )

                def pmax32(x):
                    xi = x.bitcast(fx.Int32)
                    return plswap(
                        xi,
                        32,
                        lambda a, b: fx.Int32(
                            a.bitcast(fx.Float32)
                            .maximumf(b.bitcast(fx.Float32))
                            .bitcast(fx.Int32)
                        ),
                    ).bitcast(fx.Float32)

            if fx.const_expr(RQ):
                rq_vw = [wv * fx.Int32(NROLE) + fx.Int32(p) for p in range(NROLE)]
                kd_fs = [vw * fx.Int32(8) + lane // fx.Int32(8) for vw in rq_vw]
                kd_rs = [
                    r32 ^ (((vw * fx.Int32(2) + hq) & fx.Int32(3)) * fx.Int32(4))
                    for vw in rq_vw
                ]
                kf_pieces = [(lane % fx.Int32(8)) ^ kpx(kd) for kd in kd_fs]
                rq_keys = [kk for p in range(NROLE) for kk in (kd_fs[p], kd_rs[p])]

                def idx_tile(t):
                    rows = []
                    for kk in rq_keys:
                        kidx = t * fx.Int32(KTL) + kk
                        ok = (kidx < kvlen) & (t < t1)
                        rows.append(ld_idx(ok.select(kidx, zero_i)))
                    return rows

                def ix_off(t, vw):
                    return fx.Int32(RQ_IX_OFF) + (
                        _rem_const(t, NIX) * fx.Int32(4) + vw
                    ) * fx.Int32(512)

                def dma_idx(t):
                    for p in fx.range_constexpr(NROLE):
                        for u, kk in enumerate((kd_fs[p], kd_rs[p])):
                            kidx = t * fx.Int32(KTL) + kk
                            ok = (kidx < kvlen) & (t < t1)
                            ga = ibase + fx.Int64(ok.select(kidx, zero_i)) * fx.Int64(4)
                            _rocdl_d.global_load_lds(
                                fx.to_llvm_ptr(cu._to_ptr_global(ga)),
                                fx.to_llvm_ptr(
                                    p_rq
                                    + (
                                        ix_off(t, rq_vw[p])
                                        + fx.Int32(256 * u)
                                        + lane * fx.Int32(4)
                                    )
                                ),
                                4,
                                0,
                                alias_scopes=DMA_AS,
                            )

                def idx_lds(t):
                    return [
                        fx.Int32(
                            lds_rd(
                                ix_off(t, rq_vw[p])
                                + fx.Int32(256 * u)
                                + lane * fx.Int32(4),
                                fx.Vector.make_type(1, fx.Int32),
                                4,
                            )[0]
                        )
                        for p in range(NROLE)
                        for u in range(2)
                    ]

                def dma_tile(t, rows):
                    for p in fx.range_constexpr(NROLE):
                        vw = rq_vw[p]
                        rf = fx.Int64(row_ok(rows[2 * p]))
                        rr = fx.Int64(row_ok(rows[2 * p + 1]))
                        gk = (
                            fx.Int64(fx.ptrtoint(kv_ptr))
                            + rf * fx.Int64(ROWB)
                            + fx.Int64(kf_pieces[p] * fx.Int32(16))
                        )
                        gr = (
                            fx.Int64(fx.ptrtoint(kvr_ptr))
                            + rr * fx.Int64(2 * ROPE)
                            + fx.Int64((vw * fx.Int32(2) + hq) * fx.Int32(16))
                        )
                        kfb = kf_base(t) + vw * fx.Int32(1024)
                        for cg in fx.range_constexpr(4):
                            dma16(gk + fx.Int64(128 * cg), kfb + fx.Int32(4096 * cg))
                        dma16(gr, kr_base(t) + vw * fx.Int32(1024))

                qrow_l = rb_qrow[0]
                head_l = rb_h0[0] + r32
                if fx.const_expr(BOUNDS):
                    rq_bnd = rb_bnd[0]
                qb = (fx.Int64(qrow_l) * fx.Int64(H) + fx.Int64(head_l)) * fx.Int64(
                    ROWB
                )
                qop = []
                for st in fx.range_constexpr(7):
                    o_ = qb + fx.Int64(64 * st) + fx.Int64(hq * fx.Int32(32))
                    lo = load16(q_ptr, o_)
                    hi = load16(q_ptr, o_ + fx.Int64(16))
                    qop.append(join8(lo, hi))
                qsc = load16(q_ptr, qb + fx.Int64(NOPE))
                qrb = (fx.Int64(qrow_l) * fx.Int64(H) + fx.Int64(head_l)) * fx.Int64(
                    ROPE
                )
                qrope = [
                    fx.ptr_load(
                        qr_ptr + qrb + fx.Int64(fx.Int32(16 * r) + hq * fx.Int32(8)),
                        result_type=v8bf16,
                    )
                    for r in range(4)
                ]

                kq_o = [
                    fx.Int32(16)
                    * (
                        r32 * fx.Int32(8)
                        + ((fx.Int32(4 * par + jh) + hq * fx.Int32(2)) ^ kpx(r32))
                    )
                    for par in range(2)
                    for jh in range(2)
                ]
                ksc_o = kfo(fx.Int32(28), r32)
                kr_o = [
                    fx.Int32(16)
                    * (
                        fx.Int32(32) * (fx.Int32(2 * r) + hq)
                        + (r32 ^ ((fx.Int32(2 * (r % 2)) + hq) * fx.Int32(4)))
                    )
                    for r in range(2)
                ]
                lg = lane // fx.Int32(16)
                li = lane % fx.Int32(16)
                lq_a = li // fx.Int32(4)
                lq_b = li % fx.Int32(4)
                cG = (lg & fx.Int32(1)) * fx.Int32(2) + lq_b // fx.Int32(2)
                swG = cG * fx.Int32(4)
                pv_o = [
                    fx.Int32(16)
                    * (
                        fx.Int32(32) * cG
                        + ((fx.Int32(8 * qq) + hq * fx.Int32(4)) ^ swG)
                        + lq_a
                    )
                    + (lq_b & fx.Int32(1)) * fx.Int32(8)
                    for qq in range(2)
                ]
                dq_k = tid % fx.Int32(32)
                dq_src, dq_dst = [], []
                for p in fx.range_constexpr(NROLE):
                    dq_g = (tid + fx.Int32(NT * p)) // fx.Int32(32)
                    dq_h = (dq_k // fx.Int32(16)) ^ (dq_g & fx.Int32(1))
                    dq_cf0 = dq_g // fx.Int32(2)
                    dq_srcp = [
                        kfo(dq_cf0 + fx.Int32(4 * par), dq_k) + dq_h * fx.Int32(8)
                        for par in range(2)
                    ]
                    dq_dst0 = kslot(dq_cf0 * fx.Int32(2) + dq_h, dq_k)
                    dq_src += [
                        dq_srcp[r % 2] + fx.Int32(4096 * (r // 2)) for r in range(7)
                    ]
                    dq_dst += [dq_dst0 + fx.Int32(4096 * r) for r in range(7)]
                dq_sc = kfo(fx.Int32(28), dq_k)
                NDQ = 7 * NROLE

                def dequant_steps(t):
                    st_ = {}

                    def s0():
                        st_["kfb"] = kf_base(t)
                        st_["vb"] = v_base(t)
                        st_["scd"] = lds_rd16(st_["kfb"] + dq_sc)
                        st_["src"] = [
                            lds_rd(st_["kfb"] + dq_src[r], v2i32, 8) for r in range(2)
                        ]

                    def item(ri):
                        r = ri % 7

                        def f():
                            src = st_["src"]
                            if fx.const_expr(ri + 2 < NDQ):
                                src.append(
                                    lds_rd(st_["kfb"] + dq_src[ri + 2], v2i32, 8)
                                )
                            e = (
                                fx.Int32(st_["scd"][r // 2]) >> fx.Int32(16 * (r % 2))
                            ) & fx.Int32(0xFF)
                            scl = (e << fx.Int32(23)).bitcast(fx.Float32)
                            wds = []
                            for wd in fx.range_constexpr(2):
                                for sel in fx.range_constexpr(2):
                                    pr = fx.Vector(
                                        fx.rocdl.cvt_scalef32_pk_bf16_fp8(
                                            v2bf16,
                                            fx.Int32(src[ri][wd]).ir_value(),
                                            scl.ir_value(),
                                            bool(sel),
                                        )
                                    )
                                    wds.append(pr.bitcast(fx.Int32)[0])
                            lds_wr16(
                                st_["vb"] + dq_dst[ri],
                                fx.Vector.from_elements(wds, fx.Int32),
                            )

                        return f

                    return [s0] + [item(ri) for ri in range(NDQ)]

                def k_op(kfb, krb, st):
                    if fx.const_expr(st < 7):
                        o_ = fx.Int32(4096 * (st // 2))
                        lo = lds_rd16(kfb + kq_o[2 * (st % 2)] + o_)
                        hi = lds_rd16(kfb + kq_o[2 * (st % 2) + 1] + o_)
                        return join8(lo, hi)
                    r = st - 7
                    return lds_rd16(
                        krb + kr_o[r % 2] + fx.Int32(1024 * (r // 2) * 2)
                    ).bitcast(fx.BFloat16)

                def qk_mfma(st, op, ksc, sc):
                    if fx.const_expr(st < 7):
                        return fx.Vector(
                            fx.rocdl.mfma_scale_f32_32x32x64_f8f6f4(
                                v16f32,
                                [
                                    op,
                                    qop[st],
                                    sc,
                                    0,
                                    0,
                                    2 * (st % 2),
                                    fx.Int32(ksc[st // 2]),
                                    2 * (st % 2),
                                    fx.Int32(qsc[st // 2]),
                                ],
                            )
                        )
                    return fx.Vector(
                        fx.rocdl.mfma_f32_32x32x16_bf16(v16f32, [op, qrope[st - 7], sc])
                    )

                def qk_steps(t, out):
                    st_ = {}

                    def step(st):
                        def f():
                            if fx.const_expr(st == 0):
                                st_["kfb"] = kf_base(t)
                                st_["krb"] = kr_base(t)
                                st_["ksc"] = lds_rd16(st_["kfb"] + ksc_o)
                                st_["sc"] = fx.Vector.filled(16, 0.0, fx.Float32)
                                st_["ops"] = [
                                    k_op(st_["kfb"], st_["krb"], s2)
                                    for s2 in range(KPD)
                                ]
                            ops = st_["ops"]
                            if fx.const_expr(st + KPD < 11):
                                ops.append(k_op(st_["kfb"], st_["krb"], st + KPD))
                            st_["sc"] = qk_mfma(st, ops[st], st_["ksc"], st_["sc"])
                            if fx.const_expr(st == 10):
                                out.append(anchor(st_["sc"], "v"))

                        return f

                    return [step(st) for st in range(11)]

                def softmax_steps(t, sc, m_run, l_run, masked, out):
                    st_ = {}

                    def s_scale():
                        tbase = t * fx.Int32(KTL)
                        ss = [fx.Float32(sc[r]) * scale_log2e for r in range(16)]
                        if fx.const_expr(masked):
                            lim = kvlen - tbase - hq * fx.Int32(4)
                            ss = [
                                (fx.Int32(8 * (r // 4) + r % 4) < lim).select(
                                    ss[r], ninf
                                )
                                for r in range(16)
                            ]
                        if fx.const_expr(BOUNDS):
                            kb = tbase + hq * fx.Int32(4)
                            a0, a1, b0, b1 = rq_bnd
                            la0 = a0 - kb
                            la1 = (a1 > a0).select(a1 - a0, zero_i)
                            lb0 = b0 - kb
                            lb1 = (b1 > b0).select(b1 - b0, zero_i)

                            def bmask(ss=ss, la0=la0, la1=la1, lb0=lb0, lb1=lb1):
                                return [
                                    (
                                        _ult(fx.Int32(8 * (r // 4) + r % 4) - la0, la1)
                                        | _ult(
                                            fx.Int32(8 * (r // 4) + r % 4) - lb0, lb1
                                        )
                                    ).select(ss[r], ninf)
                                    for r in range(16)
                                ]

                            inside = ((la0 <= zero_i) & (a1 - kb >= fx.Int32(28))) | (
                                (lb0 <= zero_i) & (b1 - kb >= fx.Int32(28))
                            )
                            nmask = fx.Int64(
                                fx.rocdl.ballot(T.i64, (inside == false_).ir_value())
                            )
                            if nmask != fx.Int64(0):
                                ss = bmask()
                        st_["ss"] = ss

                    def s_max():
                        ss = st_["ss"]
                        mx = ss[0]
                        for r in fx.range_constexpr(1, 16):
                            mx = mx.maximumf(ss[r])
                        st_["mx"] = mx

                    def s_m():
                        mx = pmax32(st_["mx"])
                        mn = (mx > m_run + fx.Float32(8.0)).select(mx, m_run)
                        st_["mn"] = mn
                        st_["m_safe"] = (mn == ninf).select(fx.Float32(0.0), mn)
                        st_["alpha"] = exp2(m_run - st_["m_safe"])
                        st_["pf"] = []

                    def s_exp(q4):
                        def f():
                            st_["pf"] += [
                                exp2(st_["ss"][4 * q4 + i] - st_["m_safe"])
                                for i in range(4)
                            ]

                        return f

                    def s_sum():
                        pf = st_["pf"]
                        ps0 = (pf[0] + pf[1]) + (pf[2] + pf[3])
                        ps1 = (pf[4] + pf[5]) + (pf[6] + pf[7])
                        ps2 = (pf[8] + pf[9]) + (pf[10] + pf[11])
                        ps3 = (pf[12] + pf[13]) + (pf[14] + pf[15])
                        st_["ps"] = (ps0 + ps1) + (ps2 + ps3)

                    def s_pack():
                        pf = st_["pf"]
                        pops = [
                            fx.Vector.from_elements(
                                pf[8 * ks : 8 * ks + 8], fx.Float32
                            ).to(fx.BFloat16)
                            for ks in range(2)
                        ]
                        out.extend(
                            [
                                pops,
                                st_["mn"],
                                l_run * st_["alpha"] + st_["ps"],
                                st_["alpha"],
                            ]
                        )

                    return (
                        [s_scale, s_max, s_m]
                        + [s_exp(q4) for q4 in range(4)]
                        + [s_sum, s_pack]
                    )

                def v_op(vb0, vb1, kb0, kb1, j, ks):
                    b0, b1 = (vb0, vb1) if j < 14 else (kb0, kb1)
                    o_ = 2048 * (j if j < 14 else j - 14) + 256 * ks
                    h0_ = lds_tr(b0 + fx.Int32(o_))
                    h1_ = lds_tr(b1 + fx.Int32(o_))
                    return fx.Vector.from_elements(
                        [h0_[i] for i in range(4)] + [h1_[i] for i in range(4)],
                        fx.BFloat16,
                    )

                def pv_steps(t, pops, accs, out):
                    st_ = {}
                    steps = [(j, ks) for ks in range(2) for j in range(16)]

                    def step(n):
                        def f():
                            if fx.const_expr(n == 0):
                                vb = v_base(t)
                                krb = kr_base(t)
                                st_["b"] = (
                                    vb + pv_o[0],
                                    vb + pv_o[1],
                                    krb + pv_o[0],
                                    krb + pv_o[1],
                                )
                                st_["ops"] = [
                                    v_op(*st_["b"], *steps[k]) for k in range(PVD)
                                ]
                                st_["acc"] = list(accs)
                            ops = st_["ops"]
                            if fx.const_expr(n + PVD < 32):
                                ops.append(v_op(*st_["b"], *steps[n + PVD]))
                            j, ks = steps[n]
                            acc = st_["acc"]
                            acc[j] = pv_mfma(
                                ops[n], pops[ks], acc[j], n == 31, n in (0, 16)
                            )
                            if fx.const_expr(n == 31):
                                out.extend(acc)

                        return f

                    return [step(n) for n in range(32)]

                def rescale_all(accs, alpha):
                    out = []
                    for j in fx.range_constexpr(NACC):
                        els = []
                        st2 = _llvm_d.StructType.get_literal([T.f32, T.f32])
                        for i in fx.range_constexpr(16):
                            r_ = _llvm_d.InlineAsmOp(
                                st2,
                                [
                                    fx.Float32(accs[j][i]).ir_value(),
                                    alpha.ir_value(),
                                ],
                                "v_accvgpr_read_b32 $1, $0\n"
                                " s_nop 1\n"
                                " v_mul_f32 $1, $1, $3\n"
                                " v_accvgpr_write_b32 $0, $1",
                                "=a,=&v,0,v",
                            ).res
                            els.append(fx.Float32(_llvm_d.extractvalue(T.f32, r_, [0])))
                        out.append(fx.Vector.from_elements(els, fx.Float32))
                    return out

                def rescale(accs, alpha, m_run):
                    ne = (alpha != fx.Float32(1.0)) & (m_run != ninf)
                    need = fx.Int64(fx.rocdl.ballot(T.i64, ne.ir_value())) != fx.Int64(
                        0
                    )
                    if need:
                        accs = rescale_all(accs, alpha)
                    return accs

                idx0 = [idx_tile(t0 + fx.Int32(j)) for j in range(DD)]
                VG = 7 * NROLE
                for j in fx.range_constexpr(DD):
                    dma_idx(t0 + fx.Int32(DD - 1 + j))
                    dma_tile(t0 + fx.Int32(j), idx0[j])
                if fx.const_expr(MERGED):
                    slot, myx, tk_lane = protocol_entry()
                NACC = 16
                acc0 = fx.Vector.filled(16, 0.0, fx.Float32)
                acc0s = [
                    _llvm_d.InlineAsmOp(
                        v16f32, [acc0.ir_value()], "; acc $0", "=a,0"
                    ).res
                    for _ in range(NACC)
                ]
                nits = t1 - t0
                if fx.const_expr(NVB_RQ == 1):
                    fx.rocdl.s_waitcnt(vmcnt=VG * (DD - 1))
                    lds_barrier()
                    u_sc0l = []
                    run_mixed(qk_steps(t0, u_sc0l), dequant_steps(t0))
                    fx.rocdl.s_waitcnt(vmcnt=VG * (DD - 2))
                    lds_barrier()
                    u_init = [
                        ninf.ir_value(),
                        fx.Float32(0.0).ir_value(),
                        u_sc0l[0].ir_value(),
                    ] + acc0s
                    u_nmain = (nits > fx.Int32(1)).select(nits - fx.Int32(1), zero_i)
                    u_results = u_init
                    for u_it, u_state in range(
                        zero_i, u_nmain, fx.Int32(1), init=u_init
                    ):
                        u_m_run = fx.Float32(u_state[0])
                        u_l_run = fx.Float32(u_state[1])
                        u_sc = fx.Vector(u_state[2])
                        u_accs = [fx.Vector(u_state[3 + j]) for j in range(NACC)]
                        u_tt = t0 + fx.Int32(u_it)
                        u_i_n = idx_lds(u_tt + fx.Int32(DD))
                        dma_idx(u_tt + fx.Int32(2 * DD - 1))
                        dma_tile(u_tt + fx.Int32(DD), u_i_n)
                        fence()
                        u_so = []
                        run_gen(softmax_steps(u_tt, u_sc, u_m_run, u_l_run, True, u_so))
                        u_pops, u_m_new, u_l_new, u_alpha = u_so
                        u_accs = rescale(u_accs, u_alpha, u_m_run)
                        fence()
                        u_po = []
                        run_mixed(pv_steps(u_tt, u_pops, u_accs, u_po), None)
                        lds_barrier()
                        u_qo = []
                        run_mixed(
                            qk_steps(u_tt + fx.Int32(1), u_qo),
                            dequant_steps(u_tt + fx.Int32(1)),
                        )
                        fx.rocdl.s_waitcnt(vmcnt=VG * (DD - 2))
                        lds_barrier()
                        u_results = yield (
                            [
                                u_m_new.ir_value(),
                                u_l_new.ir_value(),
                                u_qo[0].ir_value(),
                            ]
                            + [x.ir_value() for x in u_po]
                        )
                    m_l = fx.Float32(u_results[0])
                    l_l = fx.Float32(u_results[1])
                    acc_l = [fx.Vector(u_results[3 + j]) for j in range(NACC)]
                    if nits > zero_i:
                        u_tl = t1 - fx.Int32(1)
                        u_so2 = []
                        run_gen(
                            softmax_steps(
                                u_tl, fx.Vector(u_results[2]), m_l, l_l, True, u_so2
                            )
                        )
                        u_pops2, u_m_l2, u_l_l2, u_alpha2 = u_so2
                        u_acc_l2 = rescale(acc_l, u_alpha2, m_l)
                        u_po2 = []
                        run_mixed(pv_steps(u_tl, u_pops2, u_acc_l2, u_po2), None)
                        m_l = u_m_l2
                        l_l = u_l_l2
                        acc_l = u_po2
                else:
                    fx.rocdl.s_waitcnt(vmcnt=VG * (DD - 1))
                    lds_barrier()
                    run_gen(dequant_steps(t0))
                    fx.rocdl.s_waitcnt(vmcnt=VG * (DD - 2))
                    lds_barrier()
                    sc0l = []
                    run_gen(qk_steps(t0, sc0l))
                    init = [
                        ninf.ir_value(),
                        fx.Float32(0.0).ir_value(),
                        sc0l[0].ir_value(),
                    ]
                    init += acc0s
                    nmain = (nits > fx.Int32(1)).select(nits - fx.Int32(1), zero_i)
                    results = init
                    for it, state in range(zero_i, nmain, fx.Int32(1), init=init):
                        m_run = fx.Float32(state[0])
                        l_run = fx.Float32(state[1])
                        sc_cur = fx.Vector(state[2])
                        accs = [fx.Vector(state[3 + j]) for j in range(NACC)]
                        tt = t0 + fx.Int32(it)
                        i3 = idx_lds(tt + fx.Int32(DD))
                        dma_idx(tt + fx.Int32(2 * DD - 1))
                        dma_tile(tt + fx.Int32(DD), i3)
                        fence()
                        qo, so = [], []
                        run_mixed(
                            qk_steps(tt + fx.Int32(1), qo),
                            softmax_steps(tt, sc_cur, m_run, l_run, False, so),
                        )
                        pops, m_new, l_new, alpha = so
                        accs = rescale(accs, alpha, m_run)
                        fence()
                        po = []
                        run_mixed(
                            pv_steps(tt, pops, accs, po),
                            dequant_steps(tt + fx.Int32(1)),
                            every=4 // NROLE,
                        )
                        fx.rocdl.s_waitcnt(vmcnt=VG * (DD - 2))
                        lds_barrier()
                        results = yield (
                            [m_new.ir_value(), l_new.ir_value(), qo[0].ir_value()]
                            + [x.ir_value() for x in po]
                        )
                    if fx.const_expr(OWN):
                        tkv_own = early_tickets()
                    m_l = fx.Float32(results[0])
                    l_l = fx.Float32(results[1])
                    acc_l = [fx.Vector(results[3 + j]) for j in range(NACC)]
                    if nits > zero_i:
                        tl = t1 - fx.Int32(1)
                        so = []
                        run_gen(
                            softmax_steps(tl, fx.Vector(results[2]), m_l, l_l, True, so)
                        )
                        pops, m_l2, l_l2, alpha = so
                        acc_l2 = rescale(acc_l, alpha, m_l)
                        po = []
                        run_mixed(pv_steps(tl, pops, acc_l2, po), None)
                        m_l = m_l2
                        l_l = l_l2
                        acc_l = po
                fx.rocdl.s_waitcnt(vmcnt=0)
                m_fin = [m_l]
                l_fin = [l_l + l_l.shuffle_xor(fx.Int32(32), fx.Int32(WAVE))]
                if fx.const_expr(MERGED and RQM):
                    own_ip, own_lse = rq_partials(m_fin[0], l_fin[0])
                    own_acc = acc_l
                acc_fin = [
                    [
                        fx.Vector.from_elements(
                            [acc_l[n // 4][4 * (n % 4) + i] for i in range(4)],
                            fx.Float32,
                        )
                        for n in range(64)
                    ]
                ]
            elif fx.const_expr(RD):
                SPEC = RD_NQW < 4 and not Q16
                if fx.const_expr(SPEC):
                    dma_wave = wv >= fx.Int32(2)
                    dd_ = wv - fx.Int32(2)
                    kd_a = dd_ * fx.Int32(8) + lane // fx.Int32(8)
                    kd_b = kd_a + fx.Int32(16)
                    kd_r = r32 ^ (
                        ((dd_ * fx.Int32(2) + hq) & fx.Int32(3)) * fx.Int32(4)
                    )
                    kd_list = (kd_a, kd_b, kd_r)
                    NVM = 13
                else:
                    dma_wave = true_
                    kd_f = wv * fx.Int32(8) + lane // fx.Int32(8)
                    kd_r = r32 ^ (((wv * fx.Int32(2) + hq) & fx.Int32(3)) * fx.Int32(4))
                    kd_list = (kd_f, kd_r)
                    NVM = 7
                NKD = len(kd_list)

                def idx_tile(t):
                    rows = []
                    for kk in kd_list:
                        kidx = t * fx.Int32(KTL) + kk
                        ok = (kidx < kvlen) & (t < t1)
                        rows.append(ld_idx(ok.select(kidx, zero_i)))
                    return rows

                def ix_off(t):
                    return fx.Int32(RQ_IX_OFF) + (
                        _rem_const(t, NIX) * fx.Int32(2) + dd_
                    ) * fx.Int32(256 * NKD)

                def dma_idx(t):
                    for u, kk in enumerate(kd_list):
                        kidx = t * fx.Int32(KTL) + kk
                        ok = (kidx < kvlen) & (t < t1)
                        ga = ibase + fx.Int64(ok.select(kidx, zero_i)) * fx.Int64(4)
                        _rocdl_d.global_load_lds(
                            fx.to_llvm_ptr(cu._to_ptr_global(ga)),
                            fx.to_llvm_ptr(
                                p_rq
                                + (ix_off(t) + fx.Int32(256 * u) + lane * fx.Int32(4))
                            ),
                            4,
                            0,
                            alias_scopes=DMA_AS,
                        )

                def idx_lds(t):
                    return [
                        fx.Int32(
                            lds_rd(
                                ix_off(t) + fx.Int32(256 * u) + lane * fx.Int32(4),
                                fx.Vector.make_type(1, fx.Int32),
                                4,
                            )[0]
                        )
                        for u in range(NKD)
                    ]

                def dma_kf_block(t, row, kd, kb):
                    r_ = fx.Int64(row_ok(row))
                    gk = (
                        fx.Int64(fx.ptrtoint(kv_ptr))
                        + r_ * fx.Int64(ROWB)
                        + fx.Int64(((lane % fx.Int32(8)) ^ kpx(kd)) * fx.Int32(16))
                    )
                    kfb = kf_base(t) + kb * fx.Int32(1024)
                    for cg in fx.range_constexpr(4):
                        dma16(gk + fx.Int64(128 * cg), kfb + fx.Int32(4096 * cg))

                def dma_kr_item(t, row, p):
                    rr = fx.Int64(row_ok(row))
                    gr = (
                        fx.Int64(fx.ptrtoint(kvr_ptr))
                        + rr * fx.Int64(2 * ROPE)
                        + fx.Int64((p * fx.Int32(2) + hq) * fx.Int32(16))
                    )
                    dma16(gr, kr_base(t) + p * fx.Int32(1024))

                def dma_tile(t, rows):
                    dma_kf_block(t, rows[0], kd_a, dd_)
                    dma_kf_block(t, rows[1], kd_b, dd_ + fx.Int32(2))
                    dma_kr_item(t, rows[2], dd_)
                    dma_kr_item(t, rows[2], dd_ + fx.Int32(2))

                if fx.const_expr(Q16):
                    q_lim = smax_i32(
                        smin_i32(kvlen, t1 * fx.Int32(KTL)) - fx.Int32(1), zero_i
                    )
                    kf_lane = fx.Int64(fx.ptrtoint(kv_ptr)) + fx.Int64(
                        ((lane % fx.Int32(8)) ^ kpx(kd_f)) * fx.Int32(16)
                    )
                    kr_lane = fx.Int64(fx.ptrtoint(kvr_ptr)) + fx.Int64(
                        (wv * fx.Int32(2) + hq) * fx.Int32(16)
                    )

                    def ix_clamp(t, kk):
                        kidx = t * fx.Int32(KTL) + kk
                        return (kidx < q_lim).select(kidx, q_lim)

                    def row_u(raw):
                        return fx.Int64(_ult(raw, num_rows).select(raw, zero_i))

                dq_k = lane % fx.Int32(32)
                dq_h = (dq_k // fx.Int32(16)) ^ hq
                dq_sc = kfo(fx.Int32(28), dq_k)

                dqa_src = [
                    kfo(wv + fx.Int32(4 * par), dq_k) + dq_h * fx.Int32(8)
                    for par in range(2)
                ]
                dqa_dst = kslot(wv * fx.Int32(2) + dq_h, dq_k)

                def dequant_steps(t):
                    st_ = {}

                    def s0():
                        st_["kfb"] = kf_base(t)
                        st_["vb"] = v_base(t)
                        st_["scd"] = lds_rd16(st_["kfb"] + dq_sc)
                        st_["src"] = [
                            lds_rd(
                                st_["kfb"] + dqa_src[r % 2] + fx.Int32(4096 * (r // 2)),
                                v2i32,
                                8,
                            )
                            for r in range(2)
                        ]

                    def item(r):
                        def f():
                            src = st_["src"]
                            if fx.const_expr(r + 2 < 7):
                                r2 = r + 2
                                src.append(
                                    lds_rd(
                                        st_["kfb"]
                                        + dqa_src[r2 % 2]
                                        + fx.Int32(4096 * (r2 // 2)),
                                        v2i32,
                                        8,
                                    )
                                )
                            e = (
                                fx.Int32(st_["scd"][r // 2]) >> fx.Int32(16 * (r % 2))
                            ) & fx.Int32(0xFF)
                            scl = (e << fx.Int32(23)).bitcast(fx.Float32)
                            wds = []
                            for wd in fx.range_constexpr(2):
                                for sel in fx.range_constexpr(2):
                                    pr = fx.Vector(
                                        fx.rocdl.cvt_scalef32_pk_bf16_fp8(
                                            v2bf16,
                                            fx.Int32(src[r][wd]).ir_value(),
                                            scl.ir_value(),
                                            bool(sel),
                                        )
                                    )
                                    wds.append(pr.bitcast(fx.Int32)[0])
                            lds_wr16(
                                st_["vb"] + dqa_dst + fx.Int32(4096 * r),
                                fx.Vector.from_elements(wds, fx.Int32),
                            )

                        return f

                    return [s0] + [item(r) for r in range(7)]

                NQW = RD_NQW
                NPB = ROWS // 32
                NACC = 4 * NPB
                qk_wave = wv < fx.Int32(NQW)
                PP = 80

                def p_off(row, slot16):
                    return (
                        fx.Int32(RD_P_OFF)
                        + row * fx.Int32(PP)
                        + (slot16 ^ ((row >> fx.Int32(4)) & fx.Int32(1))) * fx.Int32(16)
                    )

                if fx.const_expr(Q16):
                    is_qk = wv >= fx.Int32(4)
                    wq = wv & fx.Int32(3)
                    qk_wave = is_qk

                    def load_q16():
                        q_jb = jb0 + wq
                        q_qp = _div_const(q_jb, HB)
                        qrow_l = q0 + (q_qp < qlen).select(q_qp, qlen - fx.Int32(1))
                        head_l = _rem_const(q_jb, HB) * fx.Int32(16) + m
                        qb = (
                            fx.Int64(qrow_l) * fx.Int64(H) + fx.Int64(head_l)
                        ) * fx.Int64(ROWB)
                        qop = []
                        for cc in fx.range_constexpr(4):
                            lo = load16(
                                q_ptr, qb + fx.Int64(128 * cc) + fx.Int64(gsh * 2)
                            )
                            if fx.const_expr(cc < 3):
                                hi = load16(
                                    q_ptr,
                                    qb + fx.Int64(128 * cc + 64) + fx.Int64(gsh * 2),
                                )
                            else:
                                hi = zero4
                            qop.append(join8(lo, hi))
                        qsc16 = load16(q_ptr, qb + fx.Int64(NOPE))
                        qscl16 = [_scl(qsc16[cc], cc) for cc in range(4)]
                        qrb = (
                            fx.Int64(qrow_l) * fx.Int64(H) + fx.Int64(head_l)
                        ) * fx.Int64(ROPE)
                        qrope = [
                            fx.ptr_load(
                                qr_ptr + qrb + fx.Int64(gsh + fx.Int32(32 * j)),
                                result_type=v8bf16,
                            )
                            for j in range(2)
                        ]
                        return qop, qscl16, qrope

                    def prefetch_next():
                        nb = fx.Int32(fx.block_idx.x) + fx.Int32(cfg.PFN)
                        nblk = ((nseq + fx.Int32(7)) >> fx.Int32(3)) * fx.Int32(
                            8 * NRG * S
                        )

                        def warm(sq, rgn):
                            if fx.const_expr(EPI_Q and MSQ == 1):
                                q0n, qln = sq, fx.Int32(1)
                            else:
                                q0n = _sload(qo_ptr, sq)
                                qln = _sload(qo_ptr, sq + fx.Int32(1)) - q0n
                            kv0n = _sload(indptr_ptr, sq)
                            kvln = _sload(indptr_ptr, sq + fx.Int32(1)) - kv0n
                            gl = wq * fx.Int32(WAVE) + lane
                            acc = zero_i
                            NQL = ROWS * 5
                            for j in range((NQL + 4 * WAVE - 1) // (4 * WAVE)):
                                ln = gl + fx.Int32(4 * WAVE * j)
                                lc = (ln < fx.Int32(NQL)).select(ln, zero_i)
                                r = _div_const(lc, 5)
                                part = lc - r * fx.Int32(5)
                                jb = rgn * fx.Int32(RB) + (r >> fx.Int32(4))
                                qp = _div_const(jb, HB)
                                qp = (qp < qln).select(qp, qln - fx.Int32(1))
                                qrow = (qln > zero_i).select(q0n + qp, zero_i)
                                hd = _rem_const(jb, HB) * fx.Int32(16) + (
                                    r & fx.Int32(15)
                                )
                                ri = fx.Int64(qrow) * fx.Int64(H) + fx.Int64(hd)
                                qa = (part < fx.Int32(4)).select(
                                    fx.Int64(fx.ptrtoint(q_ptr))
                                    + ri * fx.Int64(ROWB)
                                    + fx.Int64(part) * fx.Int64(128),
                                    fx.Int64(fx.ptrtoint(qr_ptr))
                                    + ri * fx.Int64(2 * ROPE),
                                )
                                acc = acc ^ fx.Int32(
                                    fx.generic_load(cu._to_ptr_global(qa))
                                )
                            ki = gl * fx.Int32(32)
                            kic = ((ki < kvln) & (gl < fx.Int32(8))).select(ki, zero_i)
                            ia = (kvln > zero_i).select(
                                fx.Int64(fx.ptrtoint(idx_ptr))
                                + (fx.Int64(kv0n) + fx.Int64(kic)) * fx.Int64(4),
                                fx.Int64(fx.ptrtoint(indptr_ptr))
                                + fx.Int64(sq) * fx.Int64(4),
                            )
                            acc = acc ^ fx.Int32(fx.generic_load(cu._to_ptr_global(ia)))
                            if (nseq < zero_i) & (acc == fx.Int32(1)):
                                fx.ptr_store(fx.Float32(0.0), sink_ptr)

                        sq, rgn, _ = map_bid(nb, nseq)
                        if (nb < nblk) & (sq < nseq):
                            warm(sq, rgn)

                    k16 = [fx.Int32(16 * kb) + m for kb in range(2)]
                    kq16_o = [
                        [
                            fx.Int32(16)
                            * (
                                k16[kb] * fx.Int32(8)
                                + ((fx.Int32(4 * hf) + g) ^ kpx(k16[kb]))
                            )
                            for hf in range(2)
                        ]
                        for kb in range(2)
                    ]
                    ksc16_o = [kfo(fx.Int32(28), k16[kb]) for kb in range(2)]
                    kr16_o = [
                        fx.Int32(16)
                        * (g * fx.Int32(32) + (k16[kb] ^ (g * fx.Int32(4))))
                        for kb in range(2)
                    ]
                    sm_row = wq * fx.Int32(16) + m
                    sm_first = g == zero_i

                else:
                    q_row_l = wv * fx.Int32(32) + r32
                    q_jb = jb0 + wv * fx.Int32(2)
                    q_qp = _div_const(q_jb, HB)
                    qrow_l = q0 + (q_qp < qlen).select(q_qp, qlen - fx.Int32(1))
                    head_l = _rem_const(q_jb, HB) * fx.Int32(16) + r32
                    qb = (fx.Int64(qrow_l) * fx.Int64(H) + fx.Int64(head_l)) * fx.Int64(
                        ROWB
                    )
                    qop = []
                    for st in fx.range_constexpr(7):
                        o_ = qb + fx.Int64(64 * st) + fx.Int64(hq * fx.Int32(32))
                        lo = load16(q_ptr, o_)
                        hi = load16(q_ptr, o_ + fx.Int64(16))
                        qop.append(join8(lo, hi))
                    qsc = load16(q_ptr, qb + fx.Int64(NOPE))
                    qrb = (
                        fx.Int64(qrow_l) * fx.Int64(H) + fx.Int64(head_l)
                    ) * fx.Int64(ROPE)
                    qrope = [
                        fx.ptr_load(
                            qr_ptr
                            + qrb
                            + fx.Int64(fx.Int32(16 * r) + hq * fx.Int32(8)),
                            result_type=v8bf16,
                        )
                        for r in range(4)
                    ]
                    kq_o = [
                        fx.Int32(16)
                        * (
                            r32 * fx.Int32(8)
                            + ((fx.Int32(4 * par + jj) + hq * fx.Int32(2)) ^ kpx(r32))
                        )
                        for par in range(2)
                        for jj in range(2)
                    ]
                    ksc_o = kfo(fx.Int32(28), r32)
                    kr_o = [
                        fx.Int32(16)
                        * (
                            fx.Int32(32) * (fx.Int32(2 * r) + hq)
                            + (r32 ^ ((fx.Int32(2 * (r % 2)) + hq) * fx.Int32(4)))
                        )
                        for r in range(2)
                    ]

                    def k_op(kfb, krb, st):
                        if fx.const_expr(st < 7):
                            o_ = fx.Int32(4096 * (st // 2))
                            lo = lds_rd16(kfb + kq_o[2 * (st % 2)] + o_)
                            hi = lds_rd16(kfb + kq_o[2 * (st % 2) + 1] + o_)
                            return join8(lo, hi)
                        r = st - 7
                        return lds_rd16(
                            krb + kr_o[r % 2] + fx.Int32(1024 * (r // 2) * 2)
                        ).bitcast(fx.BFloat16)

                    def qk_steps(t, out):
                        st_ = {}

                        def step(st):
                            def f():
                                if fx.const_expr(st == 0):
                                    st_["kfb"] = kf_base(t)
                                    st_["krb"] = kr_base(t)
                                    st_["ksc"] = lds_rd16(st_["kfb"] + ksc_o)
                                    st_["sc"] = fx.Vector.filled(16, 0.0, fx.Float32)
                                    st_["ops"] = [
                                        k_op(st_["kfb"], st_["krb"], s2)
                                        for s2 in range(KPD)
                                    ]
                                ops = st_["ops"]
                                if fx.const_expr(st + KPD < 11):
                                    ops.append(k_op(st_["kfb"], st_["krb"], st + KPD))
                                if fx.const_expr(st < 7):
                                    st_["sc"] = fx.Vector(
                                        fx.rocdl.mfma_scale_f32_32x32x64_f8f6f4(
                                            v16f32,
                                            [
                                                ops[st],
                                                qop[st],
                                                st_["sc"],
                                                0,
                                                0,
                                                2 * (st % 2),
                                                fx.Int32(st_["ksc"][st // 2]),
                                                2 * (st % 2),
                                                fx.Int32(qsc[st // 2]),
                                            ],
                                        )
                                    )
                                else:
                                    st_["sc"] = fx.Vector(
                                        fx.rocdl.mfma_f32_32x32x16_bf16(
                                            v16f32, [ops[st], qrope[st - 7], st_["sc"]]
                                        )
                                    )
                                if fx.const_expr(st == 10):
                                    out.append(st_["sc"])

                            return f

                        return [step(st) for st in range(11)]

                    def sc_list(sc):
                        return [fx.Float32(sc[0][r]) for r in range(16)]

                    def key_of(idx):
                        return fx.Int32(8 * (idx // 4) + idx % 4) + hq * fx.Int32(4)

                    def row_max(x):
                        return pmax32(x)

                    def row_sum(x):
                        return x + x.shuffle_xor(fx.Int32(32), fx.Int32(WAVE))

                    NSC = 16
                    sm_row = q_row_l
                    sm_first = hq == zero_i

                    def p_write(pf):
                        for j in fx.range_constexpr(4):
                            pb = fx.Vector.from_elements(
                                pf[4 * j : 4 * j + 4], fx.Float32
                            ).to(fx.BFloat16)
                            _llvm_d.StoreOp(
                                _arith.unwrap(pb.bitcast(fx.Int32)),
                                fx.to_llvm_ptr(
                                    p_rq
                                    + (p_off(sm_row, fx.Int32(j)) + hq * fx.Int32(8))
                                ),
                                alignment=8,
                                noalias_scopes=DMA_AS,
                            )

                    def softmax_steps(t, sc, m_run, l_run, masked, out):
                        st_ = {}

                        def s_scale():
                            tbase = t * fx.Int32(KTL)
                            ss = [x * scale_log2e for x in sc_list(sc)]
                            if fx.const_expr(masked):
                                lim = kvlen - tbase
                                ss = [
                                    (key_of(r) < lim).select(ss[r], ninf)
                                    for r in range(NSC)
                                ]
                            st_["ss"] = ss

                        def s_m():
                            ss = st_["ss"]
                            mx = ss[0]
                            for r in fx.range_constexpr(1, NSC):
                                mx = mx.maximumf(ss[r])
                            mx = row_max(mx)
                            mn = (mx > m_run + fx.Float32(8.0)).select(mx, m_run)
                            st_["mn"] = mn
                            st_["m_safe"] = (mn == ninf).select(fx.Float32(0.0), mn)
                            st_["alpha"] = exp2(m_run - st_["m_safe"])
                            st_["pf"] = []

                        def s_exp(q4):
                            def f():
                                st_["pf"] += [
                                    exp2(st_["ss"][4 * q4 + i] - st_["m_safe"])
                                    for i in range(4)
                                ]

                            return f

                        def s_out():
                            pf = st_["pf"]
                            ps = pf[0]
                            for r in fx.range_constexpr(1, NSC):
                                ps = ps + pf[r]
                            ps = row_sum(ps)
                            p_write(pf)
                            a_acc = (m_run == ninf).select(
                                fx.Float32(1.0), st_["alpha"]
                            )
                            if sm_first:
                                _llvm_d.StoreOp(
                                    _arith.unwrap(a_acc),
                                    fx.to_llvm_ptr(
                                        p_rq
                                        + (fx.Int32(RD_A_OFF) + sm_row * fx.Int32(4))
                                    ),
                                    alignment=4,
                                    noalias_scopes=DMA_AS,
                                )
                            out.extend([st_["mn"], l_run * st_["alpha"] + ps])

                        return (
                            [s_scale, s_m]
                            + [s_exp(q4) for q4 in range(NSC // 4)]
                            + [s_out]
                        )

                lg = lane // fx.Int32(16)
                li = lane % fx.Int32(16)
                lq_a = li // fx.Int32(4)
                lq_b = li % fx.Int32(4)
                cG = (lg & fx.Int32(1)) * fx.Int32(2) + lq_b // fx.Int32(2)
                swG = cG * fx.Int32(4)
                pvd_o = [
                    fx.Int32(16)
                    * (
                        fx.Int32(32) * cG
                        + ((hq * fx.Int32(8) + fx.Int32(4 * qq)) ^ swG)
                        + lq_a
                    )
                    + (lq_b & fx.Int32(1)) * fx.Int32(8)
                    for qq in range(2)
                ]
                pv_dvb = wv * fx.Int32(4)
                rope_w = wv == fx.Int32(3)

                def v_opd(t, jj, ks):
                    vb = v_base(t)
                    krb = kr_base(t)
                    j = pv_dvb + fx.Int32(jj)
                    if fx.const_expr(jj < 2):
                        base = vb + j * fx.Int32(2048)
                    else:
                        base = rope_w.select(
                            krb + (j - fx.Int32(14)) * fx.Int32(2048),
                            vb + j * fx.Int32(2048),
                        )
                    o_ = base + fx.Int32(256 * ks)
                    h0_ = lds_tr(o_ + pvd_o[0])
                    h1_ = lds_tr(o_ + pvd_o[1])
                    return fx.Vector.from_elements(
                        [h0_[i] for i in range(4)] + [h1_[i] for i in range(4)],
                        fx.BFloat16,
                    )

                def p_opd(rb, ks):
                    row = fx.Int32(32 * rb) + r32
                    return lds_rd16(p_off(row, fx.Int32(2 * ks) + hq)).bitcast(
                        fx.BFloat16
                    )

                def pv_steps(t, accs, out, last=False):
                    st_ = {}
                    steps = [
                        (ks, jj, rb)
                        for ks in range(2)
                        for jj in range(4)
                        for rb in range(NPB)
                    ]
                    NS = len(steps)

                    def ops_of(n):
                        ks, jj, rb = steps[n]
                        if fx.const_expr(rb == 0):
                            st_["va"][(ks, jj)] = v_opd(t, jj, ks)
                        if fx.const_expr(jj == 0):
                            st_["pb"][(ks, rb)] = p_opd(rb, ks)

                    def step(n):
                        def f():
                            if fx.const_expr(n == 0):
                                st_["va"] = {}
                                st_["pb"] = {}
                                st_["acc"] = list(accs)
                                for k0 in fx.range_constexpr(min(RD_PVD, NS)):
                                    ops_of(k0)
                            if fx.const_expr(n + RD_PVD < NS):
                                ops_of(n + RD_PVD)
                            ks, jj, rb = steps[n]
                            ai = jj * NPB + rb
                            st_["acc"][ai] = pv_mfma(
                                st_["va"][(ks, jj)],
                                st_["pb"][(ks, rb)],
                                st_["acc"][ai],
                                last and n == NS - 1,
                            )
                            if fx.const_expr(n == NS - 1):
                                out.extend(st_["acc"])

                        return f

                    return [step(n) for n in range(NS)]

                def rescale_d(accs):
                    al = [
                        fx.Float32(
                            lds_rd(
                                fx.Int32(RD_A_OFF)
                                + (fx.Int32(32 * rb) + r32) * fx.Int32(4),
                                fx.Vector.make_type(1, fx.Float32),
                                4,
                            )[0]
                        )
                        for rb in range(NPB)
                    ]
                    ne = al[0] != fx.Float32(1.0)
                    for rb in fx.range_constexpr(1, NPB):
                        ne = ne | (al[rb] != fx.Float32(1.0))
                    need = fx.Int64(fx.rocdl.ballot(T.i64, ne.ir_value())) != fx.Int64(
                        0
                    )
                    if need:
                        accs = [
                            rescale_blk(accs[ai], al[ai % NPB]) for ai in range(NACC)
                        ]
                    return accs

                def rescale_blk(a, alpha, n=16):
                    st2 = _llvm_d.StructType.get_literal([T.f32, T.f32])
                    els = []
                    for i in fx.range_constexpr(n):
                        r_ = _llvm_d.InlineAsmOp(
                            st2,
                            [fx.Float32(a[i]).ir_value(), alpha.ir_value()],
                            "v_accvgpr_read_b32 $1, $0\n"
                            " s_nop 1\n"
                            " v_mul_f32 $1, $1, $3\n"
                            " v_accvgpr_write_b32 $0, $1",
                            "=a,=&v,0,v",
                        ).res
                        els.append(fx.Float32(_llvm_d.extractvalue(T.f32, r_, [0])))
                    return fx.Vector.from_elements(els, fx.Float32)

                if fx.const_expr(Q16):
                    PBUF = ROWS * PP
                    ABUF = ROWS * 4

                    def ro(ph, n, stride, extra=0):
                        if isinstance(ph, int):
                            return (ph % n) * stride + extra
                        return (ph & fx.Int32(n - 1)) * fx.Int32(stride) + fx.Int32(
                            extra
                        )

                    def at(base, off):
                        return base + (fx.Int32(off) if isinstance(off, int) else off)

                    def padd(ph, k):
                        return ph + k if isinstance(ph, int) else ph + fx.Int32(k)

                    kr_b = [fx.Int32(RQ_KR_OFF) + kr16_o[kb] for kb in range(2)]
                    pw_b = (
                        fx.Int32(RD_P_OFF)
                        + sm_row * fx.Int32(PP)
                        + ((g >> fx.Int32(1)) ^ (wq & fx.Int32(1))) * fx.Int32(16)
                        + (g & fx.Int32(1)) * fx.Int32(8)
                    )
                    aw_b = fx.Int32(RD_A_OFF) + sm_row * fx.Int32(4)
                    hi_g = g >= fx.Int32(2)

                    KPF = 3
                    QK_ORDER = [(kb, st) for st in range(6) for kb in range(2)]

                    def k_op(ph, kb, st):
                        if fx.const_expr(st < 4):
                            lo = lds_rd16(
                                at(kq16_o[kb][0], ro(ph, NKF, RQ_KF, 4096 * st))
                            )
                            hi = lds_rd16(
                                at(
                                    kq16_o[kb][1],
                                    ro(ph, NKF, RQ_KF, 4096 * st if st < 3 else 0),
                                )
                            )
                            return join8(lo, hi)
                        return lds_rd16(
                            at(kr_b[kb], ro(ph, NKR, RQ_KR, 2048 * (st - 4)))
                        ).bitcast(fx.BFloat16)

                    def qk(q16, ph):
                        qop, qscl16, qrope = q16
                        ksc = [
                            lds_rd16(at(ksc16_o[kb], ro(ph, NKF, RQ_KF)))
                            for kb in range(2)
                        ]
                        ops = {}
                        for n in fx.range_constexpr(KPF):
                            ops[n] = k_op(ph, *QK_ORDER[n])
                        scl = []
                        for kb in fx.range_constexpr(2):
                            row = []
                            for cc in fx.range_constexpr(4):
                                b = fx.Int32(ksc[kb][cc]) >> gsh
                                if fx.const_expr(cc == 3):
                                    b = hi_g.select(fx.Int32(127), b)
                                row.append(b)
                            scl.append(row)
                        sc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
                        for n in fx.range_constexpr(len(QK_ORDER)):
                            if fx.const_expr(n + KPF < len(QK_ORDER)):
                                ops[n + KPF] = k_op(ph, *QK_ORDER[n + KPF])
                            kb, st = QK_ORDER[n]
                            if fx.const_expr(st < 4):
                                sc[kb] = fx.Vector(
                                    fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                        v4f32,
                                        [
                                            ops[n],
                                            qop[st],
                                            sc[kb],
                                            0,
                                            0,
                                            0,
                                            scl[kb][st],
                                            0,
                                            qscl16[st],
                                        ],
                                    )
                                )
                            else:
                                sc[kb] = fx.Vector(
                                    fx.rocdl.mfma_f32_16x16x32_bf16(
                                        v4f32, [ops[n], qrope[st - 4], sc[kb]]
                                    )
                                )
                        return sc

                    def softmax(ph, sc, m_run, l_run, lim, tt, bnd):
                        ss = [
                            fx.Float32(sc[kb][i]) * scale_log2e
                            for kb in range(2)
                            for i in range(4)
                        ]
                        if fx.const_expr(lim is not None):
                            ss = [
                                (
                                    fx.Int32(16 * (r // 4) + r % 4) + g * fx.Int32(4)
                                    < lim
                                ).select(ss[r], ninf)
                                for r in range(8)
                            ]
                        if fx.const_expr(BOUNDS):
                            a0, a1, b0, b1 = bnd
                            tb = tt * fx.Int32(KTL)
                            te = tb + fx.Int32(KTL)
                            inside = ((a0 <= tb) & (a1 >= te)) | (
                                (b0 <= tb) & (b1 >= te)
                            )
                            kl = tb + g * fx.Int32(4)
                            la0 = a0 - kl
                            lb0 = b0 - kl
                            wa = (a1 > a0).select(a1 - a0, zero_i)
                            wb = (b1 > b0).select(b1 - b0, zero_i)

                            def bmask(ss=ss, la0=la0, lb0=lb0, wa=wa, wb=wb):
                                return [
                                    (
                                        _ult(fx.Int32(16 * (r // 4) + r % 4) - la0, wa)
                                        | _ult(
                                            fx.Int32(16 * (r // 4) + r % 4) - lb0, wb
                                        )
                                    ).select(ss[r], ninf)
                                    for r in range(8)
                                ]

                            if inside == false_:
                                ss = bmask()
                        mx = ss[0]
                        for r in fx.range_constexpr(1, 8):
                            mx = mx.maximumf(ss[r])
                        mx = plswap_f(plswap_f(mx, 16, _fmax), 32, _fmax)
                        mn = (mx > m_run + fx.Float32(8.0)).select(mx, m_run)
                        m_safe = (mn == ninf).select(fx.Float32(0.0), mn)
                        alpha = exp2(m_run - m_safe)
                        pf = [exp2(ss[r] - m_safe) for r in range(8)]
                        pb_ = ro(ph, 2, PBUF)
                        for kb in fx.range_constexpr(2):
                            pbv = fx.Vector.from_elements(
                                pf[4 * kb : 4 * kb + 4], fx.Float32
                            ).to(fx.BFloat16)
                            _llvm_d.StoreOp(
                                _arith.unwrap(pbv.bitcast(fx.Int32)),
                                fx.to_llvm_ptr(
                                    p_rq
                                    + at(
                                        pw_b,
                                        (
                                            pb_ + 32 * kb
                                            if isinstance(pb_, int)
                                            else pb_ + fx.Int32(32 * kb)
                                        ),
                                    )
                                ),
                                alignment=8,
                                noalias_scopes=DMA_AS,
                            )
                        ps = pf[0]
                        for r in fx.range_constexpr(1, 8):
                            ps = ps + pf[r]
                        ps = plswap_f(plswap_f(ps, 16, _fadd), 32, _fadd)
                        a_acc = (m_run == ninf).select(fx.Float32(1.0), alpha)
                        if sm_first:
                            _llvm_d.StoreOp(
                                _arith.unwrap(a_acc),
                                fx.to_llvm_ptr(p_rq + at(aw_b, ro(ph, 2, ABUF))),
                                alignment=4,
                                noalias_scopes=DMA_AS,
                            )
                        return mn, l_run * alpha + ps

                    def qk_step(q16, ph, m_run, l_run, lim, tt, bnd):
                        sc = qk(q16, ph)
                        mn, ln = softmax(ph, sc, m_run, l_run, lim, tt, bnd)
                        lds_barrier()
                        return mn, ln

                    vt_b = [
                        fx.Int32(RQ_V_OFF) + wv * fx.Int32(8192) + pvd_o[q]
                        for q in range(2)
                    ]
                    vt23_b = [
                        rope_w.select(
                            fx.Int32(RQ_KR_OFF) + pvd_o[q],
                            fx.Int32(RQ_V_OFF + 4096) + wv * fx.Int32(8192) + pvd_o[q],
                        )
                        for q in range(2)
                    ]
                    prx = (r32 >> fx.Int32(4)) & fx.Int32(1)
                    pr_b = (
                        fx.Int32(RD_P_OFF)
                        + r32 * fx.Int32(PP)
                        + (hq ^ prx) * fx.Int32(16)
                    )
                    ar_b = fx.Int32(RD_A_OFF) + r32 * fx.Int32(4)
                    ixr_b = (
                        fx.Int32(RQ_IX_OFF) + wv * fx.Int32(512) + lane * fx.Int32(4)
                    )
                    hd = rope_w.select(zero_i, hq)
                    dqr_b = [
                        fx.Int32(16)
                        * (
                            wq * fx.Int32(256)
                            + r32 * fx.Int32(8)
                            + ((hd * fx.Int32(4) + fx.Int32(j)) ^ kpx(r32))
                        )
                        for j in range(4)
                    ]
                    dqs_b = kfo(fx.Int32(28), r32) + wq * fx.Int32(4)
                    dq_sh = fx.Int32(23) - hd * fx.Int32(16)
                    dqw_b = [
                        fx.Int32(RQ_V_OFF)
                        + wq * fx.Int32(8192)
                        + hd * fx.Int32(4096)
                        + fx.Int32(16) * (r32 ^ fx.Int32(4 * c))
                        for c in range(4)
                    ]

                    def dequant_steps(ph):
                        kb_ = ro(ph, NKF, RQ_KF)
                        vb = ro(ph, 2, RQ_V)
                        st_ = {}

                        def rd():
                            st_["d"] = fx.Int32(
                                lds_rd(
                                    at(dqs_b, kb_), fx.Vector.make_type(1, fx.Int32), 4
                                )[0]
                            )
                            st_["src"] = [lds_rd16(at(dqr_b[j], kb_)) for j in range(4)]

                        def cv(j, e):
                            def f():
                                if fx.const_expr(j == 0 and e == 0):
                                    st_["scl"] = (
                                        (st_["d"] << dq_sh) & fx.Int32(0x7F800000)
                                    ).bitcast(fx.Float32)
                                scl = st_["scl"]
                                wds = []
                                for wd in fx.range_constexpr(2):
                                    for sel in fx.range_constexpr(2):
                                        pr = fx.Vector(
                                            fx.rocdl.cvt_scalef32_pk_bf16_fp8(
                                                v2bf16,
                                                fx.Int32(
                                                    st_["src"][j][2 * e + wd]
                                                ).ir_value(),
                                                scl.ir_value(),
                                                bool(sel),
                                            )
                                        )
                                        wds.append(pr.bitcast(fx.Int32)[0])
                                c = (2 * j + e) & 3
                                lds_wr16(
                                    at(
                                        dqw_b[c],
                                        (
                                            vb + 512 * (2 * j + e)
                                            if isinstance(vb, int)
                                            else vb + fx.Int32(512 * (2 * j + e))
                                        ),
                                    ),
                                    fx.Vector.from_elements(wds, fx.Int32),
                                )

                            return f

                        return [rd] + [cv(j, e) for j in range(4) for e in range(2)]

                    def pv_ops(ph, b23, n):
                        ks, jj, rb = PV_SEQ[n]
                        vb = ro(ph, 2, RQ_V)
                        pb_ = ro(ph, 2, PBUF)
                        va = pb = None
                        if fx.const_expr(rb == 0):
                            if fx.const_expr(jj < 2):
                                hs = [
                                    lds_tr(
                                        at(
                                            vt_b[q],
                                            (
                                                vb + 2048 * jj + 256 * ks
                                                if isinstance(vb, int)
                                                else vb + fx.Int32(2048 * jj + 256 * ks)
                                            ),
                                        )
                                    )
                                    for q in range(2)
                                ]
                            else:
                                hs = [
                                    lds_tr(
                                        b23[q] + fx.Int32(2048 * (jj - 2) + 256 * ks)
                                    )
                                    for q in range(2)
                                ]
                            va = fx.Vector.from_elements(
                                [hs[0][i] for i in range(4)]
                                + [hs[1][i] for i in range(4)],
                                fx.BFloat16,
                            )
                        if fx.const_expr(jj == 0):
                            pb = lds_rd16(
                                at(
                                    pr_b,
                                    (
                                        pb_ + 2560 * rb + 32 * ks
                                        if isinstance(pb_, int)
                                        else pb_ + fx.Int32(2560 * rb + 32 * ks)
                                    ),
                                )
                            ).bitcast(fx.BFloat16)
                        return va, pb

                    PV_SEQ = [
                        (ks, jj, rb)
                        for ks in range(2)
                        for jj in range(4)
                        for rb in range(NPB)
                    ]

                    def pv_steps(ph, accs, out):
                        st_ = {}
                        NS = len(PV_SEQ)

                        def fetch(n):
                            a_, b_ = pv_ops(ph, st_["b23"], n)
                            ks, jj, rb = PV_SEQ[n]
                            if fx.const_expr(a_ is not None):
                                st_["va"][(ks, jj)] = a_
                            if fx.const_expr(b_ is not None):
                                st_["pb"][(ks, rb)] = b_

                        def step(n):
                            def f():
                                if fx.const_expr(n == 0):
                                    sel = rope_w.select(
                                        fx.Int32(ro(ph, NKR, RQ_KR)),
                                        fx.Int32(ro(ph, 2, RQ_V)),
                                    )
                                    st_["b23"] = [vt23_b[q] + sel for q in range(2)]
                                    st_["va"], st_["pb"] = {}, {}
                                    st_["acc"] = list(accs)
                                    for k0 in fx.range_constexpr(min(RD_PVD, NS)):
                                        fetch(k0)
                                if fx.const_expr(n + RD_PVD < NS):
                                    fetch(n + RD_PVD)
                                ks, jj, rb = PV_SEQ[n]
                                ai = jj * NPB + rb
                                st_["acc"][ai] = pv_mfma(
                                    st_["va"][(ks, jj)],
                                    st_["pb"][(ks, rb)],
                                    st_["acc"][ai],
                                    False,
                                )
                                if fx.const_expr(n == NS - 1):
                                    out.extend(st_["acc"])

                            return f

                        return [step(n) for n in range(NS)]

                    def al_read(ph):
                        ab = ro(ph, 2, ABUF)
                        return [
                            fx.Float32(
                                lds_rd(
                                    at(
                                        ar_b,
                                        (
                                            ab + 128 * rb
                                            if isinstance(ab, int)
                                            else ab + fx.Int32(128 * rb)
                                        ),
                                    ),
                                    fx.Vector.make_type(1, fx.Float32),
                                    4,
                                )[0]
                            )
                            for rb in range(NPB)
                        ]

                    def rescale_al(accs, al):
                        ne = al[0] != fx.Float32(1.0)
                        for rb in fx.range_constexpr(1, NPB):
                            ne = ne | (al[rb] != fx.Float32(1.0))
                        need = fx.Int64(
                            fx.rocdl.ballot(T.i64, ne.ir_value())
                        ) != fx.Int64(0)
                        if need:
                            accs = [
                                rescale_blk(accs[ai], al[ai % NPB])
                                for ai in range(NACC)
                            ]
                        return accs

                    if fx.const_expr(PV16R):
                        t16a = m >> fx.Int32(2)
                        t16b = m & fx.Int32(3)
                        t16c = t16b >> fx.Int32(1)

                        def t16_lane(par, qq):
                            k = g * fx.Int32(8) + fx.Int32(4 * qq) + t16a
                            return fx.Int32(16) * (
                                t16c * fx.Int32(32)
                                + (k ^ ((fx.Int32(2 * par) + t16c) * fx.Int32(4)))
                            ) + (t16b & fx.Int32(1)) * fx.Int32(8)

                        v16_lo = [
                            [
                                fx.Int32(RQ_V_OFF)
                                + wv * fx.Int32(8192)
                                + t16_lane(par, qq)
                                for qq in range(2)
                            ]
                            for par in range(2)
                        ]
                        v16_hi = [
                            [
                                rope_w.select(
                                    fx.Int32(RQ_KR_OFF) + t16_lane(par, qq),
                                    fx.Int32(RQ_V_OFF + 4096)
                                    + wv * fx.Int32(8192)
                                    + t16_lane(par, qq),
                                )
                                for qq in range(2)
                            ]
                            for par in range(2)
                        ]
                        p16_b = [
                            fx.Int32(RD_P_OFF)
                            + m * fx.Int32(PP)
                            + (g ^ fx.Int32(par)) * fx.Int32(16)
                            for par in range(2)
                        ]
                        a16_b = fx.Int32(RD_A_OFF) + m * fx.Int32(4)
                        PV16_SEQ = [(blk, rb) for blk in range(8) for rb in range(RB)]

                        def pv16_mfma(va, pb, acc):
                            return fx.Vector(
                                _llvm_d.InlineAsmOp(
                                    v4f32,
                                    [va.ir_value(), pb.ir_value(), acc.ir_value()],
                                    "v_mfma_f32_16x16x32_bf16 $0, $1, $2, $3",
                                    "=a,v,v,0",
                                ).res
                            )

                        def pv16_a(ph, hi, blk):
                            par = blk & 1
                            if fx.const_expr(blk < 4):
                                vb = ro(ph, 2, RQ_V)
                                hs = [
                                    lds_tr(
                                        at(
                                            v16_lo[par][qq],
                                            (
                                                vb + 1024 * blk
                                                if isinstance(vb, int)
                                                else vb + fx.Int32(1024 * blk)
                                            ),
                                        )
                                    )
                                    for qq in range(2)
                                ]
                            else:
                                hs = [
                                    lds_tr(hi[par][qq] + fx.Int32(1024 * (blk - 4)))
                                    for qq in range(2)
                                ]
                            return fx.Vector.from_elements(
                                [hs[0][i] for i in range(4)]
                                + [hs[1][i] for i in range(4)],
                                fx.BFloat16,
                            )

                        def pv16_b(ph, rb):
                            pb_ = ro(ph, 2, PBUF)
                            return lds_rd16(
                                at(
                                    p16_b[rb & 1],
                                    (
                                        pb_ + 1280 * rb
                                        if isinstance(pb_, int)
                                        else pb_ + fx.Int32(1280 * rb)
                                    ),
                                )
                            ).bitcast(fx.BFloat16)

                        PV16_AHEAD = 2

                        def pv16_steps(ph, accs, out):
                            st_ = {}
                            NS = len(PV16_SEQ)

                            def step(n):
                                def f():
                                    blk, rb = PV16_SEQ[n]
                                    if fx.const_expr(n == 0):
                                        sel = rope_w.select(
                                            fx.Int32(ro(ph, NKR, RQ_KR)),
                                            fx.Int32(ro(ph, 2, RQ_V)),
                                        )
                                        st_["hi"] = [
                                            [v16_hi[par][qq] + sel for qq in range(2)]
                                            for par in range(2)
                                        ]
                                        st_["acc"] = list(accs)
                                        st_["va"] = {}
                                        st_["pb"] = {}
                                        for b0 in fx.range_constexpr(PV16_AHEAD):
                                            st_["va"][b0] = pv16_a(ph, st_["hi"], b0)
                                            if fx.const_expr(b0 == 0):
                                                for r0 in fx.range_constexpr(RB):
                                                    st_["pb"][r0] = pv16_b(ph, r0)
                                    if fx.const_expr(rb == 0 and blk + PV16_AHEAD < 8):
                                        st_["va"][blk + PV16_AHEAD] = pv16_a(
                                            ph, st_["hi"], blk + PV16_AHEAD
                                        )
                                    ai = blk * RB + rb
                                    st_["acc"][ai] = pv16_mfma(
                                        st_["va"][blk], st_["pb"][rb], st_["acc"][ai]
                                    )
                                    if fx.const_expr(n == NS - 1):
                                        out.extend(st_["acc"])

                                return f

                            return [step(n) for n in range(NS)]

                        def al16_read(ph):
                            ab = ro(ph, 2, ABUF)
                            return [
                                fx.Float32(
                                    lds_rd(
                                        at(
                                            a16_b,
                                            (
                                                ab + 64 * rb
                                                if isinstance(ab, int)
                                                else ab + fx.Int32(64 * rb)
                                            ),
                                        ),
                                        fx.Vector.make_type(1, fx.Float32),
                                        4,
                                    )[0]
                                )
                                for rb in range(RB)
                            ]

                        def rescale16(accs, al):
                            ne = al[0] != fx.Float32(1.0)
                            for rb in fx.range_constexpr(1, RB):
                                ne = ne | (al[rb] != fx.Float32(1.0))
                            need = fx.Int64(
                                fx.rocdl.ballot(T.i64, ne.ir_value())
                            ) != fx.Int64(0)
                            if need:
                                accs = [
                                    rescale_blk(accs[ai], al[ai % RB], 4)
                                    for ai in range(8 * RB)
                                ]
                            return accs

                    def ptile(q):
                        return (q < t1 - t0).select(t0 + q - fx.Int32(1), t1)

                    def idx_rd(ph):
                        return [
                            fx.Int32(
                                lds_rd(
                                    at(ixr_b, ro(ph, NIX, 2048, 256 * u)),
                                    fx.Vector.make_type(1, fx.Int32),
                                    4,
                                )[0]
                            )
                            for u in range(NKD)
                        ]

                    def dma_ix(q, ph):
                        for u, kk in enumerate(kd_list):
                            ga = ibase + fx.Int64(ix_clamp(ptile(q), kk)) * fx.Int64(4)
                            _rocdl_d.global_load_lds(
                                fx.to_llvm_ptr(cu._to_ptr_global(ga)),
                                fx.to_llvm_ptr(
                                    p_rq
                                    + at(
                                        wv * fx.Int32(512),
                                        ro(ph, NIX, 2048, RQ_IX_OFF + 256 * u),
                                    )
                                ),
                                4,
                                0,
                                alias_scopes=DMA_AS,
                            )

                    def dma_kv(rows, ph):
                        gk = kf_lane + row_u(rows[0]) * fx.Int64(ROWB)
                        for cg in fx.range_constexpr(4):
                            dma16(
                                gk + fx.Int64(128 * cg),
                                at(wv * fx.Int32(1024), ro(ph, NKF, RQ_KF, 4096 * cg)),
                                KV_NT,
                            )
                        gr = kr_lane + row_u(rows[1]) * fx.Int64(2 * ROPE)
                        dma16(
                            gr,
                            at(wv * fx.Int32(1024), ro(ph, NKR, RQ_KR, RQ_KR_OFF)),
                            KV_NT,
                        )

                    def pv_step(ph, s, accs, live=None, v16=False):
                        i3 = idx_rd(padd(ph, 3))
                        al = al16_read(ph) if v16 else al_read(ph)
                        fence()
                        accs = rescale16(accs, al) if v16 else rescale_al(accs, al)
                        fence()

                        def dma_a():
                            dma_ix(s + fx.Int32(5), padd(ph, 5))

                        def dma_b():
                            dma_kv(i3, padd(ph, 3))

                        side = [dma_a, dma_b]
                        if fx.const_expr(live is not None):
                            if live:
                                dma_b()
                            fence()
                            side = [dma_a]
                        po = []
                        run_mixed(
                            pv16_steps(ph, accs, po) if v16 else pv_steps(ph, accs, po),
                            side + dequant_steps(padd(ph, 1)),
                            every=1,
                        )
                        if fx.const_expr(live is None):
                            fx.rocdl.s_waitcnt(vmcnt=NVM * (DD - 2))
                        else:
                            if live:
                                fx.rocdl.s_waitcnt(vmcnt=NVM * (DD - 2))
                            else:
                                fx.rocdl.s_waitcnt(vmcnt=NKD * (DD - 2))
                        fence()
                        lds_barrier()
                        fence()
                        return po

                    nits = t1 - t0
                    tp0 = (t1 > t0).select(t1 - fx.Int32(1), t0)
                    lim0 = (t1 > t0).select(kvlen - tp0 * fx.Int32(KTL), zero_i)
                    if fx.const_expr(MERGED):
                        slot, myx, tk_lane = protocol_entry()
                    acc0z = fx.Vector.filled(16, 0.0, fx.Float32)

                    def qk_role():
                        q16 = load_q16()
                        if fx.const_expr(Q16_OSC):
                            snk_q = fx.Float32(
                                fx.ptr_load(
                                    sink_ptr
                                    + fx.Int64(
                                        (jb0 * fx.Int32(16) + sm_row) & fx.Int32(H - 1)
                                    )
                                )
                            ) * fx.Float32(LOG2E)
                        bnd = None
                        if fx.const_expr(BOUNDS):
                            b_qp = _div_const(jb0 + wq, HB)
                            b_row = q0 + (b_qp < qlen).select(b_qp, qlen - fx.Int32(1))
                            bnd = [
                                _sload(bnd_ptr, b_row * fx.Int32(4) + fx.Int32(i))
                                for i in range(4)
                            ]
                        lds_barrier()
                        m_c, l_c = qk_step(
                            q16, 0, ninf, fx.Float32(0.0), lim0, tp0, bnd
                        )
                        nq = (nits > fx.Int32(1)).select(nits - fx.Int32(1), zero_i)
                        ngq = nq >> fx.Int32(2)
                        for itq, stq in range(
                            zero_i,
                            ngq,
                            fx.Int32(1),
                            init=[m_c.ir_value(), l_c.ir_value()],
                        ):
                            m_q = fx.Float32(stq[0])
                            l_q = fx.Float32(stq[1])
                            tq = t0 + fx.Int32(itq) * fx.Int32(4)
                            for u in fx.range_constexpr(4):
                                m_q, l_q = qk_step(
                                    q16, u + 1, m_q, l_q, None, tq + fx.Int32(u), bnd
                                )
                            rq = yield [m_q.ir_value(), l_q.ir_value()]
                        m_l = fx.Float32(rq[0])
                        l_l = fx.Float32(rq[1])
                        remq = nq & fx.Int32(3)
                        tr0 = t0 + ngq * fx.Int32(4)
                        for u in fx.range_constexpr(3):
                            if remq > fx.Int32(u):
                                m_r, l_r = qk_step(
                                    q16, u + 1, m_l, l_l, None, tr0 + fx.Int32(u), bnd
                                )
                                m_l = m_r
                                l_l = l_r
                        if fx.const_expr(Q16_OSC):
                            lf = l_l.maximumf(fx.Float32(1e-30))
                            lse2 = m_l + fly_math.log2(lf)
                            gm = snk_q.maximumf(lse2)
                            w_o = exp2(lse2 - gm)
                            den = exp2(snk_q - gm) + w_o
                            if fx.const_expr(BOUNDS):
                                nokey = l_l == fx.Float32(0.0)
                            else:
                                nokey = kvlen == zero_i
                            osc = nokey.select(
                                qnan, ((fx.Float32(1.0) / lf) * w_o) * rcp(den)
                            )
                        if sm_first:
                            if fx.const_expr(Q16_OSC):
                                _llvm_d.StoreOp(
                                    _arith.unwrap(osc),
                                    fx.to_llvm_ptr(
                                        p_rq
                                        + (
                                            fx.Int32(RD_ML_OFF + 8 * ROWS)
                                            + sm_row * fx.Int32(4)
                                        )
                                    ),
                                    alignment=4,
                                )
                            _llvm_d.StoreOp(
                                _arith.unwrap(m_l),
                                fx.to_llvm_ptr(
                                    p_rq + (fx.Int32(RD_ML_OFF) + sm_row * fx.Int32(4))
                                ),
                                alignment=4,
                            )
                            _llvm_d.StoreOp(
                                _arith.unwrap(l_l),
                                fx.to_llvm_ptr(
                                    p_rq
                                    + (
                                        fx.Int32(RD_ML_OFF + 4 * ROWS)
                                        + sm_row * fx.Int32(4)
                                    )
                                ),
                                alignment=4,
                            )
                        lds_barrier()
                        lds_barrier()
                        if fx.const_expr(cfg.PFN):
                            prefetch_next()
                        return (
                            m_l,
                            l_l,
                            [
                                fx.Vector(
                                    _llvm_d.InlineAsmOp(
                                        v16f32, [acc0z.ir_value()], "; acc $0", "=a,0"
                                    ).res
                                )
                                for _ in range(0 if PV16R else NACC)
                            ],
                        )

                    def pv_role():
                        if fx.const_expr(PV16R):
                            pv_kvall = _sload(indptr_ptr, nseq) - _sload(
                                indptr_ptr, zero_i
                            )
                        ptiles = [tp0, ptile(fx.Int32(1)), ptile(fx.Int32(2))]
                        idx0 = [
                            [ld_idx(ix_clamp(tt_, kk)) for kk in kd_list]
                            for tt_ in ptiles
                        ]
                        for j in fx.range_constexpr(DD):
                            dma_ix(fx.Int32(j + 2), j + 2)
                            dma_kv(idx0[j], j)
                        fx.rocdl.s_waitcnt(vmcnt=NVM * (DD - 1))
                        lds_barrier()
                        run_gen(dequant_steps(0))
                        fx.rocdl.s_waitcnt(vmcnt=NVM * (DD - 2))
                        lds_barrier()
                        if fx.const_expr(PV16R):
                            lo_, hi_ = PV16R
                            use16 = _ult(kvlen - fx.Int32(lo_ + 1), fx.Int32(hi_ - lo_))
                            use16 = use16 & (
                                fx.Int64(kvlen) * fx.Int64(nseq) * fx.Int64(4)
                                <= fx.Int64(pv_kvall) * fx.Int64(5)
                            )
                            if use16:
                                q16_store(pv_body(True), True)
                            else:
                                q16_store(pv_body(False), False)
                            return ninf, fx.Float32(0.0), []
                        return ninf, fx.Float32(0.0), pv_body(False)

                    def pv_body(v16):
                        nacc = 8 * RB if v16 else NACC
                        az = fx.Vector.filled(4 if v16 else 16, 0.0, fx.Float32)
                        acc_c = [
                            fx.Vector(
                                _llvm_d.InlineAsmOp(
                                    v4f32 if v16 else v16f32,
                                    [az.ir_value()],
                                    "; acc $0",
                                    "=a,0",
                                ).res
                            )
                            for _ in range(nacc)
                        ]
                        nva = (nits > zero_i).select(nits, zero_i)
                        nv = (nva > fx.Int32(DD)).select(nva - fx.Int32(DD), zero_i)
                        ngv = nv >> fx.Int32(2)
                        for itv, stv in range(
                            zero_i, ngv, fx.Int32(1), init=[x.ir_value() for x in acc_c]
                        ):
                            acc_v = [fx.Vector(stv[j]) for j in range(nacc)]
                            sv = fx.Int32(itv) * fx.Int32(4)
                            for u in fx.range_constexpr(4):
                                acc_v = pv_step(u, sv + fx.Int32(u), acc_v, v16=v16)
                            rv = yield [x.ir_value() for x in acc_v]
                        acc_l = [fx.Vector(rv[j]) for j in range(nacc)]
                        sv0 = ngv * fx.Int32(4)
                        remv = nva - sv0
                        for u in fx.range_constexpr(DD + 3):
                            if remv > fx.Int32(u):
                                acc_r = pv_step(
                                    u,
                                    sv0 + fx.Int32(u),
                                    acc_l,
                                    live=sv0 + fx.Int32(u) < nv,
                                    v16=v16,
                                )
                                acc_l = acc_r
                        if nits <= zero_i:
                            lds_barrier()
                        # drain the PV role's DMAs before the epilogue reuses the tile rings; the s_nops
                        # cover the inline-asm MFMA -> VALU accumulator read
                        fx.rocdl.s_waitcnt(vmcnt=0)
                        lds_barrier()
                        _llvm_d.InlineAsmOp(
                            None,
                            [],
                            "s_nop 7\n s_nop 7\n s_nop 3",
                            "",
                            has_side_effects=True,
                        )
                        return acc_l

                    def q16_store(acc_l, v16):
                        ln_ = fx.Int32(
                            _llvm_d.InlineAsmOp(
                                T.i32,
                                [],
                                "v_mbcnt_lo_u32_b32 $0, -1, 0\n"
                                " v_mbcnt_hi_u32_b32 $0, -1, $0",
                                "=v",
                                has_side_effects=True,
                            ).res
                        )
                        m_ = ln_ & fx.Int32(15)
                        g_ = ln_ >> fx.Int32(4)
                        hq_ = ln_ >> fx.Int32(5)
                        r32_ = ln_ & fx.Int32(31)
                        Q_OBP = 272
                        qo_wb = wv * fx.Int32(ROWS * Q_OBP)

                        def acc_rd(a, i):
                            return fx.Float32(
                                _llvm_d.InlineAsmOp(
                                    T.f32,
                                    [fx.Float32(a[i]).ir_value()],
                                    "v_accvgpr_read_b32 $0, $1",
                                    "=v,a",
                                    has_side_effects=True,
                                ).res
                            )

                        for rb in fx.range_constexpr(RB if v16 else NPB):
                            row = (
                                fx.Int32(16 * rb) + m_
                                if v16
                                else fx.Int32(32 * rb) + r32_
                            )
                            osc = fx.Float32(
                                lds_rd(
                                    fx.Int32(RD_ML_OFF + 8 * ROWS) + row * fx.Int32(4),
                                    fx.Vector.make_type(1, fx.Float32),
                                    4,
                                )[0]
                            )
                            for nt in fx.range_constexpr(8 if v16 else 16):
                                fx.rocdl.sched_barrier(0)
                                if fx.const_expr(v16):
                                    a = acc_l[nt * RB + rb]
                                    av = [acc_rd(a, i) for i in range(4)]
                                    col = fx.Int32(32 * nt) + g_ * fx.Int32(8)
                                else:
                                    a = acc_l[(nt // 4) * NPB + rb]
                                    av = [acc_rd(a, 4 * (nt % 4) + i) for i in range(4)]
                                    col = fx.Int32(
                                        2 * (32 * (nt // 4) + 8 * (nt % 4))
                                    ) + hq_ * fx.Int32(8)
                                v_ = (fx.Vector.from_elements(av, fx.Float32) * osc).to(
                                    fx.BFloat16
                                )
                                _llvm_d.StoreOp(
                                    _arith.unwrap(v_.bitcast(fx.Int32)),
                                    fx.to_llvm_ptr(
                                        p_rq + (qo_wb + row * fx.Int32(Q_OBP) + col)
                                    ),
                                    alignment=8,
                                )
                        for k in fx.range_constexpr(ROWS // 4):
                            qo_rr = fx.Int32(4 * k) + g_
                            qo_row = lds_rd(
                                qo_wb + qo_rr * fx.Int32(Q_OBP) + m_ * fx.Int32(16),
                                fx.Vector.make_type(4, fx.Int32),
                                16,
                            )
                            qo_cr = jb0 * fx.Int32(16) + qo_rr
                            qo_qp = qo_cr >> fx.Int32(H.bit_length() - 1)
                            qo_o = fx.Int64(q0 + qo_qp) * fx.Int64(out_s0) + fx.Int64(
                                qo_cr & fx.Int32(H - 1)
                            ) * fx.Int64(out_s1)
                            qo_a = fx.Int64(fx.ptrtoint(out_ptr)) + (
                                qo_o + fx.Int64(wv * fx.Int32(128) + m_ * fx.Int32(8))
                            ) * fx.Int64(2)
                            if fx.const_expr(MSQ > 1):
                                if qo_qp < qlen:
                                    st_agent_vec(qo_a, qo_row, 16)
                            else:
                                st_agent_vec(qo_a, qo_row, 16)

                    m_l = ninf
                    l_l = fx.Float32(0.0)
                    acc_l = [acc0z for _ in range(0 if PV16R else NACC)]
                    if is_qk:
                        m_l, l_l, acc_l = qk_role()
                    else:
                        m_l, l_l, acc_l = pv_role()
                    lane_x = fx.Int32(
                        _llvm_d.InlineAsmOp(
                            T.i32,
                            [],
                            "v_mbcnt_lo_u32_b32 $0, -1, 0\n"
                            " v_mbcnt_hi_u32_b32 $0, -1, $0",
                            "=v",
                            has_side_effects=True,
                        ).res
                    )
                    tid = wv * fx.Int32(WAVE) + lane_x
                    wave = tid >> fx.Int32(6)
                    lane = tid & fx.Int32(WAVE - 1)
                    m = lane & fx.Int32(15)
                    g = lane >> fx.Int32(4)
                    gsh = g * fx.Int32(8)
                    hq = lane >> fx.Int32(5)
                    r32 = lane & fx.Int32(31)
                    wv = fx.Int32(fx.rocdl.readfirstlane(T.i32, wave.ir_value()))
                    if fx.const_expr(MERGED):
                        tk_lane = ticket_lane(slot, lane)
                else:
                    idx0 = [idx_tile(t0 + fx.Int32(j)) for j in range(DD)]
                    if dma_wave:
                        for j in fx.range_constexpr(DD):
                            dma_idx(t0 + fx.Int32(DD - 1 + j))
                            dma_tile(t0 + fx.Int32(j), idx0[j])
                    if fx.const_expr(MERGED):
                        slot, myx, tk_lane = protocol_entry()
                    if dma_wave:
                        fx.rocdl.s_waitcnt(vmcnt=NVM * (DD - 1))
                    lds_barrier()
                    run_gen(dequant_steps(t0))
                    if dma_wave:
                        fx.rocdl.s_waitcnt(vmcnt=NVM * (DD - 2))
                    lds_barrier()
                    sc0l = []
                    run_gen(qk_steps(t0, sc0l))
                    acc0 = fx.Vector.filled(16, 0.0, fx.Float32)
                    acc0s = [
                        _llvm_d.InlineAsmOp(
                            v16f32, [acc0.ir_value()], "; acc $0", "=a,0"
                        ).res
                        for _ in range(NACC)
                    ]
                    NSCV = len(sc0l)
                    init = [ninf.ir_value(), fx.Float32(0.0).ir_value()]
                    init += [x.ir_value() for x in sc0l]
                    init += acc0s
                    nits = t1 - t0
                    nmain = (nits > fx.Int32(1)).select(nits - fx.Int32(1), zero_i)
                    results = init
                    AO = 2 + NSCV
                    for it, state in range(zero_i, nmain, fx.Int32(1), init=init):
                        m_run = fx.Float32(state[0])
                        l_run = fx.Float32(state[1])
                        sc_cur = [fx.Vector(state[2 + i]) for i in range(NSCV)]
                        accs = [fx.Vector(state[AO + j]) for j in range(NACC)]
                        tt = t0 + fx.Int32(it)
                        if dma_wave:
                            i3 = idx_lds(tt + fx.Int32(DD))
                            dma_idx(tt + fx.Int32(2 * DD - 1))
                            dma_tile(tt + fx.Int32(DD), i3)
                        fence()
                        qo = []
                        so = []
                        m_new = m_run
                        l_new = l_run
                        sc_nxt = sc_cur
                        if qk_wave:
                            run_mixed(
                                qk_steps(tt + fx.Int32(1), qo),
                                softmax_steps(tt, sc_cur, m_run, l_run, False, so),
                                every=1,
                            )
                            m_new = so[0]
                            l_new = so[1]
                            sc_nxt = qo
                        lds_barrier()
                        accs = rescale_d(accs)
                        fence()
                        po = []
                        dqs_ = dequant_steps(tt + fx.Int32(1))
                        run_mixed(
                            pv_steps(tt, accs, po),
                            dqs_,
                            every=max(1, (8 * NPB) // len(dqs_)),
                        )
                        if dma_wave:
                            fx.rocdl.s_waitcnt(vmcnt=NVM * (DD - 2))
                        lds_barrier()
                        results = yield (
                            [m_new.ir_value(), l_new.ir_value()]
                            + [x.ir_value() for x in sc_nxt]
                            + [x.ir_value() for x in po]
                        )
                    m_l = fx.Float32(results[0])
                    l_l = fx.Float32(results[1])
                    acc_l = [fx.Vector(results[AO + j]) for j in range(NACC)]
                    if nits > zero_i:
                        tl = t1 - fx.Int32(1)
                        m_l2 = m_l
                        l_l2 = l_l
                        if qk_wave:
                            so = []
                            run_gen(
                                softmax_steps(
                                    tl,
                                    [fx.Vector(results[2 + i]) for i in range(NSCV)],
                                    m_l,
                                    l_l,
                                    True,
                                    so,
                                )
                            )
                            m_l2 = so[0]
                            l_l2 = so[1]
                        lds_barrier()
                        acc_l2 = rescale_d(acc_l)
                        po = []
                        run_gen(pv_steps(tl, acc_l2, po, last=True))
                        m_l = m_l2
                        l_l = l_l2
                        acc_l = po
                if fx.const_expr(not Q16):
                    lds_barrier()
                if fx.const_expr(Q16):
                    pass
                elif qk_wave & sm_first:
                    _llvm_d.StoreOp(
                        _arith.unwrap(m_l),
                        fx.to_llvm_ptr(
                            p_rq + (fx.Int32(RD_ML_OFF) + sm_row * fx.Int32(4))
                        ),
                        alignment=4,
                    )
                    _llvm_d.StoreOp(
                        _arith.unwrap(l_l),
                        fx.to_llvm_ptr(
                            p_rq
                            + (fx.Int32(RD_ML_OFF + 4 * ROWS) + sm_row * fx.Int32(4))
                        ),
                        alignment=4,
                    )
                if fx.const_expr(not Q16):
                    lds_barrier()
                m_fin, l_fin, acc_fin = [], [], []
                for rb in fx.range_constexpr(0 if PV16R else NPB):
                    row = fx.Int32(32 * rb) + r32
                    m_fin.append(
                        fx.Float32(
                            lds_rd(
                                fx.Int32(RD_ML_OFF) + row * fx.Int32(4),
                                fx.Vector.make_type(1, fx.Float32),
                                4,
                            )[0]
                        )
                    )
                    l_fin.append(
                        fx.Float32(
                            lds_rd(
                                fx.Int32(RD_ML_OFF + 4 * ROWS) + row * fx.Int32(4),
                                fx.Vector.make_type(1, fx.Float32),
                                4,
                            )[0]
                        )
                    )
                    acc_fin.append(_AccRow(acc_l, NPB, rb, Q16))
            elif fx.const_expr(KL):
                wl = wave
                key_l = wl * fx.Int32(16) + m

                def vphys(kk):
                    lo = kk & fx.Int32(15)
                    return (
                        (kk - lo)
                        + ((lo & fx.Int32(3)) << fx.Int32(1))
                        + (lo & fx.Int32(8))
                        + ((lo >> fx.Int32(2)) & fx.Int32(1))
                    )

                vrow = p_v + vphys(key_l) * fx.Int32(VPITCH)
                rrow0 = vphys(g * fx.Int32(8) + m // fx.Int32(4))

                def idx_row1(t, key):
                    kidx = t * fx.Int32(KT_K) + key
                    ok = (t < t1) & (kidx < kvlen)
                    return ld_idx(ok.select(kidx, zero_i))

                up8 = m >= fx.Int32(8)
                colc = up8.select(g + fx.Int32(4), g) * fx.Int32(16)

                def idx_row(t):
                    if fx.const_expr(KC):
                        ka = wl * fx.Int32(16) + (m & fx.Int32(7))
                        return [idx_row1(t, ka), idx_row1(t, ka + fx.Int32(8))]
                    return [idx_row1(t, key_l)]

                def load_kv(rows):
                    rows = [row_ok(r) for r in rows]
                    if fx.const_expr(KC):
                        rowa, rowb = rows
                        ba = fx.Int64(rowa) * fx.Int64(ROWB) + fx.Int64(colc)
                        bb = fx.Int64(rowb) * fx.Int64(ROWB) + fx.Int64(colc)
                        lo, hi = [], []
                        for cc in fx.range_constexpr(4):
                            if fx.const_expr(cc < 3):
                                oa, ob = ba, bb
                            else:
                                c3 = up8.select(fx.Int32(64), g * fx.Int32(16)) - colc
                                oa, ob = ba + fx.Int64(c3), bb + fx.Int64(c3)
                            lo.append(load16(kv_ptr, oa + fx.Int64(cc * 128)))
                            hi.append(load16(kv_ptr, ob + fx.Int64(cc * 128)))
                        rown = up8.select(rowb, rowa)
                        rope = [
                            fx.ptr_load(
                                kvr_ptr
                                + fx.Int64(rown) * fx.Int64(ROPE)
                                + fx.Int64(g * fx.Int32(8) + fx.Int32(32 * j)),
                                result_type=v16u8,
                            ).bitcast(fx.Int32)
                            for j in range(2)
                        ]
                        return lo + hi + rope
                    kbase = fx.Int64(rows[0]) * fx.Int64(ROWB)
                    regs = []
                    for cc in fx.range_constexpr(4):
                        regs.append(
                            load16(
                                kv_ptr,
                                kbase + fx.Int64(cc * 128) + fx.Int64(g * fx.Int32(16)),
                            )
                        )
                    for cc in fx.range_constexpr(3):
                        regs.append(
                            load16(
                                kv_ptr,
                                kbase
                                + fx.Int64(cc * 128 + 64)
                                + fx.Int64(g * fx.Int32(16)),
                            )
                        )
                    regs.append(load16(kv_ptr, kbase + fx.Int64(NOPE)))
                    krbase = fx.Int64(rows[0]) * fx.Int64(ROPE)
                    for j in fx.range_constexpr(2):
                        regs.append(
                            fx.ptr_load(
                                kvr_ptr
                                + krbase
                                + fx.Int64(g * fx.Int32(8) + fx.Int32(32 * j)),
                                result_type=v16u8,
                            ).bitcast(fx.Int32)
                        )
                    return regs

                def kv_fix(raw):
                    if fx.const_expr(not KC):
                        return raw
                    a, b = raw[0:4], raw[4:8]

                    def mix(own, oth, bank_mask):
                        return fx.Vector.from_elements(
                            [
                                fx.Int32(
                                    fx.rocdl.update_dpp(
                                        T.i32,
                                        fx.Int32(own[i]).ir_value(),
                                        fx.Int32(oth[i]).ir_value(),
                                        0x128,
                                        0xF,
                                        bank_mask,
                                        False,
                                    )
                                )
                                for i in range(4)
                            ],
                            fx.Int32,
                        )

                    lo = [mix(a[cc], b[cc], 0xC) for cc in range(4)]
                    hi = [mix(b[cc], a[cc], 0x3) for cc in range(4)]
                    return lo + hi + [raw[8], raw[9]]

                NKV = 10
                NR = 2 if KC else 1
                rows_pf = [idx_row(t0 + fx.Int32(j)) for j in range(2)]

                qfr, qscl, qrf = load_q()
                fx.rocdl.sched_barrier(0)
                kv_pf = load_kv(rows_pf[0])

                if fx.const_expr(MERGED):
                    slot, myx, tk_lane = protocol_entry()

                def tile_step(kvc, tt, m_run, l_run, accs):
                    kvc = kv_fix(kvc)
                    tvalid = tt < t1
                    tbase = tt * fx.Int32(KT_K)
                    klo = kvc[0:4]
                    khi = kvc[4:7] + [zero4]
                    ksc = kvc[7]
                    krf = [kvc[8].bitcast(fx.BFloat16), kvc[9].bitcast(fx.BFloat16)]
                    kop = [join8(klo[cc], khi[cc]) for cc in range(4)]
                    kscl = [_scl(ksc[cc], cc) for cc in range(4)]
                    scs = []
                    for rb in fx.range_constexpr(RB):
                        sc = fx.Vector.filled(4, 0.0, fx.Float32)
                        for cc in fx.range_constexpr(4):
                            sc = fx.Vector(
                                fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                                    v4f32,
                                    [
                                        kop[cc],
                                        qfr[rb][cc],
                                        sc,
                                        0,
                                        0,
                                        0,
                                        kscl[cc],
                                        0,
                                        qscl[rb][cc],
                                    ],
                                )
                            )
                        for j in fx.range_constexpr(2):
                            sc = fx.Vector(
                                fx.rocdl.mfma_f32_16x16x32_bf16(
                                    v4f32, [krf[j], qrf[rb][j], sc]
                                )
                            )
                        scs.append(sc)
                    for cc in fx.range_constexpr(4):
                        for hh in fx.range_constexpr(2 if cc < 3 else 1):
                            blk = 2 * cc + hh
                            e = (
                                fx.Int32(ksc[blk // 2]) >> fx.Int32(16 * (blk % 2))
                            ) & (fx.Int32(0xFF))
                            scl = (e << fx.Int32(23)).bitcast(fx.Float32)
                            src = klo[cc] if hh == 0 else khi[cc]
                            parts = []
                            for wd in fx.range_constexpr(4):
                                for sel in fx.range_constexpr(2):
                                    pr = fx.Vector(
                                        fx.rocdl.cvt_scalef32_pk_bf16_fp8(
                                            v2bf16,
                                            fx.Int32(src[wd]).ir_value(),
                                            scl.ir_value(),
                                            bool(sel),
                                        )
                                    )
                                    parts += [pr[0], pr[1]]
                            vv = fx.Vector.from_elements(parts, fx.BFloat16)
                            fx.ptr_store(
                                vv,
                                vrow + fx.Int32(cc * 128 + hh * 64) + g * fx.Int32(16),
                            )
                    for j in fx.range_constexpr(2):
                        fx.ptr_store(
                            krf[j], vrow + fx.Int32(NOPE + 32 * j) + g * fx.Int32(8)
                        )
                    flat = [
                        fx.Float32(scs[rb][i]) * scale_log2e
                        for rb in range(RB)
                        for i in range(4)
                    ]
                    tend = tbase + fx.Int32(KT_K)
                    kfull = (tend <= kvlen) & tvalid
                    if fx.const_expr(BOUNDS):
                        for rb in fx.range_constexpr(RB):
                            a0_, a1_, b0_, b1_ = rb_bnd[rb]
                            kfull = kfull & (
                                ((a0_ <= tbase) & (a1_ >= tend))
                                | ((b0_ <= tbase) & (b1_ >= tend))
                            )
                    kpart = fx.Int64(
                        fx.rocdl.ballot(T.i64, (kfull == false_).ir_value())
                    ) != fx.Int64(0)
                    if kpart:
                        kval = []
                        for i in fx.range_constexpr(4):
                            kk = (
                                tbase
                                + wl * fx.Int32(16)
                                + g * fx.Int32(4)
                                + fx.Int32(i)
                            )
                            kval.append((kk < kvlen) & tvalid)
                        mflat = []
                        for rb in fx.range_constexpr(RB):
                            kv_rb = kval
                            if fx.const_expr(BOUNDS):
                                kv_rb = [
                                    kval[i]
                                    & in_bounds(
                                        tbase
                                        + wl * fx.Int32(16)
                                        + g * fx.Int32(4)
                                        + fx.Int32(i),
                                        rb_bnd[rb],
                                    )
                                    for i in range(4)
                                ]
                            mflat += [
                                kv_rb[i].select(flat[4 * rb + i], ninf)
                                for i in range(4)
                            ]
                        flat = mflat
                    ss = []
                    for rb in fx.range_constexpr(RB):
                        s = flat[4 * rb : 4 * rb + 4]
                        ss.append(s)
                        mx = s[0].maximumf(s[1]).maximumf(s[2]).maximumf(s[3])
                        mx = plswap_f(mx, 16, _fmax)
                        mx = plswap_f(mx, 32, _fmax)
                        if g == zero_i:
                            lds_st(
                                p_rmax, wl * fx.Int32(ROWS) + fx.Int32(16 * rb) + m, mx
                            )
                    fx.gpu.barrier()
                    m_new, alpha = [], []
                    lsum = []
                    for rb in fx.range_constexpr(RB):
                        tmax = lds_ldf(p_rmax, fx.Int32(16 * rb) + m)
                        for ww in fx.range_constexpr(1, NWAVES):
                            tmax = tmax.maximumf(
                                lds_ldf(p_rmax, fx.Int32(ww * ROWS + 16 * rb) + m)
                            )
                        mn = (tmax > m_run[rb] + fx.Float32(8.0)).select(
                            tmax, m_run[rb]
                        )
                        m_safe = (mn == ninf).select(fx.Float32(0.0), mn)
                        m_new.append(mn)
                        alpha.append(exp2(m_run[rb] - m_safe))
                        p = [exp2(ss[rb][i] - m_safe) for i in range(4)]
                        pb = fx.Vector.from_elements(p, fx.Float32).to(fx.BFloat16)
                        fx.ptr_store(
                            pb,
                            p_p
                            + (fx.Int32(16 * rb) + m) * fx.Int32(PPITCH)
                            + wl * fx.Int32(16)
                            + g * fx.Int32(4),
                        )
                        pr = pb.to(fx.Float32)
                        ps = (
                            fx.Float32(pr[0])
                            + fx.Float32(pr[1])
                            + fx.Float32(pr[2])
                            + fx.Float32(pr[3])
                        )
                        lsum.append(ps)
                    fx.gpu.barrier()
                    l_new, pfr = [], []
                    for rb in fx.range_constexpr(RB):
                        l_new.append(l_run[rb] * alpha[rb] + lsum[rb])
                        pfr.append(
                            [
                                fx.ptr_load(
                                    p_p
                                    + (fx.Int32(16 * rb) + m) * fx.Int32(PPITCH)
                                    + fx.Int32(32 * ks)
                                    + g * fx.Int32(8),
                                    result_type=v8bf16,
                                )
                                for ks in range(2)
                            ]
                        )
                    ne = alpha[0] != fx.Float32(1.0)
                    for rb in fx.range_constexpr(1, RB):
                        ne = ne | (alpha[rb] != fx.Float32(1.0))
                    need = fx.Int64(fx.rocdl.ballot(T.i64, ne.ir_value())) != fx.Int64(
                        0
                    )
                    accs = [
                        [fx.Vector(accs[rb][nt]) for nt in range(8)] for rb in range(RB)
                    ]
                    if need:
                        accs = [
                            [accs[rb][nt] * alpha[rb] for nt in range(8)]
                            for rb in range(RB)
                        ]
                    new_accs = [[None] * 8 for _ in range(RB)]
                    for nt in fx.range_constexpr(8):
                        a = [fx.Vector(accs[rb][nt]) for rb in range(RB)]
                        dvb = wl * fx.Int32(128) + fx.Int32(16 * nt)
                        for ks in fx.range_constexpr(2):
                            halves = []
                            for hh in fx.range_constexpr(2):
                                rk = rrow0 + fx.Int32(32 * ks + hh)
                                ptr = (
                                    p_v
                                    + rk * fx.Int32(VPITCH)
                                    + dvb
                                    + (m % fx.Int32(4)) * fx.Int32(4)
                                )
                                halves.append(
                                    fx.Vector(
                                        fx.rocdl.ds_read_tr16_b64(
                                            v4bf16, fx.to_llvm_ptr(ptr)
                                        ).result
                                    )
                                )
                            va = fx.Vector.from_elements(
                                [halves[0][i] for i in range(4)]
                                + [halves[1][i] for i in range(4)],
                                fx.BFloat16,
                            )
                            for rb in fx.range_constexpr(RB):
                                a[rb] = fx.Vector(
                                    fx.rocdl.mfma_f32_16x16x32_bf16(
                                        v4f32, [va, pfr[rb][ks], a[rb]]
                                    )
                                )
                        for rb in fx.range_constexpr(RB):
                            new_accs[rb][nt] = a[rb]
                    fx.gpu.barrier()
                    return m_new, l_new, new_accs

                NST = 2 * RB + 8 * RB
                init = [ninf.ir_value() for _ in range(RB)]
                init += [fx.Float32(0.0).ir_value() for _ in range(RB)]
                init += [
                    fx.Vector.filled(4, 0.0, fx.Float32).ir_value()
                    for _ in range(8 * RB)
                ]
                init += [x.ir_value() for x in kv_pf] + [
                    x.ir_value() for x in rows_pf[1]
                ]
                nits = t1 - t0
                results = init
                for ip, state in range(
                    zero_i, nits >> fx.Int32(1), fx.Int32(1), init=init
                ):
                    m_run = [fx.Float32(state[rb]) for rb in range(RB)]
                    l_run = [fx.Float32(state[RB + rb]) for rb in range(RB)]
                    accs = [
                        [fx.Vector(state[2 * RB + 8 * rb + nt]) for nt in range(8)]
                        for rb in range(RB)
                    ]
                    kvq = [[fx.Vector(state[NST + i]) for i in range(NKV)]]
                    rows_n = [fx.Int32(state[NST + NKV + i]) for i in range(NR)]
                    tt = t0 + (fx.Int32(ip) << fx.Int32(1))
                    for half in fx.range_constexpr(2):
                        rows_nn = idx_row(tt + fx.Int32(2 + half))
                        kvq.append(load_kv(rows_n))
                        m_run, l_run, accs = tile_step(
                            kvq[half], tt + fx.Int32(half), m_run, l_run, accs
                        )
                        rows_n = rows_nn
                    results = yield (
                        [x.ir_value() for x in m_run]
                        + [x.ir_value() for x in l_run]
                        + [x.ir_value() for rb in range(RB) for x in accs[rb]]
                        + [x.ir_value() for x in kvq[2]]
                        + [x.ir_value() for x in rows_n]
                    )
                m_fin = [fx.Float32(results[rb]) for rb in range(RB)]
                l_fin = [fx.Float32(results[RB + rb]) for rb in range(RB)]
                acc_fin = [
                    [fx.Vector(results[2 * RB + 8 * rb + nt]) for nt in range(8)]
                    for rb in range(RB)
                ]
                kv_last = [fx.Vector(results[NST + i]) for i in range(NKV)]
                if (nits & fx.Int32(1)) != zero_i:
                    m_fin, l_fin, acc_fin = tile_step(
                        kv_last, t1 - fx.Int32(1), m_fin, l_fin, acc_fin
                    )
                for rb in fx.range_constexpr(RB):
                    ls = plswap_f(l_fin[rb], 16, _fadd)
                    ls = plswap_f(ls, 32, _fadd)
                    if g == zero_i:
                        lds_st(p_rsum, wl * fx.Int32(ROWS) + fx.Int32(16 * rb) + m, ls)
                fx.gpu.barrier()
                l_tot = []
                for rb in fx.range_constexpr(RB):
                    tsum = lds_ldf(p_rsum, fx.Int32(16 * rb) + m)
                    for ww in fx.range_constexpr(1, NWAVES):
                        tsum = tsum + lds_ldf(p_rsum, fx.Int32(ww * ROWS + 16 * rb) + m)
                    l_tot.append(tsum)
                l_fin = l_tot

            empty = kvlen == zero_i

            NACCW = 64 if RQ else (16 if RD else 8)
            NRBE = 1 if RQ else (ROWS // 32 if RD else RB)
            em = r32 if (RQ or RD) else m
            eg = hq if (RQ or RD) else g
            RPB = 16 if (RQ or RD) else 8
            ROPE_RUNS = 8 if (RQ or RD) else 4

            def rope_off(k):
                if fx.const_expr(RQ or RD):
                    return fx.Int32(32 * (k // 4) + 8 * (k % 4)) + hq * fx.Int32(4)
                return fx.Int32(16 * k) + g * fx.Int32(4)

            def dv_of(nt):
                if fx.const_expr(RQ):
                    return fx.Int32(32 * (nt // 4) + 8 * (nt % 4)) + hq * fx.Int32(4)
                if fx.const_expr(RD):
                    return (
                        wv * fx.Int32(128)
                        + fx.Int32(32 * (nt // 4) + 8 * (nt % 4))
                        + hq * fx.Int32(4)
                    )
                return wl * fx.Int32(128) + fx.Int32(16 * nt) + g * fx.Int32(4)

            def direct_epi():
                if fx.const_expr(not Q16):
                    snk = [
                        fx.Float32(fx.ptr_load(sink_ptr + fx.Int64(rb_h0[rb] + em)))
                        * fx.Float32(LOG2E)
                        for rb in range(NRBE)
                    ]
                if fx.const_expr(EPI_Q):
                    frv = [
                        [
                            fx.ptr_load(
                                fr_ptr
                                + rb_pos[rb] * fx.Int64(ROPE)
                                + fx.Int64(rope_off(j)),
                                result_type=v4f32,
                            )
                            for j in range(ROPE_RUNS)
                        ]
                        for rb in range(NRBE if QROW_EPI else 1)
                    ]
                    if fx.const_expr(RQ):
                        is_rw = true_
                    elif fx.const_expr(RD):
                        is_rw = wv == fx.Int32(3)
                    else:
                        is_rw = wl == fx.Int32(3)
                for rb in fx.range_constexpr(NRBE):
                    if fx.const_expr(Q16):
                        osc = fx.Float32(
                            lds_rd(
                                fx.Int32(RD_ML_OFF + 8 * ROWS)
                                + (fx.Int32(32 * rb) + r32) * fx.Int32(4),
                                fx.Vector.make_type(1, fx.Float32),
                                4,
                            )[0]
                        )
                        inv_l, empty_r = osc, empty
                    else:
                        lf = l_fin[rb].maximumf(fx.Float32(1e-30))
                        lse2 = m_fin[rb] + fly_math.log2(lf)
                        gm = snk[rb].maximumf(lse2)
                        w = exp2(lse2 - gm)
                        den = exp2(snk[rb] - gm) + w
                        inv_l = fx.Float32(1.0) / lf
                        empty_r = (l_fin[rb] == fx.Float32(0.0)) if BOUNDS else empty
                        f = empty_r.select(qnan, rcp(den))
                    h = rb_h0[rb] + em
                    if fx.const_expr(Q16 and not EPI_Q):
                        Q_OBP = 272
                        qo_wb = wv * fx.Int32(ROWS * Q_OBP)
                        if wv < fx.Int32(4):
                            for nt in fx.range_constexpr(NACCW):
                                if fx.const_expr(nt % 2 == 0 or MERGED):
                                    fx.rocdl.sched_barrier(0)
                                v_ = (acc_fin[rb][nt] * osc).to(fx.BFloat16)
                                _llvm_d.StoreOp(
                                    _arith.unwrap(v_.bitcast(fx.Int32)),
                                    fx.to_llvm_ptr(
                                        p_rq
                                        + (
                                            qo_wb
                                            + (fx.Int32(32 * rb) + r32)
                                            * fx.Int32(Q_OBP)
                                            + fx.Int32(
                                                2 * (32 * (nt // 4) + 8 * (nt % 4))
                                            )
                                            + hq * fx.Int32(8)
                                        )
                                    ),
                                    alignment=8,
                                )
                            # all LDS reads ahead of the (asm) stores: one wait
                            qo_rows = [
                                lds_rd(
                                    qo_wb
                                    + (fx.Int32(32 * rb + 4 * k) + g) * fx.Int32(Q_OBP)
                                    + m * fx.Int32(16),
                                    fx.Vector.make_type(4, fx.Int32),
                                    16,
                                )
                                for k in range(8)
                            ]
                            for k in fx.range_constexpr(8):
                                qo_rr = fx.Int32(32 * rb + 4 * k) + g
                                qo_row = qo_rows[k]
                                qo_cr = jb0 * fx.Int32(16) + qo_rr
                                qo_qp = qo_cr >> fx.Int32(H.bit_length() - 1)
                                qo_o = fx.Int64(q0 + qo_qp) * fx.Int64(
                                    out_s0
                                ) + fx.Int64(qo_cr & fx.Int32(H - 1)) * fx.Int64(out_s1)
                                qo_a = fx.Int64(fx.ptrtoint(out_ptr)) + (
                                    qo_o
                                    + fx.Int64(wv * fx.Int32(128) + m * fx.Int32(8))
                                ) * fx.Int64(2)
                                if fx.const_expr(MSQ > 1):
                                    if qo_qp < qlen:
                                        st_agent_vec(qo_a, qo_row, 16)
                                else:
                                    st_agent_vec(qo_a, qo_row, 16)
                    elif fx.const_expr(not EPI_Q):
                        obase = fx.Int64(rb_qrow[rb]) * fx.Int64(out_s0) + fx.Int64(
                            h
                        ) * fx.Int64(out_s1)
                        if rb_ok[rb]:
                            for nt in fx.range_constexpr(NACCW):
                                fx.ptr_store(
                                    ((acc_fin[rb][nt] * inv_l) * w * f).to(fx.BFloat16),
                                    out_ptr + obase + fx.Int64(dv_of(nt)),
                                )
                    else:
                        qrow_x = rb_qrow[rb] if QROW_EPI else seq
                        fr_rb = frv[rb if QROW_EPI else 0]
                        # Q16: the PV waves (0-3) hold the rows; QROW_EPI: rows past qlen
                        st_ok = rb_ok[rb] if QROW_EPI else true_
                        if fx.const_expr(Q16):
                            st_ok = st_ok & (wv < fx.Int32(4))
                        xrow = fx.Int64(qrow_x) * fx.Int64(H) + fx.Int64(h)
                        for blk in fx.range_constexpr(NACCW // RPB):
                            ys = []
                            amb = zero_i
                            for n8 in fx.range_constexpr(RPB):
                                nt = RPB * blk + n8
                                x = acc_fin[rb][nt] * inv_l
                                if fx.const_expr(not Q16):
                                    x = (x * w) * f
                                x = x.to(fx.BFloat16).to(fx.Float32)
                                if fx.const_expr(nt >= NACCW - ROPE_RUNS):
                                    fr = fr_rb[nt - (NACCW - ROPE_RUNS)]
                                    yy = []
                                    for pp in fx.range_constexpr(2):
                                        c = fx.Float32(fr[2 * pp])
                                        sn = fx.Float32(fr[2 * pp + 1])
                                        xa = fx.Float32(x[2 * pp])
                                        xb = fx.Float32(x[2 * pp + 1])
                                        yy.append(is_rw.select(xa * c + xb * sn, xa))
                                        yy.append(is_rw.select(xb * c - xa * sn, xb))
                                    x = fx.Vector.from_elements(yy, fx.Float32)
                                y = x.to(fx.BFloat16).to(fx.Float32)
                                ys.append(y)
                                for i in fx.range_constexpr(4):
                                    amb = smax_i32(amb, fabs_bits(fx.Float32(y[i])))
                            if fx.const_expr(not RQ):
                                amb = plswap(amb, 16, smax_i32)
                            amb = plswap(amb, 32, smax_i32)
                            ex = mxfp8_exp(amb)
                            ex = empty_r.select(fx.Int32(255), ex)
                            nexp = fx.Int32(127) - ex
                            xq_base = xrow * fx.Int64(DV)
                            for n8 in fx.range_constexpr(RPB):
                                nt = RPB * blk + n8
                                y = ys[n8]
                                qv = [
                                    fp8_scaled(fx.Float32(y[i]), nexp) for i in range(4)
                                ]
                                pk = fx.rocdl.cvt_pk_fp8_f32(
                                    T.i32, qv[0], qv[1], fx.Int32(0), False
                                )
                                pk = fx.rocdl.cvt_pk_fp8_f32(
                                    T.i32, qv[2], qv[3], pk, True
                                )
                                pk = empty_r.select(fx.Int32(0x7F7F7F7F), fx.Int32(pk))
                                pkb = fx.Vector.from_elements([pk], fx.Int32).bitcast(
                                    fx.Uint8
                                )
                                if fx.const_expr(QROW_EPI or Q16):
                                    if st_ok:
                                        fx.ptr_store(
                                            pkb, out_ptr + xq_base + fx.Int64(dv_of(nt))
                                        )
                                else:
                                    fx.ptr_store(
                                        pkb, out_ptr + xq_base + fx.Int64(dv_of(nt))
                                    )
                            xs_a = (
                                xs_ptr
                                + xrow * fx.Int64(4)
                                + (
                                    fx.Int64(blk)
                                    if RQ
                                    else (fx.Int64(wv) if RD else fx.Int64(wl))
                                )
                            )
                            xs_ok = eg == zero_i
                            if fx.const_expr(QROW_EPI or Q16):
                                xs_ok = xs_ok & st_ok
                            if xs_ok:
                                fx.ptr_store(fx.Uint8(ex), xs_a)

            def pst(addr, val, nbytes):
                if fx.const_expr(WTP):
                    st_agent_vec(addr, val, nbytes)
                elif fx.const_expr(SK):
                    wthru = sk_wt
                    if wthru:
                        st_agent_vec(addr, val, nbytes)
                    else:
                        st_raw(addr, val, nbytes)
                else:
                    st_raw(addr, val, nbytes)

            def pld(addr, n):
                if fx.const_expr(WTP):
                    return ld_agent_f32(addr, n)
                return gload(addr, n)

            def pld16(addr):
                if fx.const_expr(WTP):
                    v = (ld64_agent if EPT == 4 else ld_agent_i32)(addr, False)
                    return fx.Vector.from_elements([v], type(v)).bitcast(fx.Float16)
                return fx.generic_load(
                    cu._to_ptr_global(addr, fx.Float16, 2 * EPT), count=EPT
                )

            def merged_epi():
                tkv = zero_i
                if fx.const_expr(not OWN):  # noqa: SIM102 (compile-time guard)
                    if wave == zero_i:
                        tkv = ld_agent_i32(tk_lane, volatile=False)
                sub = tid // fx.Int32(TPB)
                lb = tid % fx.Int32(TPB)

                bpcv = (
                    udiv(fx.Int32(NQB) + ns - fx.Int32(1), ns) if DYN else fx.Int32(bpc)
                )

                def slice_addrs(j, pin):
                    qa = j * bpcv
                    qe = (qa + bpcv < fx.Int32(NQB)).select(qa + bpcv, fx.Int32(NQB))
                    rs = []
                    for r in fx.range_constexpr(NRND):
                        qb = qa + fx.Int32(BPR * r) + sub
                        act = qb < qe
                        qbs = act.select(qb, zero_i)
                        row = qbs >> fx.Int32(2)
                        cblk = qbs & fx.Int32(3)
                        jb = jb0 + (row >> fx.Int32(4))
                        qpos = _div_const(jb, HB)
                        head = _rem_const(jb, HB) * fx.Int32(16) + (row & fx.Int32(15))
                        d0 = cblk * fx.Int32(128) + lb * fx.Int32(EPT)
                        la = lse_base + (
                            fx.Int64(group) * fx.Int64(ROWS) + fx.Int64(row)
                        ) * fx.Int64(SP * 4)
                        pbase = part_base + (
                            (fx.Int64(group) * fx.Int64(S * ROWS) + fx.Int64(row))
                            * fx.Int64(DV)
                            + fx.Int64(d0)
                        ) * fx.Int64(PEB)
                        pa = [
                            pbase + fx.Int64(sidx * ROWS * DV * PEB)
                            for sidx in range(S)
                        ]
                        if fx.const_expr(pin):
                            la = pin_v64(la)
                            pa = [pin_v64(x) for x in pa]
                        sa = fx.Int64(fx.ptrtoint(sink_ptr)) + fx.Int64(
                            head
                        ) * fx.Int64(4)
                        frr = None
                        if fx.const_expr(EPI_Q):
                            ro = (d0 >= fx.Int32(NOPE)).select(
                                d0 - fx.Int32(NOPE), zero_i
                            )
                            if fx.const_expr(QROW_EPI):
                                qpc = (qpos < qlen).select(qpos, qlen - fx.Int32(1))
                                pos_r = fx.Int64(
                                    fx.ptr_load(pos_ptr + fx.Int64(q0 + qpc))
                                )
                            else:
                                pos_r = pos
                            frr = fx.ptr_load(
                                fr_ptr + pos_r * fx.Int64(ROPE) + fx.Int64(ro),
                                result_type=fx.Vector.make_type(EPT, fx.Float32),
                            )
                        psa = None
                        if fx.const_expr(PF16):
                            psa = psc_base + (
                                (fx.Int64(group) * fx.Int64(ROWS) + fx.Int64(row))
                                * fx.Int64(4)
                                + fx.Int64(cblk)
                            ) * fx.Int64(SP * 4)
                        rs.append(
                            _SliceRow(act, head, qpos, d0, la, pa, sa, frr, cblk, psa)
                        )
                    return rs

                def merge_slice(rs):
                    for c0 in MCHUNKS:
                        if fx.const_expr(DYN and c0 > 0):
                            if fx.Int32(c0 * BPR) < bpcv:
                                merge_rounds(rs, list(range(c0, min(NRND, c0 + MCH))))
                        else:
                            merge_rounds(rs, list(range(c0, min(NRND, c0 + MCH))))

                def merge_finish(r, rs, acc, den, lv):
                    x = rs[r]
                    act, head, qpos, d0 = x.act, x.head, x.qpos, x.d0
                    frr, cblk = x.frr, x.cblk
                    if fx.const_expr(BOUNDS):
                        gl = lv[0]
                        for sidx in fx.range_constexpr(1, S):
                            gl = gl.maximumf(lv[sidx])
                        empty_r = gl == ninf
                    else:
                        empty_r = empty
                    fsc = empty_r.select(qnan, rcp(den))
                    if fx.const_expr(not EPI_Q):
                        if act & (qpos < qlen):
                            fx.ptr_store(
                                (acc * fsc).to(fx.BFloat16),
                                out_ptr
                                + fx.Int64(q0 + qpos) * fx.Int64(out_s0)
                                + fx.Int64(head) * fx.Int64(out_s1)
                                + fx.Int64(d0),
                            )
                    else:
                        x = (acc * fsc).to(fx.BFloat16).to(fx.Float32)
                        is_rope = d0 >= fx.Int32(NOPE)
                        ys = []
                        for pp in fx.range_constexpr(EPT // 2):
                            c = fx.Float32(frr[2 * pp])
                            sn = fx.Float32(frr[2 * pp + 1])
                            xa = fx.Float32(x[2 * pp])
                            xb = fx.Float32(x[2 * pp + 1])
                            ys.append(is_rope.select(xa * c + xb * sn, xa))
                            ys.append(is_rope.select(xb * c - xa * sn, xb))
                        y = (
                            fx.Vector.from_elements(ys, fx.Float32)
                            .to(fx.BFloat16)
                            .to(fx.Float32)
                        )
                        amb = fabs_bits(fx.Float32(y[0]))
                        for i in fx.range_constexpr(1, EPT):
                            amb = smax_i32(amb, fabs_bits(fx.Float32(y[i])))
                        amb = butterfly_i32(amb, TPB, smax_i32)
                        ex = empty_r.select(fx.Int32(255), mxfp8_exp(amb))
                        nexp = fx.Int32(127) - ex
                        qv = [fp8_scaled(fx.Float32(y[i]), nexp) for i in range(EPT)]
                        pk = fx.rocdl.cvt_pk_fp8_f32(
                            T.i32, qv[0], qv[1], fx.Int32(0), False
                        )
                        if fx.const_expr(EPT == 4):
                            pk = fx.rocdl.cvt_pk_fp8_f32(T.i32, qv[2], qv[3], pk, True)
                        pk = empty_r.select(fx.Int32(0x7F7F7F7F), fx.Int32(pk))
                        xqrow = (q0 + qpos) if QROW_EPI else seq
                        act_x = (act & (qpos < qlen)) if QROW_EPI else act
                        if act_x:
                            xq_off = (
                                fx.Int64(xqrow) * fx.Int64(H) + fx.Int64(head)
                            ) * fx.Int64(DV) + fx.Int64(d0)
                            pkb = fx.Vector.from_elements([pk], fx.Int32).bitcast(
                                fx.Uint8
                            )
                            if fx.const_expr(EPT == 4):
                                fx.ptr_store(pkb, out_ptr + xq_off)
                            else:
                                fx.ptr_store(
                                    fx.Vector.from_elements([pkb[0], pkb[1]], fx.Uint8),
                                    out_ptr + xq_off,
                                )
                            if lb == zero_i:
                                fx.ptr_store(
                                    fx.Uint8(ex),
                                    xs_ptr
                                    + (fx.Int64(xqrow) * fx.Int64(H) + fx.Int64(head))
                                    * fx.Int64(4)
                                    + fx.Int64(cblk),
                                )

                def merge_rounds_sk(rs, rounds):

                    def pass_loads(r, sb, n):
                        la, pa0 = rs[r].la, rs[r].pa[0]
                        lsv = []
                        for k4 in fx.range_constexpr(n // 4):
                            v4 = fx.Vector(
                                pld(
                                    la + fx.Int64(sb + fx.Int32(4 * k4)) * fx.Int64(4),
                                    4,
                                )
                            )
                            lsv += [fx.Float32(v4[i]) for i in range(4)]
                        lse, pv = [], []
                        for i in fx.range_constexpr(n):
                            si = sb + fx.Int32(i)
                            ok = (tps * si) < ntiles
                            lse.append(ok.select(lsv[i], ninf))
                            pa = pa0 + fx.Int64(ok.select(si, zero_i)) * fx.Int64(
                                ROWS * DV * PEB
                            )
                            pv.append(fx.Vector(pld(pa, EPT)))
                        return lse, pv

                    def pass_add(st, lse, pv):
                        m, den, acc = st
                        mn = m
                        for x in lse:
                            mn = mn.maximumf(x)
                        al = exp2(m - mn)
                        den = den * al
                        acc = acc * al
                        for x, p in zip(lse, pv):
                            w = exp2(x - mn)
                            den = den + w
                            acc = acc + p * w
                        return mn, den, acc

                    first = {}
                    for r in rounds:
                        snk = fx.Float32(gload(rs[r].sa, 1))
                        first[r] = (snk, pass_loads(r, zero_i, SKP0))
                    fx.rocdl.sched_barrier(0)
                    zv = fx.Vector.filled(EPT, 0.0, fx.Float32)
                    fin = {}
                    for r in rounds:
                        snk, (lse, pv) = first[r]
                        snk = snk * fx.Float32(LOG2E)
                        fin[r] = pass_add((snk, fx.Float32(1.0), zv), lse, pv)
                    if fx.const_expr(S > SKP0):
                        init = []
                        for r in rounds:
                            init += [x.ir_value() for x in fin[r]]
                        npass = (ns + fx.Int32(3 - SKP0)) >> fx.Int32(2)
                        results = init
                        for pp, state in range(zero_i, npass, fx.Int32(1), init=init):
                            sb = fx.Int32(pp) * fx.Int32(4) + fx.Int32(SKP0)
                            ld = {r: pass_loads(r, sb, 4) for r in rounds}
                            fx.rocdl.sched_barrier(0)
                            nxt = []
                            for ir, r in enumerate(rounds):
                                st = (
                                    fx.Float32(state[3 * ir]),
                                    fx.Float32(state[3 * ir + 1]),
                                    fx.Vector(state[3 * ir + 2]),
                                )
                                nxt += [x.ir_value() for x in pass_add(st, *ld[r])]
                            results = yield nxt
                        for ir, r in enumerate(rounds):
                            fin[r] = (
                                fx.Float32(results[3 * ir]),
                                fx.Float32(results[3 * ir + 1]),
                                fx.Vector(results[3 * ir + 2]),
                            )
                    for r in rounds:
                        _, den, acc = fin[r]
                        merge_finish(r, rs, acc, den, None)

                def merge_rounds_std(rs, rounds):
                    loaded = {}
                    for r in rounds:
                        la, pa, sa, psa = rs[r].la, rs[r].pa, rs[r].sa, rs[r].psa
                        lsc = []
                        for k0, n in lse_chunks:
                            if fx.const_expr(n == 1):
                                lsc.append(fx.Float32(pld(la + fx.Int64(4 * k0), 1)))
                            else:
                                vv_ = fx.Vector(pld(la + fx.Int64(4 * k0), n))
                                lsc += [fx.Float32(vv_[jx]) for jx in range(n)]
                        pscl = None
                        if fx.const_expr(PF16):
                            pscl = []
                            for k0, n in lse_chunks:
                                if fx.const_expr(n == 1):
                                    pscl.append(
                                        fx.Float32(pld(psa + fx.Int64(4 * k0), 1))
                                    )
                                else:
                                    vv_ = fx.Vector(pld(psa + fx.Int64(4 * k0), n))
                                    pscl += [fx.Float32(vv_[jx]) for jx in range(n)]
                            pvs = [
                                fx.Vector(pld16(pa[sidx])).to(fx.Float32)
                                for sidx in range(S)
                            ]
                        elif fx.const_expr(ADAPT):
                            pvs = [
                                fx.Vector(
                                    pld(
                                        ((tps * fx.Int32(sidx)) < ntiles).select(
                                            pa[sidx], pa[0]
                                        ),
                                        EPT,
                                    )
                                )
                                for sidx in range(S)
                            ]
                        else:
                            pvs = [fx.Vector(pld(pa[sidx], EPT)) for sidx in range(S)]
                        snk = fx.Float32(gload(sa, 1))
                        loaded[r] = (lsc, pvs, snk, pscl)
                    fx.rocdl.sched_barrier(0)
                    for r in rounds:
                        lsc, pvs, snk, pscl = loaded[r]
                        snk = snk * fx.Float32(LOG2E)
                        gmax = snk
                        lv, vsl = [], []
                        for sidx in fx.range_constexpr(S):
                            vs = (tps * fx.Int32(sidx)) < ntiles
                            vsl.append(vs)
                            lv.append(vs.select(lsc[sidx], ninf))
                            gmax = gmax.maximumf(lv[sidx])
                        den = exp2(snk - gmax)
                        zv = fx.Vector.filled(EPT, 0.0, fx.Float32)
                        acc = zv
                        for sidx in fx.range_constexpr(S):
                            w = exp2(lv[sidx] - gmax)
                            den = den + w
                            wp = w * pscl[sidx] if PF16 else w
                            acc = acc + vsl[sidx].select(fx.Vector(pvs[sidx]) * wp, zv)
                        merge_finish(r, rs, acc, den, lv)

                merge_rounds = merge_rounds_sk if SK else merge_rounds_std

                if fx.const_expr(RQM):
                    pass
                elif t0 < t1:
                    for rb in fx.range_constexpr(NRBE):
                        if fx.const_expr(RQ):
                            rloc = wv * fx.Int32(32) + r32
                        elif fx.const_expr(RD):
                            rloc = fx.Int32(32 * rb) + r32
                        else:
                            rloc = fx.Int32(16 * rb) + m
                        has = l_fin[rb] > fx.Float32(0.0)
                        inv_l = has.select(
                            fx.Float32(1.0) / l_fin[rb].maximumf(fx.Float32(1e-30)),
                            fx.Float32(0.0),
                        )
                        lse2 = has.select(
                            m_fin[rb]
                            + fly_math.log2(l_fin[rb].maximumf(fx.Float32(1e-30))),
                            ninf,
                        )
                        rec = (
                            fx.Int64(group) * fx.Int64(S) + fx.Int64(split)
                        ) * fx.Int64(ROWS) + fx.Int64(rloc)
                        if fx.const_expr(RQ):
                            lds_barrier()
                            PQ_P = 528
                            pq_wb = wv * fx.Int32(32 * PQ_P)
                            pq_rec0 = (
                                fx.Int64(group) * fx.Int64(S) + fx.Int64(split)
                            ) * fx.Int64(ROWS) + fx.Int64(wv * fx.Int32(32))
                            for qd in fx.range_constexpr(4):
                                for jj in fx.range_constexpr(4):
                                    for qq in fx.range_constexpr(4):
                                        nt = 4 * (4 * qd + jj) + qq
                                        _llvm_d.StoreOp(
                                            _arith.unwrap(acc_fin[rb][nt] * inv_l),
                                            fx.to_llvm_ptr(
                                                p_rq
                                                + (
                                                    pq_wb
                                                    + r32 * fx.Int32(PQ_P)
                                                    + fx.Int32(4 * (32 * jj + 8 * qq))
                                                    + hq * fx.Int32(16)
                                                )
                                            ),
                                            alignment=16,
                                        )
                                for pq_i2 in fx.range_constexpr(16):
                                    pq_rr = fx.Int32(2 * pq_i2) + hq
                                    pq_row = lds_rd(
                                        pq_wb
                                        + pq_rr * fx.Int32(PQ_P)
                                        + r32 * fx.Int32(16),
                                        fx.Vector.make_type(4, fx.Int32),
                                        16,
                                    )
                                    pst(
                                        part_base
                                        + (
                                            (pq_rec0 + fx.Int64(pq_rr)) * fx.Int64(DV)
                                            + fx.Int64(
                                                fx.Int32(128 * qd) + r32 * fx.Int32(4)
                                            )
                                        )
                                        * fx.Int64(4),
                                        pq_row,
                                        16,
                                    )
                                if fx.const_expr(qd < 3):
                                    _llvm_d.InlineAsmOp(
                                        None,
                                        [],
                                        "s_waitcnt lgkmcnt(0)",
                                        "~{memory}",
                                        has_side_effects=True,
                                    )
                        elif fx.const_expr(Q16):
                            Q_PQP = 528
                            qp_wb = wv * fx.Int32(ROWS * Q_PQP)
                            qp_rec0 = (
                                fx.Int64(group) * fx.Int64(S) + fx.Int64(split)
                            ) * fx.Int64(ROWS)
                            Q_PQP = 272
                            qp_wb = wv * fx.Int32(ROWS * Q_PQP)
                            if wv < fx.Int32(4):
                                pv_ = [acc_fin[rb][nt] * inv_l for nt in range(NACCW)]

                                amx = fx.Float32(0.0)
                                for nt in fx.range_constexpr(NACCW):
                                    for i in fx.range_constexpr(4):
                                        amx = maxnum_f32(amx, fabs_f32(pv_[nt][i]))
                                amx = plswap_f(amx, 32, maxnum_f32)
                                ex_ = amx.bitcast(fx.Int32) >> fx.Int32(23)
                                kx = fx.Int32(141) - ex_
                                kx = smax_i32(
                                    smin_i32(kx, fx.Int32(126)), fx.Int32(-126)
                                )
                                sdn = ((kx + fx.Int32(127)) << fx.Int32(23)).bitcast(
                                    fx.Float32
                                )
                                sup = ((fx.Int32(127) - kx) << fx.Int32(23)).bitcast(
                                    fx.Float32
                                )
                                if hq == zero_i:
                                    pst(
                                        psc_base
                                        + (
                                            (
                                                fx.Int64(group) * fx.Int64(ROWS)
                                                + fx.Int64(rloc)
                                            )
                                            * fx.Int64(4 * SP)
                                            + fx.Int64(
                                                wv * fx.Int32(SP) + fx.Int32(split)
                                            )
                                        )
                                        * fx.Int64(4),
                                        sup,
                                        4,
                                    )
                                for nt in fx.range_constexpr(NACCW):
                                    fx.rocdl.sched_barrier(0)
                                    v_ = (pv_[nt] * sdn).to(fx.Float16)
                                    _llvm_d.StoreOp(
                                        _arith.unwrap(v_.bitcast(fx.Int32)),
                                        fx.to_llvm_ptr(
                                            p_rq
                                            + (
                                                qp_wb
                                                + rloc * fx.Int32(Q_PQP)
                                                + fx.Int32(
                                                    2 * (32 * (nt // 4) + 8 * (nt % 4))
                                                )
                                                + hq * fx.Int32(8)
                                            )
                                        ),
                                        alignment=8,
                                    )
                                if fx.const_expr(rb == NRBE - 1):
                                    for qp_k in fx.range_constexpr(ROWS // 4):
                                        qp_rr = fx.Int32(4 * qp_k) + g
                                        qp_row = lds_rd(
                                            qp_wb
                                            + qp_rr * fx.Int32(Q_PQP)
                                            + m * fx.Int32(16),
                                            fx.Vector.make_type(4, fx.Int32),
                                            16,
                                        )
                                        pst(
                                            part_base
                                            + (
                                                (qp_rec0 + fx.Int64(qp_rr))
                                                * fx.Int64(DV)
                                                + fx.Int64(
                                                    wv * fx.Int32(128) + m * fx.Int32(8)
                                                )
                                            )
                                            * fx.Int64(2),
                                            qp_row,
                                            16,
                                        )
                        else:
                            for nt in fx.range_constexpr(NACCW):
                                pst(
                                    part_base
                                    + (rec * fx.Int64(DV) + fx.Int64(dv_of(nt)))
                                    * fx.Int64(4),
                                    acc_fin[rb][nt] * inv_l,
                                    16,
                                )
                        if fx.const_expr(RQ):
                            lse_w = hq == zero_i
                        elif fx.const_expr(RD):
                            lse_w = (hq == zero_i) & (wv == zero_i)
                        else:
                            lse_w = (g == zero_i) & (wl == zero_i)
                        if lse_w:
                            pst(
                                lse_base
                                + (
                                    (fx.Int64(group) * fx.Int64(ROWS) + fx.Int64(rloc))
                                    * fx.Int64(SP)
                                    + fx.Int64(split)
                                )
                                * fx.Int64(4),
                                lse2,
                                4,
                            )

                if fx.const_expr(not RQM):
                    own = slice_addrs(split, True)
                if fx.const_expr(OWN):
                    if wave == zero_i:
                        tst0 = spin_tickets_state(
                            tk_lane, tkv_own, START_POLLS, myx + fx.Int32(1)
                        )
                        if lane == zero_i:
                            lds_st(p_flag, fx.Int32(2), tst0)
                    lds_barrier()
                    tst_c = fx.Int32(fx.ptr_load(p_flag + fx.Int32(2)))
                    leave = (tst_c & fx.Int32(1)) == zero_i
                    rq_store(own_ip, own_lse, own_acc, leave)
                tst_mask = sk_last.select(fx.Int32(-1), zero_i)
                wb_on = sk_wt == false_
                # arrive: partials stored (vmcnt 0) before the tickets are judged. Stay only if every
                # sibling started; release by L2 write-back unless all siblings share this XCD (or
                # the partials were written through), then one returning atomic add of the bits
                fx.rocdl.s_waitcnt(vmcnt=0)
                fx.gpu.barrier()
                if wave == zero_i:
                    if fx.const_expr(OWN):
                        tst = fx.Int32(fx.ptr_load(p_flag + fx.Int32(2)))
                    else:
                        tst = spin_tickets_state(
                            tk_lane, tkv, START_POLLS, myx + fx.Int32(1)
                        )
                        if fx.const_expr(SK):
                            tst = tst & tst_mask
                    stay = (tst & fx.Int32(1)) != zero_i
                    col = (tst & fx.Int32(2)) != zero_i
                    av = fx.Int64(0)
                    wb = false_ if WTP else (col == false_) & wb_on
                    if lane == zero_i:
                        if wb:
                            cu.fence_agent_release()
                        bit = fx.Int64(1) << fx.Int64(split)
                        inc = bit + stay.select(bit << fx.Int64(32), fx.Int64(0))
                        av = rmw_add64_agent(slot + fx.Int64(A_OFF), inc) + inc
                    av = fx.Int64(
                        fx.rocdl.readfirstlane(T.i64, fx.Int64(av).ir_value())
                    )
                    if fx.const_expr(DYN):
                        fullv = (fx.Int64(1) << fx.Int64(ns)) - fx.Int64(1)
                        last = fx.Int32(av & fullv) == fx.Int32(fullv)
                    else:
                        fullv = fx.Int64(FULL)
                        last = fx.Int32(av & fx.Int64(FULL)) == fx.Int32(FULL32)
                    # the last arriver resets the slot for the next launch (graph replay)
                    if last:
                        if lane < fx.Int32(S):
                            st_raw(tk_lane, zero_i, 4)
                        if lane == zero_i:
                            st_raw(slot + fx.Int64(A_OFF), fx.Int64(0), 8)
                    if lane == zero_i:
                        if stay & (last == false_):
                            spin_all_arrived(
                                slot + fx.Int64(A_OFF),
                                fullv,
                                fx.Int64(1) << fx.Int64(split),
                            )
                        flv = (
                            stay.select(fx.Int32(1), zero_i)
                            | last.select(fx.Int32(2), zero_i)
                            | (stay & col).select(fx.Int32(4), zero_i)
                        )
                        lds_st(p_flag, zero_i, flv)
                        lds_st(p_flag, fx.Int32(1), fx.Int32(av >> fx.Int64(32)))
                fx.gpu.barrier()
                fl = fx.Int32(fx.ptr_load(p_flag))
                smask = fx.Int32(fx.ptr_load(p_flag + fx.Int32(1)))
                if (fl & fx.Int32(3)) != zero_i:
                    # acquire: L1 invalidate when colocated, else agent-scope
                    inv_all = false_ if WTP else ((fl & fx.Int32(4)) == zero_i)
                    if inv_all:
                        cu.fence_agent_acquire()
                    else:
                        inv_l1()
                    if fx.const_expr(OWN):
                        merge_own(split, True)
                    else:
                        merge_slice(own)
                    if (fl & fx.Int32(2)) != zero_i:
                        for jsl in range(
                            zero_i, ns if DYN else fx.Int32(S), fx.Int32(1)
                        ):
                            j = fx.Int32(jsl)
                            if (j != split) & (((smask >> j) & fx.Int32(1)) == zero_i):
                                if fx.const_expr(OWN):
                                    merge_own(j, False)
                                else:
                                    merge_slice(slice_addrs(j, False))

            def epilogue():
                if fx.const_expr(not MERGED):
                    if fx.const_expr(not PV16R):
                        direct_epi()
                elif fx.const_expr(DYN or SHORT):
                    if ns == fx.Int32(1):
                        if split == zero_i:
                            direct_epi()
                    else:
                        if live:
                            merged_epi()
                else:
                    merged_epi()

            epilogue()

    return types.SimpleNamespace(
        emit=emit,
        map_bid=map_bid,
        Smem=Smem,
        NT=NT,
        NRG=NRG,
        S=S,
        name=name,
        agpr=RQ,
    )


# the traced bodies live in helpers the compile cache does not see: key on their source
# (needs the .py files, which aiter installs)
def _source_key():
    h = hashlib.sha256()
    d = os.path.dirname(os.path.abspath(__file__))
    for f in ("mla_v4_decode_gfx950.py", "mla_v4_decode_common.py"):
        with open(os.path.join(d, f), "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


_SRC_KEY = _source_key()


@functools.cache
def build_mla_v4_decode(cfg: V4Cfg):
    main = _emitter(cfg)
    IDLE = cfg.IDLE
    one = _emitter(replace(cfg, S=1, PF16=False, IDLE=())) if IDLE else main
    emit_one = one.emit
    subs = []
    wsp = cfg.ws_seq()
    for keys, sc in cfg.SUB:
        subs.append((keys, sc, _emitter(sc, wsp)))
        g1, b1 = sc.ws_seq()
        wsp = (wsp[0] + g1, wsp[1] + b1)
    emit_main = main.emit
    map_bid = main.map_bid
    NT = main.NT
    NRG, S = main.NRG, main.S
    NSUB = len(subs)
    CPS = cfg.cps
    SUB_K = [k for k, _, _ in subs]
    SUB_S = [sc.S for _, sc, _ in subs]
    SUB_C = [sc.NRG * sc.S for _, sc, _ in subs]
    SUB_W = [em.NT // WAVE for _, _, em in subs]
    SUB_EMIT = [em.emit for _, _, em in subs]
    SPREAD = cfg.SPREAD
    SKG = cfg.SK
    SKD = cfg.SKD
    MINT = cfg.MINT
    SKI = -(-SKG // WAVE)
    SKM = 2  # key-balanced map only if the longest split > 1.25 runs + SKM tiles
    SKT = cfg.SKT
    SKO = 256  # keys charged per stream for a segment's fixed cost
    SKMEAN = cfg.SKMEAN
    SKTL = cfg.KT
    SKSH = SKTL.bit_length() - 1
    QSEQ = cfg.EPI == "invrope_mxfp8" and cfg.MSQ == 1
    name = (
        main.name
        + "".join(f"_sub{k}r{sc.RB}s{sc.S}" for k, sc in cfg.SUB)
        + ("_spr" if SPREAD else "")
        + ("_idle{}_{}_{}_{}".format(*IDLE) if IDLE else "")
    )
    if NSUB == 0 and not IDLE:
        Smem = main.Smem
    else:
        fields = {"lng": main.Smem}
        fields.update({f"sb{i}": em.Smem for i, (_, _, em) in enumerate(subs)})
        if IDLE:
            fields["one"] = one.Smem
        Smem = fx.union(type("Smem", (), {"__annotations__": fields}))

    CFG_KEY = repr(cfg) + _SRC_KEY

    @flyc.kernel(name=name, known_block_size=[NT, 1, 1])
    def kernel(
        nseq: fx.Int32,
        num_rows: fx.Int32,
        indptr_ptr: fx.Pointer,
        idx_ptr: fx.Pointer,
        qo_ptr: fx.Pointer,
        q_ptr: fx.Pointer,
        kv_ptr: fx.Pointer,
        kvr_ptr: fx.Pointer,
        qr_ptr: fx.Pointer,
        pos_ptr: fx.Pointer,
        ws_ptr: fx.Pointer,
        sink_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        xs_ptr: fx.Pointer,
        fr_ptr: fx.Pointer,
        out_s0: fx.Int32,
        out_s1: fx.Int32,
        hdr_ptr: fx.Pointer,
        bnd_ptr: fx.Pointer,
    ):
        # flydsl's compile cache keys on the source and closure values only: reading
        # CFG_KEY gives every variant (and source revision) its own binary
        _variant = CFG_KEY
        args = (
            nseq,
            num_rows,
            indptr_ptr,
            idx_ptr,
            q_ptr,
            kv_ptr,
            kvr_ptr,
            qr_ptr,
            qo_ptr,
            pos_ptr,
            ws_ptr,
            sink_ptr,
            out_ptr,
            xs_ptr,
            fr_ptr,
            out_s0,
            out_s1,
            hdr_ptr,
            bnd_ptr,
        )
        if fx.const_expr(SKG):
            lds = fx.SharedAllocator().allocate(Smem).peek()
            zero = fx.Int32(0)
            one = fx.Int32(1)
            lane = fx.Int32(fx.thread_idx.x) & fx.Int32(WAVE - 1)
            false_ = one == zero
            true_ = one != zero
            c = fx.Int32(fx.block_idx.x)
            if fx.const_expr(cfg.TAIL):
                seq_s, split_s = _tail_map(c, nseq, 1, SKD)
            else:
                bj = c >> fx.Int32(3)
                gps = (nseq + fx.Int32(7)) >> fx.Int32(3)
                split_s = udiv(bj, gps)
                seq_s = ((bj - split_s * gps) << fx.Int32(3)) | (c & fx.Int32(7))
            ok_s = seq_s < nseq
            seq_sc = ok_s.select(seq_s, zero)
            bounds = []
            for k in fx.range_constexpr(SKI):
                bk = lane + fx.Int32(WAVE * k)
                ok = bk < nseq
                bc = ok.select(bk, zero)
                a0 = fx.Int32(fx.ptr_load(indptr_ptr + fx.Int64(bc)))
                a1 = fx.Int32(fx.ptr_load(indptr_ptr + fx.Int64(bc) + fx.Int64(1)))
                bounds.append((bk, ok, a0, a1))
            # (one round trip with the batch bounds)
            ptrs = (indptr_ptr,) if QSEQ else (indptr_ptr, qo_ptr)
            md = [_sload(p, seq_sc + fx.Int32(i)) for p in ptrs for i in (0, 1)]
            md = _pin_s(md + [_sload(indptr_ptr, zero), _sload(indptr_ptr, nseq)])
            kv0_s, kvlen_s = md[0], md[1] - md[0]
            q0_s, qlen_s = (seq_sc, one) if QSEQ else (md[2], md[3] - md[2])
            base = md[-2]
            nkeys = md[-1] - base + nseq * fx.Int32(SKO)
            share = smax_i32(_div_const(nkeys + fx.Int32(SKG - 1), SKG), fx.Int32(SKTL))
            share_t = (share + fx.Int32(SKTL - 1)) >> fx.Int32(SKSH)
            mint = fx.Int32(MINT)
            if fx.const_expr(SKMEAN):
                kts = smax_i32(nseq, one) * fx.Int32(SKTL)
                mint = smax_i32(
                    mint, udiv(nkeys - nseq * fx.Int32(SKO) + kts - one, kts)
                )
            b_lo = zero
            b_end = zero
            run = share
            nrun = one

            def owners(a, n, run, nrun):
                f = smin_i32(udiv(a, run), nrun - one)
                nt = (n + fx.Int32(SKTL - 1)) >> fx.Int32(SKSH)
                la = udiv(a + (nt - one) * fx.Int32(SKTL), run)
                return f, (n > zero).select(la, f), nt

            seqs = []
            nmax = zero
            for bk, ok, a0, a1 in bounds:
                n = ok.select(a1 - a0, zero)
                seqs.append((ok, a0 - base + (bk + one) * fx.Int32(SKO), n))
                nmax = smax_i32(nmax, n)
            nmax = butterfly_i32(nmax, WAVE, smax_i32)
            ntmax = (nmax + fx.Int32(SKTL - 1)) >> fx.Int32(SKSH)
            tpmax = _div_const(ntmax + fx.Int32(SKD - 1), SKD)
            if fx.const_expr(SKMEAN):
                tpmax = smin_i32(ntmax, smax_i32(tpmax, mint))
            # short runs: a split of > SKT tiles, >= 2 runs and >= 4 shares (skew)
            runt = smax_i32(share_t, _div_const(ntmax + fx.Int32(S - 2), S - 1))
            skew = tpmax >= smax_i32(runt * fx.Int32(2), share_t * fx.Int32(4))
            use_sk = (share_t >= fx.Int32(SKT)) & (
                tpmax * fx.Int32(4) > share_t * fx.Int32(5) + fx.Int32(4 * SKM)
            ) | (tpmax > fx.Int32(SKT)) & skew
            if fx.const_expr(SKMEAN):
                use_sk = use_sk & (
                    (ntmax * fx.Int32(4) > mint * fx.Int32(5))
                    | (share_t >= fx.Int32(2 * SKT))
                )
            if use_sk:
                run = smax_i32(share, _div_const(nmax + fx.Int32(S - 2), S - 1))
                nrun = smax_i32(udiv(nkeys + run - one, run), one)
                cnt = zero
                for ok, a, n in seqs:
                    f, la, _ = owners(a, n, run, nrun)
                    cnt = cnt + (ok & (la < c)).select(one, zero)
                    cnt = cnt + (ok & (f <= c)).select(fx.Int32(1 << 16), zero)
                cnt = butterfly_i32(cnt, WAVE, _fadd)
                b_lo = cnt & fx.Int32(0xFFFF)
                b_end = cnt >> fx.Int32(16)
            if use_sk:
                for bi in range(b_lo, b_end, one):
                    b = fx.Int32(bi)
                    kv0 = fx.Int32(fx.ptr_load(indptr_ptr + fx.Int64(b)))
                    kvlen = (
                        fx.Int32(fx.ptr_load(indptr_ptr + fx.Int64(b) + fx.Int64(1)))
                        - kv0
                    )
                    q0 = fx.Int32(fx.ptr_load(qo_ptr + fx.Int64(b)))
                    qlen = (
                        fx.Int32(fx.ptr_load(qo_ptr + fx.Int64(b) + fx.Int64(1))) - q0
                    )
                    a = kv0 - base + (b + one) * fx.Int32(SKO)
                    f, la, nt = owners(a, kvlen, run, nrun)
                    lo = c * run - a
                    hi = lo + run
                    t0 = (lo > zero).select(
                        (lo + fx.Int32(SKTL - 1)) >> fx.Int32(SKSH), zero
                    )
                    t1 = (hi > zero).select(
                        (hi + fx.Int32(SKTL - 1)) >> fx.Int32(SKSH), zero
                    )
                    ns = la - f + one
                    emit_main(
                        *args,
                        lds,
                        b,
                        zero,
                        c - f,
                        (kv0, kvlen),
                        (
                            t0,
                            smin_i32(t1, nt),
                            ns,
                            b == b_end - one,
                            true_,
                            (q0, qlen),
                            one,
                            ns,
                        ),
                    )
                    fx.gpu.barrier()
            else:
                nt = (kvlen_s + fx.Int32(SKTL - 1)) >> fx.Int32(SKSH)
                tps = smax_i32(_div_const(nt + fx.Int32(SKD - 1), SKD), mint)
                if fx.const_expr(cfg.SHORT):
                    tps = (nt <= fx.Int32(cfg.SHORT)).select(nt, tps)
                t0 = split_s * tps
                ns = (nt > tps).select(fx.Int32(SKD), one)
                wt_s = false_
                if fx.const_expr(cfg.TAIL):
                    wt_s = seq_s >= (nseq & fx.Int32(-8))
                if ok_s & (split_s < ns):
                    emit_main(
                        *args,
                        lds,
                        seq_s,
                        zero,
                        split_s,
                        (kv0_s, kvlen_s),
                        (
                            t0,
                            smin_i32(t0 + tps, nt),
                            ns,
                            true_,
                            wt_s,
                            (q0_s, qlen_s),
                            tps,
                            nt,
                        ),
                    )
        elif fx.const_expr(IDLE):
            st = fx.SharedAllocator().allocate(Smem)
            lds_2, lds_1 = st.lng.peek(), st.one.peek()
            zero = fx.Int32(0)
            c = fx.Int32(fx.block_idx.x)
            xcd = c & fx.Int32(7)
            gpx = ((nseq + fx.Int32(7)) >> fx.Int32(3)) * fx.Int32(NRG)
            ext = (c >> fx.Int32(3)) >= gpx
            k = (c >> fx.Int32(3)) - ext.select(gpx, zero)
            rg = _rem_const(k, NRG)
            seq = (_div_const(k, NRG) << fx.Int32(3)) | xcd
            tl_lane = fx.Int32(fx.thread_idx.x) & fx.Int32(WAVE - 1)
            if ext:
                # splits 1.. of the stream of rank k / NRG / (S - 1) on this XCD (none:
                # past nseq)
                j, ok, rk = xcd_ranks(indptr_ptr, nseq, xcd, tl_lane, IDLE[0] // NRG)
                hit = ok & (rk == _div_const(_div_const(k, NRG), S - 1))
                j1 = butterfly_i32(hit.select(j + fx.Int32(1), zero), WAVE, _fadd)
                j1 = fx.Int32(fx.rocdl.readfirstlane(T.i32, j1.ir_value()))
                seq = (j1 == zero).select(nseq, j1 - fx.Int32(1))
            seq = fx.Int32(fx.rocdl.readfirstlane(T.i32, seq.ir_value()))
            seq_c = (seq < nseq).select(seq, zero)
            ptrs = (indptr_ptr,) if QSEQ else (qo_ptr, indptr_ptr)
            md = [_sload(p, seq_c + fx.Int32(i)) for p in ptrs for i in (0, 1)]
            md = _pin_s(md)
            kvall = _sload(indptr_ptr, nseq) - _sload(indptr_ptr, zero)
            kvm = (md[-2], md[-1] - md[-2]) + (() if QSEQ else (md[0], md[1] - md[0]))
            nt = (kvm[1] + fx.Int32(SKTL - 1)) >> fx.Int32(SKSH)
            # split: longer than the mean and f tiles, splits 1.. on CTAs of this XCD
            two = (seq < nseq) & (nt > fx.Int32(IDLE[2]))
            two = two & (
                fx.Int64(nt - fx.Int32(1)) * fx.Int64(nseq * fx.Int32(SKTL))
                >= fx.Int64(kvall)
            )
            if two:
                _, _, rk = xcd_ranks(indptr_ptr, nseq, xcd, tl_lane, IDLE[0] // NRG)
                rk = fx.Int32(
                    fx.rocdl.readlane(
                        T.i32, rk.ir_value(), (seq >> fx.Int32(3)).ir_value()
                    )
                )
                pos = gpx + (rk + fx.Int32(1)) * fx.Int32(NRG * (S - 1))
                two = (pos <= fx.Int32(IDLE[0])) | (
                    (pos <= fx.Int32(IDLE[0] + IDLE[1])) & (nt > fx.Int32(IDLE[3]))
                )
            if two:
                split = fx.Int32(1) + _rem_const(_div_const(k, NRG), S - 1)
                emit_main(*args, lds_2, seq, rg, ext.select(split, zero), kvm)
            else:
                if (c >> fx.Int32(3)) < gpx:
                    emit_one(*args, lds_1, seq, rg, zero, kvm)
        elif fx.const_expr(NSUB == 0):
            seq, rg, split = map_bid(fx.Int32(fx.block_idx.x), nseq)
            emit_main(*args, fx.SharedAllocator().allocate(Smem).peek(), seq, rg, split)
        else:
            st = fx.SharedAllocator().allocate(Smem)
            lds_m = st.lng.peek()
            lds_s = [getattr(st, f"sb{i}").peek() for i in range(NSUB)]
            bid = fx.Int32(fx.block_idx.x)
            if fx.const_expr(SPREAD):
                seq = _div_const(bid, CPS)
                c = bid - seq * fx.Int32(CPS)
            elif fx.const_expr(cfg.TAIL):
                seq, c = _tail_map(bid, nseq, 1, CPS)
            else:
                bj = bid >> fx.Int32(3)
                sg = _div_const(bj, CPS)
                c = bj - sg * fx.Int32(CPS)
                seq = (sg << fx.Int32(3)) | (bid & fx.Int32(7))
            seq_c = (seq < nseq).select(seq, fx.Int32(0))
            ptrs = (indptr_ptr,) if QSEQ else (indptr_ptr, qo_ptr)
            md = [_sload(ptr, seq_c + fx.Int32(j)) for ptr in ptrs for j in range(2)]
            kvlen = md[1] - md[0]
            kvm = (md[0], kvlen)
            if fx.const_expr(not QSEQ):
                kvm = kvm + (md[2], md[3] - md[2])
            wv = fx.Int32(
                fx.rocdl.readfirstlane(
                    T.i32, (fx.Int32(fx.thread_idx.x) >> fx.Int32(6)).ir_value()
                )
            )

            def run(emit_b, lds_b, s_b, cps_b, w_b):
                if c < fx.Int32(cps_b):
                    rgi = _div_const(c, s_b)
                    spi = c - rgi * fx.Int32(s_b)
                    if fx.const_expr(w_b * WAVE == NT):
                        emit_b(*args, lds_b, seq, rgi, spi, kvm)
                    else:
                        if wv < fx.Int32(w_b):
                            emit_b(*args, lds_b, seq, rgi, spi, kvm)

            def body(i):
                if fx.const_expr(i == NSUB):
                    run(emit_main, lds_m, S, NRG * S, NT // WAVE)
                else:
                    if kvlen <= fx.Int32(SUB_K[i]):
                        run(SUB_EMIT[i], lds_s[i], SUB_S[i], SUB_C[i], SUB_W[i])
                    else:
                        body(i + 1)

            body(0)

    def _kattrs():
        if not main.agpr:
            return {}
        va = {}
        va["passthrough"] = _ir.ArrayAttr.get(
            [
                _ir.ArrayAttr.get(
                    [
                        _ir.StringAttr.get("amdgpu-agpr-alloc"),
                        _ir.StringAttr.get("256"),
                    ]
                )
            ]
        )
        return {"value_attrs": va}

    @flyc.jit
    def launch(
        q_ptr: fx.Pointer,
        qr_ptr: fx.Pointer,
        kv_ptr: fx.Pointer,
        kvr_ptr: fx.Pointer,
        qo_ptr: fx.Pointer,
        indptr_ptr: fx.Pointer,
        idx_ptr: fx.Pointer,
        sink_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        xs_ptr: fx.Pointer,
        pos_ptr: fx.Pointer,
        fr_ptr: fx.Pointer,
        ws_ptr: fx.Pointer,
        bnd_ptr: fx.Pointer,
        hdr_ptr: fx.Pointer,
        nseq: fx.Int32,
        num_rows: fx.Int32,
        out_s0: fx.Int32,
        out_s1: fx.Int32,
        stream: fx.Stream,
    ):
        if fx.const_expr(SKG):
            nstd = ((nseq + fx.Int32(7)) // fx.Int32(8)) * fx.Int32(8 * cfg.SKD)
            if fx.const_expr(cfg.TAIL):
                nstd = nseq * fx.Int32(cfg.SKD)
            nblk = (nstd > fx.Int32(SKG)).select(nstd, fx.Int32(SKG))
        elif fx.const_expr(IDLE):
            nstd = ((nseq + fx.Int32(7)) // fx.Int32(8)) * fx.Int32(8 * NRG)
            ntl = fx.Int32(8 * (IDLE[0] + IDLE[1]))
            nblk = (nstd > ntl).select(nstd, ntl)
        elif fx.const_expr(SPREAD or cfg.TAIL):
            nblk = nseq * fx.Int32(CPS)
        else:
            nblk = ((nseq + fx.Int32(7)) // fx.Int32(8)) * fx.Int32(8 * CPS)
        kernel(
            nseq,
            num_rows,
            indptr_ptr,
            idx_ptr,
            qo_ptr,
            q_ptr,
            kv_ptr,
            kvr_ptr,
            qr_ptr,
            pos_ptr,
            ws_ptr,
            sink_ptr,
            out_ptr,
            xs_ptr,
            fr_ptr,
            out_s0,
            out_s1,
            hdr_ptr,
            bnd_ptr,
            **_kattrs(),
        ).launch(
            grid=(nblk, 1, 1),
            block=(NT, 1, 1),
            stream=stream,
        )

    llvm_opts = {
        "amdgpu-kernarg-preload": True,
        "amdgpu-kernarg-preload-count": 32,
    }
    launch.compile_hints = {"llvm_options": llvm_opts}
    return launch
