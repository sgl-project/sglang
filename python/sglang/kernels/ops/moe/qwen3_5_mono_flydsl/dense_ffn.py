# SPDX-License-Identifier: Apache-2.0
"""``dense_ffn``: the FFN half of a dense Qwen3.8-27B decoder layer as one
persistent launch, ``BLOCKS`` x ``THREADS``, S <= 8 decode rows (TP1 below):

    norm     (every CTA, redundantly; the first gate_up weights already in
             flight): h + residual (fp32) -> ``res_out`` (bf16, spread over the
             CTAs); post_attention_layernorm (Gemma, 1 + w) -> bf16 x -> MXFP4
             qdq (1 x 32 blocks, Quark "even" scales, as aiter's Triton
             ``dynamic_mxfp4_quant``) -> LDS
    gate_up  (1088 groups of 16 intermediate columns: CTA b groups 4 b .. 4 b + 3
             whole (wave w: gate / up by w % 2, K quarter w / 2), and K quarter
             b % 4 of group 1024 + b / 4, first, its fp32 partial put to the
             quarter-0 CTA): MXFP4 weights (fp4 -> bf16 by their e8m0 scales)
             against the x rows -> bf16 g, u (the stock GEMM's output) ->
             bf16(bf16(silu(g)) u) -> H; one ready flag a CTA
    down     (CTA b: K quarter b / 64 of the intermediate, 5 of the 320 hidden
             row groups, a wave a contiguous run of K steps): that quarter of H
             through MXFP4 qdq -> LDS; the weights against it -> fp32 partials;
             quarters 1..3 put theirs, quarter 0's CTA adds them in quarter order
             -> bf16 ``out``

At TP2 (``FfnBuild.tp``, ``normed``) a rank runs its half of the intermediate
(``FGeo``) from the stock post_attention_layernorm's output, qdq'd per CTA, and
writes down's partial before the all-reduce: 544 groups, 2 whole a CTA and K
quarters of 32 split groups on CTAs 0 .. 127; down's quarters are 17 steps.

Weights are the checkpoint's as SGLang keeps them for ``gemm_afp4wfp4``: fp4x2
[N, K / 2] row-major (low nibble the even column), e8m0 scales [N, K / 32].
A load past a weight's end (a dead slot) reads zeros at no memory traffic.
"""

from dataclasses import dataclass
from typing import NamedTuple

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from sglang.kernels.ops.moe.k3_mono_flydsl.common.mx import FP4, mfma_scaled
from sglang.kernels.ops.moe.k3_mono_flydsl.common.ops import (
    CM_DEV,
    CM_NT,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    memrealtime,
    row_sum,
    rsrc,
    traced,
    xred,
)
from sglang.kernels.ops.moe.k3_mono_flydsl.common.plan import (
    BLOCKS,
    THREADS,
    WAVES,
    KernelAbi,
    key_tuple,
    pair_layout,
)
from sglang.kernels.ops.moe.k3_mono_flydsl.common.sync import Mailbox, publish, sreg
from sglang.kernels.ops.moe.qwen3_5_mono_flydsl.layer import (
    pow2,
    sigmoid,
    step_tag,
)
from sglang.kernels.ops.moe.qwen3_5_mono_flydsl.sources import digest

HIDDEN = 5120
INTER = 17408
ROWS = 16
STEP = 128  # K an fp4 tile step covers (a lane: 32 fp4 of one row)
MAX_TOKENS = 8

# K partition: gate_up and down sum K in quarters, wave runs and fp32 adds, not
# in the stock GEMM's split-K (gate_up 2 splits; down 2 at TP1, 1 at TP2).
# The bits still match: an fp4 x fp4 product scaled by e8m0 codes is exact in
# fp32, and the stock output is bf16(fp32(exact sum)) on real weights, so any
# partition whose fp32 partials are exact gives the same bf16. That fails only
# when a row's exact sum spans more than 24 bits (products whose block scales
# differ by more than about 2^14), where the partitions round differently;
# hence the load-time self-check and StockAccumulationTest. The out_proj fold
# in dense_gdn (core rows, wider ranges) keeps stock's splits instead.

# gate_up
GU_STEPS = HIDDEN // STEP  # 40
GU_Q = GU_STEPS // 4  # 10: a wave's K quarter
QT_UNITS = 3  # a wave's steps of a quarter task (10 = 3 + 3 + 2 + 2)
W13_DW = HIDDEN // 8  # dwords a w13 row
W13S_DW = HIDDEN // 32 // 4  # scale dwords a w13 row
XQROW = HIDDEN // 8 + 4  # i32 (8 fp4 each) an LDS x row (padded)
XSROW = HIDDEN // 32  # e8m0 codes (an i32 each) a row

# down
DQ = 4  # K quarters of the intermediate (and of gate_up's split groups)
DCTA = BLOCKS // DQ  # 64 CTAs a quarter
D_RG = HIDDEN // ROWS // DCTA  # 5 row groups a CTA


class FGeo(NamedTuple):
    """A TP rank's share of the FFN: half the intermediate at TP2 (gate_up's N,
    down's K). TP1: 1088 groups (4 whole a CTA, 64 split), down quarters of 34
    steps; TP2: 544 (2 whole, 32 split), quarters of 17 steps."""

    tp: int

    @property
    def inter(self) -> int:
        return INTER // self.tp

    @property
    def ng(self) -> int:  # groups of 16 intermediate columns
        return self.inter // ROWS

    @property
    def g_full(self) -> int:  # whole groups a CTA
        return self.ng // BLOCKS

    @property
    def g_split(self) -> int:  # groups split in K quarters over 4 CTAs each
        return self.ng % BLOCKS

    @property
    def qt_ctas(self) -> int:  # CTAs with a quarter task
        return self.g_split * DQ

    @property
    def dk(self) -> int:
        return self.inter // DQ

    @property
    def d_steps(self) -> int:
        return self.dk // STEP

    @property
    def d_units(self) -> int:  # a wave's steps of a row group, at most
        return -(-self.d_steps // WAVES)

    @property
    def w2_dw(self) -> int:
        return self.inter // 8

    @property
    def w2s_dw(self) -> int:
        return self.inter // 32 // 4

    @property
    def h_blocks(self) -> int:  # qdq blocks a token's quarter
        return self.dk // 32

    @property
    def hqrow(self) -> int:
        return self.dk // 8 + 4

    @property
    def hsrow(self) -> int:
        return self.dk // 32


TP_SIZES = (1, 2)

# o_proj (FfnBuild.oproj): CTA b < O_CTAS owns hidden columns 32 b .. 32 b + 31
CORE = 6144
O_CTAS = HIDDEN // (2 * ROWS)  # 160
OQROW = CORE // 8 + 4
OSROW = CORE // 32
X_PAIRS = 5  # a (token, owner) x hand-off: 4 fp4 words, the code

# norm
CHUNKS = HIDDEN // 8  # 640 chunks of 8 columns a row
CH_WAVES = CHUNKS // 64  # 10 wave-sized runs of chunks a row
# The stock Triton Gemma norm (minimax_m3_rmsnorm, 8192-wide block): 16 warps,
# a thread 8 columns; DPP row_shr 8..1, row_bcast 15 / 31 in a warp.
NORM_WARPS = 16
WAVE_ORDER = (8, 4, 2, 1, 16, 32)
MAX_ROUNDS = (MAX_TOKENS * CHUNKS + THREADS - 1) // THREADS  # 10

assert CHUNKS % 64 == 0 and D_RG * DCTA * ROWS == HIDDEN
for _g in map(FGeo, TP_SIZES):
    assert _g.qt_ctas <= BLOCKS and _g.d_steps * STEP == _g.dk
assert FGeo(1).qt_ctas == BLOCKS and FGeo(1).d_units == 5

_STREAM = fx.Stream(None)
STAMPS = 8  # debug stamps a CTA


def stamp(c, k):
    """Thread 0: the 100 MHz clock -> stamp k of this CTA (debug builds)."""
    if c["stamps"] is not None:
        _stamp(c, k)


@traced
def _stamp(c, k):
    t = memrealtime()
    if c["tid"] == 0:
        bo.buffer_store(
            fx.Vector.from_elements(
                [fx.Int32(t & 0xFFFFFFFF), fx.Int32(t >> 32)], fx.Int32
            ),
            rsrc(c["stamps"]),
            (c["bid"] * STAMPS + k) * 2,
        )


@dataclass(frozen=True)
class FfnBuild:
    tokens: int
    eps: float = 1e-6
    # debug: phases < stop only (1 norm, 2 gate_up, 3 down)
    stop: int = 99
    # debug: per-CTA s_memrealtime stamps at the phase ends -> scratch "stamps"
    stamps: bool = False
    # weights in aiter's shuffle_weight(layout=(16, 16)) order: a 16-row
    # group's 16 B K chunks as [chunk][row % 16][16 B] (``shuffle_w``)
    shuffled: bool = False
    # o_proj / out_proj folded in: ``hidden`` is its input (the attention's
    # core, bf16 [S, CORE]); MXFP4 qdq + the GEMV by the owner CTAs, then the
    # norm distributed (sum-of-squares and fp4 x hand-offs). Needs ``shuffled``.
    oproj: bool = False
    # the rank's share at TP ``tp`` (``FGeo``)
    tp: int = 1
    # ``hidden`` is post_attention_layernorm's bf16 output: no residual, no
    # res_out; ``out`` gets down's output (at TP2 the rank's partial)
    normed: bool = False


ABI = KernelAbi(
    (
        "hidden",
        "residual",
        "w_o",
        "w_os",
        "ln_w",
        "w13",
        "w13s",
        "w2",
        "w2s",
        "out",
        "res_out",
        "scratch",
        "epoch",
        "layer",
    )
)


def scratch_layout(key: FfnBuild) -> dict:
    s = key.tokens
    geo = FGeo(key.tp)
    lay = pair_layout(
        (
            ("hrdy", BLOCKS),
            ("gpart", geo.g_split * (DQ - 1) * s * 2 * ROWS),
            ("dpart", (DQ - 1) * s * HIDDEN),
            ("nsum", CHUNKS * s),
            ("xmb", s * O_CTAS * X_PAIRS),
        )
    )
    end = max(o + n for o, n in lay.values())
    lay["h"] = ((end + 255) // 256 * 256, s * geo.inter * 2)
    lay["stamps"] = (lay["h"][0] + lay["h"][1], BLOCKS * STAMPS * 8)
    return lay


def scratch_bytes(key: FfnBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def mx_code_even(amax):
    """A 1 x 32 block's e8m0 code from its abs max, Quark "even" (aiter's Triton
    ``_mxfp4_scale_from_amax``): the exponent of amax rounded up past 1.75,
    minus 2, clamped to 0 .. 254."""
    bits = amax.bitcast(fx.Int32)
    e = ((bits + fx.Int32(0x200000)) >> 23) & 0xFF
    return fx.min(fx.max(e - 2, fx.Int32(0)), fx.Int32(254))


def pack8(vs, code):
    """8 fp32 -> e2m1 at e8m0 ``code`` (v_cvt_scalef32_pk_fp4_f32: nearest even,
    saturating, bit-exact with aiter's Triton quant) -> an i32, element 2 j / 2 j
    + 1 the low / high nibble of byte j. ``code`` >= 1 (``mx_code``)."""
    sc = pow2(code)
    w = fx.Int32(0)
    for j in range(4):
        w = fx.Int32(
            rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32, w, fx.Float32(vs[2 * j]), fx.Float32(vs[2 * j + 1]), sc, j
            )
        )
    return w


def mx_code(amax):
    """``mx_code_even`` with code 0 (an all-zero block: its fp4 are 0 at any
    scale) taken as 1, a scale the conversion can divide by."""
    return fx.max(mx_code_even(amax), fx.Int32(1))


def ld4(r, at, cm=0):
    return fx.Vector(bo.buffer_load(r, at, vec_width=4, dtype=T.i32, cache_modifier=cm))


def pack_bf(vs):
    """f32 list (even length) -> i32 list of packed bf16 pairs."""
    return [
        fx.Vector.from_elements([vs[2 * j], vs[2 * j + 1]], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
        for j in range(len(vs) // 2)
    ]


def tl_sqrt(x):
    """Triton's ``tl.sqrt``: the raw v_sqrt_f32."""
    return fx.Float32(
        llvm.call_intrinsic(
            T.f32, "llvm.amdgcn.sqrt.f32", [fx.Float32(x).ir_value()], [], []
        )
    )


def chunk_ss(z):
    """A chunk's sum of squares as the stock norm's ISA forms it: packed pairs,
    its mul / add tree fp-contracted."""
    sq = lambda e: fx.Float32(fmath.fma(z[e], z[e], z[e + 2] * z[e + 2]))  # noqa: E731
    return (sq(0) + sq(4)) + (sq(1) + sq(5))


def _add_live(x, y):
    """x + y, None standing for a zero partial."""
    return x if y is None else y if x is None else x + y


def runs_total(runs):
    """A row's CH_WAVES run sums combined as the stock norm's NORM_WARPS warp
    partials (xor 8, 4, 2, 1; warps past the row hold zeros)."""
    v = list(runs) + [None] * (NORM_WARPS - len(runs))
    for off in (8, 4, 2, 1):
        v = [_add_live(v[l], v[l + off]) for l in range(off)]
    return v[0]


def stock_rstd(tot, eps):
    """The stock norm's 1 / sqrt(sum / n + eps): IEEE divisions, raw sqrt."""
    return fx.Float32(1.0) / tl_sqrt(tot / fx.Float32(HIDDEN) + fx.Float32(eps))


def silu_mul_bf(g, u):
    """SiluAndMul on ROCm: silu(g) rounded to bf16 before the multiply."""
    return bf16_round(g * sigmoid(g)) * u


# ---------------------------------------------------------------- norm
def norm_loads(c):
    """Chunk ``tid + 512 i`` of the S rows (8 columns; every round's live chunks
    of a wave in one row): raw h / residual words and the 1 + w words."""
    s, tid, a = c["S"], c["tid"], c["args"]
    n = s * CHUNKS
    out = []
    for i in range((n + THREADS - 1) // THREADS):
        ch = fx.min(tid + THREADS * i, n - 1)
        out.append(
            (
                ld4(rsrc(a["hidden"]), ch * 4),
                ld4(rsrc(a["residual"]), ch * 4),
                ld4(rsrc(a["ln_w"]), ch % CHUNKS * 4),
            )
        )
    return out


@traced
def run_norm(c, raw):
    """z = h + residual -> res_out (chunk run ch / 16 by CTA run % 256); every
    row's sum of squares (a wave's 64 chunks a run, as the stock norm) -> rstd
    -> bf16 x -> qdq by 32 columns (four lanes a block) -> the LDS x rows."""
    s, tid, lane, wave, bid = c["S"], c["tid"], c["lane"], c["wave"], c["bid"]
    nrm, rst = c["nrm"], c["rst"]
    a = c["args"]
    n = s * CHUNKS
    zs = []
    for i in range_constexpr(len(raw)):
        hw, rw, _ = raw[i]
        z = []
        for d in range_constexpr(4):
            z += [bf_lo(hw[d]) + bf_lo(rw[d]), bf_hi(hw[d]) + bf_hi(rw[d])]
        zs.append(z)
        ss = butterfly(chunk_ss(z), WAVE_ORDER)
        if lane == 0:
            fx.ptr_store(ss, nrm + (i * WAVES + wave))
    gpu.barrier()
    stamp(c, 5)
    if tid < s:
        tot = runs_total(
            [fx.ptr_load(nrm + (tid * CH_WAVES + k)) for k in range(CH_WAVES)]
        )
        fx.ptr_store(stock_rstd(tot, c["eps"]), rst + tid)
    gpu.barrier()
    stamp(c, 6)
    for i in range_constexpr(len(raw)):
        ch_raw = tid + THREADS * i
        live = ch_raw < n
        ch = fx.min(ch_raw, n - 1)
        t = ch // CHUNKS
        col = ch % CHUNKS * 8
        z = zs[i]
        wv = raw[i][2]
        lnw = []
        for d in range_constexpr(4):
            lnw += [fx.Float32(1.0) + bf_lo(wv[d]), fx.Float32(1.0) + bf_hi(wv[d])]
        if live & ((ch // 16) % BLOCKS == bid):
            bo.buffer_store(
                fx.Vector.from_elements(pack_bf(z), fx.Int32),
                rsrc(a["res_out"]),
                ch * 4,
            )
        r = fx.ptr_load(rst + t)
        x = [bf16_round(z[e] * r * lnw[e]) for e in range(8)]
        amax = fx.Float32(0.0)
        for e in range_constexpr(8):
            amax = fx.max(amax, fx.Float32(fmath.absf(x[e])))
        amax = xred(xred(amax, 1, fx.max), 2, fx.max)
        code = mx_code(amax)
        w = pack8(x, code)
        if live:
            fx.ptr_store(w, c["xq"] + (t * XQROW + col // 8))
            if ch % 4 == 0:
                fx.ptr_store(code, c["xs"] + (t * XSROW + col // 32))
    gpu.barrier()


# ---------------------------------------------------------------- o_proj
def o_owner(c):
    return fx.min(c["bid"], O_CTAS - 1)


def block_loads(c, width):
    """1 x 32 block ``tid + 512 i`` of the S ``hidden`` rows (bf16, ``width``
    wide): its 64 B."""
    s, tid, a = c["S"], c["tid"], c["args"]
    nblk = width // 32
    nb = s * nblk
    out = []
    for i in range((nb + THREADS - 1) // THREADS):
        b = fx.min(tid + THREADS * i, nb - 1)
        at = (b // nblk * width + 32 * (b % nblk)) // 2
        out.append([ld4(rsrc(a["hidden"]), at + 4 * k) for k in range(4)])
    return out


@traced
def block_quant(c, raw, width, dq, ds, qrow, srow):
    """``block_loads``' blocks through MXFP4 qdq (a thread a 1 x 32 block) ->
    LDS rows (fp4 at ``dq`` + ``qrow`` i32 a row, codes at ``ds`` + ``srow``)."""
    s, tid = c["S"], c["tid"]
    nblk = width // 32
    nb = s * nblk
    for i in range_constexpr(len(raw)):
        b = fx.min(tid + THREADS * i, nb - 1)
        t = b // nblk
        blk = b % nblk
        vs = []
        for k in range_constexpr(4):
            for d in range_constexpr(4):
                vs += [bf_lo(raw[i][k][d]), bf_hi(raw[i][k][d])]
        amax = fx.Float32(0.0)
        for e in range_constexpr(32):
            amax = fx.max(amax, fx.Float32(fmath.absf(vs[e])))
        code = mx_code(amax)
        qs = [pack8(vs[8 * k : 8 * k + 8], code) for k in range(4)]
        if tid + THREADS * i < nb:
            fx.ptr_store(
                fx.Vector.from_elements(qs, fx.Int32), dq + (t * qrow + 4 * blk)
            )
            fx.ptr_store(code, ds + (t * srow + blk))
    gpu.barrier()


def core_loads(c):
    return block_loads(c, CORE)


def o_lane(c):
    """The norm thread (token t, run j of 8 of the owner's 32 columns)."""
    tid = c["tid"]
    return fx.min(tid // 4, c["S"] - 1), tid % 4


def o_res_loads(c):
    """The norm thread's residual and 1 + w words (8 columns)."""
    a = c["args"]
    t, j = o_lane(c)
    col = 2 * ROWS * o_owner(c) + 8 * j
    return ld4(rsrc(a["residual"]), (t * HIDDEN + col) // 2), ld4(
        rsrc(a["ln_w"]), col // 2
    )


def o_loads(c, width=CORE, splits=4):
    """Wave w: the owner's row group w % 2, K split w / 2 of ``splits`` (of
    o_proj's K, ``width``: the stock GEMM's split-K partition); past the weight
    (no traffic) in a CTA that owns none, or a wave past the splits."""
    lane, wave, bid, a = c["lane"], c["wave"], c["bid"], c["args"]
    o_q = width // STEP // splits
    row = (2 * bid + wave % 2) * ROWS + lane % ROWS
    live = bid < O_CTAS
    if const_expr(2 * splits < WAVES):
        live = live & (wave < 2 * splits)
    row = live.select(row, fx.Int32(HIDDEN))
    r_w = rsrc(a["w_o"], HIDDEN * width // 2)
    r_s = rsrc(a["w_os"], HIDDEN * width // 32)
    steps = [wave // 2 * o_q + i for i in range(o_q)]
    tiles = [
        ld4(r_w, row // ROWS * (ROWS * width // 8) + st * 256 + lane * 4, CM_NT)
        for st in steps
    ]
    words = [
        fx.Int32(
            bo.buffer_load(r_s, row * (width // 128) + st, vec_width=1, dtype=T.i32)
        )
        for st in steps
    ]
    return tiles, words


def o_quant(c, craw):
    """The core rows through MXFP4 qdq -> LDS."""
    block_quant(c, craw, CORE, c["oq"], c["os"], OQROW, OSROW)


def x_quant(c, xraw):
    """``normed``: the x rows (the stock norm's output) through MXFP4 qdq ->
    the LDS x rows."""
    block_quant(c, xraw, HIDDEN, c["xq"], c["xs"], XQROW, XSROW)


@traced
def o_stage(c, ops, width=CORE, splits=4):
    """The wave's K split of the owner's row group, its steps chained in order
    -> red (the core rows' qdq in LDS, ``width // 8 + 4`` i32 a row)."""
    lane, wave = c["lane"], c["wave"]
    o_q = width // STEP // splits
    gl = lane // ROWS
    t = fx.min(lane % ROWS, c["S"] - 1)
    tiles, words = ops
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(o_q):
        st = wave // 2 * o_q + i
        acc = scaled_step(
            c["oq"],
            c["os"],
            t * (width // 8 + 4),
            t * (width // 32),
            fx.min(st, width // STEP - 1),
            gl,
            tiles[i],
            words[i],
            acc,
        )
    fx.ptr_store(acc, c["red"] + (wave * 64 + lane) * 4)
    gpu.barrier()


@traced
def o_finish(c, rraw):
    """Norm thread (t, j): the o_proj rows summed over the K quarters in order
    -> bf16 (the stock GEMM's output) + residual = z -> res_out; the chunk's sum
    of squares -> NSUM. Returns z, 1 + w."""
    s, tid, bid = c["S"], c["tid"], c["bid"]
    t, j = o_lane(c)
    rw, wv = rraw
    res, lnw = [], []
    for d in range_constexpr(4):
        res += [bf_lo(rw[d]), bf_hi(rw[d])]
        lnw += [fx.Float32(1.0) + bf_lo(wv[d]), fx.Float32(1.0) + bf_hi(wv[d])]
    z = [
        bf16_round(row_sum_waves(c["red"], 8 * (j % 2) + e, t, j // 2)) + res[e]
        for e in range(8)
    ]
    live = (tid < s * 4) & (bid < O_CTAS)
    if live:
        bo.buffer_store(
            fx.Vector.from_elements(pack_bf(z), fx.Int32),
            rsrc(c["args"]["res_out"]),
            (t * HIDDEN + 2 * ROWS * bid + 8 * j) // 2,
        )
        c["put"](c["nsum"], t * CHUNKS + 4 * bid + j, chunk_ss(z))
    return z, lnw


@traced
def o_rstd(c):
    """Owner CTAs: wave t reduces token t's chunk sums as the stock norm (lane l:
    chunk l of each run of 64) -> rstd -> LDS."""
    s, tid, lane, bid = c["S"], c["tid"], c["lane"], c["bid"]
    if (tid < s * 64) & (bid < O_CTAS):
        t = tid // 64
        got = c["poll"](
            [(c["nsum"], t * CHUNKS + 64 * k + lane, 1) for k in range(CH_WAVES)]
        )
        runs = [butterfly(g[0].bitcast(fx.Float32), WAVE_ORDER) for g in got]
        if lane == 0:
            fx.ptr_store(stock_rstd(runs_total(runs), c["eps"]), c["rst"] + t)
    gpu.barrier()


@traced
def o_x(c, z, lnw):
    """Norm thread (t, j): bf16 x -> the 1 x 32 block's code (over the 4 threads)
    -> its fp4 word (and the code) put for every CTA."""
    s, tid, bid = c["S"], c["tid"], c["bid"]
    t, j = o_lane(c)
    r = fx.ptr_load(c["rst"] + t)
    x = [bf16_round(z[e] * r * lnw[e]) for e in range(8)]
    amax = fx.Float32(0.0)
    for e in range_constexpr(8):
        amax = fx.max(amax, fx.Float32(fmath.absf(x[e])))
    amax = xred(xred(amax, 1, fx.max), 2, fx.max)
    code = mx_code(amax)
    w = pack8(x, code)
    if (tid < s * 4) & (bid < O_CTAS):
        base = (t * O_CTAS + bid) * X_PAIRS
        c["put_words"](c["xmb"], base + j, [w])
        if j == 0:
            c["put_words"](c["xmb"], base + 4, [code])


@traced
def o_gather(c):
    """Every owner's fp4 x words and codes -> the LDS x rows."""
    s, tid = c["S"], c["tid"]
    n = s * O_CTAS
    rounds = (n + THREADS - 1) // THREADS
    units = [fx.min(tid + THREADS * i, n - 1) for i in range(rounds)]
    specs = []
    for u in units:
        base = u * X_PAIRS
        specs += [(c["xmb"], base, 2), (c["xmb"], base + 2, 2), (c["xmb"], base + 4, 1)]
    got = c["poll"](specs)
    for i in range_constexpr(rounds):
        u = units[i]
        t = u // O_CTAS
        b = u % O_CTAS
        ws = [got[3 * i][0], got[3 * i][1], got[3 * i + 1][0], got[3 * i + 1][1]]
        if tid + THREADS * i < n:
            fx.ptr_store(
                fx.Vector.from_elements(ws, fx.Int32), c["xq"] + (t * XQROW + 4 * b)
            )
            fx.ptr_store(got[3 * i + 2][0], c["xs"] + (t * XSROW + b))
    gpu.barrier()


# ---------------------------------------------------------------- gate_up
def w13_loads(c, row, steps):
    """This lane's w13 tiles of ``row`` at K steps ``steps`` (traced, each) and
    their scale words (byte lane / 16 of each)."""
    lane, a, geo = c["lane"], c["args"], c["G"]
    gl = lane // ROWS
    r_w = rsrc(a["w13"], 2 * geo.inter * HIDDEN // 2)
    r_s = rsrc(a["w13s"], 2 * geo.inter * HIDDEN // 32)
    if c["shuffled"]:
        at = [row // ROWS * (ROWS * W13_DW) + st * 256 + lane * 4 for st in steps]
    else:
        at = [row * W13_DW + st * 16 + 4 * gl for st in steps]
    tiles = [ld4(r_w, a_, CM_NT) for a_ in at]
    words = [
        fx.Int32(bo.buffer_load(r_s, row * W13S_DW + st, vec_width=1, dtype=T.i32))
        for st in steps
    ]
    return tiles, words


def gu_loads(c, g):
    """Whole group g: wave w's gate / up (w % 2) rows, K quarter w / 2."""
    lane, wave = c["lane"], c["wave"]
    row = ((wave % 2) * c["G"].ng + g) * ROWS + lane % ROWS
    kq = wave // 2
    return w13_loads(c, row, [kq * GU_Q + i for i in range(GU_Q)])


def qt_bounds(wave):
    """A quarter task's wave: gate / up by wave % 2; sub-run wave / 2 of the 10
    steps (3, 3, 2, 2): (first step, steps)."""
    sub = wave // 2
    start = 3 * sub - fx.max(sub - 2, 0)
    return start, (sub < 2).select(fx.Int32(3), fx.Int32(2))


def qt_loads(c):
    """CTA b's quarter task: K quarter b % 4 of split group 1024 + b / 4 (TP1);
    a dead third step, and the whole task of a CTA past ``qt_ctas``, past the
    weight."""
    lane, wave, bid, geo = c["lane"], c["wave"], c["bid"], c["G"]
    g = geo.g_full * BLOCKS + bid // DQ
    kq = bid % DQ
    start, n = qt_bounds(wave)
    row = ((wave % 2) * geo.ng + g) * ROWS + lane % ROWS
    dead = fx.Int32(2 * geo.inter)
    if const_expr(geo.qt_ctas < BLOCKS):
        row = (bid < geo.qt_ctas).select(row, dead)
    rows = [row, row, (n > 2).select(row, dead)]
    lane_ops = [w13_loads(c, rows[i], [kq * GU_Q + start + i]) for i in range(QT_UNITS)]
    return [o[0][0] for o in lane_ops], [o[1][0] for o in lane_ops]


def gu_mfmas(c, tiles, words, steps):
    lane = c["lane"]
    gl = lane // ROWS
    t = fx.min(lane % ROWS, c["S"] - 1)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(len(tiles)):
        acc = scaled_step(
            c["xq"],
            c["xs"],
            t * XQROW,
            t * XSROW,
            steps[i],
            gl,
            tiles[i],
            words[i],
            acc,
        )
    return acc


def scaled_step(aq, as_, qrow, srow, st, gl, tile, word, acc):
    """One 128-K step: the lane's 32 fp4 of its weight row (byte gl of ``word``
    its e8m0) against the 32 fp4 of its token's activation row in LDS (``aq`` +
    ``qrow`` i32, its code at ``as_`` + ``srow``), one scaled MFMA."""
    bw = fx.Vector(
        fx.ptr_load(
            aq + (qrow + st * 16 + gl * 4), result_type=fx.Vector.make_type(4, fx.Int32)
        )
    )
    sb = fx.ptr_load(as_ + (srow + st * 4 + gl))
    sa = (word >> (gl * 8)) & 0xFF
    return mfma_scaled(tile, bw, acc, sa, sb, FP4, FP4, 0, 0)


def store_h(c, g, tt, cp, gv, uv):
    """bf16 g, u of columns 16 g + 2 cp + (0, 1), token tt -> H."""
    v = [silu_mul_bf(gv[d], uv[d]) for d in range(2)]
    bo.buffer_store(
        pack_bf(v)[0],
        rsrc(c["hbuf"]),
        (tt * c["G"].inter + g * ROWS + 2 * cp) // 2,
        cache_modifier=CM_DEV,
    )


@traced
def stage_gu(c, g, ops):
    """Whole group g: the wave's 10 steps -> red; thread (token, column pair):
    gate and up summed over the K quarters in order, bf16 each -> H."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    kq = wave // 2
    acc = gu_mfmas(c, ops[0], ops[1], [kq * GU_Q + i for i in range(GU_Q)])
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < s * (ROWS // 2):
        tt = tid // (ROWS // 2)
        cp = tid % (ROWS // 2)
        gv = [
            bf16_round(row_sum(red, 2 * cp + d, tt, waves=(0, 2, 4, 6)))
            for d in range(2)
        ]
        uv = [
            bf16_round(row_sum(red, 2 * cp + d, tt, waves=(1, 3, 5, 7)))
            for d in range(2)
        ]
        store_h(c, g, tt, cp, gv, uv)
    gpu.barrier()


@traced
def stage_qt(c, ops):
    """The quarter task: fp32 partial of gate | up (16 + 16 columns, S tokens);
    quarters 1..3 put it, quarter 0 keeps its own in LDS (``qown``)."""
    s, tid, lane, wave, red, bid = (
        c["S"],
        c["tid"],
        c["lane"],
        c["wave"],
        c["red"],
        c["bid"],
    )
    kq = bid % DQ
    start, _ = qt_bounds(wave)
    # a dead step's zero tile still reads x: keep it inside the row
    steps = [kq * GU_Q + fx.min(start + i, GU_Q - 1) for i in range(QT_UNITS)]
    acc = gu_mfmas(c, ops[0], ops[1], steps)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if qt_live(c, tid < s * 2 * ROWS):
        tt = tid // (2 * ROWS)
        gc = tid % (2 * ROWS)  # gate columns 0..15, up 16..31
        gu = gc // ROWS
        col = gc % ROWS
        v = row_sum_waves(red, col, tt, gu)
        if kq > 0:
            at = (((bid // DQ) * (DQ - 1) + kq - 1) * s + tt) * 2 * ROWS + gc
            c["put"](c["gpart"], at, v)
        if kq == 0:
            fx.ptr_store(v, c["qown"] + tid)
    gpu.barrier()


def qt_live(c, pred):
    """``pred`` in a CTA with a quarter task."""
    geo = c["G"]
    if const_expr(geo.qt_ctas < BLOCKS):
        return pred & (c["bid"] < geo.qt_ctas)
    return pred


def row_sum_waves(red, r, t, gu, n=4):
    """Row r, token t summed over the waves of gate (gu 0: 0, 2, 4, 6) or up
    (1, 3, 5, 7), the first ``n`` of them, in order (gu traced)."""
    tot = fx.Float32(0.0)
    for k in range(n):
        w = 2 * k + gu
        tot = tot + fx.ptr_load(red + ((w * 64 + 16 * (r // 4) + t) * 4 + r % 4))
    return tot


@traced
def finish_qt(c):
    """Quarter 0's CTA: its partial + quarters 1..3's, in order -> bf16 g, u ->
    H of split group 1024 + b / 4."""
    s, tid, bid = c["S"], c["tid"], c["bid"]
    if qt_live(c, (bid % DQ == 0) & (tid < s * (ROWS // 2))):
        tt = tid // (ROWS // 2)
        cp = tid % (ROWS // 2)
        lg = bid // DQ
        got = c["poll"](
            [
                (
                    c["gpart"],
                    ((lg * (DQ - 1) + kq - 1) * s + tt) * 2 * ROWS + gu * ROWS + 2 * cp,
                    2,
                )
                for kq in range(1, DQ)
                for gu in range(2)
            ]
        )
        vals = []
        for gu in range_constexpr(2):
            pair = []
            for d in range_constexpr(2):
                v = fx.ptr_load(c["qown"] + (tt * 2 * ROWS + gu * ROWS + 2 * cp + d))
                for kq in range_constexpr(DQ - 1):
                    v = v + got[kq * 2 + gu][d].bitcast(fx.Float32)
                pair.append(bf16_round(v))
            vals.append(pair)
        store_h(c, c["G"].g_full * BLOCKS + lg, tt, cp, vals[0], vals[1])


@traced
def run_gate_up(c, qops, ops):
    """The quarter task and the first whole group (their weights ``qops``,
    ``ops`` in flight since the start), the other three (a group's weights in
    flight under the previous one), the split group's sum, one flag."""
    bid, tid = c["bid"], c["tid"]
    g_full = c["G"].g_full
    g0 = g_full * bid
    stage_qt(c, qops)
    for j in range_constexpr(g_full):
        if const_expr(j + 1 < g_full):
            ops_n = gu_loads(c, g0 + j + 1)
        stage_gu(c, g0 + j, ops)
        if const_expr(j + 1 < g_full):
            ops = ops_n
    finish_qt(c)
    publish(c["put"], c["hrdy"], bid, fx.Int32(1), tid == 0)


# ---------------------------------------------------------------- down
def down_bounds(geo, wave, k):
    """Row group k's steps for this wave: the waves rotated by e k, the first e
    (rotated) take one step more than the rest (TP1: e = 2, 5 and 4 steps; TP2:
    e = 1, 3 and 2): (first step, steps)."""
    base, extra = divmod(geo.d_steps, WAVES)
    wr = (wave + (WAVES - extra * k % WAVES)) % WAVES
    start = base * wr + fx.min(wr, extra)
    return start, (wr < extra).select(fx.Int32(base + 1), fx.Int32(base))


def down_loads(c, q, rg0):
    """Every (row group, step) unit of this wave; the dead fifth past the
    weight."""
    lane, wave, geo = c["lane"], c["wave"], c["G"]
    a = c["args"]
    gl = lane // ROWS
    r_w = rsrc(a["w2"], HIDDEN * geo.inter // 2)
    r_s = rsrc(a["w2s"], HIDDEN * geo.inter // 32)
    tiles, words = [], []
    for k in range(D_RG):
        start, n = down_bounds(geo, wave, k)
        row = (rg0 + k) * ROWS + lane % ROWS
        for i in range(geo.d_units):
            st = start + i
            r = row
            if i == geo.d_units - 1:
                r = (n > i).select(row, fx.Int32(HIDDEN))
            if c["shuffled"]:
                at = (
                    r // ROWS * (ROWS * geo.w2_dw)
                    + (q * geo.d_steps + st) * 256
                    + lane * 4
                )
            else:
                at = r * geo.w2_dw + q * (geo.dk // 8) + st * 16 + 4 * gl
            tiles.append(ld4(r_w, at, CM_NT))
            words.append(
                fx.Int32(
                    bo.buffer_load(
                        r_s,
                        r * geo.w2s_dw + q * geo.d_steps + st,
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
            )
    return tiles, words


@traced
def load_h(c, q):
    """Every CTA's H ready -> H's quarter q through MXFP4 qdq (a thread a 1 x 32
    block) -> LDS rows of HROW."""
    s, tid, geo = c["S"], c["tid"], c["G"]
    if tid < BLOCKS:
        c["poll"]([(c["hrdy"], tid, 1)])
    gpu.barrier()
    nb = s * geo.h_blocks
    r_h = rsrc(c["hbuf"])
    for i in range_constexpr((nb + THREADS - 1) // THREADS):
        b = fx.min(tid + THREADS * i, nb - 1)
        t = b // geo.h_blocks
        blk = b % geo.h_blocks
        at = (t * geo.inter + q * geo.dk + 32 * blk) // 2
        ws = [ld4(r_h, at + 4 * k, CM_DEV) for k in range(4)]
        vs = []
        for k in range_constexpr(4):
            for d in range_constexpr(4):
                vs += [bf_lo(ws[k][d]), bf_hi(ws[k][d])]
        amax = fx.Float32(0.0)
        for e in range_constexpr(32):
            amax = fx.max(amax, fx.Float32(fmath.absf(vs[e])))
        code = mx_code(amax)
        qs = [pack8(vs[8 * k : 8 * k + 8], code) for k in range(4)]
        if tid + THREADS * i < nb:
            fx.ptr_store(
                fx.Vector.from_elements(qs, fx.Int32),
                c["hq"] + (t * geo.hqrow + 4 * blk),
            )
            fx.ptr_store(code, c["hs"] + (t * geo.hsrow + blk))
    gpu.barrier()


@traced
def stage_down(c, q, rg0, ops):
    """The wave's units -> fp32 accumulators a row group -> LDS; then thread
    (row group, token, row pair): the CTA's partial; quarters 1..3 put it,
    quarter 0 adds theirs in order -> bf16 out."""
    s, tid, lane, wave, geo = c["S"], c["tid"], c["lane"], c["wave"], c["G"]
    dred = c["dred"]
    tiles, words = ops
    gl = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    hq_row, hs_row = t * geo.hqrow, t * geo.hsrow
    for k in range_constexpr(D_RG):
        start, _ = down_bounds(geo, wave, k)
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for i in range_constexpr(geo.d_units):
            u = k * geo.d_units + i
            st = fx.min(start + i, geo.d_steps - 1)
            acc = scaled_step(
                c["hq"], c["hs"], hq_row, hs_row, st, gl, tiles[u], words[u], acc
            )
        fx.ptr_store(acc, dred + ((k * WAVES + wave) * 64 + lane) * 4)
    gpu.barrier()
    n = D_RG * s * (ROWS // 2)
    if tid < n:
        k = tid // (s * (ROWS // 2))
        rem = tid % (s * (ROWS // 2))
        tt = rem // (ROWS // 2)
        rp = rem % (ROWS // 2)
        base = dred + k * (WAVES * 64 * 4)
        v = [row_sum(base, 2 * rp + d, tt) for d in range(2)]
        row = (rg0 + k) * ROWS + 2 * rp
        if q > 0:
            c["put_words"](
                c["dpart"],
                ((q - 1) * s + tt) * HIDDEN + row,
                [v[0].bitcast(fx.Int32), v[1].bitcast(fx.Int32)],
            )
        if q == 0:
            got = c["poll"](
                [
                    (c["dpart"], ((qq - 1) * s + tt) * HIDDEN + row, 2)
                    for qq in range(1, DQ)
                ]
            )
            y = []
            for d in range_constexpr(2):
                yd = v[d]
                for qq in range_constexpr(DQ - 1):
                    yd = yd + got[qq][d].bitcast(fx.Float32)
                y.append(yd)
            bo.buffer_store(
                pack_bf(y)[0], rsrc(c["args"]["out"]), (tt * HIDDEN + row) // 2
            )


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: FfnBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    assert 1 <= s <= MAX_TOKENS and key.tp in TP_SIZES
    # TP2 runs from the stock norm's output only
    assert key.normed == (key.tp > 1) and not (key.normed and key.oproj)
    geo = FGeo(key.tp)
    lay = scratch_layout(key)

    @fx.struct
    class XLds:
        xq: fx.Array[fx.Int32, s * XQROW, 16]
        xs: fx.Array[fx.Int32, s * XSROW, 16]

    @fx.struct
    class DLds:
        hq: fx.Array[fx.Int32, s * geo.hqrow, 16]
        hs: fx.Array[fx.Int32, s * geo.hsrow, 16]
        dred: fx.Array[fx.Float32, D_RG * WAVES * 64 * 4, 16]

    @fx.struct
    class OLds:
        oq: fx.Array[fx.Int32, s * OQROW, 16]
        os: fx.Array[fx.Int32, s * OSROW, 16]

    @fx.union
    class StageLds:
        x: XLds
        d: DLds
        o: OLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        nrm: fx.Array[fx.Float32, MAX_ROUNDS * WAVES, 16]
        rst: fx.Array[fx.Float32, MAX_TOKENS, 16]
        qown: fx.Array[fx.Float32, MAX_TOKENS * 2 * ROWS, 16]

    keyed = key_tuple(key, digest())
    name = (
        f"qwen38_dense_ffn_s{s}_p{key.stop}_t{int(key.stamps)}_w{int(key.shuffled)}"
        f"_o{int(key.oproj)}_tp{key.tp}_n{int(key.normed)}_v5"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kd(
        hidden: Int64,
        residual: Int64,
        w_o: Int64,
        w_os: Int64,
        ln_w: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        out: Int64,
        res_out: Int64,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        st_lds = alloc.allocate(StageLds)
        xl, dl, ol = st_lds.x.peek(), st_lds.d.peek(), st_lds.o.peek()
        mb = Mailbox(step_tag(epoch, layer) - 1)
        c = {
            "S": s,
            "G": geo,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "eps": key.eps,
            "shuffled": key.shuffled,
            "stamps": scratch + fx.Int64(lay["stamps"][0]) if key.stamps else None,
            "put": mb.put,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "nrm": lds.nrm.ptr,
            "rst": lds.rst.ptr,
            "qown": lds.qown.ptr,
            "xq": xl.xq.ptr,
            "xs": xl.xs.ptr,
            "hq": dl.hq.ptr,
            "hs": dl.hs.ptr,
            "dred": dl.dred.ptr,
            "oq": ol.oq.ptr,
            "os": ol.os.ptr,
            "nsum": sreg(scratch, lay["nsum"][0], "nsum"),
            "xmb": sreg(scratch, lay["xmb"][0], "xmb"),
            "hbuf": scratch + fx.Int64(lay["h"][0]),
            "hrdy": sreg(scratch, lay["hrdy"][0], "hrdy"),
            "gpart": sreg(scratch, lay["gpart"][0], "gpart"),
            "dpart": sreg(scratch, lay["dpart"][0], "dpart"),
            "args": {
                "hidden": hidden,
                "residual": residual,
                "w_o": w_o,
                "w_os": w_os,
                "ln_w": ln_w,
                "w13": w13,
                "w13s": w13s,
                "w2": w2,
                "w2s": w2s,
                "out": out,
                "res_out": res_out,
            },
        }
        stamp(c, 0)
        if const_expr(key.oproj):
            # the core first, then the residual and o weights: vmcnt drains in order
            craw = core_loads(c)
            rraw = o_res_loads(c)
            oops = o_loads(c)
            o_quant(c, craw)
            o_stage(c, oops)
            z, lnw = o_finish(c, rraw)
            stamp(c, 5)
            # the first gate_up weights before the norm's hand-offs
            qops = qt_loads(c)
            gops = gu_loads(c, geo.g_full * bid)
            o_rstd(c)
            o_x(c, z, lnw)
            o_gather(c)
        elif const_expr(key.normed):
            # x first: vmcnt drains in issue order
            xraw = block_loads(c, HIDDEN)
            qops = qt_loads(c)
            gops = gu_loads(c, geo.g_full * bid)
            x_quant(c, xraw)
        else:
            # the norm's loads first: vmcnt drains in issue order
            nraw = norm_loads(c)
            qops = qt_loads(c)
            run_norm(c, nraw)
            gops = gu_loads(c, geo.g_full * bid)
        stamp(c, 1)
        if const_expr(key.stop > 2):
            run_gate_up(c, qops, gops)
            stamp(c, 2)
        if const_expr(key.stop > 3):
            q = bid // DCTA
            rg0 = (bid % DCTA) * D_RG
            dops = down_loads(c, q, rg0)
            load_h(c, q)
            stamp(c, 3)
            stage_down(c, q, rg0, dops)
            stamp(c, 4)

    @flyc.jit
    def launch(
        hidden: Int64,
        residual: Int64,
        w_o: Int64,
        w_os: Int64,
        ln_w: Int64,
        w13: Int64,
        w13s: Int64,
        w2: Int64,
        w2s: Int64,
        out: Int64,
        res_out: Int64,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kd(
            hidden, residual, w_o, w_os, ln_w, w13, w13s, w2, w2s, out, res_out,
            scratch, epoch, layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(kd, launch)
    _BUILDS[key] = launch
    return launch


def shuffle_w(w):
    """fp4x2 [N, K / 2] -> the (16, 16) shuffled order ``FfnBuild.shuffled`` reads."""
    n, kb = w.shape
    return (
        w.view(n // ROWS, ROWS, kb // 16, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(n, kb)
    )


def dense_ffn(
    key: FfnBuild,
    *,
    hidden,
    residual,
    ln_w,
    w13,
    w13s,
    w2,
    w2s,
    out,
    res_out,
    scratch,
    epoch,
    layer,
    w_o=None,
    w_os=None,
):
    """Host launcher. ``hidden`` / ``residual`` / ``out`` / ``res_out``: bf16
    [S, HIDDEN] (``out`` / ``res_out`` may be views of larger buffers' first S
    rows); ``w13`` = [gate; up] rows of the rank's share (``FGeo``), ``w2`` its
    K columns. ``key.oproj``: ``hidden`` is o_proj's input, bf16 [S, CORE], and
    ``w_o`` / ``w_os`` its shuffled weights. ``key.normed``: ``hidden`` is x;
    ``residual`` / ``ln_w`` / ``res_out`` None."""
    s = key.tokens
    inter = FGeo(key.tp).inter
    if key.oproj:
        assert key.shuffled and hidden.shape == (s, CORE)
        assert w_o.shape == (HIDDEN, CORE // 2) and w_os.shape == (HIDDEN, CORE // 32)
    else:
        assert hidden.shape == (s, HIDDEN)
        w_o = w_os = hidden
    if key.normed:
        assert residual is None and res_out is None and ln_w is None
        residual = res_out = ln_w = hidden
    assert residual.shape == (s, HIDDEN)
    assert w13.shape == (2 * inter, HIDDEN // 2) and w13s.shape == (
        2 * inter,
        HIDDEN // 32,
    )
    assert w2.shape == (HIDDEN, inter // 2) and w2s.shape == (HIDDEN, inter // 32)
    assert scratch.numel() >= scratch_bytes(key)
    for t in (hidden, residual, out, res_out, w13, w13s, w2, w2s):
        assert t.is_contiguous()
    f = build(key)
    f(
        *ABI.pack(
            {
                "hidden": hidden.data_ptr(),
                "residual": residual.data_ptr(),
                "w_o": w_o.data_ptr(),
                "w_os": w_os.data_ptr(),
                "ln_w": ln_w.data_ptr(),
                "w13": w13.data_ptr(),
                "w13s": w13s.data_ptr(),
                "w2": w2.data_ptr(),
                "w2s": w2s.data_ptr(),
                "out": out.data_ptr(),
                "res_out": res_out.data_ptr(),
                "scratch": scratch.data_ptr(),
                "epoch": epoch.data_ptr(),
                "layer": layer,
            }
        ),
        stream=torch.cuda.current_stream(),
    )
