# SPDX-License-Identifier: Apache-2.0
"""K1 for Qwen3.8-27B at TP1 (``gdn27_pre``): one GDN layer of a decode step
from the layer's input (the previous layer's FFN output + residual) to the
gated core (out_proj's input, ``dense_ffn``'s with ``oproj``), S <= 8 rows,
one token a request. One launch, ``BLOCKS`` x ``THREADS``:

    front    (CTAs 0 .. 159, 32 hidden columns each): h + residual -> ``res_out``;
             sum-of-squares partials -> rstd -> input_layernorm (Gemma, 1 + w)
             -> bf16 x -> MXFP4 blocks put for every CTA -> the LDS x rows
             (``dense_ffn``'s distributed norm)
    in_proj  in_proj_qkvz: 1024 groups of 16 rows, CTA b groups 4 b ..; wave w
             group w / 2, K half w % 2 -> bf16 PROJ entries. in_proj_ba: 6
             groups x 8 K eighths on CTAs 0 .. 47 -> fp32 partials (BAP)
    gdn      tasks of 8 state rows (48 value heads x 16 chunks a token: 3 S a
             CTA), phase by phase over the CTA's tasks so each hand-off is
             waited on once:
               conv   the task's 8 v channels; the key head's first value head
                      also its 8 q and 8 k channels (out through QK). Window
                      updated in place, products rounded to bf16, SiLU -> bf16
               gates  q / k of the key head (QK) l2-normed, q * HD^-1/2; b, a
                      (the BAP partials in order, bf16) -> beta, decay
               recur  the delta rule on the 8 rows (fp32 state in place) ->
                      bf16 o; the chunk's sum of squares -> ONRM
               norm   RMSNormGated over the head (16 chunks) * silu(z) -> core

Rounding follows the stock decode path elementwise (aiter's Triton conv update,
``fused_recurrent_gated_delta_rule_packed_decode``, RMSNormGated); reductions
differ in order. A row whose state slot is <= 0 (slot 0 is the pool's padding
slot, -1 a CUDA-graph pad row) writes no state and a zero core row.

Weights: in_proj_qkvz [q | k | v | z] and in_proj_ba [b | a] as MXFP4 in the
(16, 16)-shuffled order (``dense_ffn.shuffle_w``), e8m0 scales [N, K / 32].
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from sglang.kernels.ops.moe.k3_mono_flydsl.common.ops import (
    CM_NT,
    bf16_round,
    bf_hi,
    bf_lo,
    row_sum,
    rsrc,
    traced,
    uniform,
    wave_sum,
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
from sglang.kernels.ops.moe.k3_mono_flydsl.common.sync import Mailbox, sreg
from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import dense_ffn as D
from sglang.kernels.ops.moe.qwen3_5_mono_flydsl.layer import step_tag
from sglang.kernels.ops.moe.qwen3_5_mono_flydsl.sources import digest

HIDDEN, ROWS, MAX_TOKENS = D.HIDDEN, D.ROWS, D.MAX_TOKENS
HD = 128
NK = 16
NV = 48
HPK = NV // NK  # 3 value heads a key head
KEY = NK * HD  # 2048
VAL = NV * HD  # 6144
QKVZ = 2 * KEY + 2 * VAL  # 16384
BA = 2 * NV  # 96
CONV = 2 * KEY + VAL  # 10240 conv channels: q | k | v
CORE = VAL
CONV_W = 4
SL = CONV_W - 1
V0 = 2 * KEY  # conv channel / PROJ entry of v
Z0 = CONV  # PROJ entry of z

STEPS = HIDDEN // D.STEP  # 40
QG = QKVZ // ROWS  # 1024 qkvz groups
QG_CTA = QG // BLOCKS  # 4
Q_HALF = STEPS // 2  # 20
BA_G = BA // ROWS  # 6
BA_E = 8  # K eighths a ba group
BA_STEPS = STEPS // BA_E  # 5
BA_CTAS = BA_G * BA_E  # 48
W_DW = HIDDEN // 8  # 640 dwords a weight row
WS_DW = HIDDEN // 32 // 4  # 40 scale dwords a row

VCH = 8  # state rows a task
NVCH = HD // VCH  # 16
TPT = NV * NVCH  # 768 tasks a token
TPC = TPT // BLOCKS  # 3 a CTA a token
CI = 3 * VCH  # conv items a task: 8 v, 8 q, 8 k
SOFTPLUS_THRESHOLD = 20.0

assert QG == QG_CTA * BLOCKS and TPT == TPC * BLOCKS and VCH == WAVES
assert BA_STEPS * BA_E == STEPS and QG_CTA * 2 == WAVES

_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class K1Build:
    tokens: int
    first: bool = False  # the model's first layer: no residual (residual := hidden)
    eps: float = 1e-6  # input_layernorm
    norm_eps: float = 1e-6  # the gated norm
    stamps: bool = False  # debug: per-CTA s_memrealtime stamps -> scratch "stamps"
    stop: int = 99  # debug: phases < stop (1 front, 2 in_proj, 3 gdn)


ABI = KernelAbi(
    (
        "hidden",
        "residual",
        "res_out",
        "ln_w",
        "w_qkvz",
        "w_qkvzs",
        "w_ba",
        "w_bas",
        "conv_w",
        "conv_st",
        "cs_seq",
        "cs_dim",
        "cs_tok",
        "a_log",
        "dt_bias",
        "norm_w",
        "rstate",
        "rs_seq",
        "st_idx",
        "core",
        "scratch",
        "epoch",
        "layer",
    )
)


def scratch_layout(key: K1Build) -> dict:
    s = key.tokens
    lay = pair_layout(
        (
            ("nsum", D.O_CTAS * s),
            ("xmb", s * D.O_CTAS * D.X_PAIRS),
            ("proj", s * QKVZ),
            ("bap", BA_CTAS * s * ROWS),
            ("qk", s * NK * 2 * HD),
            ("onrm", s * NV * NVCH),
        )
    )
    end = max(o + n for o, n in lay.values())
    lay["stamps"] = ((end + 255) // 256 * 256, BLOCKS * D.STAMPS * 8)
    return lay


def scratch_bytes(key: K1Build) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def f32_of(w):
    return w.bitcast(fx.Float32)


def ld_bf(ptr, i):
    return fx.Float32(fx.BFloat16(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16)))


def ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


def silu(z):
    return z / (fx.Float32(1.0) + fmath.exp(-z))


def _addf(x, y):
    return x + y


# ---------------------------------------------------------------- front
def front_loads(c):
    """The norm thread (t, j)'s hidden, residual and 1 + w words (8 columns of
    the owner's 32)."""
    a = c["args"]
    t, j = D.o_lane(c)
    col = 2 * ROWS * D.o_owner(c) + 8 * j
    hw = D.ld4(rsrc(a["hidden"]), (t * HIDDEN + col) // 2)
    rw = hw if c["first"] else D.ld4(rsrc(a["residual"]), (t * HIDDEN + col) // 2)
    return hw, rw, D.ld4(rsrc(a["ln_w"]), col // 2)


@traced
def front_z(c, raw):
    """z = h + residual (fp32; h on the first layer) -> res_out; z's sum of
    squares over the owner's 32 columns -> NSUM. Returns z, 1 + w."""
    s, tid, bid = c["S"], c["tid"], c["bid"]
    t, j = D.o_lane(c)
    hw, rw, wv = raw
    z, lnw = [], []
    for d in range_constexpr(4):
        if const_expr(c["first"]):
            z += [bf_lo(hw[d]), bf_hi(hw[d])]
        else:
            z += [bf_lo(hw[d]) + bf_lo(rw[d]), bf_hi(hw[d]) + bf_hi(rw[d])]
        lnw += [fx.Float32(1.0) + bf_lo(wv[d]), fx.Float32(1.0) + bf_hi(wv[d])]
    ss = fx.Float32(0.0)
    for v in z:
        ss = ss + v * v
    ss = xred(xred(ss, 1, _addf), 2, _addf)
    if (tid < s * 4) & (bid < D.O_CTAS):
        bo.buffer_store(
            fx.Vector.from_elements(D.pack_bf(z), fx.Int32),
            rsrc(c["args"]["res_out"]),
            (t * HIDDEN + 2 * ROWS * bid + 8 * j) // 2,
        )
        if j == 0:
            c["put"](c["nsum"], bid * s + t, ss)
    return z, lnw


# ---------------------------------------------------------------- in_proj
def wloads(c, w, ws, nrows, row, steps):
    """This lane's tiles of (shuffled) weight row ``row`` at K steps ``steps``
    and their scale words; a row past ``nrows`` reads zeros, no traffic."""
    lane = c["lane"]
    r_w = rsrc(w, nrows * HIDDEN // 2)
    r_s = rsrc(ws, nrows * HIDDEN // 32)
    tiles = [D.ld4(r_w, row // ROWS * (ROWS * W_DW) + st * 256 + lane * 4, CM_NT) for st in steps]
    words = [
        fx.Int32(bo.buffer_load(r_s, row * WS_DW + st, vec_width=1, dtype=T.i32))
        for st in steps
    ]
    return tiles, words


def qkvz_loads(c):
    a, lane, wave, bid = c["args"], c["lane"], c["wave"], c["bid"]
    row = (QG_CTA * bid + wave // 2) * ROWS + lane % ROWS
    return wloads(c, a["w_qkvz"], a["w_qkvzs"], QKVZ, row, [wave % 2 * Q_HALF + i for i in range(Q_HALF)])


def ba_loads(c):
    """CTA b < 48: group b / 8, K eighth b % 8, wave w < 5 its step w; the other
    waves and CTAs past the weight."""
    a, lane, wave, bid = c["args"], c["lane"], c["wave"], c["bid"]
    row = (bid // BA_E) * ROWS + lane % ROWS
    row = ((bid < BA_CTAS) & (wave < BA_STEPS)).select(row, fx.Int32(BA))
    st = (bid % BA_E) * BA_STEPS + fx.min(wave, BA_STEPS - 1)
    return wloads(c, a["w_ba"], a["w_bas"], BA, row, [st])


@traced
def stage_qkvz(c, ops):
    """The CTA's 4 groups: the two K halves in order -> bf16 PROJ."""
    s, tid, lane, wave, red, bid = c["S"], c["tid"], c["lane"], c["wave"], c["red"], c["bid"]
    acc = D.gu_mfmas(c, ops[0], ops[1], [wave % 2 * Q_HALF + i for i in range(Q_HALF)])
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < QG_CTA * ROWS * s:
        q = tid // (ROWS * s)
        rem = tid % (ROWS * s)
        t = rem // ROWS
        r = rem % ROWS
        v = bf16_round(row_sum(red, r, t, waves=(2 * q, 2 * q + 1)))
        c["put"](c["proj"], t * QKVZ + (QG_CTA * bid + q) * ROWS + r, v)
    gpu.barrier()


@traced
def stage_ba(c, ops):
    """CTAs 0 .. 47: the eighth's fp32 partial (waves 0 .. 4 in order) -> BAP."""
    s, tid, lane, wave, red, bid = c["S"], c["tid"], c["lane"], c["wave"], c["red"], c["bid"]
    gl = lane // ROWS
    t = fx.min(lane % ROWS, s - 1)
    st = (bid % BA_E) * BA_STEPS + fx.min(wave, BA_STEPS - 1)
    acc = D.scaled_step(c["xq"], c["xs"], t * D.XQROW, t * D.XSROW, st, gl, ops[0][0], ops[1][0],
                        fx.Vector.filled(4, 0.0, fx.Float32))
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if (bid < BA_CTAS) & (tid < ROWS * s):
        t2 = tid // ROWS
        r = tid % ROWS
        v = row_sum(red, r, t2, waves=tuple(range(BA_STEPS)))
        c["put"](c["bap"], (bid * s + t2) * ROWS + r, v)


# ---------------------------------------------------------------- gdn
def task_of(c, k):
    """The CTA's task k (a Python int): token, value head, chunk."""
    w = (k % TPC) * BLOCKS + c["bid"]
    return k // TPC, w // NVCH, w % NVCH


def gdn_loads(c):
    """Every task's slot and state rows (thread: row wave, columns 2 lane ..),
    issued before the first GDN hand-off; per conv item its window and weights."""
    s, tid, a = c["S"], c["tid"], c["args"]
    nt = TPC * s
    g = {"slot": [], "sbase": [], "st": [], "soff": []}
    for k in range(nt):
        t, h, ch = task_of(c, k)
        slot = uniform(fx.Int32(bo.buffer_load(rsrc(a["st_idx"]), t, vec_width=1, dtype=T.i32)))
        sbase = a["rstate"] + fx.Int64(fx.max(slot, 0)) * fx.Int64(a["rs_seq"]) * 4
        soff = (h * HD + ch * VCH + c["wave"]) * HD + 2 * c["lane"]
        g["slot"].append(slot)
        g["sbase"].append(sbase)
        g["soff"].append(soff)
        g["st"].append(fx.Vector(bo.buffer_load(rsrc(sbase), soff, vec_width=2, dtype=T.f32)))
    # conv items: item i = tid + 512 r -> task i / 24, channel i % 24 (8 v, 8 q, 8 k)
    ni = nt * CI
    items = []
    for r in range((ni + THREADS - 1) // THREADS):
        i = fx.min(tid + THREADS * r, ni - 1)
        k = i // CI
        cc = i % CI
        # task k's (t, h, ch) from a traced k
        w = (k % TPC) * BLOCKS + c["bid"]
        t = k // TPC
        h = w // NVCH
        ch = w % NVCH
        kh = h // HPK
        part = cc // VCH  # 0 v, 1 q, 2 k
        d = cc % VCH
        chan = (part == 0).select(V0 + h * HD + ch * VCH + d, (part - 1) * KEY + kh * HD + ch * VCH + d)
        slot = fx.Int32(bo.buffer_load(rsrc(a["st_idx"]), t, vec_width=1, dtype=T.i32))
        cs = a["conv_st"] + fx.Int64(fx.max(slot, 0)) * fx.Int64(a["cs_seq"]) * 2
        co = chan * a["cs_dim"]
        win = [ld_bf(cs, co + j * a["cs_tok"]) for j in range(SL)]
        cw = fx.Vector(bo.buffer_load(rsrc(a["conv_w"]), chan * 2, vec_width=2, dtype=T.i32))
        live = (tid + THREADS * r < ni) & ((part == 0) | (h % HPK == 0))
        items.append(
            {"k": k, "t": t, "h": h, "ch": ch, "kh": kh, "part": part, "d": d, "chan": chan,
             "slot": slot, "cs": cs, "co": co, "win": win,
             "cw": [bf_lo(cw[0]), bf_hi(cw[0]), bf_lo(cw[1]), bf_hi(cw[1])], "live": live}
        )
    g["items"] = items
    return g


@traced
def gdn_conv(c, g):
    """Every conv item: x (PROJ) -> window shifted in place; the width-4 conv
    (products rounded to bf16, fp32 sum in tap order), SiLU -> bf16; v -> LDS,
    q / k -> QK."""
    s, a = c["S"], c["args"]
    items = g["items"]
    got = c["poll"]([(c["proj"], it["t"] * QKVZ + it["chan"], 1) for it in items])
    for n in range_constexpr(len(items)):
        it = items[n]
        x = f32_of(got[n][0])
        win, w = it["win"], it["cw"]
        if it["live"] & (it["slot"] > 0):
            rs = rsrc(it["cs"])
            bo.buffer_store(fx.Float32(win[1]).to(fx.BFloat16), rs, it["co"])
            bo.buffer_store(fx.Float32(win[2]).to(fx.BFloat16), rs, it["co"] + a["cs_tok"])
            bo.buffer_store(fx.Float32(x).to(fx.BFloat16), rs, it["co"] + 2 * a["cs_tok"])
        acc = bf16_round(w[0] * win[0]) + bf16_round(w[1] * win[1])
        acc = acc + bf16_round(w[2] * win[2])
        acc = acc + bf16_round(w[3] * x)
        o = bf16_round(silu(acc))
        if it["live"] & (it["part"] == 0):
            fx.ptr_store(o, c["vloc"] + (it["k"] * VCH + it["d"]))
        if it["live"] & (it["part"] > 0):
            at = ((it["t"] * NK + it["kh"]) * 2 + it["part"] - 1) * HD + it["ch"] * VCH + it["d"]
            c["put"](c["qk"], at, o)
    gpu.barrier()


@traced
def gdn_gates(c):
    """Every task's q, k (QK) -> LDS; b, a (BAP, the eighths in order) -> beta,
    decay; then q / k l2-normed (q * HD^-1/2)."""
    s, tid, lane, wave, bid, a = c["S"], c["tid"], c["lane"], c["wave"], c["bid"], c["args"]
    nt = TPC * s
    ni = nt * 2 * HD
    specs, dst = [], []
    for r in range_constexpr((ni + THREADS - 1) // THREADS):
        i = fx.min(tid + THREADS * r, ni - 1)
        k = i // (2 * HD)
        e = i % (2 * HD)
        w = (k % TPC) * BLOCKS + bid
        kh = w // NVCH // HPK
        specs.append((c["qk"], ((k // TPC) * NK + kh) * 2 * HD + e, 1))
        dst.append(i)
    got = c["poll"](specs)
    for r in range_constexpr(len(specs)):
        if tid + THREADS * r < ni:
            fx.ptr_store(f32_of(got[r][0]), c["qkl"] + dst[r])
    if tid < 2 * nt:
        k = tid // 2
        which = tid % 2  # 0 b, 1 a
        w = (k % TPC) * BLOCKS + bid
        h = w // NVCH
        t = k // TPC
        row = which * NV + h
        gb = row // ROWS
        r = row % ROWS
        got2 = c["poll"]([(c["bap"], ((gb * BA_E + e) * s + t) * ROWS + r, 1) for e in range(BA_E)])
        v = f32_of(got2[0][0])
        for e in range_constexpr(1, BA_E):
            v = v + f32_of(got2[e][0])
        v = bf16_round(v)
        x = v + ld_bf(a["dt_bias"], h)
        sp = fmath.log(fx.Float32(1.0) + fmath.exp(x))
        sp = (x <= fx.Float32(SOFTPLUS_THRESHOLD)).select(sp, x)
        decay = fmath.exp(-fmath.exp(ld_f32(a["a_log"], h)) * sp)
        beta = bf16_round(fx.Float32(1.0) / (fx.Float32(1.0) + fmath.exp(-v)))
        fx.ptr_store((which == 0).select(beta, decay), c["scal"] + (k * 2 + which))
    gpu.barrier()
    for n in range_constexpr((2 * nt + WAVES - 1) // WAVES):
        vi = fx.min(wave + WAVES * n, 2 * nt - 1)
        base = c["qkl"] + vi * HD
        x0 = fx.ptr_load(base + lane)
        x1 = fx.ptr_load(base + (64 + lane))
        nrm = fmath.sqrt(wave_sum(x0 * x0 + x1 * x1) + fx.Float32(1e-6))
        y0, y1 = x0 / nrm, x1 / nrm
        isq = vi % 2 == 0
        y0 = isq.select(y0 * fx.Float32(HD**-0.5), y0)
        y1 = isq.select(y1 * fx.Float32(HD**-0.5), y1)
        if wave + WAVES * n < 2 * nt:
            fx.ptr_store(y0, base + lane)
            fx.ptr_store(y1, base + (64 + lane))
    gpu.barrier()


@traced
def gdn_recur(c, g):
    """Task by task: state row ``wave`` (columns 2 lane, + 1): decay, the delta
    rule, the new state (in place), o = bf16(h q) -> LDS; then the chunk sums of
    squares -> ONRM."""
    s, tid, lane, wave, bid = c["S"], c["tid"], c["lane"], c["wave"], c["bid"]
    nt = TPC * s
    for k in range_constexpr(nt):
        live = g["slot"][k] > 0
        beta = fx.ptr_load(c["scal"] + 2 * k)
        decay = fx.ptr_load(c["scal"] + (2 * k + 1))
        qb = c["qkl"] + k * 2 * HD
        q0 = fx.ptr_load(qb + 2 * lane)
        q1 = fx.ptr_load(qb + (2 * lane + 1))
        k0 = fx.ptr_load(qb + (HD + 2 * lane))
        k1 = fx.ptr_load(qb + (HD + 2 * lane + 1))
        st = g["st"][k]
        h0 = live.select(st[0], fx.Float32(0.0)) * decay
        h1 = live.select(st[1], fx.Float32(0.0)) * decay
        p = wave_sum(h0 * k0 + h1 * k1)
        vn = (fx.ptr_load(c["vloc"] + (k * VCH + wave)) - p) * beta
        h0 = h0 + vn * k0
        h1 = h1 + vn * k1
        o = bf16_round(wave_sum(h0 * q0 + h1 * q1))
        if live:
            bo.buffer_store(fx.Vector.from_elements([h0, h1], fx.Float32), rsrc(g["sbase"][k]), g["soff"][k])
        if lane == 0:
            fx.ptr_store(live.select(o, fx.Float32(0.0)), c["ol"] + (k * VCH + wave))
    gpu.barrier()
    if tid < nt:
        k = tid
        w = (k % TPC) * BLOCKS + bid
        ss = fx.Float32(0.0)
        for e in range_constexpr(VCH):
            x = fx.ptr_load(c["ol"] + (k * VCH + e))
            ss = ss + x * x
        c["put"](c["onrm"], ((k // TPC) * NV + w // NVCH) * NVCH + w % NVCH, ss)


@traced
def gdn_norm(c, g):
    """Thread (task, row): RMSNormGated over the head (every chunk's partial):
    bf16(o rstd w silu(z)) -> core."""
    s, tid, bid, a = c["S"], c["tid"], c["bid"], c["args"]
    nt = TPC * s
    if tid < nt * VCH:
        k = tid // VCH
        v = tid % VCH
        w = (k % TPC) * BLOCKS + bid
        t = k // TPC
        h = w // NVCH
        ch = w % NVCH
        vr = ch * VCH + v
        got = c["poll"](
            [(c["onrm"], (t * NV + h) * NVCH + q, 1) for q in range(NVCH)]
            + [(c["proj"], t * QKVZ + Z0 + h * HD + vr, 1)]
        )
        ss = f32_of(got[0][0])
        for q in range_constexpr(1, NVCH):
            ss = ss + f32_of(got[q][0])
        r = fx.Float32(1.0) / fmath.sqrt(ss * fx.Float32(1.0 / HD) + fx.Float32(c["norm_eps"]))
        y = fx.ptr_load(c["ol"] + tid) * r * ld_bf(a["norm_w"], vr)
        y = y * silu(f32_of(got[NVCH][0]))
        slot = fx.Int32(bo.buffer_load(rsrc(a["st_idx"]), t, vec_width=1, dtype=T.i32))
        y = (slot > 0).select(y, fx.Float32(0.0))
        bo.buffer_store(y.to(fx.BFloat16), rsrc(a["core"]), t * CORE + h * HD + vr)


# ---------------------------------------------------------------- kernel
_BUILDS: dict = {}


def build(key: K1Build):
    if key in _BUILDS:
        return _BUILDS[key]
    s = key.tokens
    assert 1 <= s <= MAX_TOKENS
    lay = scratch_layout(key)
    nt = TPC * s

    @fx.struct
    class XLds:
        xq: fx.Array[fx.Int32, s * D.XQROW, 16]
        xs: fx.Array[fx.Int32, s * D.XSROW, 16]

    @fx.struct
    class GLds:
        qkl: fx.Array[fx.Float32, nt * 2 * HD, 16]
        vloc: fx.Array[fx.Float32, nt * VCH, 16]
        ol: fx.Array[fx.Float32, nt * VCH, 16]
        scal: fx.Array[fx.Float32, nt * 2, 16]

    @fx.union
    class StageLds:
        x: XLds
        g: GLds

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]
        rst: fx.Array[fx.Float32, MAX_TOKENS, 16]

    keyed = key_tuple(key, digest())
    name = f"qwen38_27b_k1_s{s}_f{int(key.first)}_t{int(key.stamps)}_p{key.stop}"

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def k1(
        hidden: Int64,
        residual: Int64,
        res_out: Int64,
        ln_w: Int64,
        w_qkvz: Int64,
        w_qkvzs: Int64,
        w_ba: Int64,
        w_bas: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        norm_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        core: Int64,
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
        xl, gl = st_lds.x.peek(), st_lds.g.peek()
        mb = Mailbox(step_tag(epoch, layer) - 1)
        c = {
            "S": s,
            "tid": tid,
            "bid": bid,
            "lane": tid % 64,
            "wave": tid // 64,
            "first": key.first,
            "eps": key.eps,
            "norm_eps": key.norm_eps,
            "stream_only": False,
            "stamps": scratch + fx.Int64(lay["stamps"][0]) if key.stamps else None,
            "put": mb.put,
            "put_words": mb.put_words,
            "poll": mb.poll,
            "red": lds.red.ptr,
            "rst": lds.rst.ptr,
            "xq": xl.xq.ptr,
            "xs": xl.xs.ptr,
            "qkl": gl.qkl.ptr,
            "vloc": gl.vloc.ptr,
            "ol": gl.ol.ptr,
            "scal": gl.scal.ptr,
            "args": {
                "hidden": hidden,
                "residual": residual,
                "res_out": res_out,
                "ln_w": ln_w,
                "w_qkvz": w_qkvz,
                "w_qkvzs": w_qkvzs,
                "w_ba": w_ba,
                "w_bas": w_bas,
                "conv_w": conv_w,
                "conv_st": conv_st,
                "cs_seq": cs_seq,
                "cs_dim": cs_dim,
                "cs_tok": cs_tok,
                "a_log": a_log,
                "dt_bias": dt_bias,
                "norm_w": norm_w,
                "rstate": rstate,
                "rs_seq": rs_seq,
                "st_idx": st_idx,
                "core": core,
            },
        }
        for region in ("nsum", "xmb", "proj", "bap", "qk", "onrm"):
            c[region] = sreg(scratch, lay[region][0], region)
        D.stamp(c, 0)
        # the norm's inputs first, then the in_proj weights (vmcnt drains in order)
        raw = front_loads(c)
        qops = qkvz_loads(c)
        bops = ba_loads(c)
        z, lnw = front_z(c, raw)
        D.o_rstd(c)
        D.o_x(c, z, lnw)
        D.o_gather(c)
        D.stamp(c, 1)
        if const_expr(key.stop > 2):
            stage_qkvz(c, qops)
            stage_ba(c, bops)
            D.stamp(c, 2)
        if const_expr(key.stop > 3):
            g = gdn_loads(c)
            gdn_conv(c, g)
            D.stamp(c, 3)
            gdn_gates(c)
            D.stamp(c, 4)
            gdn_recur(c, g)
            D.stamp(c, 5)
            gdn_norm(c, g)
            D.stamp(c, 6)

    @flyc.jit
    def launch(
        hidden: Int64,
        residual: Int64,
        res_out: Int64,
        ln_w: Int64,
        w_qkvz: Int64,
        w_qkvzs: Int64,
        w_ba: Int64,
        w_bas: Int64,
        conv_w: Int64,
        conv_st: Int64,
        cs_seq: Int32,
        cs_dim: Int32,
        cs_tok: Int32,
        a_log: Int64,
        dt_bias: Int64,
        norm_w: Int64,
        rstate: Int64,
        rs_seq: Int32,
        st_idx: Int64,
        core: Int64,
        scratch: Int64,
        epoch: Int64,
        layer: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        k1(
            hidden, residual, res_out, ln_w, w_qkvz, w_qkvzs, w_ba, w_bas, conv_w,
            conv_st, cs_seq, cs_dim, cs_tok, a_log, dt_bias, norm_w, rstate, rs_seq,
            st_idx, core, scratch, epoch, layer,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(k1, launch)
    _BUILDS[key] = launch
    return launch


def gdn27_pre(
    key: K1Build,
    *,
    hidden,
    residual,
    res_out,
    ln_w,
    w_qkvz,
    w_qkvzs,
    w_ba,
    w_bas,
    conv_w,
    conv_state,
    a_log,
    dt_bias,
    norm_w,
    rstate,
    st_idx,
    core,
    scratch,
    epoch,
    layer,
):
    """Host launcher. ``residual`` None on the first layer (``key.first``);
    ``w_qkvz`` / ``w_ba`` shuffled (``dense_ffn.shuffle_w``); ``conv_state``
    [slots, CONV, 3] (strides read); ``rstate`` [slots, NV, HD, HD] fp32, a slot
    dense; ``st_idx`` [>= S] int32, <= 0 a pad row; ``epoch`` a device int32
    bumped once a step, ``layer`` this launch's tag slot."""
    s = key.tokens
    assert (residual is None) == key.first
    assert hidden.shape == (s, HIDDEN) and res_out.shape == (s, HIDDEN)
    assert core.shape == (s, CORE) and ln_w.shape == (HIDDEN,)
    assert w_qkvz.shape == (QKVZ, HIDDEN // 2) and w_qkvzs.shape == (QKVZ, HIDDEN // 32)
    assert w_ba.shape == (BA, HIDDEN // 2) and w_bas.shape == (BA, HIDDEN // 32)
    assert conv_w.numel() == CONV * CONV_W and conv_w.dtype == torch.bfloat16
    assert conv_state.dtype == torch.bfloat16 and conv_state.shape[1:] == (CONV, SL)
    assert rstate.dtype == torch.float32 and rstate.shape[1:] == (NV, HD, HD)
    assert rstate[0].is_contiguous()
    assert a_log.dtype == torch.float32 and dt_bias.dtype == torch.bfloat16
    assert norm_w.dtype == torch.bfloat16 and norm_w.numel() == HD
    assert st_idx.dtype == torch.int32 and st_idx.numel() >= s
    assert scratch.numel() >= scratch_bytes(key)
    for t in (hidden, res_out, w_qkvz, w_qkvzs, w_ba, w_bas, conv_w, core):
        assert t.is_contiguous()
    f = build(key)
    f(
        *ABI.pack(
            {
                "hidden": hidden.data_ptr(),
                "residual": hidden.data_ptr() if residual is None else residual.data_ptr(),
                "res_out": res_out.data_ptr(),
                "ln_w": ln_w.data_ptr(),
                "w_qkvz": w_qkvz.data_ptr(),
                "w_qkvzs": w_qkvzs.data_ptr(),
                "w_ba": w_ba.data_ptr(),
                "w_bas": w_bas.data_ptr(),
                "conv_w": conv_w.data_ptr(),
                "conv_st": conv_state.data_ptr(),
                "cs_seq": conv_state.stride(0),
                "cs_dim": conv_state.stride(1),
                "cs_tok": conv_state.stride(2),
                "a_log": a_log.data_ptr(),
                "dt_bias": dt_bias.data_ptr(),
                "norm_w": norm_w.data_ptr(),
                "rstate": rstate.data_ptr(),
                "rs_seq": rstate.stride(0),
                "st_idx": st_idx.data_ptr(),
                "core": core.data_ptr(),
                "scratch": scratch.data_ptr(),
                "epoch": epoch.data_ptr(),
                "layer": layer,
            }
        ),
        stream=torch.cuda.current_stream(),
    )
