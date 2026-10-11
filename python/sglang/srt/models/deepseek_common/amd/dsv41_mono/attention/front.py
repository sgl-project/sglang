# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""K1 ``front``: one DeepSeek-V4.1 attention layer from its normed input row to
the rotated query and the step's KV rows in the sliding-window cache.

One launch a layer and rank, ``BLOCKS`` CTAs x ``THREADS`` threads, S rows.
Every CTA issues its weight loads (its wqkv row tile and its wq_b tasks) at the
kernel start, before it waits on anything; the stages:

    (X8, the input's MXFP8 rows, comes from the mono layer's seam, a flag a
    token and K half)
    kt     (S): the token's attention keys for K2 -- top-k compressed slots
           through the kv source's block table, then window slots -> KT
    wqkv_a (112 row tiles, 16 rows, all of K): X8's halves in turn, a token
           tile at a time (the next rows in flight), times wqkv's in the K
           order of SGLang's GEMM at that width (``plan.wqkv_order``) -> fp32
           rows -> QKVP
    qkv    (S): the wqkv row -> bf16; q_norm -> bf16 -> MXFP8 -> QX8
           (published, a flag a token); kv_norm -> bf16 -> GPT-J RoPE -> bf16
           -> the fp8_ds_mla record at the token's slot
    wq_b   (H x 512 / 16, 16 rows): QX8 times wq_b -> bf16 -> GPT-J RoPE of
           each head's dims 448.. -> q (global, bf16)

Each elementwise step follows vLLM's ROCm path (``mxfp8_quantize``,
``FusedQKVRMSNorm``, ``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert``);
the GEMV reductions differ from the Triton GEMMs' only in their order (wqkv_a's
and the kv record's follow SGLang's bit for bit).
Adapted from ATOM's V4.1 ``attn_pre`` (K1), less its mHC seam.
"""

from dataclasses import dataclass

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .device import (
    BLOCKS,
    CM_DEV,
    THREADS,
    Mailbox,
    bf16_round,
    butterfly,
    ceil_exp,
    clamp_fp8,
    div_rn,
    first_task,
    fma,
    fp8_pack4,
    fresh,
    gstore,
    hw_rsq,
    ld_f32,
    ld_i32,
    mx_code,
    mx_mul,
    rsrc,
    stamp,
    traced,
    xshfl,
)
from .gemv import (
    RED,
    chain_loads,
    chain_mfmas,
    chain_slots,
    chain_store,
    chain_sum,
    gemv_loads,
    gemv_mfmas,
    lds_row,
    row_sum,
    rows_loads,
    rows_store,
)
from .plan import (
    DATA,
    EPS,
    HALF,
    HEAD_DIM,
    HIDDEN,
    KEYS,
    KV_DIM,
    NOPE,
    Q_RANK,
    QKV_ROWS,
    ROPE,
    ROWS,
    TOPK,
    WINDOW,
    WQKV_KPARTS,
    WQKV_TASKS,
    Dims,
    front_scratch,
    tile_rows,
    token_tiles,
    wqkv_order,
)

X_WORDS = HIDDEN // 4
X_GROUPS = HIDDEN // 32
QX_WORDS = Q_RANK // 4
KSTEPS = HIDDEN // 128 // WQKV_KPARTS  # an X8 K half's 128-steps
KWORDS = KSTEPS * 32
KGROUPS = KSTEPS * 4
FRONT_POINTS = 6  # start, kt done, wqkv done, qkv done, wq_b done, (spare)


@dataclass(frozen=True)
class FrontBuild:
    tokens: int
    tp: int
    ratio: int  # the layer's compress ratio (0: no top-k keys)
    timeline: bool = False


def plus(t0, v):
    return v if isinstance(t0, int) and t0 == 0 else v + t0


@traced
def publish(c, region, i):
    """This CTA's stores visible device-wide, then the flag pair (region, i)."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if c["tid"] == 0:
        c["mb"].put(region, i, fx.Int32(1))


@traced
def await_flags(c, region, n, index):
    """Thread u < n polls flag ``index(u)``; then the CTA proceeds."""
    tid = c["tid"]
    if tid < n:
        c["mb"].poll([(region, index(fx.min(tid, n - 1)), 1)])
    gpu.barrier()


# ---------------------------------------------------------------- kt


@traced
def stage_kt(c, t):
    """Token t's keys for K2: ``KT[t][k] = slot | 1 << 31`` for its top-k compressed
    rows (k < ntopk), the window slot for k in [ntopk, kv_len), -1 for an
    invalid or absent key; KLEN[t] = kv_len (0 for a pad row), KLEN[S + t] =
    ntopk."""
    tid = c["tid"]
    a, ratio = c["args"], c["ratio"]
    slot = ld_i32(a["slot"], 2 * t)
    live = slot >= 0
    nswa = live.select(ld_i32(a["swa_lens"], t), fx.Int32(0))
    if const_expr(ratio > 0):
        pos = ld_i32(a["pos"], 2 * t)
        ntopk = live.select(fx.min((pos + 1) // ratio, fx.Int32(TOPK)), fx.Int32(0))
        req = ld_i32(a["t2r"], t)
    else:
        ntopk = fx.Int32(0)
    kv_len = ntopk + nswa
    for i in range_constexpr((KEYS + THREADS - 1) // THREADS):
        k = fresh(tid) + THREADS * i
        sidx = fx.min(fx.max(k - ntopk, 0), WINDOW - 1)
        s_slot = ld_i32(a["swa_idx"], t * WINDOW + sidx)
        val = (k < kv_len).select(s_slot, fx.Int32(-1))
        if const_expr(ratio > 0):
            local = ld_i32(a["topk"], t * TOPK + fx.min(k, TOPK - 1))
            lc = fx.max(local, 0)
            e = a["comp_block"]
            blk = ld_i32(a["comp_bt"], req * a["bt_stride"] + lc // e)
            c_slot = (blk * e + lc % e) | fx.Int32(-2147483648)
            c_val = (local >= 0).select(c_slot, fx.Int32(-1))
            val = (k < ntopk).select(c_val, val)
        if k < KEYS:
            bo.buffer_store(val, rsrc(a["kt"]), t * KEYS + k)
    if tid == 0:
        bo.buffer_store(kv_len, rsrc(a["klen"]), t)
        bo.buffer_store(ntopk, rsrc(a["klen"]), c["S"] + t)


# ---------------------------------------------------------------- wqkv_a


def stage_wqkv(c, ops, slots, n, halves):
    """The wqkv chains against a token tile of n (all of K in LDS); the window
    is free again on return."""
    accs = chain_mfmas(c, c["xl"], c["xsl"], ops, n, HIDDEN // 128, slots, halves)
    gpu.barrier()
    return accs


@traced
def wqkv_publish(c, task, accs, slots, order, t0, n, live):
    """Rows 16 task .. of wqkv against tokens t0 .. t0 + n: the chains' partials
    summed in SGLang's order -> fp32 QKVP (``live``: a CTA with a wqkv task)."""
    tid, red = c["tid"], c["red"]
    chain_store(c, red, accs, slots)
    gpu.barrier()
    if (tid < ROWS * n) & live:
        r = tid % ROWS
        tl = tid // ROWS
        t = plus(t0, tl)
        v = chain_sum(red, r, tl, order)
        c["mb"].put(c["qkvp"], t * QKV_ROWS + task * ROWS + r, v)
    gpu.barrier()


# ---------------------------------------------------------------- q / kv norms


@traced
def stage_qkv(c, t):
    """Token t: the wqkv row -> bf16; q_norm -> bf16 -> MXFP8 (QX8);
    kv_norm -> bf16 -> RoPE -> bf16 -> the fp8_ds_mla record at
    slot_mapping[t]. 256 threads (8 q columns and 2 kv columns each); threads
    256.. mirror thread 255 and store nothing."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    tt = fx.min(tid, 255)
    mine = tid < 256
    qc = [fx.min(tt * 8 + j, Q_RANK - 1) for j in range(8)]
    # independent loads first: they overlap the partials' wait
    qw = [c["ldbf"](a["qn_w"], qc[j]) for j in range(8)]
    kw = [c["ldbf"](a["kvn_w"], tt * 2 + j) for j in range(2)]
    pos = ld_i32(a["pos"], 2 * t)
    slot = ld_i32(a["slot"], 2 * t)
    jr = tt - NOPE // 2
    is_rope = jr >= 0
    jj = fx.max(jr, 0)
    cs = is_rope.select(ld_f32(a["cos_sin"], pos * ROPE + jj), fx.Float32(1.0))
    sn = is_rope.select(ld_f32(a["cos_sin"], pos * ROPE + HALF + jj), fx.Float32(0.0))
    specs = [(c["qkvp"], t * QKV_ROWS + qc[j], 1) for j in range(8)]
    specs += [(c["qkvp"], t * QKV_ROWS + Q_RANK + tt * 2 + j, 1) for j in range(2)]
    got = c["mb"].poll(specs)
    vals = [bf16_round(got[j][0].bitcast(fx.Float32)) for j in range(10)]
    xq = [((tt * 8 + j) < Q_RANK).select(vals[j], fx.Float32(0.0)) for j in range(8)]
    xk = vals[8:10]
    ssq = xq[0] * xq[0]
    for j in range_constexpr(1, 8):
        ssq = ssq + xq[j] * xq[j]
    ssq = butterfly(ssq, (32, 16, 8, 4, 2, 1))
    # kv's sum of squares as SGLang's K store (fused_k_norm_rope_flashmla, gfx950
    # build) adds it: fma(x0, x0, x1 x1) a thread, a 32-lane butterfly, then the
    # 8 warps' partials in warp::reduce_sum<8>'s order
    ssk = butterfly(fma(xk[0], xk[0], xk[1] * xk[1]), (16, 8, 4, 2, 1))
    if (lane == 0) & mine:
        fx.ptr_store(ssq, red + wave)
    if (lane % 32 == 0) & mine:
        fx.ptr_store(ssk, red + 16 + tt // 32)
    gpu.barrier()
    tq = (fx.ptr_load(red + 0) + fx.ptr_load(red + 1)) + (
        fx.ptr_load(red + 2) + fx.ptr_load(red + 3)
    )
    pk = [fx.ptr_load(red + 16 + w) for w in range(8)]
    tk = ((pk[0] + pk[4]) + (pk[2] + pk[6])) + ((pk[1] + pk[5]) + (pk[3] + pk[7]))
    rq = hw_rsq(
        div_rn(tq, fx.Float32(float(Q_RANK)), fx.Float32(1.0 / Q_RANK))
        + fx.Float32(EPS)
    )
    rk = hw_rsq(
        div_rn(tk, fx.Float32(float(KV_DIM)), fx.Float32(1.0 / KV_DIM))
        + fx.Float32(EPS)
    )
    # q: bf16 normed row, then vLLM's MXFP8 (a 32-group = 4 threads)
    ys = [bf16_round((xq[j] * rq) * qw[j]) for j in range(8)]
    amax = abs(ys[0])
    for y in ys[1:]:
        amax = fx.max(amax, abs(y))
    code = mx_code(butterfly(amax, (1, 2), fx.max))
    mul = mx_mul(code)
    if mine & (tt * 8 < Q_RANK):
        w0 = fp8_pack4(*[clamp_fp8(ys[i] * mul) for i in range(4)])
        w1 = fp8_pack4(*[clamp_fp8(ys[4 + i] * mul) for i in range(4)])
        bo.buffer_store(
            fx.Vector.from_elements([w0, w1], fx.Int32),
            rsrc(c["qx8"]),
            t * QX_WORDS + 2 * tt,
            cache_modifier=CM_DEV,
        )
        if tt % 4 == 0:
            bo.buffer_store(
                code,
                rsrc(c["qx8s"]),
                t * (Q_RANK // 32) + tt // 4,
                cache_modifier=CM_DEV,
            )
    # kv: this thread's pair (2 tt, 2 tt + 1)
    # fp32 like SGLang's V4 K store: nope quantized unrounded, rope rounded once;
    # its rope rounds one product and fuses the other (that build's ISA)
    kn = [(xk[j] * rk) * kw[j] for j in range(2)]
    e0 = is_rope.select(fma(kn[0], cs, -(kn[1] * sn)), kn[0])
    e1 = is_rope.select(fma(kn[0], sn, kn[1] * cs), kn[1])
    # record quant of the nope dims: a 64-block = 32 threads (half a wave)
    amax = butterfly(fx.max(abs(e0), abs(e1)), (1, 2, 4, 8, 16), fx.max)
    rcode = ceil_exp(
        div_rn(
            fx.max(amax, fx.Float32(1e-4)), fx.Float32(448.0), fx.Float32(1.0 / 448.0)
        )
    )
    inv = mx_mul(rcode)
    q0 = clamp_fp8(e0 * inv)
    q1 = clamp_fp8(e1 * inv)
    n0 = xshfl(q0, 1)
    n1 = xshfl(q1, 1)
    word = fp8_pack4(q0, q1, n0, n1)
    rope_word = (
        fx.Vector.from_elements([e0, e1], fx.Float32)
        .to(fx.BFloat16)
        .bitcast(fx.Int32)[0]
    )
    blk = a["swa_block"]
    sblock = slot // blk
    soff = slot % blk
    base = a["swa"] + fx.Int64(sblock) * fx.Int64(a["swa_stride"])
    rec = base + fx.Int64(soff * DATA)
    if mine & (tt % 32 == 0) & (tt < NOPE // 2):
        fx.ptr_store(rcode.bitcast(fx.Float32), red + 8 + tt // 32)
    gpu.barrier()
    if mine & (slot >= 0):
        if tt < NOPE // 2:
            if tt % 2 == 0:
                gstore(rec + fx.Int64(2 * tt), word, words=1)
        else:
            gstore(rec + fx.Int64(NOPE + 4 * (tt - NOPE // 2)), rope_word, words=1)
        if tt == 0:
            sb = [fx.ptr_load(red + 8 + b).bitcast(fx.Int32) for b in range(7)]
            w0 = sb[0] | (sb[1] << 8) | (sb[2] << 16) | (sb[3] << 24)
            w1 = sb[4] | (sb[5] << 8) | (sb[6] << 16)
            sc = base + fx.Int64(blk * DATA + soff * 8)
            gstore(sc, fx.Vector.from_elements([w0, w1], fx.Int32), words=2)
    publish(c, c["qrdy"], t)


# ---------------------------------------------------------------- wq_b


def wqb_rope(c, task0, pos_t, n):
    """(cos, sin) of this thread's pair in ``stage_wqb``'s epilogue over a tile
    of n rows (``pos_t``: its token's position; every task of the CTA shares
    the pair's dims: tasks BLOCKS apart are whole heads apart)."""
    a = c["args"]
    rp = c["tid"] % (ROWS // 2)
    dim = (task0 * ROWS) % HEAD_DIM + 2 * rp
    jj = fx.max((dim - NOPE) >> 1, 0)
    return ld_f32(a["cos_sin"], pos_t * ROPE + jj), ld_f32(
        a["cos_sin"], pos_t * ROPE + HALF + jj
    )


def wqb_pos(c, t0, n):
    """This thread's token position for ``stage_wqb``'s epilogue of tile
    t0 .. t0 + n."""
    per = ROWS * n // 2
    tl = fx.min((c["tid"] % per) // (ROWS // 2), n - 1)
    return ld_i32(c["args"]["pos"], 2 * (t0 + tl))


@traced
def stage_wqb(c, task0, bops, t0, n, rope):
    """Rows 16 (task0 + BLOCKS k) .. of the rank's H x 512 query rows, every
    task k of the CTA: GEMVs -> bf16 -> GPT-J RoPE of a head's dims 448..
    (fp32) -> bf16 -> q."""
    tid, red, d = c["tid"], c["red"], c["d"]
    a = c["args"]
    nt = len(bops)
    for k in range_constexpr(nt):
        gemv_mfmas(c, Q_RANK, c["qxl"], c["qxsl"], red + k * RED, bops[k], rows=n)
    gpu.barrier()
    per = ROWS * n // 2  # bf16 pairs a task
    if tid < nt * per:
        k = tid // per
        u = tid % per
        rp = u % (ROWS // 2)
        tl = u // (ROWS // 2)
        t = plus(t0, tl)
        r0 = (task0 + BLOCKS * k) * ROWS + 2 * rp
        rk = red + k * RED
        v0 = bf16_round(row_sum(rk, 2 * rp, tl))
        v1 = bf16_round(row_sum(rk, 2 * rp + 1, tl))
        is_rope = (task0 * ROWS) % HEAD_DIM + 2 * rp >= NOPE
        cs, sn = rope
        e0 = is_rope.select(v0 * cs - v1 * sn, v0)
        e1 = is_rope.select(v0 * sn + v1 * cs, v1)
        word = (
            fx.Vector.from_elements([e0, e1], fx.Float32)
            .to(fx.BFloat16)
            .bitcast(fx.Int32)[0]
        )
        bo.buffer_store(word, rsrc(a["q"]), (t * d.q_rows + r0) // 2)
    gpu.barrier()


# ---------------------------------------------------------------- kernel


def front_lds(s: int, d: Dims):
    """K1's LDS (the attention front's stages)."""
    rows = tile_rows(s)
    wqb_per_cta = d.wqb_tasks // BLOCKS

    @fx.struct
    class FrontLds:
        red: fx.Array[fx.Float32, wqb_per_cta * RED + 16, 16]
        qxl: fx.Array[fx.Int32, rows * lds_row(QX_WORDS), 16]
        qxsl: fx.Array[fx.Int32, rows * lds_row(Q_RANK // 32), 16]
        xl: fx.Array[fx.Int32, rows * lds_row(X_WORDS), 16]
        xsl: fx.Array[fx.Int32, rows * lds_row(X_GROUPS), 16]

    return FrontLds


def front_context(key, lds, args: dict, scratch, tag, layout=None) -> dict:
    """The front stages' context: LDS views, mailbox (tag ``tag``), scratch
    regions, arguments (``args``: the kernel's, by name)."""
    s, d = key.tokens, Dims(key.tp)
    tid = fx.thread_idx.x
    c = {
        "S": s,
        "tid": tid,
        "bid": fx.block_idx.x,
        "lane": tid % 64,
        "wave": tid // 64,
        "red": lds.red.ptr,
        "xl": lds.xl.ptr,
        "xsl": lds.xsl.ptr,
        "qxl": lds.qxl.ptr,
        "qxsl": lds.qxsl.ptr,
        "mb": Mailbox(tag),
        "d": d,
        "ratio": key.ratio,
        "ldbf": lambda ptr, i: fx.Float32(
            fx.BFloat16(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16))
        ),
        "args": args,
    }
    for region, (off, _) in (layout or front_scratch(s, d)).items():
        c[region] = scratch + fx.Int64(off)
    return c


@traced
def run_front(c, key, bid, x8_given=False, before_wqkv=None, after_wqkv=None):
    """The front's stages on this CTA. ``x8_given``: a stage before this one
    (the mono layer's seam norm) publishes X8 / XRDY for every token (no input
    quantization here). ``before_wqkv()`` runs ahead of the wqkv weights' loads
    (a latency-critical stage of the caller's), ``after_wqkv()`` right after."""
    s = key.tokens
    # the wqkv GEMV takes all of K from X8, which the seam publishes
    assert x8_given
    wqb_tasks = Dims(key.tp).wqb_tasks
    a = c["args"]
    tiles = token_tiles(s)
    wqb_per_cta = wqb_tasks // BLOCKS
    assert wqb_tasks % BLOCKS == 0
    n_wqkv = WQKV_TASKS  # a row tile a CTA, all of K in SGLang's order
    n_free = BLOCKS - n_wqkv  # CTAs without a wqkv task: kt
    QKV0 = 0  # on wqkv CTAs
    WQB0 = 0
    waves, halves, order = wqkv_order(s)
    assert waves * 64 * 4 <= wqb_per_cta * RED  # the chains' partials fit ``red``
    stamp(c, 0)

    # ---- Loads complete in issue order, so every CTA issues its latency-
    # critical loads before its later stages' weights: a wqkv CTA its wqkv
    # row tile, its first input rows, then its wq_b tasks; a CTA with a
    # key-table task those after that work. The q / kv norms run on wqkv
    # CTAs, their partials' wait behind the wq_b weights that are due by
    # then anyway.
    wqkv_unit = first_task(bid, 0)
    has_wqkv = wqkv_unit < n_wqkv
    task = fx.min(wqkv_unit, n_wqkv - 1)
    wqb_task = first_task(bid, WQB0)
    free = bid - n_wqkv
    is_free = free >= 0
    if const_expr(before_wqkv is not None):
        before_wqkv()
    slots = chain_slots(c, HIDDEN // 128, waves)
    kops = chain_loads(c, a["wqkv"], a["wqkv_s"], HIDDEN, task * ROWS, slots, has_wqkv)
    if const_expr(after_wqkv is not None):
        after_wqkv()
    for t in range(is_free.select(free, fx.Int32(s)), s, n_free):
        stage_kt(c, t)
    stamp(c, 1)
    poss = [wqb_pos(c, t0, n) for t0, n in tiles]

    # ---- wqkv_a: a row tile a CTA, a token tile at a time with all of K in
    # LDS (X8's K halves side by side), the next tile's rows in flight; the
    # wq_b weights issued behind the input rows' loads
    def wqb_weights():
        return [
            gemv_loads(c, a["wqb"], a["wqb_s"], Q_RANK, (wqb_task + BLOCKS * i) * ROWS)
            for i in range(wqb_per_cta)
        ]

    def x_loads(t0, n):
        return [
            rows_loads(
                c,
                c["x8"],
                c["x8s"],
                X_WORDS,
                p * KWORDS,
                KWORDS,
                X_GROUPS,
                p * KGROUPS,
                KGROUPS,
                t0,
                n,
                pred=has_wqkv,
            )
            for p in range(WQKV_KPARTS)
        ]

    await_flags(c, c["xrdy"], WQKV_KPARTS * s, lambda u: u)
    regs = x_loads(*tiles[0])
    if const_expr(len(tiles) == 1):
        bops = wqb_weights()
    for ti in range_constexpr(len(tiles)):
        t0, n = tiles[ti]
        for p in range_constexpr(WQKV_KPARTS):
            rows_store(
                c,
                regs[p],
                KWORDS,
                KGROUPS,
                c["xl"] + p * KWORDS,
                c["xsl"] + p * KGROUPS,
                n,
                lds_row(X_WORDS),
                lds_row(X_GROUPS),
            )
        if const_expr(ti + 1 < len(tiles)):
            regs = x_loads(*tiles[ti + 1])
        if const_expr(ti + 2 == len(tiles)):
            bops = wqb_weights()
        accs = stage_wqkv(c, kops, slots, n, halves)
        wqkv_publish(c, task, accs, slots, order, t0, n, has_wqkv)
    stamp(c, 2)

    # ---- q / kv norms, KV insert
    for t in range(first_task(bid, QKV0), s, BLOCKS):
        stage_qkv(c, t)
    stamp(c, 3)

    # ---- wq_b: every task's rows of a token tile at a time, the next
    # tile's rows in flight
    ropes = [wqb_rope(c, wqb_task, poss[ti], tiles[ti][1]) for ti in range(len(tiles))]
    await_flags(c, c["qrdy"], s, lambda u: u)
    qx = (c["qx8"], c["qx8s"], QX_WORDS, 0, QX_WORDS, Q_RANK // 32, 0, Q_RANK // 32)
    regs = rows_loads(c, *qx, *tiles[0])
    for ti in range_constexpr(len(tiles)):
        t0, n = tiles[ti]
        rows_store(c, regs, QX_WORDS, Q_RANK // 32, c["qxl"], c["qxsl"], n)
        if const_expr(ti + 1 < len(tiles)):
            regs = rows_loads(c, *qx, *tiles[ti + 1])
        stage_wqb(c, wqb_task, bops, t0, n, ropes[ti])
    stamp(c, 4)
