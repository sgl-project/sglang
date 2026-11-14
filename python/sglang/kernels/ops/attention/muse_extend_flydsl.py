"""Muse Spark full-attention extend (prefill) on gfx950 in FlyDSL.

FP8 e4m3 Q [tokens, 8 * KV heads, 64] (per-tensor descale) over a paged FP8 e4m3 KV cache [pages, 16, KV heads, 64],
causal, BF16 output. Same API as ``muse_extend_gluon.muse_extend_attention`` (``muse_extend_attention_fly``).

One workgroup = (16-token query block, KV head) = 128 query columns (16 tokens x GQA 8); 4 waves of 32 columns
(4 tokens x 8 heads), 3 waves / SIMD (waves_per_eu=3, ~156 VGPRs). Transposed form on 32x32x64 scaled FP8 MFMAs:
  * S^T = K Q^T per 128-key tile: 4 MFMAs (key tiles of 32), K A-fragments from LDS (2 x ds_read_b128), Q^T as the
    B fragment in 8 VGPRs for the whole kernel (lane half hl holds dims 32 hl .. 32 hl + 31 for both operands).
  * Softmax along keys = accumulator rows, in-lane over the 64 rows a lane holds (v_max3); the rescale runs only
    when a score exceeds the running max (ballot, wave-uniform, llvm.expect -> out-of-line block): P = 448 *
    exp2(s c - m) as in the Gluon kernel. All 32 pk_fma exponent arguments are formed before the exps.
  * P^T accumulators, packed to FP8, are the PV B operand as-is (byte b of half hl = key
    64 ks + 32 (b // 16) + 8 ((b % 16) // 4) + 4 hl + b % 4); V^T A-fragments come from ds_read_b64_tr_b8 in that key
    order. Row sums on the matrix core (ones x P^T, 2 extra MFMAs / tile: cheaper than 32 VALU adds).
  * K / V tiles (128 keys x 64 B each) are LDS-DMA'd (buffer_load ... lds) by the 4 waves, one 16-key page per
    instruction, through a per-page buffer descriptor (64-bit page base, num_records = valid rows): rows past the
    request read as zero (no stale NaN bytes reach P.V) and there is no 2^31 offset limit. 2 LDS stages, one bare
    s_barrier per tile, the loop unrolled per stage with one LLVM alias scope per stage so the compiler waits only
    for the DMA of the stage being read. 16-byte chunks XOR-swizzled by row ((r>>1) + 2 (r>>3)) & 3: the b128 K
    reads and the transposed V reads are bank-conflict free.
  * Page ids are vector loads (SMEM loads placed after LDS DMAs get a conservative vmcnt(0)).
See DEVLOG.md for the version history, measurements and dead ends.
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, _to_raw, buf_base_i64
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import arith
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

GQA = 8
HD = 64
BQ = 16  # query tokens per workgroup
NW = 4  # waves per workgroup
TILE = 128  # keys per tile
PAGE = 16
LOG2E = 1.4426950408889634
M_INIT = -1.0e30
LOG2_448 = math.log2(
    448.0
)  # P = 448 at the running max: the dominant P is exact in FP8
# sum check: rescale when a column's packed-P tile sum exceeds SC_TH (< 448: every P <= sum), re-aim at SC_TGT
SC_TH = 384.0
SC_TGT = 96.0
STAGE = 2 * TILE * HD  # K + V bytes of one LDS stage


def _f32(v):
    return arith.constant(float(v), type=T.f32)


def _i32c(v):
    return arith.constant(int(v), type=T.i32)


def _vec(raw, n, dt=fx.Float32):
    return fx.Vector(_to_raw(raw), shape=n, dtype=dt)


def _cat(a, b, n, dt=fx.Int32):
    return _to_raw(_vec(a, n, dt).shuffle(_vec(b, n, dt), list(range(2 * n))))


def _allreduce32(x_f32, use_max):
    """Combine lanes l and l ^ 32: permlane32_swap(x, x) -> (vdst, vsrc) hold {own, partner} in some order per
    lane, so op(vdst, vsrc) is the all-reduce."""
    i32 = ir.IntegerType.get_signless(32)
    pair = ir.Type.parse("!llvm.struct<(i32, i32)>")
    raw = arith.bitcast(T.i32, _to_raw(x_f32))
    sw = _rocdl.permlane32_swap(pair, raw, raw, False, True)
    a = arith.bitcast(T.f32, llvm.extractvalue(i32, sw, [0]))
    b = arith.bitcast(T.f32, llvm.extractvalue(i32, sw, [1]))
    return arith.maximumf(a, b) if use_max else arith.addf(a, b)


def _tree_max(vals):
    """Max of a list of raw f32 (pairwise tree; the compiler folds it into v_max3)."""
    vals = list(vals)
    while len(vals) > 1:
        nxt = [arith.maximumf(vals[i], vals[i + 1]) for i in range(0, len(vals) - 1, 2)]
        if len(vals) % 2:
            nxt.append(vals[-1])
        vals = nxt
    return vals[0]


def _scope_kwargs(scopes, sc):
    """alias / noalias scope kwargs for an access to LDS stage sc (None: no metadata)."""
    if sc is None or scopes is None:
        return {}
    return dict(
        alias_scopes=ir.ArrayAttr.get([scopes[sc]]),
        noalias_scopes=ir.ArrayAttr.get(
            [scopes[i] for i in range(len(scopes)) if i != sc]
        ),
    )


def _sclamp16(x):
    """clamp(x, 0, 16) for a wave-uniform i32 on the scalar unit."""
    return llvm.InlineAsmOp(
        T.i32,
        [_to_raw(fx.Int32(x))],
        "s_max_i32 $0, $1, 0\n\ts_min_i32 $0, $0, 16",
        "=&s,s,~{scc}",
    ).result


def _uniform(x):
    return fx.Int32(_rocdl.readfirstlane(T.i32, _to_raw(fx.Int32(x))))


@functools.lru_cache(maxsize=64)
def _launcher(
    hk,
    kv_s0,
    kv_s1,
    scale,
    logb,
    wpe,
    bias,
    rescale,
    nbuf,
    sub,
    scheck=False,
    sc_th=SC_TH,
    sc_tgt=SC_TGT,
    ppb=1,
):
    # ppb: 16-row blocks per physical page (physical page = 16 ppb tokens; kv_s0 = physical page stride)
    # scheck: sum-checked lazy softmax (no per-tile max)
    vsum = False  # row sums on the ones-MFMA (True: on the VALU, packed adds)
    exp_scale = float(scale) * LOG2E
    s1w = kv_s1 // 4
    nsub = TILE // sub  # softmax steps per tile
    mts = sub // 32  # 32-key MFMA row tiles per step
    vm_wait = (6 if ppb == 1 else 5) * (
        nbuf - 2
    )  # VMEM ops per iteration: 2 page-id loads + 4 DMAs

    @fx.struct
    class Lds:
        kv: fx.Array[fx.Int8, nbuf * STAGE, 16]

    @flyc.kernel(name="muse_extend_fly")
    def kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        out: fx.Tensor,
        cuq: fx.Tensor,
        seqlens: fx.Tensor,
        table: fx.Tensor,
        qd: fx.Tensor,
        kd: fx.Tensor,
        vd: fx.Tensor,
        num_seqs: fx.Int32,
        nqb: fx.Int32,
        bt_s0: fx.Int32,
        q_s0: fx.Int32,
        o_s0: fx.Int32,
    ):
        pid = fx.block_idx.x
        tid = fx.thread_idx.x
        wave = _uniform(tid // 64)
        lane = tid % 64
        col = lane % 32
        hl = lane // 32
        h = pid % hk
        # longest blocks (last of a request) first
        qb = nqb - 1 - pid // hk

        cu_r = buffer_ops.create_buffer_resource(cuq, max_size=True)
        sl_r = buffer_ops.create_buffer_resource(seqlens, max_size=True)
        bt_r = buffer_ops.create_buffer_resource(table, max_size=True)
        sc_r = [
            buffer_ops.create_buffer_resource(x, max_size=True) for x in (qd, kd, vd)
        ]

        def sload(r, i):
            return fx.Int32(
                buffer_ops.buffer_load(r, i, vec_width=1, dtype=T.i32, is_scalar=True)
            )

        # request owning q-block qb in the grid numbering cu_q[s] // BQ + s
        lo = fx.Int32(0)
        hi = num_seqs
        for _ in range_constexpr(logb):
            mid = (lo + hi) // 2
            go = lo < hi
            val = sload(cu_r, mid) // BQ + mid
            lo = (go & (val <= qb)).select(mid + 1, lo)
            hi = (go & (val > qb)).select(mid, hi)
        seq = lo - 1
        seq = (seq > 0).select(seq, fx.Int32(0))
        q_start = sload(cu_r, seq)
        q_len = sload(cu_r, seq + 1) - q_start
        local = qb - (q_start // BQ + seq)
        valid = (local * BQ < q_len) & (local >= 0)
        seq_len = sload(sl_r, seq)
        ctx = seq_len - q_len
        descale = [arith.bitcast(T.f32, _to_raw(sload(r, 0))) for r in sc_r]
        c_s = arith.mulf(arith.mulf(descale[0], descale[1]), _f32(exp_scale))
        inv_c = arith.divf(_f32(1.0), c_s)
        c_pair = _to_raw(
            fx.Vector.from_elements([fx.Float32(c_s), fx.Float32(c_s)], fx.Float32)
        )

        blk_end = (local * BQ + BQ < q_len).select(local * BQ + BQ, q_len)
        n_tiles = (ctx + blk_end + TILE - 1) // TILE
        n_tiles = valid.select(n_tiles, fx.Int32(0))
        # tiles below `interior` are visible to every query of the block: no masks
        interior = (ctx + local * BQ + 1) // TILE
        interior = (interior < n_tiles).select(interior, n_tiles)
        last_page = (seq_len - 1) // PAGE
        bt_row = seq * bt_s0

        # Q^T B fragment: column col -> token local*16 + 4 wave + col // 8, head h*8 + col % 8; lane half hl holds
        # dims 32 hl .. 32 hl + 31 (rows past the request read as zero through the descriptor bound)
        q_base = buf_base_i64(q)
        q_rs = buffer_ops.create_buffer_resource_from_addr(
            _to_raw(q_base + fx.Int64(q_start) * fx.Int64(q_s0)),
            num_records_bytes=_to_raw(fx.Int64(q_len) * fx.Int64(q_s0)),
        )
        tok = local * BQ + wave * 4 + col // GQA
        head = h * GQA + col % GQA
        q_off = (tok * q_s0 + head * HD + hl * 32) // 4
        qf = _cat(
            buffer_ops.buffer_load(q_rs, q_off, vec_width=4, dtype=T.i32),
            buffer_ops.buffer_load(q_rs, q_off + 4, vec_width=4, dtype=T.i32),
            4,
        )
        qpos = ctx + tok

        k_base = buf_base_i64(k)
        v_base = buf_base_i64(v)

        lds = fx.SharedAllocator().allocate(Lds).peek()
        lptr = lds.kv.ptr

        # one alias scope per LDS stage: the backend tracks LDS-DMA waits per scope, so reads of stage s do not
        # wait for the DMA into the other stages
        _scopes = ir.Attribute.parse(
            "["
            + ", ".join(
                f"#llvm.alias_scope<id = distinct[{i + 1}]<>, domain = #llvm.alias_scope_domain<id = distinct[0]<>>>"
                for i in range(nbuf)
            )
            + "]"
        )
        _sc = [_scopes[i] for i in range(nbuf)]

        def _scope_kw(sc):
            return _scope_kwargs(_sc, sc)

        # loader: wave w fills rows 16 (2w + n) .. + 15 (= page 8 j + 2 w + n of the request) of K and V; lane ->
        # row 16 p + lane // 4, physical chunk lane % 4 holding logical chunk (lane % 4) ^ swz(row)
        swz_w = ((lane // 8) + 2 * (lane // 32)) % 4
        lc_w = fx.Int32(arith.xori(_to_raw(lane % 4), _to_raw(swz_w)))
        voff_b = ((lane // 4) * s1w + h * (HD // 4) + lc_w * 4) * 4

        def pages(tile):
            # page ids of tile's two pages for this wave, clamped into the table row; vector loads with a uniform
            # address (SMEM loads after LDS DMAs get a conservative vmcnt(0) from the compiler)
            ids = []
            # ppb >= 2: the wave's two 16-row blocks (2 k, 2 k + 1) share one physical page -> one id load
            for n in range_constexpr(2 if ppb == 1 else 1):
                pi = tile * 8 + wave * 2 + n
                pc = (pi < last_page).select(pi, last_page)
                if fx.const_expr(ppb > 1):
                    pc = pc // ppb
                ids.append(
                    _to_raw(
                        buffer_ops.buffer_load(
                            bt_r, bt_row + pc, vec_width=1, dtype=T.i32
                        )
                    )
                )
            return ids

        wave_lds = _uniform(wave * 2048)

        def issue_dma(tile, stage_off, ids, sc):
            # LDS DMA: lane l of instruction (tensor ui, page n) lands at M0 + 16 l. Per-page descriptor: 64-bit page
            # base, num_records = valid rows * row stride (rows past the request read as zero: no stale NaN bytes
            # reach P.V, no 2^31 offset limit)
            nrec = []
            sub_off = []
            if fx.const_expr(ppb > 1):
                # 16-row block inside the physical page (block index clamped like the page id); block 2k+1 is the
                # next 16 rows of the same page
                pi = tile * 8 + wave * 2
                pc = (pi < last_page).select(pi, last_page)
                o0 = fx.Int64(_uniform(ids[0])) * fx.Int64(kv_s0) + fx.Int64(
                    (pc % ppb) * (PAGE * kv_s1)
                )
                sub_off = [o0, o0 + fx.Int64(PAGE * kv_s1)]
            for n in range_constexpr(2):
                nv = seq_len - (tile * 8 + wave * 2 + n) * PAGE
                # clamp(nv, 0, 16) on the scalar unit (the compiler picks v_med3 + readfirstlane otherwise)
                nrec.append(fx.Int32(_sclamp16(nv)) * kv_s1)
            for ui, base in enumerate((k_base, v_base)):
                for n in range_constexpr(2):
                    rs = buffer_ops.create_buffer_resource_from_addr(
                        _to_raw(
                            base
                            + (
                                sub_off[n]
                                if ppb > 1
                                else fx.Int64(_uniform(ids[n])) * fx.Int64(kv_s0)
                            )
                        ),
                        num_records_bytes=_to_raw(fx.Int64(nrec[n])),
                    )
                    dst = wave_lds + (stage_off + ui * (TILE * HD) + n * 1024)
                    _rocdl.raw_ptr_buffer_load_lds(
                        rs,
                        fx.to_llvm_ptr(lptr + fx.Int64(dst)),
                        _i32c(16),
                        _to_raw(voff_b),
                        _i32c(0),
                        _i32c(0),
                        **_scope_kw(sc),
                    )

        # reader offsets. K: lane (col, hl) reads key 32 mt + col, logical chunks 2 hl, 2 hl + 1
        swz_k = ((col // 2) + 2 * ((col // 8) % 2)) % 4
        k_rd = [
            col * HD
            + fx.Int32(arith.xori(_to_raw(fx.Int32(2) * hl + u), _to_raw(swz_k))) * 16
            for u in range_constexpr(2)
        ]
        # V (transposed): lane (G, i) of read r addresses key 64 ks + 32 (r // 2) + 16 (r % 2) + key0, logical
        # chunk 2 dt + G % 2, byte 8 (i % 2)
        gq = lane // 16
        li = lane % 16
        kk = li // 2
        key0 = 8 * (kk // 4) + 4 * hl + kk % 4
        swz_v = (2 * hl + (kk % 4) // 2 + 2 * (kk // 4)) % 4
        v_rd = [
            TILE * HD
            + key0 * HD
            + fx.Int32(arith.xori(_to_raw(2 * dt + gq % 2), _to_raw(swz_v))) * 16
            + (li % 2) * 8
            for dt in range_constexpr(2)
        ]

        s127 = _i32c(127)
        zero16 = _to_raw(fx.full(16, 0.0, fx.Float32))
        ones8 = _to_raw(fx.full(8, 0x38383838, fx.Int32))  # FP8 e4m3 1.0 bytes

        def mfma(a, b, acc):
            return fx.rocdl.mfma_scale_f32_32x32x64_f8f6f4(
                T.vec(16, T.f32), [a, b, acc, 0, 0, 0, s127, 0, s127]
            )

        def softmax_step(j, stage_off, si, o_acc, negm, thr, l_run, masked, sc):
            # keys 64 * (mts * si) .. of the tile: mts row tiles of 32 keys
            kfr = []
            for t in range_constexpr(mts):
                mt = si * mts + t
                parts = [
                    llvm.load(
                        T.vec(4, T.i32),
                        fx.to_llvm_ptr(
                            lptr + fx.Int64(stage_off + mt * 32 * HD + k_rd[u])
                        ),
                        alignment=16,
                        **_scope_kw(sc),
                    )
                    for u in range_constexpr(2)
                ]
                kfr.append(_cat(parts[0], parts[1], 4))
            s = [mfma(kfr[t], qf, zero16) for t in range_constexpr(mts)]
            ninf = fx.Float32(float("-inf"))
            flat = []
            for t in range_constexpr(mts):
                mt = si * mts + t
                x = _vec(s[t], 16)
                if fx.const_expr(masked):
                    lim = qpos - j * TILE - 4 * hl
                    flat.extend(
                        _to_raw(
                            (fx.Int32(32 * mt + 8 * (r // 4) + r % 4) <= lim).select(
                                x[r], ninf
                            )
                        )
                        for r in range(16)
                    )
                else:
                    flat.extend(_to_raw(x[r]) for r in range(16))
            mx = _tree_max(flat)
            need = arith.cmpf(arith.CmpFPredicate.OGT, mx, _to_raw(thr))
            vote = fx.Int64(_rocdl.ballot(T.i64, need))
            # rare: llvm.expect(false) lets block placement move the rescale out of line (the common path falls
            # through instead of taking a branch per check)
            hit = fx.Boolean(
                llvm.intr_expect(
                    _to_raw(vote != fx.Int64(0)),
                    arith.constant(0, type=ir.IntegerType.get_signless(1)),
                )
            )
            if hit:
                mxa = _allreduce32(mx, True)
                m_old = arith.subf(_f32(bias), _to_raw(_vec(negm, 2)[0]))
                mn = arith.maximumf(m_old, arith.mulf(mxa, c_s))
                alpha = fx.rocdl.exp2(T.f32, arith.subf(m_old, mn))
                nm = arith.subf(_f32(bias), mn)
                negm = _to_raw(
                    fx.Vector.from_elements(
                        [fx.Float32(nm), fx.Float32(nm)], fx.Float32
                    )
                )
                thr = arith.mulf(arith.addf(mn, _f32(rescale)), inv_c)
                l_run = _to_raw(_vec(l_run, 2) * fx.Float32(alpha))
                o_acc = [_to_raw(_vec(o, 16) * fx.Float32(alpha)) for o in o_acc]
            # P = exp2(s c + bias - m_used): all exponent arguments first (in place over S; interleaving
            # fma -> exp through one temporary pair serializes on the trans-unit operand reads), then exp + pack
            zs = []
            for i in range_constexpr(len(flat) // 2):
                pr = fx.Vector.from_elements(
                    [fx.Float32(flat[2 * i]), fx.Float32(flat[2 * i + 1])], fx.Float32
                )
                zs.append(_vec(fmath.fma(_to_raw(pr), c_pair, _to_raw(negm)), 2))
            fx.rocdl.sched_barrier(0)
            words = []
            psum = None
            for t in range_constexpr(mts):
                ps = []
                for r in range_constexpr(0, 16, 2):
                    z = zs[t * 8 + r // 2]
                    ps.append(_to_raw(fx.rocdl.exp2(T.f32, _to_raw(z[0]))))
                    ps.append(_to_raw(fx.rocdl.exp2(T.f32, _to_raw(z[1]))))
                    if fx.const_expr(vsum):
                        pp = fx.Vector.from_elements(
                            [fx.Float32(ps[-2]), fx.Float32(ps[-1])], fx.Float32
                        )
                        psum = pp if psum is None else psum + pp
                for w4 in range_constexpr(4):
                    lo_w = _rocdl.cvt_pk_fp8_f32(
                        T.i32,
                        ps[4 * w4],
                        ps[4 * w4 + 1],
                        llvm.mlir_poison(T.i32),
                        False,
                    )
                    words.append(
                        _rocdl.cvt_pk_fp8_f32(
                            T.i32, ps[4 * w4 + 2], ps[4 * w4 + 3], lo_w, True
                        )
                    )
            # P^T B operand of k-step ks (64 keys = row tiles 2 ks, 2 ks + 1)
            o_new = list(o_acc)
            lt = None
            for kq in range_constexpr(mts // 2):
                ks = si * (mts // 2) + kq
                pb = _to_raw(
                    fx.Vector.from_elements(
                        [fx.Int32(w) for w in words[8 * kq : 8 * kq + 8]], fx.Int32
                    )
                )
                # row sums on the matrix core first (ones x P^T): their consumer then sits after the PV MFMAs
                if fx.const_expr(not vsum):
                    lt = mfma(ones8, pb, zero16 if kq == 0 else lt)
                for dt in range_constexpr(2):
                    rd = []
                    for r in range_constexpr(4):
                        off = (
                            stage_off
                            + v_rd[dt]
                            + (64 * ks + 32 * (r // 2) + 16 * (r % 2)) * HD
                        )
                        raw = _rocdl.ds_read_tr8_b64(
                            T.vec(2, T.i32),
                            fx.to_llvm_ptr(lptr + fx.Int64(off)),
                            **_scope_kw(sc),
                        ).result
                        rd.append(raw)
                    va = _cat(_cat(rd[0], rd[1], 2), _cat(rd[2], rd[3], 2), 4)
                    o_new[dt] = mfma(va, pb, o_new[dt])
            if fx.const_expr(vsum):
                l_run = _to_raw(_vec(l_run, 2) + psum)
            else:
                l_run = _to_raw(
                    _vec(l_run, 2)
                    + fx.Vector.from_elements(
                        [_vec(lt, 16)[0], fx.Float32(0.0)], fx.Float32
                    )
                )
            return o_new, negm, thr, l_run

        def scores(j, stage_off, masked, sc):
            kfr = []
            for mt in range_constexpr(4):
                parts = [
                    llvm.load(
                        T.vec(4, T.i32),
                        fx.to_llvm_ptr(
                            lptr + fx.Int64(stage_off + mt * 32 * HD + k_rd[u])
                        ),
                        alignment=16,
                        **_scope_kw(sc),
                    )
                    for u in range_constexpr(2)
                ]
                kfr.append(_cat(parts[0], parts[1], 4))
            s = [mfma(kfr[mt], qf, zero16) for mt in range_constexpr(4)]
            ninf = fx.Float32(float("-inf"))
            flat = []
            for mt in range_constexpr(4):
                x = _vec(s[mt], 16)
                if fx.const_expr(masked):
                    lim = qpos - j * TILE - 4 * hl
                    flat.extend(
                        _to_raw(
                            (fx.Int32(32 * mt + 8 * (r // 4) + r % 4) <= lim).select(
                                x[r], ninf
                            )
                        )
                        for r in range(16)
                    )
                else:
                    flat.extend(_to_raw(x[r]) for r in range(16))
            return flat

        def exps(flat, negm, scale_pair=None):
            # z = s c + negm (packed, in place), p = exp2(z) (fp32 list)
            zs = []
            for i in range_constexpr(32):
                pr = fx.Vector.from_elements(
                    [fx.Float32(flat[2 * i]), fx.Float32(flat[2 * i + 1])], fx.Float32
                )
                zs.append(_vec(fmath.fma(_to_raw(pr), c_pair, _to_raw(negm)), 2))
            fx.rocdl.sched_barrier(0)
            ps = []
            for i in range_constexpr(32):
                ps.append(_to_raw(fx.rocdl.exp2(T.f32, _to_raw(zs[i][0]))))
                ps.append(_to_raw(fx.rocdl.exp2(T.f32, _to_raw(zs[i][1]))))
            return ps

        def pack(ps):
            words = []
            for w in range_constexpr(16):
                lo_w = _rocdl.cvt_pk_fp8_f32(
                    T.i32, ps[4 * w], ps[4 * w + 1], llvm.mlir_poison(T.i32), False
                )
                words.append(
                    _rocdl.cvt_pk_fp8_f32(
                        T.i32, ps[4 * w + 2], ps[4 * w + 3], lo_w, True
                    )
                )
            pb = [
                _to_raw(
                    fx.Vector.from_elements(
                        [fx.Int32(w) for w in words[8 * ks : 8 * ks + 8]], fx.Int32
                    )
                )
                for ks in range_constexpr(2)
            ]
            lt = mfma(ones8, pb[0], zero16)
            lt = mfma(ones8, pb[1], lt)
            return pb, lt

        def softmax_sc(j, stage_off, o_acc, negm, l_run, masked, sc):
            # sum-checked lazy softmax: no per-tile max. P = exp2(s c + negm) packed to FP8; the ones-MFMA column
            # sum T of the packed P bounds every P (P >= 0), so T <= SC_TH (unordered compare: NaN / inf trigger)
            # proves the FP8 pack exact-range. Otherwise (rare, wave-uniform): recompute S, rescale to the true
            # max, and shift further by log2(T / SC_TGT) so a flat tile does not re-trigger.
            flat = scores(j, stage_off, masked, sc)
            ps = exps(flat, negm)
            pb, lt = pack(ps)
            tsum = _vec(lt, 16)[0]
            need = arith.cmpf(arith.CmpFPredicate.UGT, _to_raw(tsum), _f32(sc_th))
            vote = fx.Int64(_rocdl.ballot(T.i64, need))
            hit = fx.Boolean(
                llvm.intr_expect(
                    _to_raw(vote != fx.Int64(0)),
                    arith.constant(0, type=ir.IntegerType.get_signless(1)),
                )
            )
            if hit:
                flat2 = scores(j, stage_off, masked, sc)
                mx = _allreduce32(_tree_max(flat2), True)
                m_old = arith.subf(_f32(bias), _to_raw(_vec(negm, 2)[0]))
                mn = arith.maximumf(m_old, arith.mulf(mx, c_s))
                nm = arith.subf(_f32(bias), mn)
                nmp = _to_raw(
                    fx.Vector.from_elements(
                        [fx.Float32(nm), fx.Float32(nm)], fx.Float32
                    )
                )
                ps2 = exps(flat2, nmp)
                acc = None
                for i in range_constexpr(32):
                    pp = fx.Vector.from_elements(
                        [fx.Float32(ps2[2 * i]), fx.Float32(ps2[2 * i + 1])], fx.Float32
                    )
                    acc = pp if acc is None else acc + pp
                tot = _allreduce32(arith.addf(_to_raw(acc[0]), _to_raw(acc[1])), False)
                # delta = max(0, log2(T / SC_TGT)) (T > 0: the column sees key 0 .. qpos, or is a padding column)
                lg = _rocdl.log(
                    T.f32, arith.maximumf(tot, _f32(1e-30))
                )  # v_log_f32 = log2
                dl = arith.maximumf(_f32(0.0), arith.subf(lg, _f32(math.log2(sc_tgt))))
                sh = fx.rocdl.exp2(T.f32, arith.subf(_f32(0.0), dl))
                ps2 = [arith.mulf(x, sh) for x in ps2]
                mn = arith.addf(mn, dl)
                alpha = fx.rocdl.exp2(T.f32, arith.subf(m_old, mn))
                nm = arith.subf(_f32(bias), mn)
                negm = _to_raw(
                    fx.Vector.from_elements(
                        [fx.Float32(nm), fx.Float32(nm)], fx.Float32
                    )
                )
                l_run = _to_raw(_vec(l_run, 2) * fx.Float32(alpha))
                o_acc = [_to_raw(_vec(o, 16) * fx.Float32(alpha)) for o in o_acc]
                pb, lt = pack(ps2)
            o_new = list(o_acc)
            for ks in range_constexpr(2):
                for dt in range_constexpr(2):
                    rd = []
                    for r in range_constexpr(4):
                        off = (
                            stage_off
                            + v_rd[dt]
                            + (64 * ks + 32 * (r // 2) + 16 * (r % 2)) * HD
                        )
                        raw = _rocdl.ds_read_tr8_b64(
                            T.vec(2, T.i32),
                            fx.to_llvm_ptr(lptr + fx.Int64(off)),
                            **_scope_kw(sc),
                        ).result
                        rd.append(raw)
                    va = _cat(_cat(rd[0], rd[1], 2), _cat(rd[2], rd[3], 2), 4)
                    o_new[dt] = mfma(va, pb[ks], o_new[dt])
            l_run = _to_raw(
                _vec(l_run, 2)
                + fx.Vector.from_elements(
                    [_vec(lt, 16)[0], fx.Float32(0.0)], fx.Float32
                )
            )
            return o_new, negm, l_run

        def tile_barrier():
            # VMEM ops per iteration, in order: 2 page-id loads then 4 DMAs. Tile j's DMAs were issued nbuf - 1
            # iterations ago; wait until only the later iterations' ops are in flight, then a bare s_barrier
            # (gpu.barrier's release fence would wait for every DMA in flight)
            fx.rocdl.sched_barrier(0)
            fx.rocdl.s_waitcnt(vmcnt=vm_wait)
            _rocdl.s_barrier()
            fx.rocdl.sched_barrier(0)

        def step(j, st, stage, masked):
            # stage: Python int (static LDS stage, scoped reads) or None (dynamic j % nbuf, unscoped)
            o_acc = list(st[0:2])
            negm, thr, l_run = st[2], st[3], st[4]
            ids = list(st[5:7])
            tile_barrier()
            nids = pages(j + nbuf)
            fx.rocdl.sched_barrier(0)
            if fx.const_expr(stage is None):
                issue_dma(j + nbuf - 1, ((j + nbuf - 1) % nbuf) * STAGE, ids, None)
                stage_off = (j % nbuf) * STAGE
                sc = None
            else:
                nst = (stage + nbuf - 1) % nbuf
                issue_dma(j + nbuf - 1, nst * STAGE, ids, nst)
                stage_off = stage * STAGE
                sc = stage
            fx.rocdl.sched_barrier(0)
            if fx.const_expr(scheck):
                o_acc, negm, l_run = softmax_sc(
                    j, stage_off, o_acc, negm, l_run, masked, sc
                )
            else:
                for si in range_constexpr(nsub):
                    o_acc, negm, thr, l_run = softmax_step(
                        j, stage_off, si, o_acc, negm, thr, l_run, masked, sc
                    )
            return (
                [_to_raw(x) for x in o_acc]
                + [_to_raw(negm), _to_raw(thr), _to_raw(l_run)]
                + [_to_raw(x) for x in nids]
            )

        # prologue: tiles 0 .. nbuf - 2 in flight, page ids of tile nbuf - 1
        for t in range_constexpr(nbuf - 1):
            issue_dma(fx.Int32(t), t * STAGE, pages(fx.Int32(t)), t)
        ids0 = pages(fx.Int32(nbuf - 1))
        nm0 = _to_raw(
            fx.Vector.from_elements(
                [fx.Float32(bias - M_INIT), fx.Float32(bias - M_INIT)], fx.Float32
            )
        )
        thr0 = arith.mulf(_f32(M_INIT + rescale), inv_c)
        l0 = _to_raw(fx.full(2, 0.0, fx.Float32))
        init = [zero16, zero16, nm0, thr0, l0] + ids0
        # interior tiles in groups of nbuf with static LDS stages, then the rest with dynamic stages
        ngr = interior // nbuf
        for gi, st in range(0, ngr, 1, init=init):
            j0 = fx.Int32(arith.index_cast(T.i32, _to_raw(gi))) * nbuf
            for u in range_constexpr(nbuf - 1):
                st = step(j0 + u, st, u, False)
            fin0 = yield step(j0 + nbuf - 1, st, nbuf - 1, False)
        for jj, st in range(ngr * nbuf, interior, 1, init=list(fin0)):
            j = fx.Int32(arith.index_cast(T.i32, _to_raw(jj)))
            fin = yield step(j, st, None, False)
        for jj, st in range(0, n_tiles - interior, 1, init=list(fin)):
            j = interior + fx.Int32(arith.index_cast(T.i32, _to_raw(jj)))
            fin2 = yield step(j, st, None, True)
        o_acc = list(fin2[0:2])
        l_run = _vec(fin2[4], 2)
        lsum = arith.addf(_to_raw(l_run[0]), _to_raw(l_run[1]))
        if fx.const_expr(vsum):
            lsum = _allreduce32(lsum, False)
        inv = arith.divf(descale[2], lsum)

        o_base = buf_base_i64(out)
        o_rs = buffer_ops.create_buffer_resource_from_addr(
            _to_raw(o_base + fx.Int64(q_start) * fx.Int64(o_s0 * 2)),
            num_records_bytes=_to_raw(fx.Int64(q_len) * fx.Int64(o_s0 * 2)),
        )
        o_row = tok * o_s0 + head * HD + hl * 4
        for dt in range_constexpr(2):
            ov = _vec(o_acc[dt], 16) * fx.Float32(inv)
            for i in range_constexpr(4):
                seg = ov.shuffle(ov, [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]).to(
                    fx.BFloat16
                )
                buffer_ops.buffer_store(_to_raw(seg), o_rs, o_row + 32 * dt + 8 * i)

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        out: fx.Tensor,
        cuq: fx.Tensor,
        seqlens: fx.Tensor,
        table: fx.Tensor,
        qd: fx.Tensor,
        kd: fx.Tensor,
        vd: fx.Tensor,
        num_seqs: fx.Int32,
        nqb: fx.Int32,
        bt_s0: fx.Int32,
        q_s0: fx.Int32,
        o_s0: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            q,
            k,
            v,
            out,
            cuq,
            seqlens,
            table,
            qd,
            kd,
            vd,
            num_seqs,
            nqb,
            bt_s0,
            q_s0,
            o_s0,
            value_attrs={"rocdl.waves_per_eu": wpe} if wpe else None,
        ).launch(grid=(fx.Int64(grid), 1, 1), block=(NW * 64, 1, 1), stream=stream)

    return launch


def muse_extend_attention_fly(
    q,
    k_cache,
    v_cache,
    out,
    cu_seqlens_q,
    seq_lens,
    block_table,
    q_descale,
    k_descale,
    v_descale,
    scale,
    window=0,
    wpe=3,
    bias=LOG2_448,
    rescale=0.0,
    nbuf=2,
    sub=128,
    sc_th=440.0,
    sc_tgt=400.0,
    sum_check=False,
):
    """Causal full-attention extend: ``out [tokens, heads, 64]`` (BF16) from FP8 ``q`` and the paged FP8 cache
    (same contract as ``muse_extend_gluon.muse_extend_attention``).

    Defaults: P = 448 * exp2(s c - m) with m the running max (the rescale runs only when some score exceeds it,
    wave-uniform branch), as accurate as the Gluon kernel. ``sum_check=True``: opt-in faster variant without the
    per-tile max (P range proven by the ones-MFMA tile sum, rescale on overflow); ~3% faster at long prefixes,
    rel-L2 up to ~5% higher than Gluon on peaky test data."""
    import torch

    assert window == 0, "sliding window not implemented in the FlyDSL extend kernel"
    n = q.shape[0]
    hk = k_cache.shape[2]
    bs = seq_lens.numel()
    assert (
        q.shape[1] == GQA * hk
        and q.shape[2] == HD
        and k_cache.shape[1] in (16, 32, 64, 128)
    )
    assert (
        q.stride(2) == 1
        and q.stride(1) == HD
        and out.stride(2) == 1
        and out.stride(1) == HD
    )
    assert (
        k_cache.stride() == v_cache.stride()
        and k_cache.stride(3) == 1
        and k_cache.stride(2) == HD
    )
    if seq_lens.dtype != torch.int32:
        seq_lens = seq_lens.to(torch.int32)
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    if cu_seqlens_q.dtype != torch.int32:
        cu_seqlens_q = cu_seqlens_q.to(torch.int32)
    nqb = n // BQ + bs
    one = None
    descs = []
    for d in (q_descale, k_descale, v_descale):
        if d is None:
            if one is None:
                one = torch.ones(1, dtype=torch.float32, device=q.device)
            d = one
        descs.append(d)
    _run_compiled(
        _launcher(
            hk,
            k_cache.stride(0),
            k_cache.stride(1),
            float(scale),
            max(1, bs.bit_length()),
            wpe,
            bias,
            rescale,
            nbuf,
            sub,
            bool(sum_check),
            sc_th,
            sc_tgt,
            k_cache.shape[1] // PAGE,
        ),
        q.view(torch.uint8),
        k_cache.view(torch.uint8),
        v_cache.view(torch.uint8),
        out,
        cu_seqlens_q,
        seq_lens,
        block_table,
        *descs,
        int(bs),
        int(nqb),
        int(block_table.stride(0)),
        int(q.stride(0)),
        int(out.stride(0)),
        int(nqb * hk),
        fx.Stream(torch.cuda.current_stream(q.device).cuda_stream),
    )
    return out
