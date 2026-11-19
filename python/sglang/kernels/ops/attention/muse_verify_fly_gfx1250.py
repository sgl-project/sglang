"""Muse Spark target-verify attention for gfx1250: FlyDSL, FP8 WMMA, wave32.

Per rank (TP8): 2 KV heads, GQA 8, head_dim 64, 8 verify tokens, FP8 e4m3 paged KV ([pages, 16, 2, 64]).
Workgroup = (request, KV head, split); 4 waves = the 4 query tiles (16 query columns each: 8 tokens x 8 heads).
The page-table row is staged in LDS once; K / V move in 64-key chunks (4 pages) by async global->LDS copies into a
4-deep ring (one barrier per chunk). Per chunk every wave computes
  S^T = K Q^T  on V_WMMA_F32_16X16X64_FP8_FP8: K rows straight from LDS (32 contiguous dims per lane), Q quantized per
               query row to e4m3 in the prologue (same per-lane byte -> dim map on both operands),
  P^T          from the accumulator registers, e4m3 with a 2^8 bias folded into the exponent (shared by acc and denom,
               so the split reduce's ratio cancels it),
  O^T += V^T P^T on the same FP8 WMMA, V^T via ds_load_tr8_b64 (4 x 16 keys per operand).
Online softmax per query column in the log2 domain (finite M_INIT); masks only on the chunks that need them (sliding
window head, causal tail). Numerics = the FP8 verify mode of gqa_verify_splitkv_2kv (Q/K/P/V in e4m3, FP32 accumulate).
Output: FP32 partials / max / denom ([T, H, SPLITS, 64] / [T, H, SPLITS]), the gqa_verify_splitkv defer_reduce layout.
WINDOW > 0: sliding window (key in (absq - WINDOW, absq]); WINDOW == 0: full causal (key <= absq).
Faster than the FP8 KV-head-pair Triton kernel on the sliding layers at 16 and 64 requests (8K contexts); about even
on the full layers.
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, _to_raw
from flydsl._mlir import ir as _ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import math as _math
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

QL, GQA, HD, PAGE, KVH = 8, 8, 64, 16, 2
WAVE, NW = 32, 4
CHUNK_TILES = 4  # 64 keys per chunk
NS = 4  # async ring depth (chunks)
ROW = 80  # padded LDS row (bytes) per key
M_INIT = -1.0e30
LOG2E = 1.0 / math.log(2.0)
# ds_load_tr8_b64 lane map (measured on gfx1250): lane reads key row / 8-byte dim column of a 16 x 16 tile
TR8_ROW_OF_LANE = lambda lane: (lane // 16) * 8 + lane % 4 + ((lane % 16) // 8) * 4  # noqa: E731
TR8_COL_OF_LANE = lambda lane: (((lane % 16) // 4) % 2) * 8  # noqa: E731


def _gptr(addr):
    return llvm.inttoptr(_ir.Type.parse("!llvm.ptr<1>"), _to_raw(addr))


def _wmma8(a, b, c):
    """V_WMMA_F32_16X16X64_FP8_FP8: a, b = 32 FP8 bytes per lane (v8i32), same per-lane byte -> K map on both"""
    return _rocdl.wmma_f32_16x16x64_fp8_fp8_(
        T.vec(8, T.f32), _to_raw(a), _to_raw(b), _to_raw(c)
    )


def _pack_fp8(vals):
    """32 f32 -> 8 dwords of e4m3 (byte j = vals[j])"""
    out = []
    for d in range(8):
        lo = _rocdl.cvt_pk_fp8_f32(
            T.i32,
            _to_raw(fx.Float32(vals[4 * d])),
            _to_raw(fx.Float32(vals[4 * d + 1])),
            _to_raw(fx.Int32(0)),
            False,
        )
        out.append(
            _rocdl.cvt_pk_fp8_f32(
                T.i32,
                _to_raw(fx.Float32(vals[4 * d + 2])),
                _to_raw(fx.Float32(vals[4 * d + 3])),
                lo,
                True,
            )
        )
    return fx.Vector.from_elements([fx.Int32(x) for x in out], fx.Int32)


def _fmax(a, b):
    return fx.Float32(a).maximumf(fx.Float32(b))


def _fma(a, b, c):
    return fx.Float32(
        _math.fma(
            _to_raw(fx.Float32(a)), _to_raw(fx.Float32(b)), _to_raw(fx.Float32(c))
        )
    )


def _exp2(x):
    return fx.Float32(rocdl.exp2(T.f32, _to_raw(fx.Float32(x))))


@functools.lru_cache(maxsize=64)
def _launcher(splits: int, scale_log2: float, window: int, q_s0: int, heads: int):
    ns = NS
    NDT = 4
    QS0W = q_s0 // 2
    CK = CHUNK_TILES * PAGE  # keys per chunk
    BUF = 2 * CK * ROW  # K + V of one chunk
    full = window == 0
    NTAB = (
        1024 if full else 256
    )  # page-table slots staged in LDS (callers check the row fits)

    @fx.struct
    class Lds:
        kv: fx.Array[fx.Int8, ns * BUF, 16]
        tab: fx.Array[fx.Int32, NTAB, 16]

    @flyc.kernel(name=f"muse_verify_fly_{'full' if full else 'swa'}_gfx1250")
    def kernel(
        q: fx.Tensor,
        kbase: fx.Int64,
        vbase: fx.Int64,
        kvlens: fx.Tensor,
        table: fx.Tensor,
        kscale: fx.Tensor,
        part: fx.Tensor,
        maxi: fx.Tensor,
        den: fx.Tensor,
        tb_s0: fx.Int32,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        lane = tid % WAVE
        wave = tid // WAVE  # = query tile
        n = lane % 16
        h = lane // 16
        split = bid % splits
        pair = bid // splits
        kvh = pair % KVH
        seq = pair // KVH
        r_q = bo.create_buffer_resource(q, max_size=True)
        r_len = bo.create_buffer_resource(kvlens, max_size=True)
        r_tab = bo.create_buffer_resource(table, max_size=True)
        r_ks = bo.create_buffer_resource(kscale, max_size=True)
        r_p = bo.create_buffer_resource(part, max_size=True)
        r_m = bo.create_buffer_resource(maxi, max_size=True)
        r_d = bo.create_buffer_resource(den, max_size=True)

        kvlen = fx.Int32(
            bo.buffer_load(r_len, seq, vec_width=1, dtype=T.i32, is_scalar=True)
        )
        context = kvlen - QL
        if const_expr(full):
            tile_lo = fx.Int32(0)
        else:
            low0 = context - (window - 1)
            low = (low0 > 0).select(low0, fx.Int32(0))
            tile_lo = low // PAGE
        tile_hi = (kvlen + (PAGE - 1)) // PAGE
        nchunks = (tile_hi - tile_lo + (CHUNK_TILES - 1)) // CHUNK_TILES
        per = (nchunks + (splits - 1)) // splits
        c_begin = split * per
        c_end0 = c_begin + per
        c_end = (c_end0 < nchunks).select(c_end0, nchunks)
        c_end = (c_end > c_begin).select(c_end, c_begin)
        scale2 = fx.Float32(scale_log2) * fx.Int32(
            bo.buffer_load(r_ks, 0, vec_width=1, dtype=T.i32, is_scalar=True)
        ).bitcast(fx.Float32)

        # Q^T B operand of this wave's query tile
        tok = wave * 2 + n // 8
        hd = kvh * GQA + n % 8
        dw = (seq * QL + tok) * QS0W + hd * (HD // 2) + h * 16
        qb = []
        for s in range_constexpr(2):
            a = fx.Vector(bo.buffer_load(r_q, dw + s * 8, vec_width=4, dtype=T.i32))
            b = fx.Vector(bo.buffer_load(r_q, dw + s * 8 + 4, vec_width=4, dtype=T.i32))
            qb.append(a.shuffle(b, list(range(8))).bitcast(fx.BFloat16))
        qf = [qb[s_].to(fx.Float32)[e] for s_ in range(2) for e in range(16)]
        qam = fx.Float32(1e-20)
        for x in qf:
            qam = _fmax(qam, _fmax(x, fx.Float32(0.0) - x))
        qam = _fmax(qam, qam.shuffle_xor(fx.Int32(16), fx.Int32(WAVE)))
        qinv = fx.Float32(448.0) / qam
        q8 = _pack_fp8([x * qinv for x in qf])
        qscale_col = qam * fx.Float32(1.0 / 448.0) * scale2
        absq = context + tok
        lo_q = absq - (window - 1)

        lds = fx.SharedAllocator().allocate(Lds).peek()
        base = lds.kv.ptr

        # cooperative chunk load: thread t -> page t // 32, key (t % 32) // 2, 32-byte half t % 2 (K and V)
        lp = tid // 32
        lr = (tid % 32) // 2
        lh = tid % 2
        loff = fx.Int64(lr * 128 + kvh * 64 + lh * 32)

        tabp = lds.tab.ptr
        # stage this request's page-table row from tile_lo into LDS once (slot j -> tile min(tile_lo + j, tile_hi - 1))
        for jj in range_constexpr(NTAB // (NW * WAVE)):
            j = tid + jj * (NW * WAVE)
            tt = tile_lo + j
            tc = (tt < tile_hi).select(tt, tile_hi - 1)
            fx.ptr_store(
                fx.Int32(
                    bo.buffer_load(r_tab, seq * tb_s0 + tc, vec_width=1, dtype=T.i32)
                ),
                tabp + fx.Int64(j),
            )
        gpu.barrier()

        def page_of(c):
            """This thread's page id for chunk c (from the LDS copy; clamped slots hold the last page)."""
            j = c * CHUNK_TILES + lp
            j = (j < NTAB).select(j, NTAB - 1)
            return fx.Int32(fx.ptr_load(tabp + fx.Int64(j), result_type=T.i32))

        zero4 = fx.full(4, 0, fx.Int32)
        row = (lp * PAGE + lr) * ROW + lh * 32

        def issue_chunk(pg, buf):
            """async global->LDS copy of this thread's 2 x 16 B of K and of V for one chunk into ring slot buf"""
            a = fx.Int64(pg) * 2048 + loff
            for kv, src in ((0, kbase), (1, vbase)):
                for j in range_constexpr(2):
                    lp_ = fx.recast_iter(
                        fx.Int8,
                        base + fx.Int64(buf * BUF + kv * CK * ROW + row + j * 16),
                    ).llvm_ptr
                    rocdl.global_load_async_to_lds_b128(_gptr(src + a + j * 16), lp_, 0)

        def zero_tail(c, buf):
            """V rows at or past kvlen -> 0 in LDS (stale FP8 bytes can encode NaN; P = 0 does not cancel NaN)"""
            vvalid = (tile_lo + c * CHUNK_TILES + lp) * PAGE + lr < kvlen
            for j in range_constexpr(2):
                pv = base + fx.Int64(buf * BUF + CK * ROW + row + j * 16)
                old = fx.Vector(fx.ptr_load(pv, result_type=T.vec(4, T.i32)))
                fx.ptr_store(vvalid.select(old, zero4), pv)

        def clampc(c):
            return (c < c_end).select(c, c_begin)

        # prologue: chunks c_begin .. c_begin + ns - 2 in flight
        for i in range_constexpr(ns - 1):
            issue_chunk(page_of(clampc(c_begin + i)), i)
        pg1 = page_of(clampc(c_begin + ns - 1))  # page ids run one issue ahead

        zero8 = fx.full(8, 0.0, fx.Float32)
        tr_off = TR8_ROW_OF_LANE(lane) * ROW + TR8_COL_OF_LANE(lane)
        init = [_to_raw(fx.Float32(M_INIT)), _to_raw(fx.Float32(0.0))] + [
            _to_raw(zero8) for _ in range(NDT)
        ]
        init.append(_to_raw(pg1))

        def step(ci, st, masked):
            m_run = fx.Float32(st[0])
            l_run = fx.Float32(st[1])
            acc = [st[2 + i] for i in range(NDT)]
            pg_n = fx.Int32(st[2 + NDT])
            c = fx.Int32(arith.index_cast(T.i32, _to_raw(ci)))
            buf = (c - c_begin) % ns
            rocdl.s_wait_asynccnt((ns - 2) * 4)  # chunk c landed (this thread's part)
            if const_expr(masked):
                zero_tail(c, buf)
            gpu.barrier()  # every thread's part landed; slot (c-1)%ns free
            issue_chunk(
                pg_n, (c + ns - 1 - c_begin) % ns
            )  # chunk c + ns - 1 (clamped page) into the freed slot
            pg_nn = page_of(clampc(c + ns))
            kb = fx.Int64(buf * BUF)
            vb = fx.Int64(buf * BUF + CK * ROW)
            key0 = (tile_lo + c * CHUNK_TILES) * PAGE
            vals = []
            for kt in range_constexpr(CHUNK_TILES):
                w0 = fx.Vector(
                    fx.ptr_load(
                        base + kb + fx.Int64((kt * PAGE + n) * ROW + h * 32),
                        result_type=T.vec(4, T.i32),
                    )
                )
                w1 = fx.Vector(
                    fx.ptr_load(
                        base + kb + fx.Int64((kt * PAGE + n) * ROW + h * 32 + 16),
                        result_type=T.vec(4, T.i32),
                    )
                )
                cc = fx.Vector(_wmma8(w0.shuffle(w1, list(range(8))), q8, zero8))
                pos0 = key0 + kt * PAGE + h * 8
                for r in range_constexpr(8):
                    pos = pos0 + r
                    if const_expr(not masked):
                        vals.append(cc[r])
                        continue
                    if const_expr(full):
                        ok = pos <= absq
                    else:
                        xr = pos - lo_q
                        ok = (xr >= 0) & (xr < window)
                    vals.append(ok.select(cc[r], fx.Float32(float("-inf"))))
            mx = vals[0]
            for v_ in vals[1:]:
                mx = _fmax(mx, v_)
            mx = _fmax(mx, mx.shuffle_xor(fx.Int32(16), fx.Int32(WAVE)))
            m_new = _fmax(
                m_run, mx * qscale_col
            )  # raw scores x per-column scale (> 0) -> log2 units
            alpha = _exp2(m_run - m_new)
            nb_ = (
                fx.Float32(8.0) - m_new
            )  # P biased by 2^8 inside the exponent (e4m3 normal range)
            ps = [_exp2(_fma(v_, qscale_col, nb_)) for v_ in vals]
            sm = ps[0]
            for p_ in ps[1:]:
                sm = sm + p_
            sm = sm + sm.shuffle_xor(fx.Int32(16), fx.Int32(WAVE))
            l_new = l_run * alpha + sm
            p8 = _pack_fp8(
                ps
            )  # acc and denom share the 2^8 bias; the reduce's ratio cancels it
            for dt in range_constexpr(NDT):
                o = fx.Vector(acc[dt]) * alpha
                raws = []
                for t in range_constexpr(CHUNK_TILES):
                    p_ = fx.to_llvm_ptr(
                        base + vb + fx.Int64((t * PAGE) * ROW + dt * 16 + tr_off)
                    )
                    raws.append(fx.Vector(_rocdl.ds_load_tr8_b64(T.vec(2, T.i32), p_)))
                va = (
                    raws[0]
                    .shuffle(raws[1], [0, 1, 2, 3])
                    .shuffle(raws[2].shuffle(raws[3], [0, 1, 2, 3]), list(range(8)))
                )
                o = fx.Vector(_wmma8(va, p8, o))
                acc[dt] = o
            return (
                [_to_raw(m_new), _to_raw(l_new)]
                + [_to_raw(x) for x in acc]
                + [_to_raw(pg_nn)]
            )

        if const_expr(full):
            ia = c_begin * 0
        else:
            ia = (context + 8 - window - tile_lo * PAGE + (CK - 1)) // CK
        ib = (
            context - (CK - 1) - tile_lo * PAGE + CK
        ) // CK  # chunks c < ib end at or before context
        a_ = (ia > c_begin).select(ia, c_begin)
        a_ = (a_ < c_end).select(a_, c_end)
        b_ = (ib < c_end).select(ib, c_end)
        b_ = (b_ > a_).select(b_, a_)
        st0 = init
        for ci, st in range(c_begin, a_, 1, init=st0):
            r_ = yield step(ci, st, True)
        for ci, st in range(a_, b_, 1, init=list(r_)):
            r2 = yield step(ci, st, False)
        for ci, st in range(b_, c_end, 1, init=list(r2)):
            fin = yield step(ci, st, True)

        rocdl.s_wait_asynccnt(0)
        m_fin = fx.Float32(fin[0])
        l_fin = fx.Float32(fin[1])
        sidx = ((seq * QL + tok) * heads + hd) * splits + split
        for dt in range_constexpr(NDT):
            o = fx.Vector(fin[2 + dt])
            pbase = sidx * HD + dt * 16 + h * 8
            bo.buffer_store(o.shuffle(o, [0, 1, 2, 3]), r_p, pbase)
            bo.buffer_store(o.shuffle(o, [4, 5, 6, 7]), r_p, pbase + 4)
        bo.buffer_store(m_fin, r_m, sidx)
        bo.buffer_store(l_fin, r_d, sidx)

    @flyc.jit
    def launch(
        q: fx.Tensor,
        kbase: fx.Int64,
        vbase: fx.Int64,
        kvlens: fx.Tensor,
        table: fx.Tensor,
        kscale: fx.Tensor,
        part: fx.Tensor,
        maxi: fx.Tensor,
        den: fx.Tensor,
        tb_s0: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(q, kbase, vbase, kvlens, table, kscale, part, maxi, den, tb_s0).launch(
            grid=(fx.Int64(grid), 1, 1), block=(NW * WAVE, 1, 1), stream=stream
        )

    return launch


_ONE = {}


def applies(q, k, v, seqused_k, block_table, window) -> bool:
    """The shapes this kernel serves: TP8 Muse verify (16 Q / 2 KV heads, HD 64), contiguous FP8 paged KV (page 16),
    contiguous Q, int32 lengths / table; full layers only while a table row fits the LDS copy."""
    return (
        q.dtype == torch.bfloat16
        and q.dim() == 3
        and q.shape[1] == KVH * GQA
        and q.shape[2] == HD
        and q.is_contiguous()
        and q.shape[0] == seqused_k.numel() * QL
        and k.dtype == torch.float8_e4m3fn
        and v.dtype == k.dtype
        and k.shape[1:] == (PAGE, KVH, HD)
        and v.shape == k.shape
        and k.stride() == (PAGE * KVH * HD, KVH * HD, HD, 1)
        and v.stride() == k.stride()
        and seqused_k.dtype == torch.int32
        and block_table.dtype == torch.int32
        and (window > 0 or block_table.shape[1] <= 1024)
    )


def muse_verify_fly(
    q,
    k,
    v,
    out,
    seqused_k,
    block_table,
    softmax_scale,
    window,
    k_descale,
    v_descale,
    *,
    splits,
    defer_reduce,
):
    """Verify attention; ``splits`` a power of two (verify_reduce_norm_gate). Returns (partial, max, denom) with
    ``defer_reduce``, else reduces into ``out``."""
    n = seqused_k.numel()
    heads = q.shape[1]
    dev = q.device
    if k_descale is None:
        ks = _ONE.get(dev)
        if ks is None:
            ks = _ONE[dev] = torch.ones(1, dtype=torch.float32, device=dev)
    else:
        ks = k_descale.reshape(-1)[:1].float()
    state = (q.shape[0], heads, splits)
    part = torch.empty((*state, HD), dtype=torch.float32, device=dev)
    maxi = torch.empty(state, dtype=torch.float32, device=dev)
    den = torch.empty(state, dtype=torch.float32, device=dev)
    _run_compiled(
        _launcher(
            int(splits),
            float(softmax_scale) * LOG2E,
            int(window),
            int(q.stride(0)),
            int(heads),
        ),
        q.view(torch.int32),
        int(k.data_ptr()),
        int(v.data_ptr()),
        seqused_k,
        block_table,
        ks,
        part,
        maxi,
        den,
        int(block_table.stride(0)),
        int(n * KVH * splits),
        fx.Stream(torch.cuda.current_stream(dev).cuda_stream),
    )
    if defer_reduce:
        return part, maxi, den
    from sglang.kernels.ops.attention.gqa_verify_splitkv import _verify_reduce

    _verify_reduce[(q.shape[0], heads)](
        part,
        maxi,
        den,
        out,
        v_descale,
        heads,
        splits,
        out.stride(0),
        out.stride(1),
        num_warps=2,
    )
    return out
