"""Muse Spark full-layer target verify on gfx950 in FlyDSL: 8 verify tokens x GQA 8 (64 query columns) per KV head,
one wave per KV head, one workgroup per (request, split, group of heads_per_group KV heads); BF16 Q, FP8 e4m3 paged
KV [pages, 16, kv_heads, 64], split-KV FP32 partials. Exact mode: BF16 QK / PV (K and V upcast from FP8); FP8 modes:
FP8 Q x K and / or FP8 P x V MFMAs straight on the cache bytes.

Transposed form per wave: S^T = K Q^T (keys x query columns) and O^T = V^T P^T on 16x16x32 BF16 MFMAs.
  * K is the A operand straight from HBM: lane (g, i) loads 16 contiguous bytes (dims 16g..16g+15) of key i of a
    16-key page; dims are permuted consistently in Q (k-step ks holds dims 16g + 8ks + 0..7).
  * Softmax runs along keys = MFMA rows: in-thread over the 8 rows of a lane, across the four 16-lane groups with
    two permlane swaps, and only when some score clears the running max by 2^8 (lazy rescale, exact: P <= 256).
  * P^T's accumulator registers are the PV B operand once the 32 keys of a chunk are relabelled
    (k-position 8g + i <-> key i < 4 ? 4g + i : 12 + 4g + i); V is written to wave-private LDS in that order with
    contiguous ds_write_b128 and read back as the V^T A operand with ds_read_b64_tr_b8 (16-byte chunks XOR-swizzled
    by row so the transposed reads are bank-conflict free).
  * Page ids are loaded one pair ahead of use (vector loads in order with K / V by default, or SMEM).
  * K / V go through buffer loads (32-bit byte offsets), or 64-bit global loads when a layer's pool reaches 2^31
    bytes (``addr64``: 3-7% slower, so only where the offsets would overflow).
  * NS register load slots refilled right after use keep ~NS chunks in flight per wave.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import rocdl as _rocdl
from flydsl.expr import arith, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, _to_raw

QL = 8
GQA = 8
HD = 64
PAGE = 16
CHUNK = 32
LOG2E = 1.4426950408889634
# Lazy rescale threshold (log2 domain): P = exp2(s - m_used) <= 2^8, exact in FP32 accumulators.
RESCALE = 8.0
# FP8 P (PVF8): P biased by 2^8 into e4m3's normal range (the bias rides through acc and denom and cancels in the
# reduce); the lazy threshold shrinks so the biased P stays <= 2^8.75 < 448.
PV_BIAS = 8.0
RESCALE_PVF8 = 0.75
FP8_MAX = 448.0
# Finite "minus infinity" for the running max so fully masked chunks give exp2(-inf - m) = 0, never NaN.
M_INIT = -1.0e30
LDS_CU = 160 * 1024  # gfx950 LDS per CU


def _f32(v):
    return arith.constant(float(v), type=T.f32)


def _fence(vmem_only):
    # vmem_only: keep the K / V / page-id loads in issue order (their vmcnt waits depend on it) but let ALU, MFMA and
    # LDS work move across chunk boundaries
    fx.rocdl.sched_barrier(0x78F if vmem_only else 0)


def _swap(op_name, x):
    """Partner values across lane groups: permlane{32,16}_swap of x with itself -> (vdst, vsrc) pair, i32."""
    i32 = ir.IntegerType.get_signless(32)
    pair = ir.Type.parse("!llvm.struct<(i32, i32)>")
    raw = _to_raw(x)
    sw = getattr(_rocdl, op_name)(pair, raw, raw, False, True)
    return llvm.extractvalue(i32, sw, [0]), llvm.extractvalue(i32, sw, [1])


def _allreduce_groups(x_f32, use_max):
    """Reduce an f32 across lanes l, l^16, l^32, l^48 (the four 16-lane groups)."""
    v = _to_raw(x_f32)
    for op in ("permlane32_swap", "permlane16_swap"):
        bits = arith.bitcast(T.i32, v)
        a, b = _swap(op, bits)
        fa = arith.bitcast(T.f32, a)
        fb = arith.bitcast(T.f32, b)
        v = arith.maximumf(fa, fb) if use_max else arith.addf(fa, fb)
    return v


def _cvt8(lo_word, hi_word):
    """8 FP8 e4m3 bytes (two i32) -> v8bf16 in byte order (exact upcast, scale 1)."""
    one = _f32(1.0)
    parts = []
    for w in (lo_word, hi_word):
        for sel in (False, True):
            pr = _rocdl.cvt_scalef32_pk_bf16_fp8(T.vec(2, T.bf16), _to_raw(w), one, sel)
            parts.append(fx.Vector(pr).bitcast(fx.Int32)[0])
    return _to_raw(fx.Vector.from_elements(parts, fx.Int32).bitcast(fx.BFloat16))


def _global_load16(addr):
    """16 bytes (vector<4 x i32>) at the 64-bit global byte address ``addr``."""
    ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
    return llvm.LoadOp(T.vec(4, T.i32), _to_raw(ptr), alignment=16).result


def _i64(lo, hi):
    return _to_raw(fx.Vector.from_elements([fx.Int32(lo), fx.Int32(hi)], fx.Int32).bitcast(fx.Int64)[0])


def _fp8x8(vals):
    """8 f32 (already inside +-448) -> 8 OCP e4m3 bytes packed as i64."""
    words = []
    for w in range(2):
        # the high half is written by the second convert: start from poison, not a materialized zero
        lo = _rocdl.cvt_pk_fp8_f32(
            T.i32, _to_raw(vals[4 * w]), _to_raw(vals[4 * w + 1]), llvm.mlir_poison(T.i32), False
        )
        words.append(_rocdl.cvt_pk_fp8_f32(T.i32, _to_raw(vals[4 * w + 2]), _to_raw(vals[4 * w + 3]), lo, True))
    return _i64(words[0], words[1])


def _v2(x):
    return fx.Vector.from_elements([fx.Float32(x), fx.Float32(x)], fx.Float32)


def _half(vec4, hi):
    return fx.Vector(vec4, shape=4, dtype=fx.Float32).shuffle(fx.Vector(vec4, shape=4, dtype=fx.Float32), [2, 3] if hi else [0, 1])


def _max4(vec):
    return arith.maximumf(
        arith.maximumf(_to_raw(vec[0]), _to_raw(vec[1])), arith.maximumf(_to_raw(vec[2]), _to_raw(vec[3]))
    )


def _sum4(vec):
    return arith.addf(arith.addf(_to_raw(vec[0]), _to_raw(vec[1])), arith.addf(_to_raw(vec[2]), _to_raw(vec[3])))


@functools.lru_cache(maxsize=64)
def _launcher(heads, kvh, hpg, kv_s0, kv_s1, scale, waves_per_eu, vec_ids, NS, q_lds, qf8, pvf8, fence_vmem, wg_per_cu, addr64):
    W = hpg
    groups = kvh // hpg
    lds_wave = 2 * CHUNK * HD  # two V stages of 32 rows x 64 bytes per wave
    exp_scale = float(scale) * LOG2E
    s0w, s1w = kv_s0 // 4, kv_s1 // 4  # dword strides

    q_wave = 8 * 64 * 16  # 8 B-fragments x 64 lanes x 16 bytes
    rescale = RESCALE_PVF8 if pvf8 else RESCALE
    p_bias = PV_BIAS if pvf8 else 0.0

    @fx.struct
    class Lds:
        # occupancy pin: V stages padded so at most wg_per_cu workgroups share a CU (the grid is sized for exactly
        # that; when registers allow 2 waves / SIMD the dispatcher otherwise stacks workgroups unevenly)
        v: fx.Array[fx.Int8, max(W * lds_wave, (LDS_CU // wg_per_cu - 1024) if wg_per_cu else 0), 16]
        q: fx.Array[fx.Int8, W * q_wave if q_lds else 16, 16]

    @flyc.kernel(name="muse_verify_fly")
    def kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        table: fx.Tensor,
        seqlens: fx.Tensor,
        kd: fx.Tensor,
        part: fx.Tensor,
        maxi: fx.Tensor,
        den: fx.Tensor,
        splits: fx.Int32,
        work_splits: fx.Int32,
        t_stride: fx.Int32,
        q_s0: fx.Int32,
    ):
        pid = fx.block_idx.x
        tid = fx.thread_idx.x
        wave = tid // 64
        lane = tid % 64
        g = lane // 16
        li = lane % 16
        # head groups of the same (request, split) are adjacent programs: they share page-table and KV lines; only
        # the work_splits busy splits get programs (dispatch spreads them evenly over XCDs / CUs), and split j also
        # writes the empty padding slot j + work_splits of the partials
        work = pid // groups
        h = (pid % groups) * W + wave
        seq = work // work_splits
        split = work % work_splits

        sl_r = buffer_ops.create_buffer_resource(seqlens, max_size=True)
        tb_r = buffer_ops.create_buffer_resource(table, max_size=True)
        kd_r = buffer_ops.create_buffer_resource(kd, max_size=True)
        q_r = buffer_ops.create_buffer_resource(q, max_size=True)
        if fx.const_expr(addr64):
            k_addr = fx.Int64(arith.index_cast(T.i64, buffer_ops.extract_base_index(k)))
            v_addr = fx.Int64(arith.index_cast(T.i64, buffer_ops.extract_base_index(v)))
        else:
            k_r = buffer_ops.create_buffer_resource(k, max_size=True)
            v_r = buffer_ops.create_buffer_resource(v, max_size=True)
        p_r = buffer_ops.create_buffer_resource(part, max_size=True)
        m_r = buffer_ops.create_buffer_resource(maxi, max_size=True)
        d_r = buffer_ops.create_buffer_resource(den, max_size=True)

        kvlen = fx.Int32(buffer_ops.buffer_load(sl_r, seq, vec_width=1, dtype=T.i32, is_scalar=True))
        kd_bits = fx.Int32(buffer_ops.buffer_load(kd_r, fx.Int32(0), vec_width=1, dtype=T.i32, is_scalar=True))
        c_scale = arith.mulf(arith.bitcast(T.f32, _to_raw(kd_bits)), _f32(exp_scale))
        ctx = kvlen - QL
        last_page = (kvlen - 1) // PAGE
        nchunks = (kvlen + CHUNK - 1) // CHUNK
        last_chunk = nchunks - 1
        # chunks per split rounded up to whole slot groups, so only the last busy split runs dummy chunks; splits
        # >= work_splits (padding up to the power-of-two partials layout) stay empty
        cps = ((nchunks + work_splits - 1) // work_splits + NS - 1) // NS * NS
        begin = split * cps
        end = (begin + cps < nchunks).select(begin + cps, nchunks)
        begin = (begin < end).select(begin, end)
        # chunks below `interior` are visible to every query position: no masks
        interior = (ctx + 1) // CHUNK
        interior = (interior < end).select(interior, end)
        interior = (interior > begin).select(interior, begin)
        row_base = seq * t_stride

        # Q^T fragments (B operand): column n = li + 16 nt -> token n // 8, head h * 8 + n % 8; k-step ks holds
        # dims 16 g + 8 ks + 0..7. QF8: per-column scale amax / 448 (row of Q), folded into the column's multiplier.
        qf = []
        c_col = []
        for nt in range_constexpr(4):
            n = li + 16 * nt
            qrow = seq * QL + n // GQA
            qhead = h * GQA + n % GQA
            halves = [
                buffer_ops.buffer_load(q_r, qrow * q_s0 + qhead * HD + g * 16 + ks * 8, vec_width=8, dtype=T.bf16)
                for ks in range_constexpr(2)
            ]
            if fx.const_expr(qf8):
                vals = []
                for ks in range_constexpr(2):
                    hv = fx.Vector(halves[ks], shape=8, dtype=fx.BFloat16).to(fx.Float32)
                    vals.extend(_to_raw(hv[j]) for j in range(8))
                amax = _f32(1e-20)
                for x in vals:
                    amax = arith.maximumf(amax, fmath.absf(x))
                amax = _allreduce_groups(amax, True)
                inv = arith.divf(_f32(FP8_MAX), amax)
                c_col.append(arith.mulf(arith.divf(amax, _f32(FP8_MAX)), c_scale))
                sv = [
                    arith.minimumf(arith.maximumf(arith.mulf(x, inv), _f32(-FP8_MAX)), _f32(FP8_MAX)) for x in vals
                ]
                qf.append(_fp8x8(sv[:8]))
                qf.append(_fp8x8(sv[8:]))
            else:
                c_col.append(c_scale)
                qf.extend(halves)
        c_pair = [_v2(c) for c in c_col]
        inv_c = [arith.divf(_f32(1.0), c) for c in c_col]

        lds = fx.SharedAllocator().allocate(Lds).peek()
        vptr = lds.v.ptr
        lds_base = wave * lds_wave
        if fx.const_expr(q_lds and not qf8):
            # Q B-fragments parked in LDS (lane-contiguous 16-byte slots, conflict free) to free 32 VGPRs
            qptr = lds.q.ptr
            for f in range_constexpr(8):
                fx.ptr_store(fx.Vector(qf[f]).bitcast(fx.Int32), qptr + fx.Int64(wave * q_wave + (f * 64 + lane) * 16))

        def qfrag(f):
            if fx.const_expr(q_lds and not qf8):
                raw = fx.ptr_load(qptr + fx.Int64(wave * q_wave + (f * 64 + lane) * 16), result_type=T.vec(4, T.i32))
                return _to_raw(fx.Vector(raw, shape=4, dtype=fx.Int32).bitcast(fx.BFloat16))
            return qf[f]

        def page_ids(chunk):
            # the two 16-key pages of a chunk, clamped to the request's last page (SMEM loads, wave-uniform)
            p0 = chunk * 2
            p1 = p0 + 1
            p0 = (p0 < last_page).select(p0, last_page)
            p1 = (p1 < last_page).select(p1, last_page)
            if fx.const_expr(vec_ids):
                # vector loads, in order with the K / V loads: the vmcnt wait for an older chunk covers them
                a = fx.Int32(buffer_ops.buffer_load(tb_r, row_base + p0, vec_width=1, dtype=T.i32))
                b = fx.Int32(buffer_ops.buffer_load(tb_r, row_base + p1, vec_width=1, dtype=T.i32))
            else:
                a = fx.Int32(buffer_ops.buffer_load(tb_r, row_base + p0, vec_width=1, dtype=T.i32, is_scalar=True))
                b = fx.Int32(buffer_ops.buffer_load(tb_r, row_base + p1, vec_width=1, dtype=T.i32, is_scalar=True))
            return a, b

        def clamp(chunk):
            return (chunk < last_chunk).select(chunk, last_chunk)

        # per-lane dword offsets inside a page: K row li, dims 16 g..; V load u fills LDS row
        # kp = 16 u + lane // 4 (k-position order) at physical 16-byte chunk lane % 4
        k_off = li * s1w + h * (HD // 4) + g * 4
        v_keys = []
        v_offs = []
        for u in range_constexpr(2):
            kp = lane // 4 + 16 * u
            gp = kp // 8
            ip = kp % 8
            key = (ip < 4).select(gp * 4 + ip, gp * 4 + ip + 12)
            logical = fx.Int32(arith.xori(_to_raw(lane % 4), _to_raw((kp // 4) % 4)))
            v_keys.append(key)
            v_offs.append((key % 16) * s1w + h * (HD // 4) + logical * 4)

        if fx.const_expr(addr64):
            # per-lane 64-bit base addresses; a load adds its page's 64-bit byte offset
            k_lane = k_addr + fx.Int64(k_off * 4)
            v_lane = [v_addr + fx.Int64(v_offs[u] * 4) for u in range_constexpr(2)]

        def issue(pids):
            p0, p1 = pids
            loads = []
            for mt in range_constexpr(2):
                page = p0 if mt == 0 else p1
                if fx.const_expr(addr64):
                    loads.append(_global_load16(k_lane + fx.Int64(page) * kv_s0))
                else:
                    loads.append(buffer_ops.buffer_load(k_r, page * s0w + k_off, vec_width=4, dtype=T.i32))
            for u in range_constexpr(2):
                page = (v_keys[u] < 16).select(p0, p1)
                if fx.const_expr(addr64):
                    loads.append(_global_load16(v_lane[u] + fx.Int64(page) * kv_s0))
                else:
                    loads.append(buffer_ops.buffer_load(v_r, page * s0w + v_offs[u], vec_width=4, dtype=T.i32))
            return loads

        def prepare(chunk, loaded, stage, masked):
            # consume a load slot: V rows to wave-private LDS (k-position order, swizzled), K to A fragments
            kraw = loaded[:2]
            vraw = loaded[2:]
            if fx.const_expr(masked):
                # zero V rows past the request so stale FP8 bytes (NaN encodings) cannot reach P.V
                zero = fx.full(4, 0, fx.Int32)
                vraw = [
                    _to_raw(
                        (chunk * CHUNK + v_keys[u] < kvlen).select(fx.Vector(vraw[u], shape=4, dtype=fx.Int32), zero)
                    )
                    for u in range_constexpr(2)
                ]
            vbase = lds_base + stage * (CHUNK * HD)
            for u in range_constexpr(2):
                fx.ptr_store(fx.Vector(vraw[u], shape=4, dtype=fx.Int32), vptr + fx.Int64(vbase + u * 1024 + lane * 16))
            ka = []
            for mt in range_constexpr(2):
                words = fx.Vector(kraw[mt], shape=4, dtype=fx.Int32)
                if fx.const_expr(qf8):
                    # FP8 QK: K straight from the cache
                    ka.append(_i64(words[0], words[1]))
                    ka.append(_i64(words[2], words[3]))
                else:
                    ka.append(_cvt8(words[0], words[1]))
                    ka.append(_cvt8(words[2], words[3]))
            return ka, vbase

        def scores(chunk, ka, masked):
            # S^T = K Q^T: tile (mt, nt) rows = keys 16 mt + 4 g + j, column li + 16 nt
            ninf = fx.Float32(float("-inf"))
            # k-step outer: the 8 tiles' first MFMAs are independent, so the dependent second ones never stall
            accs = [_to_raw(fx.full(4, 0.0, fx.Float32)) for _ in range(8)]
            for ks in range_constexpr(2):
                for mt in range_constexpr(2):
                    for nt in range_constexpr(4):
                        a_op, b_op = ka[mt * 2 + ks], qfrag(nt * 2 + ks)
                        if fx.const_expr(qf8):
                            accs[mt * 4 + nt] = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
                                T.f32x4, [a_op, b_op, accs[mt * 4 + nt], 0, 0, 0]
                            )
                        else:
                            accs[mt * 4 + nt] = fx.rocdl.mfma_f32_16x16x32_bf16(
                                T.f32x4, [a_op, b_op, accs[mt * 4 + nt], 0, 0, 0]
                            )
            s = []
            for mt in range_constexpr(2):
                for nt in range_constexpr(4):
                    sv = fx.Vector(accs[mt * 4 + nt], shape=4, dtype=fx.Float32)
                    if fx.const_expr(masked):
                        qpos = ctx + (li + 16 * nt) // GQA
                        elems = []
                        for j in range_constexpr(4):
                            key = chunk * CHUNK + mt * 16 + g * 4 + j
                            ok = (key <= qpos) & (chunk < end)
                            elems.append(ok.select(sv[j], ninf))
                        sv = fx.Vector.from_elements(elems, fx.Float32)
                    s.append(sv)
            return s

        def finish(chunk, ka, vbase, o_acc, negm, thr, l_run, masked):
            # running state per column tile nt: negm = (bias - m_used) as an f32 pair (the exp fma operand), thr =
            # (m_used + rescale) / c (lazy-rescale trigger in raw score units), l = f32 pair of partial denominators
            s = scores(chunk, ka, masked)
            lmax = []
            need = None
            for nt in range_constexpr(4):
                mx = arith.maximumf(_max4(s[nt]), _max4(s[4 + nt]))
                lmax.append(mx)
                hit = arith.cmpf(arith.CmpFPredicate.OGT, mx, _to_raw(thr[nt]))
                need = hit if need is None else arith.ori(need, hit)
            vote = fx.Int64(_rocdl.ballot(T.i64, need))
            if vote != fx.Int64(0):
                negm_new = []
                thr_new = []
                l_new = []
                o_new = list(o_acc)
                for nt in range_constexpr(4):
                    mx = _allreduce_groups(lmax[nt], True)
                    m_old = arith.subf(_f32(p_bias), _to_raw(fx.Vector(negm[nt], shape=2, dtype=fx.Float32)[0]))
                    mn = arith.maximumf(m_old, arith.mulf(mx, c_col[nt]))
                    alpha = fx.rocdl.exp2(T.f32, arith.subf(m_old, mn))
                    negm_new.append(_to_raw(_v2(arith.subf(_f32(p_bias), mn))))
                    thr_new.append(arith.mulf(arith.addf(mn, _f32(rescale)), inv_c[nt]))
                    l_new.append(_to_raw(fx.Vector(l_run[nt], shape=2, dtype=fx.Float32) * fx.Float32(alpha)))
                    for mt in range_constexpr(4):
                        o_new[mt * 4 + nt] = _to_raw(
                            fx.Vector(o_acc[mt * 4 + nt], shape=4, dtype=fx.Float32) * fx.Float32(alpha)
                        )
                negm = negm_new
                thr = thr_new
                l_run = l_new
                o_acc = o_new
            # P = exp2(s * c + bias - m_used); P^T B fragment of column tile nt = (tile (0, nt), tile (1, nt))
            pf = []
            l_out = []
            for nt in range_constexpr(4):
                nm = fx.Vector(negm[nt], shape=2, dtype=fx.Float32)
                ps = []
                for mt in range_constexpr(2):
                    for hi in range_constexpr(2):
                        z = fx.Vector(
                            fmath.fma(_to_raw(_half(s[mt * 4 + nt], hi)), _to_raw(c_pair[nt]), _to_raw(nm)),
                            shape=2,
                            dtype=fx.Float32,
                        )
                        ps.extend(fx.Float32(fx.rocdl.exp2(T.f32, _to_raw(z[j]))) for j in range(2))
                t = fx.Vector.from_elements(ps[:4], fx.Float32) + fx.Vector.from_elements(ps[4:], fx.Float32)
                l_out.append(
                    _to_raw(fx.Vector(l_run[nt], shape=2, dtype=fx.Float32) + (_half(t, False) + _half(t, True)))
                )
                if fx.const_expr(pvf8):
                    pf.append(_fp8x8(ps))
                else:
                    pf.append(_to_raw(fx.Vector.from_elements(ps, fx.Float32).to(fx.BFloat16)))
            # O^T += V^T P^T, V^T fragments read transposed from LDS just in time
            out = [_to_raw(x) for x in o_acc]
            kp = g * 8 + li // 2
            swz = (kp // 4) % 4
            for mt in range_constexpr(4):
                phys = fx.Int32(arith.xori(_to_raw(fx.Int32(mt)), _to_raw(swz)))
                addr = vbase + kp * HD + phys * 16 + (li % 2) * 8
                raw = _rocdl.ds_read_tr8_b64(T.vec(2, T.i32), fx.to_llvm_ptr(vptr + fx.Int64(addr))).result
                pair = fx.Vector(raw, shape=2, dtype=fx.Int32)
                if fx.const_expr(pvf8):
                    vf = _i64(pair[0], pair[1])
                    for nt in range_constexpr(4):
                        out[mt * 4 + nt] = fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
                            T.f32x4, [vf, pf[nt], out[mt * 4 + nt], 0, 0, 0]
                        )
                else:
                    vf = _cvt8(pair[0], pair[1])
                    for nt in range_constexpr(4):
                        out[mt * 4 + nt] = fx.rocdl.mfma_f32_16x16x32_bf16(
                            T.f32x4, [vf, pf[nt], out[mt * 4 + nt], 0, 0, 0]
                        )
            return out, [_to_raw(x) for x in negm], [_to_raw(x) for x in thr], l_out

        def group_step(c0, st, masked):
            # chunks c0 .. c0 + NS - 1 from load slots 0 .. NS - 1; slot j is refilled with chunk c0 + NS + j right
            # after use, with page ids loaded one group earlier (carried)
            o_a, n_a, t_a, l_a = list(st[:16]), list(st[16:20]), list(st[20:24]), list(st[24:28])
            pids = [fx.Int32(x) for x in st[28 + 4 * NS : 28 + 6 * NS]]
            nxt = []
            for j in range_constexpr(NS):
                nxt.extend(page_ids(clamp(c0 + 2 * NS + j)))
            refill = []
            for j in range_constexpr(NS):
                ka, vb = prepare(c0 + j, list(st[28 + 4 * j : 32 + 4 * j]), j % 2, masked)
                refill.extend(issue((pids[2 * j], pids[2 * j + 1])))
                _fence(fence_vmem)
                o_a, n_a, t_a, l_a = finish(c0 + j, ka, vb, o_a, n_a, t_a, l_a, masked)
                _fence(fence_vmem)
            return o_a + n_a + t_a + l_a + refill + [_to_raw(x) for x in nxt]

        o0 = [_to_raw(fx.full(4, 0.0, fx.Float32)) for _ in range(16)]
        n0 = [_to_raw(_v2(_f32(p_bias - M_INIT))) for _ in range(4)]
        t0 = [arith.mulf(_f32(M_INIT + rescale), inv_c[nt]) for nt in range(4)]
        l0 = [_to_raw(fx.full(2, 0.0, fx.Float32)) for _ in range(4)]

        # unmasked chunk groups, then the rest masked (chunks past the split end are fully masked dummies); every
        # group prefetches the next one, so the masked loop starts from the interior loop's slots
        ngroups = (interior - begin) // NS
        mid = begin + ngroups * NS
        tail_groups = (end - mid + NS - 1) // NS
        slots0 = []
        ids0 = []
        for j in range_constexpr(NS):
            slots0.extend(issue(page_ids(clamp(begin + j))))
        for j in range_constexpr(NS):
            ids0.extend(_to_raw(x) for x in page_ids(clamp(begin + NS + j)))
        for gi, st in range(0, ngroups, 1, init=o0 + n0 + t0 + l0 + slots0 + ids0):
            c0 = begin + fx.Int32(arith.index_cast(T.i32, _to_raw(gi))) * NS
            fin = yield group_step(c0, st, False)
        for gi, st in range(0, tail_groups, 1, init=list(fin)):
            c0 = mid + fx.Int32(arith.index_cast(T.i32, _to_raw(gi))) * NS
            fin2 = yield group_step(c0, st, True)
        o_acc, n_run, l_run = list(fin2[:16]), list(fin2[16:20]), list(fin2[24:28])

        # epilogue: denominators summed over the four lane groups; all four groups hold the same max / denom and
        # store them unmasked (a masked buffer store relies on the OOB check, off for max_size resources)
        for nt in range_constexpr(4):
            n = li + 16 * nt
            t = n // GQA
            qh = n % GQA
            state = ((seq * QL + t) * heads + h * GQA + qh) * splits + split
            l2 = fx.Vector(l_run[nt], shape=2, dtype=fx.Float32)
            l_tot = _allreduce_groups(arith.addf(_to_raw(l2[0]), _to_raw(l2[1])), False)
            m_fin = arith.subf(_f32(p_bias), _to_raw(fx.Vector(n_run[nt], shape=2, dtype=fx.Float32)[0]))
            buffer_ops.buffer_store(m_fin, m_r, state)
            buffer_ops.buffer_store(l_tot, d_r, state)
            for mt in range_constexpr(4):
                buffer_ops.buffer_store(o_acc[mt * 4 + nt], p_r, state * HD + mt * 16 + g * 4)
            if split + work_splits < splits:
                pad = state + work_splits
                buffer_ops.buffer_store(_f32(M_INIT), m_r, pad)
                buffer_ops.buffer_store(_f32(0.0), d_r, pad)
                for mt in range_constexpr(4):
                    buffer_ops.buffer_store(_to_raw(fx.full(4, 0.0, fx.Float32)), p_r, pad * HD + mt * 16 + g * 4)

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        table: fx.Tensor,
        seqlens: fx.Tensor,
        kd: fx.Tensor,
        part: fx.Tensor,
        maxi: fx.Tensor,
        den: fx.Tensor,
        splits: fx.Int32,
        work_splits: fx.Int32,
        t_stride: fx.Int32,
        q_s0: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            q, k, v, table, seqlens, kd, part, maxi, den, splits, work_splits, t_stride, q_s0,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu} if waves_per_eu else None,
        ).launch(grid=(fx.Int64(grid), 1, 1), block=(W * 64, 1, 1), stream=stream)

    return launch


def verify_fly_partials(
    q,
    k,
    v,
    seqused_k,
    block_table,
    softmax_scale,
    k_descale,
    splits,
    waves_per_eu=2,
    heads_per_group=None,
    vec_ids=True,
    slots=2,
    q_lds=False,
    qf8=False,
    pvf8=False,
    fence_vmem=False,
    work_splits=None,
    wg_per_cu=0,
):
    """FP32 split-KV partials [T, H, S, 64] + max / denom [T, H, S] (log2 domain) of the full-layer verify.

    ``splits`` is the partials' split dimension (a power of two for the reduce kernels); ``work_splits`` <= splits of
    them carry work (e.g. 21 of 32 so 12 requests x 21 programs fill 252 of 256 CUs), the rest are written empty.
    Exact by default (BF16 QK / PV). ``qf8``: FP8 Q (per-row amax / 448) x FP8 K MFMA; ``pvf8``: FP8 P (biased 2^8)
    x FP8 V MFMA. Both FP8 modes keep the partials layout and the reduce unchanged."""
    import torch

    n = seqused_k.numel()
    heads, kvh = q.shape[1], k.shape[2]
    hpg = heads_per_group or kvh
    assert kvh % hpg == 0
    assert q.dtype == torch.bfloat16 and q.shape[-1] == HD and heads == GQA * kvh and q.shape[0] == n * QL
    assert k.shape[1] == PAGE and k.stride() == v.stride() and k.stride(2) == HD and k.stride(3) == 1
    assert q.stride(2) == 1 and q.stride(1) == HD
    assert work_splits is None or splits <= 2 * work_splits <= 2 * splits
    # buffer offsets are 32-bit byte offsets: a pool past them takes the 64-bit loads
    addr64 = max(k.numel(), v.numel()) * k.element_size() >= 2**31
    st = (q.shape[0], heads, splits)
    part = torch.empty((*st, HD), dtype=torch.float32, device=q.device)
    maxi = torch.empty(st, dtype=torch.float32, device=q.device)
    den = torch.empty(st, dtype=torch.float32, device=q.device)
    kd = k_descale if k_descale is not None else torch.ones(1, dtype=torch.float32, device=q.device)
    _run_compiled(
        _launcher(heads, kvh, hpg, k.stride(0), k.stride(1), float(softmax_scale), waves_per_eu, vec_ids, slots, q_lds, qf8, pvf8, fence_vmem, wg_per_cu, addr64),
        q,
        k.view(torch.uint8),
        v.view(torch.uint8),
        block_table,
        seqused_k,
        kd,
        part,
        maxi,
        den,
        int(splits),
        int(work_splits or splits),
        int(block_table.stride(0)),
        int(q.stride(0)),
        int(n * (work_splits or splits) * (kvh // hpg)),
        fx.Stream(torch.cuda.current_stream(q.device).cuda_stream),
    )
    return part, maxi, den


def pick_splits_flydsl(num_seqs: int, kv_heads: int) -> tuple[int, int]:
    """(partials split dimension, working splits): one workgroup per CU on MI355X (256 CUs), e.g. 64 at 4
    requests, 32 at 8, 21 of 32 at 12, 16 at 16; the split dimension is a power of two for the reduce kernels."""
    work = max(1, min(256 // max(num_seqs, 1), 128))
    splits = 1 << (work - 1).bit_length()
    return splits, work


def gqa_verify_flydsl(
    q,
    k,
    v,
    out,
    seqused_k,
    block_table,
    softmax_scale,
    k_descale,
    v_descale,
    *,
    defer_reduce=False,
):
    """Full-attention verify (exact BF16 QK / PV) on the FlyDSL kernel, merged by ``_vv2_reduce`` into ``out``
    unless ``defer_reduce`` returns the (partial, maximum, denom) for the caller's merge. MI355X TP4 n8: 123 us
    (4.75 TB/s) vs 158 us for the Triton pair kernel in the same harness."""
    import torch

    from sglang.kernels.ops.attention.gqa_verify_splitkv_2kv import _vv2_reduce

    num_seqs = seqused_k.numel()
    heads = q.shape[1]
    # the kernel reads both tables as 32-bit (buffer loads); sglang's seq_lens are int64
    if seqused_k.dtype != torch.int32:
        seqused_k = seqused_k.to(torch.int32)
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    splits, work = pick_splits_flydsl(num_seqs, k.shape[2])
    partial, maximum, denom = verify_fly_partials(
        q, k, v, seqused_k, block_table, softmax_scale, k_descale, splits,
        waves_per_eu=1, fence_vmem=True, wg_per_cu=1, work_splits=work,
    )  # fmt: skip
    if defer_reduce:
        return partial, maximum, denom
    _vv2_reduce[(q.shape[0], heads)](
        partial, maximum, denom, out, v_descale, heads, splits, out.stride(0), out.stride(1), num_warps=2,
    )  # fmt: skip
    return out

