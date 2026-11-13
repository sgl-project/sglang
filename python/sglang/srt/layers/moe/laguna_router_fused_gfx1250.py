"""Fused Laguna decode router on gfx1250 (FlyDSL): one launch per MoE layer for M <= 64 tokens.

    logits[M, E] = x[M, K] (bf16) @ W[E, K]^T (fp32 weights, fp32 accumulation)
    activated    = sigmoid(logits);  biased = activated + bias
    top-k by (biased desc, expert id asc); weight = activated of the winner; renormalize by the sum

Same selection, tie rule, output order and renormalization as sglang `moe_fused_gate._router_triton_kernel`
(sigmoid, no grouping, no shared slots, no scaling; biased NaN ranks as -1e30). Design and measurements:
proxy-moe/OPT_TP16_h13-14_1111.md, "Item R".

Grid (KS, NET, 1) with cluster (KS, 1, 1): a cluster is the KS K-splits of one tile of ET experts.
  Phase 1, every WG: partial logits [M, ET] over its K chunk, written to `ws` with device-scope stores.
    flavour "valu": wave = ET / 8 experts x the whole chunk, lane FMAs, wave butterfly (small M);
                    lane owns `vec` consecutive k (two b128 per row, 32 B lane stride)
    flavour "valuc": as "valu" but each b128 instruction of a wave reads 512 contiguous bytes of one row
                    (lane i -> k = base + 4 i; the vec / 4 sub-loads of a step are WAVE * 4 floats apart)
    flavour "wmma": wave = 16 experts x 1 / NSUB of the chunk; W fp32 split exactly into hi + mid + lo bf16
                    (3 x 8 significand bits = fp32's 24), x is bf16 already -> 3 x wmma_f32_16x16x32_bf16
  cluster barrier
  Phase 2, every WG: a disjoint stripe of the [M, ET] tile; sum of the KS x NSUB partials in a FIXED order
    (deterministic), sigmoid, + bias -> s_act / s_biased [M, E].
  cluster barrier; the cluster leader takes one agent-scope acq_rel ticket on `cnt`; the last cluster resets `cnt`
  and raises its `flag`; cluster barrier.
  Phase 3, last cluster only: one wave per token, TOPK rounds of a (biased, id) wave argmax over 512 experts
    (16 per lane), weights = activated, renormalized -> out_w [M, TOPK] fp32, out_id [M, TOPK] int32.
No spinning anywhere (cluster barriers + one ticket), so there is no resident-grid precondition.
Scratch (`ws`, `s_*`, `flag`, `cnt`) is shared by all layers: one router launch in flight per device.
"""
import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl._mlir import ir
from flydsl.expr.rocdl import cluster

E, K = 512, 16384
WAVE, WAVES = 32, 8
BLOCK = WAVE * WAVES
MAXM = 64
SLOTS = E // WAVE  # experts per lane in phase 3
CPOL_DEV = 16  # gfx1250 cache policy, device scope (as aiter splitk_fused_epilogue_gfx1250 CPOL_DEVICE)
CPOL_ST_DEV = 16 | 3  # its CPOL_STORE_DEVICE
NEG_BIG = -1e30  # what the triton router ranks a NaN biased score as


def _io(elem, cpol=0):
    """load(addr, off, vec) / store1(addr, off, value) of `elem` at element offset `off` from byte address `addr`."""

    def view(addr, vec):
        pt = fx.PointerType.get(elem.ir_type, fx.AddressSpace.Global, max(1, elem.width * vec // 8))
        return rocdl.make_buffer_tensor(
            fx.Tensor(fx.make_view(fx.inttoptr(pt, fx.Int64(addr)), fx.make_layout((vec, 1), (1, 1))))
        )

    def load(addr, off, vec):
        frag = fx.make_rmem_tensor(vec, elem)
        fx.copy(fx.make_copy_atom(rocdl.BufferCopy(elem.width * vec, cpol), elem), fx.slice(view(addr, vec), (None, off)), frag)
        return fx.Vector(frag.load())

    def store1(addr, off, value):
        frag = fx.make_rmem_tensor(1, elem)
        frag.store(fx.Vector.from_elements([value], elem))
        fx.copy(fx.make_copy_atom(rocdl.BufferCopy(elem.width, cpol), elem), frag, fx.slice(view(addr, 1), (None, off)))

    return load, store1


def _gptr(addr, elem=fx.Int32):
    return fx.inttoptr(fx.PointerType.get(elem.ir_type, fx.AddressSpace.Global, elem.width // 8), fx.Int64(addr))


VALU_FLAVOURS = ("valu", "valuc", "rd", "rdc", "rdcx", "rd0")  # rd0 = no loads at all (launch + store floor)  # rd* = read probes (attribution only, wrong results)


def _p1_accum(flavour, m, epw, vec, coal, acc, wp, xload_c, xload_v):
    """Phase-1 VALU work of one step (plain python, traced). wp[e][q] = float4 W of expert e, sub-load q.
    valu / valuc: acc[t][e] += x[t] . w[e].  rd / rdc: acc[0][e] += sum(w[e]) only (no x, no per-token FMA).
    rdcx: rdc + the x loads of every token, summed once into acc[t][0]."""
    if flavour == "rd0":
        return acc
    if flavour.startswith("rd"):
        for e in range(epw):
            a = acc[0][e]
            for q in range(vec // 4):
                for j in range(4):
                    a = a + fx.Float32(wp[e][q][j])
            acc[0][e] = a
        if flavour == "rdcx":
            for t in range(m):
                xq = xload_c(t)
                a = acc[t][0]
                for q in range(vec // 4):
                    for j in range(4):
                        a = a + fx.Float32(xq[q][j])
                acc[t][0] = a
        return acc
    wv = [[wp[e][j // 4][j % 4] for j in range(vec)] for e in range(epw)]
    for t in range(m):
        if coal:
            xq = xload_c(t)
            xv = [xq[j // 4][j % 4] for j in range(vec)]
        else:
            xv = xload_v(t)
        for e in range(epw):
            a = acc[t][e]
            for j in range(vec):
                a = a + fx.Float32(xv[j]) * fx.Float32(wv[e][j])
            acc[t][e] = a
    return acc


def _key(h, l):
    """int64 key, (h, l) lexicographic as a signed compare (h signed, 0 <= l < 2^31)."""
    return fx.Int64(h) * fx.Int64(1 << 32) + fx.Int64(l)


def _p3_sorted_heads(bv, av, lane, topk):
    """Phase-3 top-k of one token in one wave: lane owns experts lane + WAVE * j (j < SLOTS).
    Key = (ordered bits of biased, E - 1 - id): a total order with the triton rule (biased desc, ties -> lowest id;
    -0 == +0 via + 0.0; NaN was mapped to NEG_BIG in phase 2). Each lane sorts its SLOTS keys (bitonic network,
    independent compare-exchanges); then TOPK rounds of a wave max over the lane heads, the owning lane pops its head.
    Returns (sel_w, sel_i, total) with lane kk holding the kk-th selection; total summed in selection order."""
    H, L, A = [], [], []
    for j in range(SLOTS):
        bb = bv[j] + fx.Float32(0.0)
        h = bb.bitcast(fx.Int32)
        H.append((h < fx.Int32(0)).select(h ^ fx.Int32(0x7FFFFFFF), h))
        L.append(fx.Int32(E - 1 - WAVE * j) - lane)
        A.append(av[j])
    n = SLOTS
    k = 2
    while k <= n:  # bitonic sort, descending
        j = k // 2
        while j >= 1:
            for i in range(n):
                q = i ^ j
                if q > i:
                    a, b = (i, q) if (i & k) == 0 else (q, i)  # want key[a] >= key[b]
                    g = _key(H[a], L[a]) > _key(H[b], L[b])
                    H[a], H[b] = g.select(H[a], H[b]), g.select(H[b], H[a])
                    L[a], L[b] = g.select(L[a], L[b]), g.select(L[b], L[a])
                    A[a], A[b] = g.select(A[a], A[b]), g.select(A[b], A[a])
            j //= 2
        k *= 2
    sel_w = fx.Float32(0.0)
    sel_i = fx.Int32(0)
    total = fx.Float32(0.0)
    for kk in range(topk):
        h, l, a = H[0], L[0], A[0]
        for sh in range(5):  # wave max of the lane heads (keys are unique -> every lane ends on the same winner)
            off = WAVE >> (sh + 1)
            oh = fx.Int32(h.shuffle_xor(off, WAVE))
            ol = fx.Int32(l.shuffle_xor(off, WAVE))
            oa = fx.Float32(a.shuffle_xor(off, WAVE))
            tk = _key(oh, ol) > _key(h, l)
            h, l, a = tk.select(oh, h), tk.select(ol, l), tk.select(oa, a)
        mine = lane == fx.Int32(kk)
        sel_w = mine.select(a, sel_w)
        sel_i = mine.select(fx.Int32(E - 1) - l, sel_i)
        total = total + a
        if kk + 1 < topk:
            pop = L[0] == l
            for i in range(n - 1):
                H[i], L[i], A[i] = pop.select(H[i + 1], H[i]), pop.select(L[i + 1], L[i]), pop.select(A[i + 1], A[i])
            H[n - 1] = pop.select(fx.Int32(-(1 << 31)), H[n - 1])
            L[n - 1] = pop.select(fx.Int32(-1), L[n - 1])
    return sel_w, sel_i, total

def _p3_scan(bv, av, lane, topk):
    """First phase-3 version (kept for A/B): TOPK rounds of a lane scan over SLOTS + wave argmax butterfly."""
    ids = [lane + fx.Int32(WAVE * j) for j in range(SLOTS)]
    sel_w = fx.Float32(0.0)
    sel_i = fx.Int32(0)
    total = fx.Float32(0.0)
    for kk in range_constexpr(topk):
        # lane-local best; ids ascend with j, so a strict > keeps the lowest id on ties
        v, vi, va = bv[0], ids[0], av[0]
        for j in range_constexpr(1, SLOTS):
            better = bv[j] > v
            v = better.select(bv[j], v)
            vi = better.select(ids[j], vi)
            va = better.select(av[j], va)
        # wave argmax on (value desc, id asc): a total order, so every lane ends on the same winner
        for sh in range_constexpr(5):
            off = WAVE >> (sh + 1)
            ov = fx.Float32(v.shuffle_xor(off, WAVE))
            oi = fx.Int32(vi.shuffle_xor(off, WAVE))
            oa = fx.Float32(va.shuffle_xor(off, WAVE))
            take = (ov > v).select(fx.Int32(1), (ov == v).select((oi < vi).select(fx.Int32(1), fx.Int32(0)), fx.Int32(0)))
            tk = take != fx.Int32(0)
            v = tk.select(ov, v)
            vi = tk.select(oi, vi)
            va = tk.select(oa, va)
        mine = lane == fx.Int32(kk)
        sel_w = mine.select(va, sel_w)
        sel_i = mine.select(vi, sel_i)
        total = total + va
        for j in range_constexpr(SLOTS):
            bv[j] = (ids[j] == vi).select(fx.Float32(float("-inf")), bv[j])
    return sel_w, sel_i, total


def nsub_of(flavour: str, et: int) -> int:
    return 1 if flavour in VALU_FLAVOURS else WAVES // (et // 16)


def ws_floats(m: int, et: int, ks: int, flavour: str) -> int:
    return E * ks * nsub_of(flavour, et) * m


@functools.lru_cache(maxsize=128)
def compile_router_fused(m: int, topk: int = 16, flavour: str = "valu", et: int = 32, ks: int = 16, vec: int = 8,
                         stop: int = 0, wpe: int = 0, xs: int = 0, nw: int = WAVES, p3: int = 0):
    """stop (attribution only, wrong results): 1 = phase 1 only, plain launch, no barrier; 2 = + cluster launch and
    one cluster barrier; 3 = + phase 2 and its barrier; 4 = + ticket / flag (no phase 3); 0 = full kernel.
    (Before 2026-11-13 11:30 UTC stop did not exit early: every stop variant ran the whole kernel.)"""
    use_cluster = stop != 1
    assert nw == WAVES or (stop == 1 and flavour in VALU_FLAVOURS), "nw != 8 only for phase-1 probes"
    nb = WAVE * nw  # threads per WG
    assert 1 <= m <= MAXM and 1 <= topk <= WAVE and flavour in VALU_FLAVOURS + ("wmma",)
    assert E % et == 0 and K % ks == 0 and 1 <= ks <= 16
    net = E // et
    kc = K // ks  # K chunk per WG
    nsub = nsub_of(flavour, et)
    nparts = ks * nsub  # partials per (token, expert)
    n_el = m * et  # elements of one expert tile
    if flavour in VALU_FLAVOURS:
        assert et % nw == 0 and vec % 4 == 0 and kc % (WAVE * vec) == 0
        epw = et // nw
        steps = kc // (WAVE * vec)
    else:
        assert et % 16 == 0 and WAVES % (et // 16) == 0
        n_ew = et // 16
        kw = kc // nsub  # K per wave
        assert kw % 32 == 0
        steps = kw // 32
        tiles = (m + 15) // 16
    p2_iters = (n_el + ks * BLOCK - 1) // (ks * BLOCK)
    p3_iters = (m + ks * WAVES - 1) // (ks * WAVES)

    @flyc.kernel(name=f"laguna_router_fused_{flavour}_m{m}_t{topk}_e{et}_k{ks}" + (f"_v{vec}" if vec != 8 else "") + (f"_w{wpe}" if wpe else "") + ("_xs" if xs else "") + (f"_n{nw}" if nw != WAVES else "") + (f"_p{p3}" if p3 else "") + (f"_stop{stop}" if stop else ""), known_block_size=[nb, 1, 1])
    def router(x_ptr: fx.Int64, w_ptr: fx.Int64, b_ptr: fx.Int64, ws_ptr: fx.Int64, sa_ptr: fx.Int64, sb_ptr: fx.Int64,
               flag_ptr: fx.Int64, cnt_ptr: fx.Int64, ow_ptr: fx.Int64, oi_ptr: fx.Int64):
        load_w, _ = _io(fx.Float32)
        load_x, _ = _io(fx.BFloat16)
        load_dev, store_dev = _io(fx.Float32, CPOL_DEV)
        _, store_dev_st = _io(fx.Float32, CPOL_ST_DEV)
        load_i_dev, _ = _io(fx.Int32, CPOL_DEV)
        _, store_i_dev = _io(fx.Int32, CPOL_ST_DEV)
        _, store_out_f = _io(fx.Float32)
        _, store_out_i = _io(fx.Int32)

        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % fx.Int32(WAVE)
        wave = tid // fx.Int32(WAVE)
        split = fx.Int32(gpu.block_id("x"))  # K split = position in the cluster
        et_i = fx.Int32(gpu.block_id("y"))  # expert tile = cluster id
        tile_base = et_i * fx.Int32(nparts * n_el)  # this tile's partial planes in ws

        # ---------------- phase 1: partial logits of this WG's K chunk ----------------
        if flavour in VALU_FLAVOURS:
            e0 = et_i * fx.Int32(et) + wave * fx.Int32(epw)
            coal = flavour in ("valuc", "rdc", "rdcx")
            kbase = split * fx.Int32(kc) + lane * fx.Int32(4 if coal else vec)
            qstride = WAVE * 4 if coal else 4  # float distance between the vec / 4 sub-loads of a step
            f32v = fx.T.vec(vec, fx.T.f32)
            f32x4 = fx.T.vec(4, fx.T.f32)
            acc = [[fx.Float32(0.0) for _ in range(epw)] for _ in range(m)]
            for s in range_constexpr(steps):
                k = kbase + fx.Int32(s * WAVE * vec)
                wp = []
                for e in range_constexpr(epw):
                    row = (e0 + fx.Int32(e)) * fx.Int32(K) + k
                    wp.append([load_w(w_ptr, row + fx.Int32(qstride * q), 4) for q in range(vec // 4)] if flavour != "rd0" else [])
                # python-level helper (an `if` statement in this body would be rewritten into a runtime branch)
                acc = _p1_accum(flavour, m, epw, vec, coal, acc, wp,
                                lambda t: [load_x(x_ptr, fx.Int32(t * K) + k + fx.Int32(qstride * q), 4).extf(f32x4) for q in range(vec // 4)],
                                lambda t: load_x(x_ptr, fx.Int32(t * K) + k, vec).extf(f32v))
            plane = tile_base + split * fx.Int32(n_el)
            for t in range_constexpr(m):
                for e in range_constexpr(epw):
                    v = acc[t][e]
                    for sh in range_constexpr(5):  # wave32 butterfly
                        v = v + fx.Float32(v.shuffle_xor(WAVE >> (sh + 1), WAVE))
                    if lane == fx.Int32(0):
                        store_dev_st(ws_ptr, plane + fx.Int32(t * et) + wave * fx.Int32(epw) + fx.Int32(e), v)
        else:
            ew = wave % fx.Int32(n_ew)
            sub = wave // fx.Int32(n_ew)
            col = lane % fx.Int32(16)
            half = lane // fx.Int32(16)
            e = et_i * fx.Int32(et) + ew * fx.Int32(16) + col
            kbase = split * fx.Int32(kc) + sub * fx.Int32(kw) + half * fx.Int32(16)
            rows = []
            for tt in range_constexpr(tiles):
                r = fx.Int32(tt * 16) + col
                rows.append((r < fx.Int32(m)).select(r, fx.Int32(m - 1)))
            f32x16 = fx.T.vec(16, fx.T.f32)
            acc = [fx.Vector.from_elements([fx.Float32(0.0) for _ in range(8)], fx.Float32) for _ in range(tiles)]
            def mma(a_vec, b_vec, c_vec):  # as aiter pa_decode/op_gemm.py Gfx1250Gemm.mma(bf16=True)
                return fx.Vector(rocdl.wmma_f32_16x16x32_bf16(
                    fx.Vector.make_type(8, fx.Float32), fx.as_ir_value(a_vec), fx.as_ir_value(b_vec),
                    fx.as_ir_value(c_vec), modC=ir.Attribute.parse("#rocdl<wmma_c_modifier none>"),
                    reuseA=False, reuseB=False).result)

            for s in range_constexpr(steps):
                k = kbase + fx.Int32(s * 32)
                w4 = [load_w(w_ptr, e * fx.Int32(K) + k + fx.Int32(4 * q), 4) for q in range(4)]
                wf = [fx.Float32(w4[j // 4][j % 4]) for j in range(16)]
                # exact 3-way split: w == hi + mid + lo (each residual fits the next bf16 exactly)
                hi = fx.Vector.from_elements(wf, fx.Float32).to(fx.BFloat16)
                hif = hi.extf(f32x16)
                r1 = [wf[j] - fx.Float32(hif[j]) for j in range(16)]
                mid = fx.Vector.from_elements(r1, fx.Float32).to(fx.BFloat16)
                midf = mid.extf(f32x16)
                r2 = [r1[j] - fx.Float32(midf[j]) for j in range(16)]
                lo = fx.Vector.from_elements(r2, fx.Float32).to(fx.BFloat16)
                for tt in range_constexpr(tiles):
                    xa0 = load_x(x_ptr, rows[tt] * fx.Int32(K) + k, 8)
                    xa1 = load_x(x_ptr, rows[tt] * fx.Int32(K) + k + fx.Int32(8), 8)
                    xa = fx.Vector.from_elements([fx.BFloat16(xa0[j]) for j in range(8)]
                                                 + [fx.BFloat16(xa1[j]) for j in range(8)], fx.BFloat16)
                    c = acc[tt]
                    c = mma(xa, lo, c)
                    c = mma(xa, mid, c)
                    c = mma(xa, hi, c)
                    acc[tt] = c
            # C layout (probe, wmma_f32_layout_probe.py): element (row, col) in lane col + 16 * (row // 8), VGPR row % 8
            plane = tile_base + (split * fx.Int32(nsub) + sub) * fx.Int32(n_el)
            for tt in range_constexpr(tiles):
                cv = fx.Vector(acc[tt])
                for v in range_constexpr(8):
                    mrow = fx.Int32(tt * 16 + v) + half * fx.Int32(8)
                    if mrow < fx.Int32(m):
                        store_dev_st(ws_ptr, plane + mrow * fx.Int32(et) + ew * fx.Int32(16) + col, fx.Float32(cv[v]))
        # stop levels (attribution): zero-trip range_constexpr loops are skipped at trace time; an `if stop: return`
        # is NOT an early exit in FlyDSL (the if is rewritten into a runtime branch and the return is dropped)
        for _s1 in range_constexpr(0 if stop == 1 else 1):
            rocdl.s_wait_storecnt(0)
            cluster.cluster_barrier()
            for _s2 in range_constexpr(0 if stop == 2 else 1):

                # ---------------- phase 2: fixed-order reduce of this tile's partials, sigmoid, bias ----------------
                for i in range_constexpr(p2_iters):
                    idx = fx.Int32(i * ks * BLOCK) + split * fx.Int32(BLOCK) + tid
                    if idx < fx.Int32(n_el):
                        s_ = fx.Float32(load_dev(ws_ptr, tile_base + idx, 1)[0])
                        for p in range_constexpr(1, nparts):
                            s_ = s_ + fx.Float32(load_dev(ws_ptr, tile_base + fx.Int32(p * n_el) + idx, 1)[0])
                        t = idx // fx.Int32(et)
                        eg = et_i * fx.Int32(et) + idx % fx.Int32(et)
                        act = fx.Float32(1.0) / (fx.Float32(1.0) + fx.math.exp(fx.Float32(0.0) - s_))
                        bsc = act + fx.Float32(load_w(b_ptr, eg, 1)[0])
                        bsc = (bsc != bsc).select(fx.Float32(NEG_BIG), bsc)
                        store_dev_st(sa_ptr, t * fx.Int32(E) + eg, act)
                        store_dev_st(sb_ptr, t * fx.Int32(E) + eg, bsc)
                rocdl.s_wait_storecnt(0)
                cluster.cluster_barrier()
                for _s3 in range_constexpr(0 if stop == 3 else 1):
                    # ---------------- one ticket per cluster; the last cluster runs phase 3 ----------------
                    if split == fx.Int32(0):
                        if tid == fx.Int32(0):
                            old = fx.atomic_add(_gptr(cnt_ptr), fx.Int32(1), syncscope=rocdl.SyncScope.Agent,
                                                ordering=fx.AtomicOrdering.AcqRel)
                            last = fx.Int32(old) == fx.Int32(net - 1)
                            if last:
                                store_i_dev(cnt_ptr, fx.Int32(0), fx.Int32(0))
                            store_i_dev(flag_ptr, et_i, last.select(fx.Int32(1), fx.Int32(0)))
                    rocdl.s_wait_storecnt(0)
                    cluster.cluster_barrier()
                    is_last = fx.Int32(load_i_dev(flag_ptr, et_i, 1)[0])

                    for _s4 in range_constexpr(0 if stop == 4 else 1):
                        # ---------------- phase 3: biased top-k per token, one wave per token ----------------
                        if is_last != fx.Int32(0):
                            fx.memory_fence(syncscope=rocdl.SyncScope.Agent, ordering=fx.AtomicOrdering.Acquire)
                            gw = split * fx.Int32(WAVES) + wave
                            for it in range_constexpr(p3_iters):
                                t = fx.Int32(it * ks * WAVES) + gw
                                if t < fx.Int32(m):
                                    row = t * fx.Int32(E) + lane
                                    bv = [fx.Float32(load_dev(sb_ptr, row + fx.Int32(WAVE * j), 1)[0]) for j in range(SLOTS)]
                                    av = [fx.Float32(load_dev(sa_ptr, row + fx.Int32(WAVE * j), 1)[0]) for j in range(SLOTS)]
                                    # p3 = 0: lane-sorted heads (default); 1: the first version (TOPK rounds of a 16-slot scan + 3-value butterfly)
                                    sel_w, sel_i, total = (_p3_sorted_heads(bv, av, lane, topk) if p3 == 0 else _p3_scan(bv, av, lane, topk))
                                    norm = (total > fx.Float32(0.0)).select(total, fx.Float32(1.0))
                                    if lane < fx.Int32(topk):
                                        store_out_f(ow_ptr, t * fx.Int32(topk) + lane, sel_w / norm)
                                        store_out_i(oi_ptr, t * fx.Int32(topk) + lane, sel_i)

    @flyc.jit
    def launch(x_ptr: fx.Int64, w_ptr: fx.Int64, b_ptr: fx.Int64, ws_ptr: fx.Int64, sa_ptr: fx.Int64, sb_ptr: fx.Int64,
               flag_ptr: fx.Int64, cnt_ptr: fx.Int64, ow_ptr: fx.Int64, oi_ptr: fx.Int64, stream: fx.Stream):
        if use_cluster:
            router(x_ptr, w_ptr, b_ptr, ws_ptr, sa_ptr, sb_ptr, flag_ptr, cnt_ptr, ow_ptr, oi_ptr,
                   value_attrs={"rocdl.cluster_dims": f"{ks},1,1"}).launch(
                grid=(ks, net, 1), block=(nb, 1, 1), stream=stream, cluster=(ks, 1, 1))
        else:
            router(x_ptr, w_ptr, b_ptr, ws_ptr, sa_ptr, sb_ptr, flag_ptr, cnt_ptr, ow_ptr, oi_ptr).launch(
                grid=(ks, net, 1), block=(nb, 1, 1), stream=stream)

    hints = {}
    if wpe:
        hints["waves_per_eu"] = wpe
    if xs:  # gfx1250 expert scheduling mode (what Triton's gfx1250 kernels run with: s_setreg WAVE_SCHED_MODE)
        hints["llvm_options"] = {"amdgpu-expert-scheduling-mode": True}
    return _with_hints(launch, hints or None)


class _with_hints:
    """Calls (and compile-only calls) of a @flyc.jit launcher under FlyDSL compile hints."""

    def __init__(self, fn, hints):
        self.fn, self.hints = fn, hints

    def _ctx(self):
        import contextlib
        from flydsl.compiler.kernel_function import CompilationContext

        return CompilationContext.compile_hints(self.hints) if self.hints else contextlib.nullcontext()

    def __call__(self, *a, **k):
        with self._ctx():
            return self.fn(*a, **k)

    def _compile_only(self, *a, **k):
        with self._ctx():
            return self.fn._compile_only(*a, **k)


# ---------------------------------------------------------------- host side
DEFAULT_CFG = {  # M upper bound -> (flavour, et, ks); op-bench sweep 2026-11-13 11:36 (bench_1136.log, h13-14 GPU 0)
    1: ("valuc", 16, 8),
    2: ("valuc", 32, 8),
    4: ("valuc", 16, 16),
    64: ("wmma", 32, 16),
}


def pick_cfg(m: int):
    for mmax in sorted(DEFAULT_CFG):
        if m <= mmax:
            return DEFAULT_CFG[mmax]
    raise ValueError(m)


class _Scratch:
    """Per-device persistent scratch: sized for M = MAXM and the largest ws of any config; cnt zeroed once."""

    _by_dev = {}

    def __init__(self, device):
        import torch

        self.ws = torch.empty(E * 16 * WAVES * MAXM, device=device, dtype=torch.float32)  # ks <= 16, nsub <= 8
        self.s_act = torch.empty(MAXM, E, device=device, dtype=torch.float32)
        self.s_biased = torch.empty(MAXM, E, device=device, dtype=torch.float32)
        self.flag = torch.zeros(E, device=device, dtype=torch.int32)
        self.cnt = torch.zeros(4, device=device, dtype=torch.int32)

    @classmethod
    def get(cls, device):
        key = (device.type, device.index)
        if key not in cls._by_dev:
            cls._by_dev[key] = cls(device)
        return cls._by_dev[key]


def laguna_router_fused(x, w, bias, topk: int = 16, cfg=None, out=None, stop: int = 0):
    """x [M, 16384] bf16, w [512, 16384] fp32, bias [512] fp32 -> (weights [M, topk] fp32, ids [M, topk] int32).

    One launch; M <= 64. cfg = (flavour, et, ks) overrides the default table. Not reentrant across streams."""
    import torch

    assert x.dim() == 2 and x.dtype == torch.bfloat16 and w.dtype == torch.float32 and tuple(w.shape) == (E, K)
    assert bias is not None and bias.dtype == torch.float32 and bias.numel() == E
    m = x.shape[0]
    assert 1 <= m <= MAXM
    x = x.contiguous()
    cfg = tuple(cfg) if cfg is not None else pick_cfg(m)
    flavour, et, ks = cfg[:3]
    vec = cfg[3] if len(cfg) > 3 else 8
    wpe = cfg[4] if len(cfg) > 4 else 0  # FlyDSL waves_per_eu compile hint (0 = none)
    xs = cfg[5] if len(cfg) > 5 else 0  # 1 = LLVM amdgpu-expert-scheduling-mode
    nw = cfg[6] if len(cfg) > 6 else WAVES  # waves per WG (probes with stop=1 only)
    p3 = cfg[7] if len(cfg) > 7 else 0  # phase-3 algorithm: 0 lane-sorted heads, 1 first version (scan)
    sc = _Scratch.get(x.device)
    if out is None:
        out = (torch.empty(m, topk, device=x.device, dtype=torch.float32),
               torch.empty(m, topk, device=x.device, dtype=torch.int32))
    stream = fx.Stream(torch.cuda.current_stream(x.device))
    fn = compile_router_fused(m, topk, flavour, et, ks, vec, stop=stop, wpe=wpe, xs=xs, nw=nw, p3=p3)
    fn(
        x.data_ptr(), w.data_ptr(), bias.contiguous().data_ptr(), sc.ws.data_ptr(), sc.s_act.data_ptr(),
        sc.s_biased.data_ptr(), sc.flag.data_ptr(), sc.cnt.data_ptr(), out[0].data_ptr(), out[1].data_ptr(), stream)
    return out
