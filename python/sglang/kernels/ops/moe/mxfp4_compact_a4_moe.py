"""Compact A4W4 MoE experts for small target-verify batches (MXFP4 weights and activations, gfx950).

Muse Spark EP8 / EP4 layout: 32 or 64 local experts, top-8 routing, hidden 4096, intermediate 4096, Quark MXFP4
checkpoint weights (per-32 E8M0 scales) used as stored. For M (a multiple of 8, up to MAX_ROWS) rows:

1. ``_prepare_a4``: per-row MXFP4 quantization of the input (or the caller passes it, ``quantized``);
2. ``_scatter_a4``: per local expert, place its 16-row blocks after the lower experts' and rank its routes in
   token order into them;
3. ``_expert_a4`` (ACT): gate/up ``dot_scaled`` GEMMs, SiLU-and-mul rounded to BF16, MXFP4 re-quantization;
4. ``_expert_a4``: down GEMM into unweighted per-(token, slot) BF16 rows.

Weights are read either as stored in the checkpoint or in aiter's FlyDSL a4w4 preshuffle layout (``shuffled``), so a
rank short on memory keeps one copy for both paths.

``compact_a4_experts`` returns those per-slot rows ``[M * 8, 4096]``; the caller's finalize applies the
per-expert norm and the route-weighted sum. Quark MXFP4 declares dynamic MXFP4 activations, so A4 is the
checkpoint numerics.

``compact_a4_experts_replicated`` is the EP4 variant with redundant expert replicas (``expert_replicas``): the
weights hold the rank's 64 home experts followed by its replicas, and ``_scatter_a4_replica`` first decides, the
same way on every rank, which of its two ranks computes each active replicated expert this step.
"""

import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

MAX_ROWS = 512
BLOCK_M = 16  # BM 32 / 64 give wrong rows and are no faster


@triton.jit
def _prepare_a4(X, Q, QS):
    # Per-row MXFP4 quantization of the input, four 1024-column tiles per row.
    pid = tl.program_id(0)
    row = pid // 4
    tile = pid % 4
    col = tile * 1024 + tl.arange(0, 1024)
    x = tl.load(X + row * 4096 + col).to(tl.float32)
    q, qscale = _mxfp4_quant_op(x[None, :], 1024, 1, 32)
    packed_col = tile * 512 + tl.arange(0, 512)
    scale_col = tile * 32 + tl.arange(0, 32)
    tl.store(Q + row * 2048 + packed_col, tl.reshape(q, (512,)))
    tl.store(QS + row * 128 + scale_col, tl.reshape(qscale, (32,)))


@triton.jit
def _scatter_a4(
    IDS,
    SORTED,
    EXPERT,
    M: tl.constexpr,
    START: tl.constexpr,
    E: tl.constexpr,
    S: tl.constexpr,
    CH: tl.constexpr,
    PER_E: tl.constexpr,
    BM: tl.constexpr,
    NB: tl.constexpr,
    NBB: tl.constexpr,
):
    # One program per local expert. Each counts every local expert's routes and places its blocks after the
    # lower experts' (fixed block order, nothing to clear between calls); the last also marks the unused tail
    # blocks (-1). Its routes are ranked in token order into its blocks (sentinel M*8).
    e = tl.program_id(0)
    ee = tl.arange(0, E)
    counts = tl.zeros((E,), tl.int32)
    for c0 in tl.static_range(0, S, CH):
        sc = c0 + tl.arange(0, CH)
        idc = tl.load(IDS + sc, sc < M * 8, other=-1)
        counts += tl.sum(((idc[:, None] - START) == ee[None, :]).to(tl.int32), axis=0)
    blocks_all = (counts + BM - 1) // BM
    first = tl.sum(tl.where(ee < e, blocks_all, 0), axis=0)
    if e == E - 1:
        b = tl.arange(0, NBB)
        tl.store(EXPERT + b, -1, (b >= tl.sum(blocks_all, axis=0)) & (b < NB))
    s = tl.arange(0, S)
    ids = tl.load(IDS + s, s < M * 8, other=-1)
    hit = (ids == START + e) & (s < M * 8)
    count = tl.sum(hit.to(tl.int32), axis=0)
    blocks = tl.cdiv(count, BM)
    if blocks > 0:
        eb = tl.arange(0, PER_E)
        tl.store(EXPERT + first + eb, e, eb < blocks)
        rank = tl.cumsum(hit.to(tl.int32), axis=0) - 1
        base = first * BM
        c = tl.arange(0, PER_E * BM)
        tl.store(SORTED + base + c, M * 8, c < blocks * BM)
        tl.debug_barrier()
        tl.store(SORTED + base + rank, s, hit)


@triton.jit
def _scatter_a4_replica(
    IDS,
    SORTED,
    EXPERT,
    LIDS,
    TAB,
    M: tl.constexpr,
    NE: tl.constexpr,
    E0: tl.constexpr,
    E: tl.constexpr,
    EP: tl.constexpr,
    K: tl.constexpr,
    KP: tl.constexpr,
    PASSES: tl.constexpr,
    S: tl.constexpr,
    PER_E: tl.constexpr,
    BM: tl.constexpr,
    NB: tl.constexpr,
    NBB: tl.constexpr,
):
    # _scatter_a4 over this rank's E = E0 + R local experts (home slots, then replica slots), after every program
    # derives the replica assignment every rank derives alike from the global ids. TAB is the rank's int32 table
    # (expert_replicas.build_table): per global id its entry, home slot and replica slot (-1 when none), then per
    # entry its home rank, then the entries' home / holder ranks packed 2 bits each in 32-bit words.
    e = tl.program_id(0)
    s = tl.arange(0, S)
    valid = s < M * 8
    ids = tl.load(IDS + s, valid, other=0)
    # Distinct active experts per rank, packed 8 bits per rank. min(hist, 1), not hist > 0: Triton 3.7 folds a
    # comparison of a histogram result to false.
    hist = tl.histogram(ids, NE, mask=valid)
    b = tl.arange(0, NE)
    load = tl.sum(tl.minimum(hist, 1) << (8 * (b // E0)), axis=0)
    ent = tl.load(TAB + ids, valid, other=-1)
    k = tl.arange(0, KP)
    act = tl.minimum(tl.histogram(ent, KP, mask=valid & (ent >= 0)), 1)
    act_bits = tl.sum(act << k, axis=0)
    # Fixed load = distinct active experts that only their home rank holds.
    load -= tl.sum(act << (8 * tl.load(TAB + 3 * NE + k)), axis=0)
    # Greedy: each active replicated expert to the less loaded of home / holder (ties home); later passes take
    # each entry out and place it again against the others.
    pk = TAB + 3 * NE + KP
    hp0, hp1, op0, op1 = tl.load(pk), tl.load(pk + 1), tl.load(pk + 2), tl.load(pk + 3)
    home_bits = act_bits * 0
    for p in tl.static_range(PASSES):
        for i in tl.static_range(K):
            a = (act_bits >> i) & 1
            if i < 16:
                h = (hp0 >> (2 * i)) & 3
                o = (op0 >> (2 * i)) & 3
            else:
                h = (hp1 >> (2 * (i - 16))) & 3
                o = (op1 >> (2 * (i - 16))) & 3
            if p > 0:
                was = tl.where(((home_bits >> i) & 1) != 0, h, o)
                load -= a << (8 * was)
            go_home = ((load >> (8 * h)) & 255) <= ((load >> (8 * o)) & 255)
            load += a << (8 * tl.where(go_home, h, o))
            home_bits = tl.where(go_home, home_bits | (1 << i), home_bits & ~(1 << i))
    # Local expert index of every route (-1: another rank runs it).
    on_home = ((home_bits >> tl.maximum(ent, 0)) & 1) != 0
    home_slot = tl.load(TAB + NE + ids, valid, other=-1)
    rep_slot = tl.load(TAB + 2 * NE + ids, valid, other=-1)
    lid = tl.where((ent < 0) | on_home, home_slot, rep_slot)
    lid = tl.where(valid, lid, -1)
    if e == 0:
        tl.store(LIDS + s, lid, valid)
    ee = tl.arange(0, EP)
    counts = tl.histogram(lid, EP, mask=lid >= 0)
    blocks_all = (counts + BM - 1) // BM
    first = tl.sum(tl.where(ee < e, blocks_all, 0), axis=0)
    if e == E - 1:
        bb = tl.arange(0, NBB)
        tl.store(EXPERT + bb, -1, (bb >= tl.sum(blocks_all, axis=0)) & (bb < NB))
    hit = lid == e
    count = tl.sum(hit.to(tl.int32), axis=0)
    blocks = tl.cdiv(count, BM)
    if blocks > 0:
        eb = tl.arange(0, PER_E)
        tl.store(EXPERT + first + eb, e, eb < blocks)
        rank = tl.cumsum(hit.to(tl.int32), axis=0) - 1
        base = first * BM
        c = tl.arange(0, PER_E * BM)
        tl.store(SORTED + base + c, M * 8, c < blocks * BM)
        tl.debug_barrier()
        tl.store(SORTED + base + rank, s, hit)


@triton.jit
def _load_w(W, n0, k0, BN: tl.constexpr, KB: tl.constexpr, SHUF: tl.constexpr):
    """[KB bytes, BN rows] tile of one expert's [N, 2048] packed E2M1 rows from row n0, byte k0. Preshuffled layout:
    [N/16, 2048/16, 16 rows, 16 bytes] (aiter shuffle_weight (16, 16) and shuffle_weight_a16w4 (16) agree on it for
    K 2048), loaded as whole contiguous 16-row groups and permuted in registers."""
    if SHUF:
        g = tl.arange(0, BN // 16)
        kc = tl.arange(0, KB // 16)
        r = tl.arange(0, 16)
        b = tl.arange(0, 16)
        offset = (
            ((n0 // 16 + g) * 32768)[:, None, None, None]
            + ((k0 // 16 + kc) * 256)[None, :, None, None]
            + (r * 16)[None, None, :, None]
            + b[None, None, None, :]
        )
        x = tl.permute(tl.load(W + offset), (1, 3, 0, 2))
        return tl.reshape(x, (KB, BN))
    n = n0 + tl.arange(0, BN)
    k = k0 + tl.arange(0, KB)
    return tl.load(W + n[None, :] * 2048 + k[:, None])


@triton.jit
def _s_offset(n, c, SHUF: tl.constexpr):
    """[N, scale col] offsets in one expert's [N, 128] E8M0 scales. Preshuffled (aiter shuffle_scale, plain and a16w4
    alike): [N/32, 128/8, 4 (c % 4), 16 (n % 16), 2 ((c % 8) // 4), 2 ((n % 32) // 16)]."""
    if SHUF:
        n2, c2 = n[:, None], c[None, :]
        return (
            ((((n2 // 32) * 16 + c2 // 8) * 4 + c2 % 4) * 16 + n2 % 16) * 4
            + ((c2 % 8) // 4) * 2
            + (n2 % 32) // 16
        )
    return n[:, None] * 128 + c[None, :]


@triton.jit
def _expert_a4(
    A,
    AS,
    W,
    WS,
    SORTED,
    EXPERT,
    OUT,
    OS,
    ROWS: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ACT: tl.constexpr,
    BM: tl.constexpr,
    SHUF: tl.constexpr,
):
    expert_block, n_block = tl.program_id(1), tl.program_id(0)
    expert = tl.load(EXPERT + expert_block)
    if expert >= 0:
        slot = tl.load(SORTED + expert_block * BM + tl.arange(0, BM))
        valid = slot < ROWS
        src = slot // 8 if ACT else slot
        n = n_block * BN + tl.arange(0, BN)
        kp = tl.arange(0, BK // 2)
        ks = tl.arange(0, BK // 32)
        wn: tl.constexpr = 8192 if ACT else 4096
        acc = tl.full((BM, BN), 0, tl.float32)
        if ACT:
            up = tl.full((BM, BN), 0, tl.float32)
        for tile in range(4096 // BK):
            a = tl.load(
                A + src[:, None] * 2048 + tile * (BK // 2) + kp[None, :],
                valid[:, None],
                other=0,
            )
            asc = tl.load(
                AS + src[:, None] * 128 + tile * (BK // 32) + ks[None, :],
                valid[:, None],
                other=127,
            )
            w = _load_w(
                W + expert * (wn * 2048),
                n_block * BN,
                tile * (BK // 2),
                BN,
                BK // 2,
                SHUF,
            )
            ws = tl.load(
                WS + expert * (wn * 128) + _s_offset(n, tile * (BK // 32) + ks, SHUF)
            )
            acc = tl.dot_scaled(a, asc, "e2m1", w, ws, "e2m1", acc)
            if ACT:
                wu = _load_w(
                    W + expert * (wn * 2048),
                    n_block * BN + 4096,
                    tile * (BK // 2),
                    BN,
                    BK // 2,
                    SHUF,
                )
                wsu = tl.load(
                    WS
                    + expert * (wn * 128)
                    + _s_offset(n + 4096, tile * (BK // 32) + ks, SHUF)
                )
                up = tl.dot_scaled(a, asc, "e2m1", wu, wsu, "e2m1", up)
        if ACT:
            activated = (acc * tl.sigmoid(acc) * up).to(tl.bfloat16).to(tl.float32)
            q, s = _mxfp4_quant_op(activated, BN, BM, 32)
            pn = n_block * (BN // 2) + tl.arange(0, BN // 2)
            sn = n_block * (BN // 32) + tl.arange(0, BN // 32)
            tl.store(OUT + slot[:, None] * 2048 + pn[None, :], q, valid[:, None])
            tl.store(OS + slot[:, None] * 128 + sn[None, :], s, valid[:, None])
        else:
            # Preserve the BF16 per-slot boundary before the Muse finalizer.
            tl.store(OUT + slot[:, None] * 4096 + n[None, :], acc, valid[:, None])


def compact_a4_applies(m: int, h: int, w13, w2, s13, s2) -> bool:
    e = w13.shape[0]
    return (
        0 < m <= MAX_ROWS
        and m % 8 == 0
        and h == 4096
        and e in (32, 64)
        and w13.shape == (e, 8192, 2048)
        and w2.shape == (e, 4096, 2048)
        and s13.shape == (e, 8192, 128)
        and s2.shape == (e, 4096, 128)
        and all(
            t.dtype == torch.uint8 and t.is_contiguous() for t in (w13, w2, s13, s2)
        )
    )


def compact_a4_experts(
    x, ids, w13, w2, s13, s2, *, local_start, shuffled=False, quantized=None
):
    """Per-(token, slot) expert outputs BF16 [M * 8, 4096] for the 32 or 64 local experts starting at ``local_start``.

    Weights and scales are uint8 [E, N, K] views, as stored or (``shuffled``) in the a4w4 preshuffle layout.
    ``quantized`` = the input's per-row MXFP4 (uint8 [M, 2048], uint8 [M, 128]) in place of quantizing ``x``."""
    m, h = ids.shape[0], w2.shape[1]
    if not (
        compact_a4_applies(m, h, w13, w2, s13, s2)
        and ids.shape == (m, 8)
        and (quantized is not None or (x.dtype == torch.bfloat16 and x.shape == (m, h)))
    ):
        raise ValueError(
            f"A4 requires M a multiple of 8 up to {MAX_ROWS}, local32 / local64 MXFP4 expert weights"
        )
    dev = ids.device
    bm, e_local = BLOCK_M, w13.shape[0]
    rows, cap = m * 8, m * 8 + e_local * (bm - 1)
    nb = triton.cdiv(cap, bm)
    sid = torch.empty(cap, device=dev, dtype=torch.int32)
    eid = torch.empty(nb, device=dev, dtype=torch.int32)
    inter = torch.empty((rows, 2048), device=dev, dtype=torch.uint8)
    inter_scale = torch.empty((rows, 128), device=dev, dtype=torch.uint8)
    y = torch.empty((rows, h), device=dev, dtype=torch.bfloat16)
    s_pow2, per_e = (
        triton.next_power_of_2(rows),
        triton.next_power_of_2(triton.cdiv(rows, bm)),
    )
    if quantized is None:
        q = torch.empty((m, 2048), device=dev, dtype=torch.uint8)
        qs = torch.empty((m, 128), device=dev, dtype=torch.uint8)
        _prepare_a4[(m * 4,)](x, q, qs, num_warps=4)
    else:
        q, qs = quantized
    _scatter_a4[(e_local,)](
        ids,
        sid,
        eid,
        m,
        local_start,
        e_local,
        s_pow2,
        min(s_pow2, 256),
        per_e,
        bm,
        nb,
        triton.next_power_of_2(nb),
        num_warps=4,
    )
    grid = (4096 // 64, nb)
    options = dict(num_warps=4, num_stages=2, matrix_instr_nonkdim=16)
    _expert_a4[grid](
        q,
        qs,
        w13,
        s13,
        sid,
        eid,
        inter,
        inter_scale,
        rows,
        64,
        512,
        True,
        bm,
        shuffled,
        **options,
    )
    _expert_a4[grid](
        inter,
        inter_scale,
        w2,
        s2,
        sid,
        eid,
        y,
        y,
        rows,
        64,
        512,
        False,
        bm,
        shuffled,
        **options,
    )
    return y


def _launch_experts(q, qs, w13, w2, s13, s2, sid, eid, rows, nb, shuffled):
    """Gate/up + down ``_expert_a4`` launches over the scattered blocks; per-slot BF16 rows [rows, 4096]."""
    dev, bm = q.device, BLOCK_M
    inter = torch.empty((rows, 2048), device=dev, dtype=torch.uint8)
    inter_scale = torch.empty((rows, 128), device=dev, dtype=torch.uint8)
    y = torch.empty((rows, 4096), device=dev, dtype=torch.bfloat16)
    grid = (4096 // 64, nb)
    options = dict(num_warps=4, num_stages=2, matrix_instr_nonkdim=16)
    _expert_a4[grid](
        q,
        qs,
        w13,
        s13,
        sid,
        eid,
        inter,
        inter_scale,
        rows,
        64,
        512,
        True,
        bm,
        shuffled,
        **options,
    )
    _expert_a4[grid](
        inter,
        inter_scale,
        w2,
        s2,
        sid,
        eid,
        y,
        y,
        rows,
        64,
        512,
        False,
        bm,
        shuffled,
        **options,
    )
    return y


def _quantized_input(x, m, quantized):
    if quantized is not None:
        return quantized
    q = torch.empty((m, 2048), device=x.device, dtype=torch.uint8)
    qs = torch.empty((m, 128), device=x.device, dtype=torch.uint8)
    _prepare_a4[(m * 4,)](x, q, qs, num_warps=4)
    return q, qs


# Greedy sweeps over the replicated experts per step (replica_plan.py: 2 sweeps reach most of the gain of 3).
REPLICA_PASSES = 2


def compact_a4_replicated_applies(m: int, h: int, w13, w2, s13, s2, home: int) -> bool:
    e = w13.shape[0]
    return (
        0 < m <= MAX_ROWS
        and m % 8 == 0
        and h == 4096
        and home == 64
        and home < e <= 128
        and w13.shape == (e, 8192, 2048)
        and w2.shape == (e, 4096, 2048)
        and s13.shape == (e, 8192, 128)
        and s2.shape == (e, 4096, 128)
        and all(
            t.dtype == torch.uint8 and t.is_contiguous() for t in (w13, w2, s13, s2)
        )
    )


def compact_a4_experts_replicated(
    x,
    ids,
    w13,
    w2,
    s13,
    s2,
    *,
    table,
    num_entries: int,
    home: int = 64,
    num_experts: int = 256,
    shuffled=False,
    quantized=None,
):
    """(per-slot rows BF16 [M * 8, 4096], local expert index int32 [M, 8]) of one rank with replicas.

    The weights are uint8 [home + R, N, K] views: the rank's ``home`` experts, then its R replicas. ``ids`` are the
    global slot ids every rank holds alike; ``table`` is the rank's int32 replica table
    (``expert_replicas.build_table``) over ``num_entries`` replicated experts. A route's local index is -1 when
    another rank computes it, so the finalize takes local_start 0 and ``home + R`` local experts."""
    m, h = ids.shape[0], w2.shape[1]
    kp = triton.next_power_of_2(num_entries)
    if not (
        compact_a4_replicated_applies(m, h, w13, w2, s13, s2, home)
        and ids.shape == (m, 8)
        and ids.dtype == torch.int32
        and num_experts % home == 0
        and num_experts // home <= 4
        and 0 < num_entries <= 31
        and table.dtype == torch.int32
        and table.shape == (3 * num_experts + kp + 4,)
        and (quantized is not None or (x.dtype == torch.bfloat16 and x.shape == (m, h)))
    ):
        raise ValueError("A4 replicas: unsupported shapes")
    dev = ids.device
    bm, e_local = BLOCK_M, w13.shape[0]
    rows, cap = m * 8, m * 8 + e_local * (bm - 1)
    nb = triton.cdiv(cap, bm)
    sid = torch.empty(cap, device=dev, dtype=torch.int32)
    eid = torch.empty(nb, device=dev, dtype=torch.int32)
    lids = torch.empty((m, 8), device=dev, dtype=torch.int32)
    q, qs = _quantized_input(x, m, quantized)
    _scatter_a4_replica[(e_local,)](
        ids,
        sid,
        eid,
        lids,
        table,
        m,
        num_experts,
        home,
        e_local,
        triton.next_power_of_2(e_local),
        num_entries,
        kp,
        REPLICA_PASSES,
        triton.next_power_of_2(rows),
        triton.next_power_of_2(triton.cdiv(rows, bm)),
        bm,
        nb,
        triton.next_power_of_2(nb),
        num_warps=4,
    )
    y = _launch_experts(q, qs, w13, w2, s13, s2, sid, eid, rows, nb, shuffled)
    return y, lids
