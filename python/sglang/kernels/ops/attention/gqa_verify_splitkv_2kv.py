"""Target-verify attention, a KV-head pair per program: full layers, 8 verify tokens, GQA 8, 2 or 4 KV heads per rank
(16 / 32 Q heads: TP8 / TP4), HD64, FP8 KV, page 16.

One program per (request, split, KV-head pair) owns both heads of its pair (2 x 64 rows = 8 tokens x 16 heads), so
every 128-byte KV line (head 2p 64 B | head 2p+1 64 B) and every page-table entry is fetched once by one CU.
QK on FP8 MFMA: Q is quantized per row (token x head, amax/448) in the prologue, K stays FP8 straight from the
cache; the row scale, the K descale, softmax scale and log2(e) fold into one per-row multiplier used by the exp2 fma.
PV in BF16 (V upcast) by default, FP8 PV (P biased by 2^8) as a variant.
Interior tiles run without any mask (causal/verify mask and kv length only matter on the last 1-2 tiles).
Partials use the layout [T, H, S, 64] fp32 + max/denom [T, H, S] of gqa_verify_splitkv; the reduce is the same
math. The exact mode (BF16 QK and PV) is bitwise equal to the GQA-folded and unified 3D paths; the ``fp8`` mode
(QF8 + PVF8 + page-id prefetch) is the FP8 numerics of the prefill FP8 path, ~15-20% faster at 4 KV heads.
"""

import torch
import triton
import triton.language as tl

QL = 8
HD = 64
GQA = 8


@triton.jit
def _vv2_tile(
    acc0,
    acc1,
    m0,
    m1,
    l0,
    l1,
    q0,
    q1,
    q0l,
    q1l,
    c0,
    c1,
    K,
    V,
    blocks,
    tile,
    kv_s0,
    kv_s1,
    nr,
    d,
    context,
    kvlen,
    qrel,
    BN: tl.constexpr,
    PAGE: tl.constexpr,
    MASKED: tl.constexpr,
    QF8: tl.constexpr,
    PVF8: tl.constexpr,
    QSPLIT: tl.constexpr,
):
    positions = tile * BN + nr
    if MASKED:
        valid = positions < kvlen
    # 64-bit page offsets: a layer's pool can exceed 2^31 bytes
    kb = blocks.to(tl.int64) * kv_s0 + (nr % PAGE) * kv_s1
    if MASKED:
        k0 = tl.load(K + kb[None, :] + d[:, None], mask=valid[None, :], other=0.0)
        k1 = tl.load(K + kb[None, :] + 64 + d[:, None], mask=valid[None, :], other=0.0)
        v0 = tl.load(V + kb[:, None] + d[None, :], mask=valid[:, None], other=0.0)
        v1 = tl.load(V + kb[:, None] + 64 + d[None, :], mask=valid[:, None], other=0.0)
    else:
        k0 = tl.load(K + kb[None, :] + d[:, None])
        k1 = tl.load(K + kb[None, :] + 64 + d[:, None])
        v0 = tl.load(V + kb[:, None] + d[None, :])
        v1 = tl.load(V + kb[:, None] + 64 + d[None, :])
    if QSPLIT:
        # Q = hi + lo, both FP8 (about bf16 precision for Q); K stays FP8 from the cache
        s0 = tl.dot(q0l, k0, acc=tl.dot(q0, k0))
        s1 = tl.dot(q1l, k1, acc=tl.dot(q1, k1))
    elif QF8:
        s0 = tl.dot(q0, k0)
        s1 = tl.dot(q1, k1)
    else:
        s0 = tl.dot(q0, k0.to(tl.bfloat16))
        s1 = tl.dot(q1, k1.to(tl.bfloat16))
    if MASKED:
        vis = (positions[None, :] <= (context + qrel)[:, None]) & valid[None, :]
        s0 = tl.where(vis, s0, float("-inf"))
        s1 = tl.where(vis, s1, float("-inf"))
    n0 = tl.maximum(m0, tl.max(s0, axis=1) * c0)
    n1 = tl.maximum(m1, tl.max(s1, axis=1) * c1)
    if MASKED:
        n0 = tl.where(n0 > float("-inf"), n0, 0.0)
        n1 = tl.where(n1 > float("-inf"), n1, 0.0)
    if PVF8:
        # bias 2^8 keeps P inside e4m3's normal range; it rides through acc and denom and cancels in the reduce
        p0 = tl.exp2(tl.fma(s0, c0[:, None], (8.0 - n0)[:, None]))
        p1 = tl.exp2(tl.fma(s1, c1[:, None], (8.0 - n1)[:, None]))
    else:
        p0 = tl.exp2(tl.fma(s0, c0[:, None], -n0[:, None]))
        p1 = tl.exp2(tl.fma(s1, c1[:, None], -n1[:, None]))
    r0 = tl.exp2(m0 - n0)
    r1 = tl.exp2(m1 - n1)
    l0 = l0 * r0 + tl.sum(p0, axis=1)
    l1 = l1 * r1 + tl.sum(p1, axis=1)
    if PVF8:
        acc0 = tl.dot(p0.to(tl.float8e4nv), v0, acc=acc0 * r0[:, None])
        acc1 = tl.dot(p1.to(tl.float8e4nv), v1, acc=acc1 * r1[:, None])
    else:
        acc0 = tl.dot(p0.to(tl.bfloat16), v0.to(tl.bfloat16), acc=acc0 * r0[:, None])
        acc1 = tl.dot(p1.to(tl.bfloat16), v1.to(tl.bfloat16), acc=acc1 * r1[:, None])
    return acc0, acc1, n0, n1, l0, l1


@triton.jit
def _vv2_partials(
    Q,
    K,
    V,
    SEQ,
    TABLE,
    PARTIAL,
    MAXIMUM,
    DENOM,
    K_SCALE,
    q_stride_0,
    q_stride_1,
    kv_s0: tl.constexpr,
    kv_s1: tl.constexpr,
    table_stride,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
    QF8: tl.constexpr,
    PVF8: tl.constexpr,
    PVBIAS_LOG2: tl.constexpr,
    QSPLIT: tl.constexpr,
    TPF: tl.constexpr,
):
    slot, pair = tl.program_id(0), tl.program_id(1)
    seq = slot // SPLITS
    split = slot % SPLITS
    K += pair * 128
    V += pair * 128
    kvlen = tl.load(SEQ + seq)
    context = kvlen - 8
    mr = tl.arange(0, 64)
    d = tl.arange(0, 64)
    nr = tl.arange(0, BN)
    qrel = mr // 8
    qrow = seq * 8 + qrel
    head0 = pair * 16 + mr % 8
    head1 = 8 + head0
    q0 = tl.load(
        Q + qrow[:, None] * q_stride_0 + head0[:, None] * q_stride_1 + d[None, :]
    ).to(tl.float32)
    q1 = tl.load(
        Q + qrow[:, None] * q_stride_0 + head1[:, None] * q_stride_1 + d[None, :]
    ).to(tl.float32)
    scale = SCALE * 1.4426950408889634
    if K_SCALE is not None:
        scale *= tl.load(K_SCALE)
    if QF8:
        qs0 = tl.maximum(tl.max(tl.abs(q0), axis=1), 1e-20) / 448.0
        qs1 = tl.maximum(tl.max(tl.abs(q1), axis=1), 1e-20) / 448.0
        q0s = q0 / qs0[:, None]
        q1s = q1 / qs1[:, None]
        q0c = q0s.to(tl.float8e4nv)
        q1c = q1s.to(tl.float8e4nv)
        if QSPLIT:
            q0l = (q0s - q0c.to(tl.float32)).to(tl.float8e4nv)
            q1l = (q1s - q1c.to(tl.float32)).to(tl.float8e4nv)
        else:
            q0l = q0c
            q1l = q1c
        c0 = qs0 * scale
        c1 = qs1 * scale
    else:
        q0c = q0.to(tl.bfloat16)
        q1c = q1.to(tl.bfloat16)
        q0l = q0c
        q1l = q1c
        c0 = tl.full((64,), 1.0, tl.float32) * scale
        c1 = tl.full((64,), 1.0, tl.float32) * scale
    tiles = tl.cdiv(kvlen, BN)
    tps = tl.cdiv(tiles, SPLITS)
    begin = split * tps
    end = tl.minimum(tiles, begin + tps)
    interior_end = tl.minimum(end, (context + 1) // BN)
    m0 = tl.full((64,), float("-inf"), tl.float32)
    m1 = tl.full((64,), float("-inf"), tl.float32)
    l0 = tl.zeros((64,), tl.float32)
    l1 = tl.zeros((64,), tl.float32)
    acc0 = tl.zeros((64, 64), tl.float32)
    acc1 = tl.zeros((64, 64), tl.float32)
    table_row = TABLE + seq * table_stride
    # page ids are software-pipelined one tile ahead of the KV loads (the table load latency is otherwise exposed
    # every iteration: table -> wait -> KV address)
    pos0 = begin * BN + nr
    nxt_blocks = tl.load(table_row + pos0 // PAGE, mask=pos0 < kvlen, other=0)
    for tile in range(begin, interior_end):
        blocks = nxt_blocks
        if TPF:
            pos1 = (tile + 1) * BN + nr
            nxt_blocks = tl.load(table_row + pos1 // PAGE, mask=pos1 < kvlen, other=0)
        else:
            blocks = tl.load(table_row + (tile * BN + nr) // PAGE)
        acc0, acc1, m0, m1, l0, l1 = _vv2_tile(
            acc0,
            acc1,
            m0,
            m1,
            l0,
            l1,
            q0c,
            q1c,
            q0l,
            q1l,
            c0,
            c1,
            K,
            V,
            blocks,
            tile,
            kv_s0,
            kv_s1,
            nr,
            d,
            context,
            kvlen,
            qrel,
            BN,
            PAGE,
            False,
            QF8,
            PVF8,
            QSPLIT,
        )
    for tile in range(tl.maximum(begin, interior_end), end):
        pos1 = tile * BN + nr
        blocks = tl.load(table_row + pos1 // PAGE, mask=pos1 < kvlen, other=0)
        acc0, acc1, m0, m1, l0, l1 = _vv2_tile(
            acc0,
            acc1,
            m0,
            m1,
            l0,
            l1,
            q0c,
            q1c,
            q0l,
            q1l,
            c0,
            c1,
            K,
            V,
            blocks,
            tile,
            kv_s0,
            kv_s1,
            nr,
            d,
            context,
            kvlen,
            qrel,
            BN,
            PAGE,
            True,
            QF8,
            PVF8,
            QSPLIT,
        )
    i0 = (qrow * HEADS + head0) * SPLITS + split
    i1 = (qrow * HEADS + head1) * SPLITS + split
    tl.store(MAXIMUM + i0, m0)
    tl.store(MAXIMUM + i1, m1)
    tl.store(DENOM + i0, l0)
    tl.store(DENOM + i1, l1)
    tl.store(PARTIAL + i0[:, None] * 64 + d[None, :], acc0)
    tl.store(PARTIAL + i1[:, None] * 64 + d[None, :], acc1)


@triton.jit
def _vv2_reduce(
    PARTIAL,
    MAXIMUM,
    DENOM,
    OUT,
    V_SCALE,
    HEADS: tl.constexpr,
    SPLITS: tl.constexpr,
    o_stride_0,
    o_stride_1,
):
    # identical math to production _verify_reduce
    token, head = tl.program_id(0), tl.program_id(1)
    segments = tl.arange(0, SPLITS)
    d = tl.arange(0, 64)
    index = (token * HEADS + head) * SPLITS + segments
    maximum = tl.load(MAXIMUM + index)
    denom = tl.load(DENOM + index)
    global_max = tl.max(tl.where(denom > 0, maximum, float("-inf")), axis=0)
    global_max = tl.where(global_max > float("-inf"), global_max, 0.0)
    weights = tl.where(denom > 0, tl.exp2(maximum - global_max), 0.0)
    total = tl.sum(denom * weights, axis=0)
    partial = tl.load(PARTIAL + index[:, None] * 64 + d[None, :])
    accum = tl.sum(partial * weights[:, None], axis=0)
    inverse = tl.where(total > 0, 1.0 / total, 0.0)
    if V_SCALE is not None:
        inverse *= tl.load(V_SCALE)
    tl.store(OUT + token * o_stride_0 + head * o_stride_1 + d, accum * inverse)


def pick_splits_v2(num_seqs: int, max_kv: int = 0, pairs: int = 1) -> int:
    # >= 512 CTAs (2 per CU) when the batch allows it: bs 1 -> 256 ... bs 8 -> 64, bs 16..64 -> 32 (one pair)
    s = 32
    while s < 256 and s * num_seqs * pairs < 512:
        s *= 2
    return s


def gqa_verify_splitkv_2kv(
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
    splits=None,
    bn=64,
    num_warps=4,
    num_stages=2,
    defer_reduce=False,
    fp8=False,
):
    """Full-attention verify, 8 tokens x 16 Q heads x 2 KV heads per (request, split, pair) program, BF16 QK / PV;
    ``fp8`` = per-row FP8 Q on the FP8 K straight from the cache and FP8 P for P.V (TP4 n8: 150 vs 178 us)."""
    num_seqs = seqused_k.numel()
    heads, kv_heads = q.shape[1], k.shape[2]
    pairs = kv_heads // 2
    assert q.dtype == torch.bfloat16 and k.dtype == v.dtype
    assert q.shape[-1] == HD and kv_heads in (2, 4) and heads == GQA * kv_heads
    assert k.shape[1] == 16
    assert (
        q.shape[0] == num_seqs * QL
        and k.stride() == v.stride()
        and k.stride(2) == 64
        and k.stride(3) == 1
    )
    splits = splits or pick_splits_v2(num_seqs, pairs=pairs)
    state_shape = (q.shape[0], heads, splits)
    partial = torch.empty((*state_shape, HD), dtype=torch.float32, device=q.device)
    maximum = torch.empty(state_shape, dtype=torch.float32, device=q.device)
    denom = torch.empty(state_shape, dtype=torch.float32, device=q.device)
    _vv2_partials[(num_seqs * splits, pairs)](
        q,
        k,
        v,
        seqused_k,
        block_table,
        partial,
        maximum,
        denom,
        k_descale,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        block_table.stride(0),
        heads,
        k.shape[1],
        splits,
        float(softmax_scale),
        bn,
        fp8,
        fp8,
        8,
        False,
        fp8,
        num_warps=num_warps,
        num_stages=num_stages,
        matrix_instr_nonkdim=16,
    )
    if defer_reduce:
        # the caller merges the partials itself (verify_reduce_norm_gate)
        return partial, maximum, denom
    _vv2_reduce[(q.shape[0], heads)](
        partial,
        maximum,
        denom,
        out,
        v_descale,
        heads,
        splits,
        out.stride(0),
        out.stride(1),
        num_warps=2,
    )
    return out
