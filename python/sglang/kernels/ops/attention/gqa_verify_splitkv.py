"""GQA-folded split-KV attention for target verify (8 query tokens, GQA 8, HD64, FP8 KV, page 16).

One program owns every query row of a request for one KV head (8 tokens x 8 heads = 64 rows), so each
KV tile is read once per (request, KV head). AITER's unified 3D path uses 16-row tiles here and reads the
same KV four times plus an empty tile, which dominates full-attention layers at long context.
Handles full layers (WINDOW == 0) and sliding layers (WINDOW > 0) with the causal and left-window masks
of AITER unified attention; BF16 Q, FP8 KV upcast to BF16, exp2 online softmax, FP32 partials.
"""

import torch
import triton
import triton.language as tl

QL = 8  # verify tokens per request
GQA = 8  # query heads per KV head
HD = 64


@triton.jit
def _verify_partials(
    Q,
    K,
    V,
    CU,
    SEQ,
    TABLE,
    PARTIAL,
    MAXIMUM,
    DENOM,
    K_SCALE,
    q_stride_0: tl.constexpr,
    q_stride_1: tl.constexpr,
    k_stride_0: tl.constexpr,
    k_stride_1: tl.constexpr,
    k_stride_2: tl.constexpr,
    k_stride_3: tl.constexpr,
    v_stride_0: tl.constexpr,
    v_stride_1: tl.constexpr,
    v_stride_2: tl.constexpr,
    v_stride_3: tl.constexpr,
    table_stride: tl.constexpr,
    HEADS: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    BN: tl.constexpr,
):
    kvhead, slot = tl.program_id(0), tl.program_id(1)
    seq, split = slot // SPLITS, slot % SPLITS
    qstart = tl.load(CU + seq)
    qlen = tl.load(CU + seq + 1) - qstart
    kvlen = tl.load(SEQ + seq)
    context = kvlen - qlen
    BM: tl.constexpr = 64
    mr = tl.arange(0, BM)
    d = tl.arange(0, 64)
    nr = tl.arange(0, BN)
    qrel = mr // 8
    qhead = kvhead * 8 + mr % 8
    qvalid = qrel < qlen
    qrow = qstart + qrel
    q = tl.load(
        Q + qrow[:, None] * q_stride_0 + qhead[:, None] * q_stride_1 + d[None, :],
        mask=qvalid[:, None],
        other=0,
    )
    if WINDOW > 0:
        low = tl.maximum(0, context - WINDOW + 1)
    else:
        low = 0
    high = tl.maximum(0, kvlen)
    tile_start = low // BN
    tile_stop = tl.cdiv(high, BN)
    tiles_per_split = tl.cdiv(tl.maximum(tile_stop - tile_start, 0), SPLITS)
    begin = tile_start + split * tiles_per_split
    end = tl.minimum(tile_stop, begin + tiles_per_split)
    scale = SCALE * 1.4426950408889634
    if K_SCALE is not None:
        scale *= tl.load(K_SCALE)
    maximum = tl.full((BM,), float("-inf"), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    accum = tl.zeros((BM, 64), tl.float32)
    absolute_query = context + qrel
    for tile in range(begin, end):
        positions = tile * BN + nr
        valid = (positions >= low) & (positions < high)
        blocks = tl.load(
            TABLE + seq * table_stride + positions // PAGE, mask=valid, other=0
        ).to(tl.int64)
        in_page = positions % PAGE
        k = tl.load(
            K
            + blocks[None, :] * k_stride_0
            + in_page[None, :] * k_stride_1
            + kvhead * k_stride_2
            + d[:, None] * k_stride_3,
            mask=valid[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        v = tl.load(
            V
            + blocks[:, None] * v_stride_0
            + in_page[:, None] * v_stride_1
            + kvhead * v_stride_2
            + d[None, :] * v_stride_3,
            mask=valid[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        scores = tl.dot(q, k) * scale
        mask = (
            qvalid[:, None]
            & valid[None, :]
            & (positions[None, :] <= absolute_query[:, None])
        )
        if WINDOW > 0:
            mask = mask & (absolute_query[:, None] - positions[None, :] < WINDOW)
        scores = tl.where(mask, scores, float("-inf"))
        new_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        new_maximum = tl.where(new_maximum > float("-inf"), new_maximum, 0.0)
        probabilities = tl.exp2(scores - new_maximum[:, None])
        rescale = tl.exp2(maximum - new_maximum)
        denom = denom * rescale + tl.sum(probabilities, axis=1)
        accum = tl.dot(probabilities.to(tl.bfloat16), v, acc=accum * rescale[:, None])
        maximum = new_maximum
    state_index = (qrow * HEADS + qhead) * SPLITS + split
    tl.store(MAXIMUM + state_index, maximum, mask=qvalid)
    tl.store(DENOM + state_index, denom, mask=qvalid)
    tl.store(
        PARTIAL + state_index[:, None] * 64 + d[None, :], accum, mask=qvalid[:, None]
    )


@triton.jit
def _verify_reduce(
    PARTIAL,
    MAXIMUM,
    DENOM,
    OUT,
    V_SCALE,
    HEADS: tl.constexpr,
    SPLITS: tl.constexpr,
    o_stride_0: tl.constexpr,
    o_stride_1: tl.constexpr,
):
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


def pick_splits(num_seqs: int, window: int) -> int:
    # Aim for >= ~256 workgroups over 2 KV heads; a sliding window holds at most 2055 keys.
    target = 256 // max(1, num_seqs * 2)
    cap = 16 if window > 0 else 128
    splits = 1
    while splits * 2 <= min(cap, max(1, target)):
        splits *= 2
    return max(splits, 4 if window <= 0 else 2)


def gqa_verify_splitkv(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    block_table,
    softmax_scale,
    window,
    k_descale,
    v_descale,
    *,
    splits=None,
    bn=64,
    num_warps=4,
    num_stages=2,
    defer_reduce=False,
):
    """Verify attention, 8 tokens x GQA 8 per (request, KV head) program; ``window`` 0 = full attention."""
    num_seqs = seqused_k.numel()
    heads = q.shape[1]
    assert q.dtype == torch.bfloat16 and k.dtype == v.dtype
    assert q.shape[-1] == k.shape[-1] == v.shape[-1] == HD and heads == k.shape[2] * GQA
    assert q.shape[0] == num_seqs * QL and k.shape[1] == v.shape[1]
    splits = splits or pick_splits(num_seqs, window)
    state_shape = (q.shape[0], heads, splits)
    partial = torch.empty((*state_shape, HD), dtype=torch.float32, device=q.device)
    maximum = torch.empty(state_shape, dtype=torch.float32, device=q.device)
    denom = torch.empty(state_shape, dtype=torch.float32, device=q.device)
    _verify_partials[(k.shape[2], num_seqs * splits)](
        q,
        k,
        v,
        cu_seqlens_q,
        seqused_k,
        block_table,
        partial,
        maximum,
        denom,
        k_descale,
        q.stride(0),
        q.stride(1),
        *k.stride(),
        *v.stride(),
        block_table.stride(0),
        heads,
        k.shape[1],
        int(window),
        splits,
        float(softmax_scale),
        bn,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if defer_reduce:
        # the caller merges the partials itself (verify_reduce_norm_gate)
        return partial, maximum, denom
    _verify_reduce[(q.shape[0], heads)](
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
