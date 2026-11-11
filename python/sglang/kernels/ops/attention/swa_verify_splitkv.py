"""Split-KV target-verify attention on a sliding window (Muse Spark shape).

8 verify tokens per request, GQA 8, head_dim 64, BF16 Q, FP8 KV in 16-token pages, 2047-token window. The
window is split across workgroups (exact causal and left-window masks); FP32 partials are merged by a second
kernel. Used for up to 8 requests, where the unified kernel's grid is too small.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _swa_verify_partials(
    Q,
    K,
    V,
    CU,
    SEQ,
    TABLE,
    PARTIAL,
    MAXIMUM,
    DENOM,
    Q_SCALE,
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
    GQA: tl.constexpr,
    PAGE: tl.constexpr,
    WINDOW: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    qblock, kvhead, slot = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seq, split = slot // SPLITS, slot % SPLITS
    qstart = tl.load(CU + seq)
    qlen = tl.load(CU + seq + 1) - qstart
    kvlen = tl.load(SEQ + seq)
    context = kvlen - qlen
    mr = tl.arange(0, BM)
    d = tl.arange(0, 64)
    nr = tl.arange(0, BN)
    qrel = qblock * (BM // GQA) + mr // GQA
    qhead = kvhead * GQA + mr % GQA
    qvalid = qrel < qlen
    qrow = qstart + qrel
    q = tl.load(
        Q + qrow[:, None] * q_stride_0 + qhead[:, None] * q_stride_1 + d[None, :],
        mask=qvalid[:, None],
        other=0,
    )
    first_query = qblock * (BM // GQA)
    last_query = tl.minimum(first_query + BM // GQA - 1, qlen - 1)
    low = tl.maximum(0, context + first_query - WINDOW + 1)
    high = tl.maximum(0, tl.minimum(kvlen, context + last_query + 1))
    tile_start = low // BN
    tile_stop = tl.cdiv(high, BN)
    tiles_per_split = tl.cdiv(tl.maximum(tile_stop - tile_start, 0), SPLITS)
    begin = tile_start + split * tiles_per_split
    end = tl.minimum(tile_stop, begin + tiles_per_split)
    scale = SCALE * 1.4426950408889634
    if Q_SCALE is not None:
        scale *= tl.load(Q_SCALE)
    if K_SCALE is not None:
        scale *= tl.load(K_SCALE)
    maximum = tl.full((BM,), float("-inf"), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    accum = tl.zeros((BM, 64), tl.float32)
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
        absolute_query = context + qrel
        mask = (
            qvalid[:, None]
            & valid[None, :]
            & (positions[None, :] <= absolute_query[:, None])
            & (absolute_query[:, None] - positions[None, :] < WINDOW)
        )
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
def _swa_verify_reduce(
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


def swa_verify_splitkv(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    *,
    num_splits=8,
    num_stages=2,
    defer_reduce=False,
):
    assert q.dtype == torch.bfloat16 and k.dtype == v.dtype == torch.float8_e4m3fn
    assert (
        out.dtype == torch.bfloat16 and q.shape[-1] == k.shape[-1] == v.shape[-1] == 64
    )
    assert q.shape[1] // k.shape[2] == 8 and k.shape[2] == v.shape[2]
    assert k.shape[1] == v.shape[1] == 16
    assert max_seqlen_q == 8 and q.shape[0] == seqused_k.numel() * 8
    assert causal and window_size == (2046, 0) and softcap == 0
    assert num_splits in (2, 4, 8, 16, 32)
    state_shape = (q.shape[0], q.shape[1], num_splits)
    partial = torch.empty((*state_shape, 64), dtype=torch.float32, device=q.device)
    maximum = torch.empty(state_shape, dtype=torch.float32, device=q.device)
    denom = torch.empty(state_shape, dtype=torch.float32, device=q.device)
    _swa_verify_partials[(4, k.shape[2], seqused_k.numel() * num_splits)](
        q,
        k,
        v,
        cu_seqlens_q,
        seqused_k,
        block_table,
        partial,
        maximum,
        denom,
        q_descale,
        k_descale,
        q.stride(0),
        q.stride(1),
        *k.stride(),
        *v.stride(),
        block_table.stride(0),
        q.shape[1],
        8,
        16,
        2047,
        num_splits,
        softmax_scale,
        16,
        64,
        num_warps=2,
        num_stages=num_stages,
    )
    if defer_reduce:
        # the caller merges the partials itself (verify_reduce_norm_gate)
        return partial, maximum, denom
    _swa_verify_reduce[(q.shape[0], q.shape[1])](
        partial,
        maximum,
        denom,
        out,
        v_descale,
        q.shape[1],
        num_splits,
        out.stride(0),
        out.stride(1),
        num_warps=4,
    )
    return out
