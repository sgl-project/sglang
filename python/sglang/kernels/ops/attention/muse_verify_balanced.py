"""Muse Spark full-layer target verify with a length-balanced split-KV work list.

The FlyDSL verify kernel (``muse_verify_flydsl``) used to give every request the same number of splits, so in a batch
of very uneven contexts the longest request set the time. Here one tiny planning kernel splits the batch's total KV
chunks into equal segments of whole slot groups (each request gets ``ceil(chunks / segment)`` of them, at least one),
the verify kernel runs one segment per program and writes its partials to the segment's slot, and the merge reads
each request's own segments. Everything stays on the device, so the step is graph-capturable.

Partials: ``[segments + 1, 8, heads, 64]`` FP32 with max / denom ``[segments + 1, 8, heads]`` (log2 domain); the last
slot takes the idle programs' writes.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.muse_verify_flydsl import (
    CHUNK,
    QL,
    _launcher,
    _run_compiled,
    fx,
)

# slot groups per chunk group of the kernel's main loop (``slots`` of verify_fly_partials)
NS = 2
NUM_CUS = 256
# smallest segment, in chunks (a segment pays ~2 us of fixed cost: Q load, prefetch fill, epilogue)
MIN_SEGMENT_CHUNKS = 4
MAX_REQUESTS = 128


@triton.jit
def _plan_kernel(
    SEQ_LENS,
    PLAN,
    SEG_START,
    SEG_COUNT,
    num_seqs,
    WINDOW: tl.constexpr,
    SEGMENTS: tl.constexpr,
    BLOCK_R: tl.constexpr,
    CHUNK_: tl.constexpr,
    NS_: tl.constexpr,
    MIN_CHUNKS: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # every program recomputes the per-request counts (BLOCK_R values) and fills BLOCK_P plan rows
    r = tl.arange(0, BLOCK_R)
    rmask = r < num_seqs
    lens = tl.load(SEQ_LENS + r, mask=rmask, other=0).to(tl.int32)
    nchunks = (lens + CHUNK_ - 1) // CHUNK_
    if WINDOW > 0:
        # sliding layers: from the chunk of the first key any of the 8 queries sees
        first = tl.maximum(lens - 8 + 1 - WINDOW, 0) // CHUNK_
    else:
        first = tl.zeros_like(nchunks)
    chunks = nchunks - first
    total = tl.sum(chunks, axis=0)
    # segment length: the total over the segments left after one per request, in whole slot groups
    budget = tl.maximum(SEGMENTS - num_seqs, 1)
    seg = (total + budget - 1) // budget
    seg = tl.maximum(seg, MIN_CHUNKS)
    seg = (seg + NS_ - 1) // NS_ * NS_
    count = tl.where(rmask, tl.maximum((chunks + seg - 1) // seg, 1), 0)
    start = tl.cumsum(count, axis=0) - count
    if tl.program_id(0) == 0:
        tl.store(SEG_START + r, start, mask=rmask)
        tl.store(SEG_COUNT + r, count, mask=rmask)
    used = tl.sum(count, axis=0)
    p = tl.program_id(0) * BLOCK_P + tl.arange(0, BLOCK_P)
    # owner request of program p: the last request whose first segment is <= p
    owner = tl.sum((start[None, :] <= p[:, None]) & rmask[None, :], axis=1) - 1
    owner = tl.maximum(owner, 0)
    own_start = tl.sum(
        tl.where(r[None, :] == owner[:, None], start[None, :], 0), axis=1
    )
    own_chunks = tl.sum(
        tl.where(r[None, :] == owner[:, None], chunks[None, :], 0), axis=1
    )
    own_first = tl.sum(
        tl.where(r[None, :] == owner[:, None], first[None, :], 0), axis=1
    )
    i = p - own_start
    begin = own_first + i * seg
    end = own_first + tl.minimum(i * seg + seg, own_chunks)
    busy = p < used
    tl.store(PLAN + p * 4, tl.where(busy, owner, 0))
    tl.store(PLAN + p * 4 + 1, tl.where(busy, begin, 0))
    tl.store(PLAN + p * 4 + 2, tl.where(busy, end, 0))
    # idle programs write the spare slot SEGMENTS
    tl.store(PLAN + p * 4 + 3, tl.where(busy, p, SEGMENTS))


def verify_plan(seqused_k, segments, window=0):
    """(plan [segments, 4] int32, seg_start [n], seg_count [n]) for the batch's KV lengths."""
    n = seqused_k.numel()
    assert (
        0 < n <= MAX_REQUESTS and segments >= 2 * n and segments & (segments - 1) == 0
    )
    dev = seqused_k.device
    plan = torch.empty((segments, 4), dtype=torch.int32, device=dev)
    seg_start = torch.empty(n, dtype=torch.int32, device=dev)
    seg_count = torch.empty(n, dtype=torch.int32, device=dev)
    block_p = min(segments, 64)
    _plan_kernel[(segments // block_p,)](
        seqused_k, plan, seg_start, seg_count, n, window, segments, triton.next_power_of_2(max(n, 2)),
        CHUNK, NS, MIN_SEGMENT_CHUNKS, block_p, num_warps=2,
    )  # fmt: skip
    return plan, seg_start, seg_count


@triton.jit
def _seg_reduce(
    PARTIAL,
    MAXIMUM,
    DENOM,
    SEG_START,
    SEG_COUNT,
    V_SCALE,
    G,
    W,
    Y,
    eps,
    N,
    H,
    y_stride_t,
    y_stride_h,
    stride_g0,
    stride_g1,
    HEADS_PER_GROUP: tl.constexpr,
    QL_: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NORM: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
    HAS_GATE: tl.constexpr,
):
    # BLOCK_M (token, head) rows per program; row = token * H + head; token = request * QL + t
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < N
    token = rows // H
    head = rows % H
    req = token // QL_
    t = token % QL_
    s0 = tl.load(SEG_START + req, mask=row_mask, other=0)
    cnt = tl.load(SEG_COUNT + req, mask=row_mask, other=0)
    d = tl.arange(0, 64)
    s = tl.arange(0, BLOCK_S)
    gmax = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    for base in range(0, tl.max(cnt, axis=0), BLOCK_S):
        valid = (base + s[None, :]) < cnt[:, None]
        index = ((s0[:, None] + base + s[None, :]) * QL_ + t[:, None]) * H + head[
            :, None
        ]
        m = tl.load(MAXIMUM + index, mask=valid, other=float("-inf"))
        l = tl.load(DENOM + index, mask=valid, other=0.0)
        gmax = tl.maximum(gmax, tl.max(tl.where(l > 0, m, float("-inf")), axis=1))
    gmax = tl.where(gmax > float("-inf"), gmax, 0.0)
    total = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, 64), tl.float32)
    for base in range(0, tl.max(cnt, axis=0), BLOCK_S):
        valid = (base + s[None, :]) < cnt[:, None]
        index = ((s0[:, None] + base + s[None, :]) * QL_ + t[:, None]) * H + head[
            :, None
        ]
        m = tl.load(MAXIMUM + index, mask=valid, other=float("-inf"))
        l = tl.load(DENOM + index, mask=valid, other=0.0)
        w = tl.where(l > 0, tl.exp2(m - gmax[:, None]), 0.0)
        total += tl.sum(l * w, axis=1)
        part = tl.load(
            PARTIAL + index[:, :, None] * 64 + d[None, None, :],
            mask=valid[:, :, None],
            other=0.0,
        )
        acc += tl.sum(part * w[:, :, None], axis=1)
    inverse = tl.where(total > 0, 1.0 / total, 0.0)
    if V_SCALE is not None:
        inverse *= tl.load(V_SCALE)
    x = acc * inverse[:, None]
    if NORM:
        x = x.to(tl.bfloat16).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / 64 + eps)
        x = x * rstd[:, None]
        if HAS_WEIGHT:
            x = x * tl.load(W + d).to(tl.float32)[None, :]
        if HAS_GATE:
            offset = (head // HEADS_PER_GROUP) * stride_g1 + (
                head % HEADS_PER_GROUP
            ) * 64
            g = tl.load(
                G + token[:, None] * stride_g0 + offset[:, None] + d[None, :],
                mask=row_mask[:, None],
                other=0.0,
            )
            x = x * tl.sigmoid(g.to(tl.float32))
    tl.store(
        Y + token[:, None] * y_stride_t + head[:, None] * y_stride_h + d[None, :],
        x,
        mask=row_mask[:, None],
    )


def _launch_reduce(
    state, y, y_stride_t, y_stride_h, norm, gate=None, eps=0.0, weight=None
):
    # deferred states come back from the attention backend as (partial, maximum, denom, segments, v_scale)
    partial, maximum, denom, segs, v_scale = (
        state if isinstance(state[3], tuple) else (*state[:3], state[4], state[3])
    )
    seg_start, seg_count = segs
    heads = maximum.shape[2]
    tokens = seg_start.numel() * QL
    n = tokens * heads
    if gate is None:
        stride_g0 = stride_g1 = 0
        heads_per_group = heads
    else:
        heads_per_group = gate.shape[-1] // 64
        stride_g0, stride_g1 = gate.stride(0), gate.stride(1)
    block_m = 1
    _seg_reduce[(triton.cdiv(n, block_m),)](
        partial, maximum, denom, seg_start, seg_count, v_scale, gate, weight, y, eps, n, heads,
        y_stride_t, y_stride_h, stride_g0, stride_g1, heads_per_group, QL, 32, block_m,
        norm, weight is not None, gate is not None, num_warps=1,
    )  # fmt: skip
    return y


def segmented_reduce_norm_gate(state, out_shape, out_dtype, gate, eps, weight=None):
    """``verify_reduce_norm_gate`` for the segmented partials of ``gqa_verify_flydsl_balanced``."""
    y = torch.empty(out_shape, device=state[0].device, dtype=out_dtype)
    heads = state[1].shape[2]
    return _launch_reduce(state, y, heads * 64, 64, True, gate, eps, weight)


def segments_for(num_seqs: int, kv_heads: int, heads_per_group: int) -> int:
    """Programs per call: one workgroup per CU (every CU takes one segment of every head group)."""
    groups = kv_heads // heads_per_group
    segs = NUM_CUS // groups
    return max(segs, triton.next_power_of_2(2 * num_seqs))


def gqa_verify_flydsl_balanced(
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
    heads_per_group=None,
    segments=None,
    wg_per_cu=1,
    spw=None,
    window=0,
):
    """Full-attention verify (exact BF16 QK / PV) on the FlyDSL kernel with the length-balanced work list."""
    num_seqs = seqused_k.numel()
    heads, kvh = q.shape[1], k.shape[2]
    hpg = heads_per_group or kvh
    if seqused_k.dtype != torch.int32:
        seqused_k = seqused_k.to(torch.int32)
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    # two segments per workgroup at 2 KV heads: 4 waves per CU keep enough KV bytes in flight
    spw = spw or (2 if kvh // hpg * hpg == kvh and hpg == 2 else 1)
    segs = segments or segments_for(num_seqs, kvh, hpg) * wg_per_cu * spw
    plan, seg_start, seg_count = verify_plan(seqused_k, segs, window)
    st = (segs + 1, QL, heads)
    part = torch.empty((*st, 64), dtype=torch.float32, device=q.device)
    maxi = torch.empty(st, dtype=torch.float32, device=q.device)
    den = torch.empty(st, dtype=torch.float32, device=q.device)
    kd = (
        k_descale
        if k_descale is not None
        else torch.ones(1, dtype=torch.float32, device=q.device)
    )
    addr64 = max(k.numel(), v.numel()) * k.element_size() >= 2**31
    _run_compiled(
        _launcher(heads, kvh, hpg, k.stride(0), k.stride(1), float(softmax_scale), 1, True, NS, False, False, False,
                  True, wg_per_cu, addr64, True, spw, window, k.shape[1]),
        q, k.view(torch.uint8), v.view(torch.uint8), block_table, seqused_k, kd, part, maxi, den, plan,
        segs, segs, int(block_table.stride(0)), int(q.stride(0)), int(segs // spw * (kvh // hpg)),
        fx.Stream(torch.cuda.current_stream(q.device).cuda_stream),
    )  # fmt: skip
    if defer_reduce:
        return part, maxi, den, (seg_start, seg_count)
    return _launch_reduce(
        (part, maxi, den, (seg_start, seg_count), v_descale),
        out,
        out.stride(0),
        out.stride(1),
        False,
    )


def swa_verify_flydsl(
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
    window,
    defer_reduce=False,
):
    """Sliding-layer verify on the FlyDSL kernel: fixed splits over each request's window (every window has the same
    length, so no plan), two workgroups per CU. Faster than the Triton kernels from 16 requests at 4 KV heads per
    rank and from 48 at 2 (MI355X, per layer: n64 TP4 40 vs 72 us, TP8 35 vs 45 us)."""
    from sglang.kernels.ops.attention.gqa_verify_splitkv_2kv import _vv2_reduce
    from sglang.kernels.ops.attention.muse_verify_flydsl import verify_fly_partials

    num_seqs = seqused_k.numel()
    heads = q.shape[1]
    if seqused_k.dtype != torch.int32:
        seqused_k = seqused_k.to(torch.int32)
    if block_table.dtype != torch.int32:
        block_table = block_table.to(torch.int32)
    work = 16 if num_seqs <= 16 else (8 if num_seqs <= 32 else 4)
    splits = work
    partial, maximum, denom = verify_fly_partials(
        q, k, v, seqused_k, block_table, softmax_scale, k_descale, splits,
        waves_per_eu=1, fence_vmem=True, wg_per_cu=2, work_splits=work, window=window,
    )  # fmt: skip
    if defer_reduce:
        return partial, maximum, denom
    _vv2_reduce[(q.shape[0], heads)](
        partial, maximum, denom, out, v_descale, heads, splits, out.stride(0), out.stride(1), num_warps=2,
    )  # fmt: skip
    return out


def use_balanced_full(num_seqs: int, kv_heads: int) -> bool:
    return num_seqs >= (4 if kv_heads >= 4 else 8)


def use_flydsl_sliding(num_seqs: int, kv_heads: int) -> bool:
    return num_seqs >= (16 if kv_heads >= 4 else 48)
