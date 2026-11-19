"""Paged decode and speculative verify for gfx1250 on uTLX: GQA 8, head dim 64, BF16 queries.

Ported from the standard-layout path of the TLX kernel in facebookexperimental/triton#5055.
K/V are BF16 or FP8 e4m3 token-major caches [pages, PAGE, kv_heads, 64] with PAGE 16 or
64; every 64-key tile gathers 64 // PAGE pages. FP8 values dequantize exactly to BF16
after the LDS load; the K scale folds into Q and the V scale into the numerators.

Each sequence has query_length 1, 7 or 8 queries, causal against its cached keys. Long
sequences split over CTAs whose FP32 partials a second kernel merges. Windows count keys
including self; window 0 means global.
"""

from functools import lru_cache

import torch
import triton
import triton.language as tl
import utlx_plugin as tlx

from sglang.kernels.ops.attention.tlx.gfx1250_prefill import _WmmaLayout

HEAD_DIM = 64
GQA = 8
QUERY_LENGTHS = (1, 7, 8)
PAGE_SIZES = (16, 64)
KV_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)


@triton.jit
def _dot(a, b, c, LAYOUT: tl.constexpr):
    a = tlx.require_layout(a, tlx.dot_operand_layout(0, LAYOUT, k_width=8))
    b = tlx.require_layout(b, tlx.dot_operand_layout(1, LAYOUT, k_width=8))
    c = tlx.require_layout(c, LAYOUT)
    return tl.dot(a, b, c)


@triton.jit
def _qk_update(
    q,
    k,
    maximum,
    denom,
    base,
    length,
    m,
    n,
    QLEN: tl.constexpr,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    LAYOUT: tl.constexpr,
    SCALE: tl.constexpr,
):
    pos = base + n
    scores = _dot(q, k, tl.zeros((BM, BN), tl.float32), LAYOUT) * (
        SCALE * 1.4426950408889634
    )
    full = base + BN <= length - QLEN + 1
    if WINDOW:
        full &= base >= length - WINDOW
    if not full:
        qpos = length - QLEN + m // 8
        valid = (pos[None, :] <= qpos[:, None]) & (pos[None, :] < length)
        if WINDOW:
            valid &= pos[None, :] > qpos[:, None] - WINDOW
        scores = tl.where(valid, scores, -float("inf"))
    new_max = tl.maximum(maximum, tl.max(scores, 1))
    safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
    p = tl.exp2(scores - safe_max[:, None])
    alpha = tl.exp2(maximum - safe_max)
    denom = denom * alpha + tl.sum(p, 1)
    return p, alpha, new_max, denom, full


@triton.jit
def _pv_update(
    p,
    v,
    acc,
    alpha,
    base,
    length,
    n,
    full,
    QLEN: tl.constexpr,
    WINDOW: tl.constexpr,
    BN: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    if not full:
        # Zero values outside the sequence: unwritten page slots may hold NaN.
        pos = base + n
        valid_v = pos < length
        if WINDOW:
            valid_v &= pos >= tl.maximum(length - QLEN - WINDOW + 1, 0)
        v = tl.where(valid_v[:, None], v, 0.0)
    # BF16 matrix operands, with FP32 score/output and normalizer accumulation.
    return _dot(p.to(tl.bfloat16), v, acc * alpha[:, None], LAYOUT)


@triton.jit
def _partition_step(length, tiles, NS: tl.constexpr, CAPS: tl.constexpr):
    step = tl.cdiv(tiles, NS)
    for i in tl.static_range(len(CAPS) - 1, -1, -1):
        bound = CAPS[i][0]
        count = CAPS[i][1]
        step = tl.where(length <= bound, tl.cdiv(tiles, count), step)
    return step


@triton.jit
def _put(
    Cache, pages_view, slot, Table, page, TS, NP, KH: tl.constexpr, PAGE: tl.constexpr
):
    # Queue the 64-key tile starting at logical page `page` into stage `slot`. uTLX
    # drops a descriptor's shared layout across calls; build it where it is used.
    desc = tlx.make_tensor_descriptor(
        None, Cache, [NP * PAGE, 64], [KH * 64, 1], [PAGE, 64]
    )
    for i in tl.static_range(64 // PAGE):
        # Pages past the table end feed only masked keys; clamping avoids a branch.
        phys = tl.load(Table + tl.minimum(page + i, TS - 1))
        tlx.tdm_async_load(
            desc, tlx.local_view(pages_view, slot * (64 // PAGE) + i), [phys * PAGE, 0]
        )


@triton.jit(do_not_specialize=["TS", "NP"])
def _partition(
    Q,
    K,
    V,
    Table,
    Lengths,
    Part,
    Stats,
    Out,
    KScale,
    VScale,
    TS,
    NP,
    QLEN: tl.constexpr,
    NS: tl.constexpr,
    BM: tl.constexpr,
    WINDOW: tl.constexpr,
    LAYOUT: tl.constexpr,
    SCALE: tl.constexpr,
    KH: tl.constexpr,
    ROTATE: tl.constexpr,
    BATCH: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    CAPS: tl.constexpr,
    LIVE: tl.constexpr,
    PAGE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    BN: tl.constexpr = 64
    b = tl.program_id(1) if HEAD_FIRST else tl.program_id(0)
    h = tl.program_id(0) if HEAD_FIRST else tl.program_id(1)
    s = tl.program_id(2)
    # Each (head, split) rotates a complete permutation of the batch.
    # Output/workspace addresses always use the resulting logical sequence.
    if ROTATE:
        b = (b + 17 * h + 3 * s) % BATCH
    length = tl.load(Lengths + b).to(tl.int32)
    start = tl.maximum(length - QLEN - WINDOW + 1, 0) // 64 * 64 if WINDOW else 0
    tiles = tl.cdiv(length - start, BN)
    step = _partition_step(length, tiles, NS, CAPS)
    first = start + s * step * BN
    end = tl.minimum(length, first + step * BN)
    m = tl.arange(0, BM)
    d = tl.arange(0, 64)
    n = tl.arange(0, BN)
    qoff = ((b.to(tl.int64) * QLEN + m // 8) * (KH * 8) + h * 8 + m % 8)[
        :, None
    ] * 64 + d[None, :]
    q = tl.load(Q + qoff, m[:, None] < QLEN * 8, 0)
    if HAS_SCALE:
        q = (q.to(tl.float32) * tl.load(KScale)).to(tl.bfloat16)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, 64), tl.float32)

    # Two stages of 64x64 K and V tiles. A page-16 cache fills each stage with four
    # TDM page copies through a page view of the same allocation; indexed views carry
    # physical LDS bases, unlike local_slice, whose offsets this TDM lowering drops.
    PAD: tl.constexpr = 16 if K.dtype.element_ty.is_fp8() else 8
    tile_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
        [(64, PAD)], [64, 64], [1, 0]
    )
    if PAGE == 64:
        kb = tlx.local_alloc((64, 64), K.dtype.element_ty, 2, layout=tile_layout)
        vb = tlx.local_alloc((64, 64), K.dtype.element_ty, 2, layout=tile_layout)
        kp = kb
        vp = vb
    else:
        whole_layout: tl.constexpr = (
            tlx.padded_shared_layout_encoding.with_identity_for(
                [(64, PAD)], [128, 64], [1, 0]
            )
        )
        page_layout: tl.constexpr = tlx.padded_shared_layout_encoding.with_identity_for(
            [(64, PAD)], [PAGE, 64], [1, 0]
        )
        k_whole = tlx.local_view(
            tlx.local_alloc((128, 64), K.dtype.element_ty, 1, layout=whole_layout), 0
        )
        v_whole = tlx.local_view(
            tlx.local_alloc((128, 64), K.dtype.element_ty, 1, layout=whole_layout), 0
        )
        kp = tlx.local_reinterpret(
            k_whole, K.dtype.element_ty, [128 // PAGE, PAGE, 64], layout=page_layout
        )
        vp = tlx.local_reinterpret(
            v_whole, K.dtype.element_ty, [128 // PAGE, PAGE, 64], layout=page_layout
        )
        kb = tlx.local_reinterpret(
            k_whole, K.dtype.element_ty, [2, 64, 64], layout=tile_layout
        )
        vb = tlx.local_reinterpret(
            v_whole, K.dtype.element_ty, [2, 64, 64], layout=tile_layout
        )
    Kh = K + h * 64
    Vh = V + h * 64
    Tb = Table + b.to(tl.int64) * TS
    if first < end:
        _put(Kh, kp, 0, Tb, first // PAGE, TS, NP, KH, PAGE)
        _put(Vh, vp, 0, Tb, first // PAGE, TS, NP, KH, PAGE)
    # Keep the original unsplit q_len=7 schedule: earlier prefetch regresses
    # some uneven and large windowed batches.
    EARLY: tl.constexpr = QLEN != 7 or NS != 1
    for base in range(first, end, BN):
        slot = ((base - first) // BN) % 2
        tlx.tdm_async_wait(0)
        if EARLY:
            # Complete all previous-stage consumers before refilling it.
            tlx.workgroup_barrier()
        else:
            kr = tlx.local_load(tlx.local_view(kb, slot))
        if base + BN < end:
            _put(Kh, kp, 1 - slot, Tb, (base + BN) // PAGE, TS, NP, KH, PAGE)
            _put(Vh, vp, 1 - slot, Tb, (base + BN) // PAGE, TS, NP, KH, PAGE)
        if EARLY:
            kr = tlx.local_load(tlx.local_view(kb, slot))
        k = tl.trans(kr.to(tl.bfloat16))
        p, alpha, new_max, denom, full = _qk_update(
            q,
            k,
            maximum,
            denom,
            base,
            length,
            m,
            n,
            QLEN,
            WINDOW,
            BM,
            BN,
            LAYOUT,
            SCALE,
        )
        # The wait at the top of this iteration orders both K and V loads.
        # Delay reading V into registers until QK and softmax have consumed K.
        v = tlx.local_load(tlx.local_view(vb, slot)).to(tl.bfloat16)
        if not EARLY:
            # The next iteration may refill this stage after all reads finish.
            tlx.workgroup_barrier()
        acc = _pv_update(
            p, v, acc, alpha, base, length, n, full, QLEN, WINDOW, BN, LAYOUT
        )
        maximum = new_max
    if HAS_SCALE:
        acc *= tl.load(VScale)
    if NS == 1:
        inv = tl.div_rn(1.0, tl.where(denom > 0, denom, 1.0))
        tl.store(Out + qoff, acc * inv[:, None], m[:, None] < QLEN * 8)
    elif not LIVE or QLEN == 1 or WINDOW or first < end:
        # Store only real query rows, including for the padded q_len=1 tile.
        # Raw FP32 numerators avoid a divide/multiply pair across the merge.
        offset = ((b.to(tl.int64) * KH + h) * NS + s) * (QLEN * 8) + m
        tl.store(Part + offset[:, None] * 64 + d[None, :], acc, m[:, None] < QLEN * 8)
        # Keep max and sum separate: max + log2(sum) loses the sum when
        # logits have very large magnitude, causing incorrect split weights.
        tl.store(Stats + offset * 2, maximum, m < QLEN * 8)
        tl.store(Stats + offset * 2 + 1, denom, m < QLEN * 8)


@triton.jit
def _reduce_impl(
    Part,
    Stats,
    Out,
    QLEN: tl.constexpr,
    BM: tl.constexpr,
    NS: tl.constexpr,
    BS: tl.constexpr,
    RM: tl.constexpr,
    KH: tl.constexpr,
    RD: tl.constexpr,
    ACTIVE: tl.constexpr,
    used,
):
    group = tl.program_id(0)
    h = tl.program_id(1) % KH
    b = tl.program_id(1) // KH
    m = group * RM + tl.arange(0, RM)
    s = tl.arange(0, BS)
    valid_s = s < used if ACTIVE else s < NS
    if RD == 64:
        d = tl.arange(0, 64)
    else:
        d = tl.program_id(2) * RD + tl.arange(0, RD)
    offset = ((b.to(tl.int64) * KH + h) * NS + s[None, :]) * BM + m[:, None]
    rows = m[:, None] < QLEN * 8
    maxima = tl.load(Stats + offset * 2, valid_s[None, :] & rows, -float("inf"))
    denom = tl.load(Stats + offset * 2 + 1, valid_s[None, :] & rows, 0.0)
    mx = tl.max(maxima, 1)
    mx = tl.where(mx == -float("inf"), 0.0, mx)
    weights = tl.exp2(maxima - mx[:, None])
    total = tl.sum(weights * denom, 1)
    part = tl.load(
        Part + offset[:, :, None] * 64 + d[None, None, :],
        valid_s[None, :, None] & (m[:, None, None] < QLEN * 8),
        0.0,
    )
    result = tl.sum(part * weights[:, :, None], 1) / tl.where(
        total[:, None] > 0, total[:, None], 1.0
    )
    outoff = ((b.to(tl.int64) * QLEN + m // 8) * (KH * 8) + h * 8 + m % 8)[
        :, None
    ] * 64 + d[None, :]
    tl.store(Out + outoff, result, m[:, None] < QLEN * 8)


@triton.jit
def _reduce(
    Part,
    Stats,
    Out,
    Lengths,
    WINDOW: tl.constexpr,
    QLEN: tl.constexpr,
    BM: tl.constexpr,
    NS: tl.constexpr,
    BS: tl.constexpr,
    RM: tl.constexpr,
    KH: tl.constexpr,
    RD: tl.constexpr,
    CAPS: tl.constexpr,
    LIVE: tl.constexpr,
):
    if not LIVE or WINDOW:
        _reduce_impl(Part, Stats, Out, QLEN, BM, NS, BS, RM, KH, RD, False, NS)
    elif QLEN == 1:
        # q1 writes neutral values for empty splits, so every lane of the
        # chosen tile is initialized. Avoid a per-lane active mask here.
        length = tl.load(Lengths + tl.program_id(1) // KH).to(tl.int32)
        bound = tl.minimum(tl.cdiv(length, 64), NS)
        if BS > 16 and bound <= 16:
            _reduce_impl(Part, Stats, Out, QLEN, BM, NS, 16, RM, KH, RD, False, NS)
        elif BS > 32 and bound <= 32:
            _reduce_impl(Part, Stats, Out, QLEN, BM, NS, 32, RM, KH, RD, False, NS)
        elif BS > 64 and bound <= 64:
            _reduce_impl(Part, Stats, Out, QLEN, BM, NS, 64, RM, KH, RD, False, NS)
        elif BS > 128 and bound <= 128:
            _reduce_impl(Part, Stats, Out, QLEN, BM, NS, 128, RM, KH, RD, False, NS)
        elif BS > 256 and bound <= 256:
            _reduce_impl(Part, Stats, Out, QLEN, BM, NS, 256, RM, KH, RD, False, NS)
        else:
            _reduce_impl(Part, Stats, Out, QLEN, BM, NS, BS, RM, KH, RD, False, NS)
    else:
        length = tl.load(Lengths + tl.program_id(1) // KH).to(tl.int32)
        tiles = tl.cdiv(length, 64)
        step = _partition_step(length, tiles, NS, CAPS)
        full = (NS - 1) * step < tiles
        if full:
            _reduce_impl(Part, Stats, Out, QLEN, BM, NS, BS, RM, KH, RD, False, NS)
        else:
            used = tl.cdiv(tiles, tl.maximum(step, 1))
            if BS > 16 and used <= 16:
                _reduce_impl(Part, Stats, Out, QLEN, BM, NS, 16, RM, KH, RD, True, used)
            elif BS > 32 and used <= 32:
                _reduce_impl(Part, Stats, Out, QLEN, BM, NS, 32, RM, KH, RD, True, used)
            elif BS > 64 and used <= 64:
                _reduce_impl(Part, Stats, Out, QLEN, BM, NS, 64, RM, KH, RD, True, used)
            else:
                _reduce_impl(Part, Stats, Out, QLEN, BM, NS, BS, RM, KH, RD, True, used)


def _select_config(batch, qlen, max_context_len, window=0, kv_heads=2):
    """Frozen dispatch rules from #5055; no on-device tuning or host length reads."""
    active = min(max_context_len, window + qlen + 62) if window else max_context_len
    short = active <= 2176
    # Occupancy depends on sequence/head groups, not batch size alone.
    groups = batch * kv_heads
    if qlen == 1:
        if groups <= 4:
            budget = 512 if active > 32768 else 256
        elif groups <= 16:
            budget = (
                256 if short else (512 if groups == 16 and active <= 16384 else 1024)
            )
        else:
            budget = 1024 if short and groups <= 64 else 2048
        warps = 1
    else:
        if groups <= 4:
            budget = 512 if groups == 4 and active > 32768 else 256
        elif groups <= 16:
            budget = 256 if short else 512
        elif groups <= 64:
            budget = 512 if short else 1024
        else:
            budget = 1024
        warps = 4
    if qlen > 1:
        # Long speculative partitions need more CTAs at small group counts.
        # Short windowed partitions need fewer CTAs to limit reduction overhead.
        if (
            not window
            and 4 < groups <= 16
            and (active >= 98304 or (groups == 16 and active >= 65536))
        ):
            budget = 1024
        elif window and short and 64 < groups <= 128:
            budget = 512
    # Medium global workloads spend a significant fraction of their time writing
    # and merging partials. Limit their CTA count without rounding up to a power
    # of two: that can nearly double the work for intermediate batch sizes.
    medium_global = (
        not window and 16 < groups <= 64 and 2176 < active and groups * active <= 524288
    )
    if medium_global:
        if qlen == 1:
            budget = 1024
        else:
            budget = 1024 if groups == 64 and active > 4096 else 512
        splits = max(1, min(triton.cdiv(active, 64), triton.cdiv(budget, groups)))
    else:
        splits = max(
            1,
            min(
                triton.cdiv(active, 64),
                triton.next_power_of_2(max(1, budget // groups)),
            ),
        )
    if qlen > 1 and not window and splits >= 16:
        # Rounding up for intermediate group counts can exceed the intended
        # work budget by almost 2x.
        splits = min(splits, triton.cdiv(budget, groups))
    if qlen > 1 and not window and splits == 8 and groups > 128:
        # Short dense batches need fewer partials. Longer contexts benefit
        # from more parallel reads once there are over 160 sequence/head groups.
        if short:
            splits = 4
        elif active <= 8192:
            splits = triton.cdiv(1024, groups)
        elif groups > 160:
            splits = triton.cdiv(2048, groups)
    reduce_rows = (
        min(8, max(1, 128 // triton.next_power_of_2(splits)))
        if groups >= 64 and qlen > 1
        else 1
    )
    # Pair q1 head rows; each reduction CTA covers 32 output channels.
    if qlen == 1 and 1 < splits <= 128:
        reduce_rows = 2
    reduce_warps = 4 if splits >= 256 else 1
    return splits, warps, reduce_rows, reduce_warps


@lru_cache(maxsize=256)
def _runtime_caps(batch, qlen, capacity, window, heads, reserved):
    if qlen == 1 or window or reserved <= 1 or capacity < 32768:
        return ()
    bounds = sorted(
        {
            448,
            960,
            2176,
            4096,
            8192,
            16384,
            32768,
            65535,
            98303,
            524288 // (batch * heads),
        }
    )
    # Short requests benefit from fewer partials; long requests retain parallelism.
    choices = [
        (bound, min(reserved, _select_config(batch, qlen, bound, window, heads)[0]))
        for bound in bounds
        if 0 < bound < capacity and bound <= 16384
    ]
    # Drop an alternative when its successor uses the same partition count.
    result = []
    following = reserved
    for bound, count in reversed(choices):
        if count != following and triton.cdiv(bound, 64) > min(count, following):
            result.append((bound, count))
            following = count
    return tuple(reversed(result))


def unsupported_reason(
    q,
    k,
    v,
    out,
    block_table,
    seq_lens,
    query_length,
    max_seqlen_k,
    window,
    k_scale=None,
    v_scale=None,
):
    """None when paged_decode serves these arguments, else why it does not."""
    if query_length not in QUERY_LENGTHS:
        return f"needs query_length in {QUERY_LENGTHS}"
    if q.dtype != torch.bfloat16 or q.ndim != 3 or q.shape[2] != HEAD_DIM:
        return "needs BF16 queries [tokens, heads, 64]"
    if out.shape != q.shape:
        return "needs an output shaped like the queries"
    if k.ndim != 4 or k.shape[1] not in PAGE_SIZES or k.shape[3] != HEAD_DIM:
        return f"needs K/V caches [pages, page size in {PAGE_SIZES}, kv heads, 64]"
    if q.shape[1] != GQA * k.shape[2]:
        return f"needs {GQA} query heads per KV head"
    if k.dtype not in KV_DTYPES or v.dtype != k.dtype or v.shape != k.shape:
        return "needs matching BF16 or FP8 e4m3 K/V caches"
    if (k_scale is None) != (v_scale is None):
        return "needs both K/V scales or neither"
    if k_scale is not None and any(
        s.numel() != 1 or s.dtype != torch.float32 for s in (k_scale, v_scale)
    ):
        return "needs scalar FP32 K/V scales"
    if q.shape[0] != seq_lens.numel() * query_length:
        return "needs query_length queries per sequence"
    if seq_lens.dtype not in (torch.int32, torch.int64) or seq_lens.ndim != 1:
        return "needs 1-D integer sequence lengths"
    if type(window) is not int or window < 0:
        return "needs a non-negative window"
    if not all(t.is_contiguous() for t in (q, k, v, out, seq_lens)):
        return "needs contiguous Q/K/V/output and lengths"
    if block_table.dtype != torch.int32 or block_table.stride(1) != 1:
        return "needs an int32 page table with contiguous rows"
    if block_table.shape[0] != seq_lens.numel():
        return "needs one page table row per sequence"
    if block_table.shape[1] * k.shape[1] < max_seqlen_k:
        return "needs a page table covering max_seqlen_k"
    if k.shape[0] * k.shape[1] >= 2**31 or max_seqlen_k >= 2**31 - 32768:
        return "TDM row indices must fit signed int32"
    if any(t.data_ptr() % 16 for t in (q, k, v)):
        return "needs 16-byte-aligned Q/K/V"
    return None


def paged_decode(
    q,
    k,
    v,
    out,
    seq_lens,
    block_table,
    query_length,
    max_seqlen_k,
    softmax_scale,
    window=0,
    k_scale=None,
    v_scale=None,
):
    """Paged decode or verify into out; arguments must pass unsupported_reason.

    q/out are [batch * query_length, 8 * kv_heads, 64]; k/v [pages, page size,
    kv_heads, 64]. seq_lens counts each sequence's cached keys, including its
    queries; block_table [batch, pages] holds int32 page IDs covering max_seqlen_k.
    """
    batch = seq_lens.numel()
    kv_heads = k.shape[2]
    if batch == 0:
        return out
    if window >= max_seqlen_k:
        window = 0
    splits, warps, reduce_rows, reduce_warps = _select_config(
        batch, query_length, max_seqlen_k, window, kv_heads
    )
    # Large capacities can reserve many more splits than a short request uses.
    # Cache the constexpr alternatives; lengths stay on the GPU. Only wide
    # q1 merges amortize the runtime length check.
    live = max_seqlen_k >= 32768 and (query_length > 1 or splits >= 256)
    caps = _runtime_caps(batch, query_length, max_seqlen_k, window, kv_heads, splits)
    if splits > 1:
        rows = query_length * GQA
        partial = torch.empty(
            (batch, kv_heads, splits, rows, HEAD_DIM),
            dtype=torch.float32,
            device=q.device,
        )
        stats = torch.empty(
            (batch, kv_heads, splits, rows, 2), dtype=torch.float32, device=q.device
        )
    else:
        partial = stats = None
    layout = _WmmaLayout(
        [[2**i, 0] for i in range(warps.bit_length() - 1)], [], (16, 16, 32)
    )
    # Spread long sequences over the device without reading lengths on the host.
    # With many heads, dispatch adjacent heads together; with fewer heads,
    # rotate sequences between head/split planes to avoid repeated length order.
    reorder = batch >= 16
    head_first = (
        reorder
        and kv_heads >= 16
        and batch % 16 == 0
        and (window == 0 or batch * kv_heads >= 512)
    )
    rotate = (
        reorder
        and kv_heads < 16
        and batch * kv_heads >= 128
        and (query_length > 1 or (window == 0 and batch % 16 == 0))
    )
    has_scale = k_scale is not None
    grid = (kv_heads, batch, splits) if head_first else (batch, kv_heads, splits)
    _partition[grid](
        q, k, v, block_table, seq_lens, partial, stats, out,
        k_scale if has_scale else seq_lens, v_scale if has_scale else seq_lens,
        block_table.stride(0), k.shape[0], query_length, splits,
        max(16, triton.next_power_of_2(query_length * GQA)), window, layout,
        float(softmax_scale), kv_heads, rotate, batch, head_first, caps, live,
        k.shape[1], has_scale, num_warps=warps,
    )  # fmt: skip
    if splits > 1:
        reduce_columns = 32 if query_length == 1 and splits <= 128 else 64
        _reduce[
            (
                triton.cdiv(query_length * GQA, reduce_rows),
                batch * kv_heads,
                HEAD_DIM // reduce_columns,
            )
        ](
            partial, stats, out, seq_lens, window, query_length, query_length * GQA,
            splits, triton.next_power_of_2(splits), reduce_rows, kv_heads,
            reduce_columns, caps, live, num_warps=reduce_warps,
        )  # fmt: skip
    return out
