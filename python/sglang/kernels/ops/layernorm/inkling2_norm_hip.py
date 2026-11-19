"""Triton RMSNorm, L2 clip and block attention residual for ROCm, where the CuTe DSL
kernels do not build. All math is FP32 with one final rounding, like the CuTe kernels."""

import torch
import triton
import triton.language as tl

ATTN_RES_MAX_SOURCES = 16


@triton.jit
def _rms_norm_kernel(
    X,
    W,
    PW,
    Y,
    stride_x,
    stride_y,
    N,
    eps,
    HAS_W: tl.constexpr,
    HAS_PW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    u = tl.load(X + row * stride_x + cols, mask, other=0).to(tl.float32)
    if HAS_PW:
        u = u * tl.load(PW + cols, mask, other=0).to(tl.float32)
    inv = tl.math.rsqrt(tl.sum(u * u, 0) / N + eps)
    y = u * inv
    if HAS_W:
        y = y * tl.load(W + cols, mask, other=0).to(tl.float32)
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask)


def rms_norm(x, weight, eps, *, pre_weight=None, output_dtype=None, out=None):
    width = x.shape[-1]
    rows = x.reshape(-1, width)
    if rows.stride(-1) != 1:
        rows = rows.contiguous()
    y = out if out is not None else x.new_empty(x.shape, dtype=output_dtype or x.dtype)
    y_rows = y.view(-1, width)
    if rows.shape[0]:
        block = triton.next_power_of_2(width)
        _rms_norm_kernel[(rows.shape[0],)](
            rows,
            weight,
            pre_weight,
            y_rows,
            rows.stride(0),
            y_rows.stride(0),
            width,
            eps,
            weight is not None,
            pre_weight is not None,
            block,
            num_warps=min(16, max(1, block // 512)),
        )
    return y


@triton.jit
def _l2_clip_kernel(
    X,
    S,
    T,
    Y,
    stride_x,
    threshold,
    eps,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    HEADS_P: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HAS_T: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    heads = tl.arange(0, HEADS_P)
    d = tl.arange(0, D)
    mask = (heads < HEADS)[:, None]
    offs = heads[:, None] * D + d[None, :]
    u = tl.load(X + row * stride_x + offs, mask, other=0).to(tl.float32)
    if HAS_SCALE:
        u = u * tl.load(S + offs, mask, other=0).to(tl.float32)
    norm = tl.sqrt(tl.sum(u * u, 1))
    y = u * tl.minimum(threshold / (norm + eps), 1.0)[:, None]
    if HAS_T:
        # Like the reference: the clipped value is rounded before the length scale.
        y = y.to(Y.dtype.element_ty).to(tl.float32) * tl.load(T + row)
    tl.store(Y + row * HEADS * D + offs, y.to(Y.dtype.element_ty), mask)


def l2_clip(x, threshold, eps, *, scale=None, output_scale=None):
    """x (tokens, heads, d): y = u * min(1, threshold / (|u| + eps)) with u = scale * x,
    then times the per-token output_scale."""
    tokens, heads, d = x.shape
    if x.stride(2) != 1 or x.stride(1) != d:
        x = x.contiguous()
    y = torch.empty((tokens, heads, d), dtype=x.dtype, device=x.device)
    if tokens:
        _l2_clip_kernel[(tokens,)](
            x,
            scale,
            output_scale,
            y,
            x.stride(0),
            threshold,
            eps,
            heads,
            d,
            triton.next_power_of_2(heads),
            scale is not None,
            output_scale is not None,
            num_warps=4,
        )
    return y


@triton.jit
def _source_chunk(
    sources,
    addend,
    n: tl.constexpr,
    N: tl.constexpr,
    offs,
    mask,
    HAS_ADDEND: tl.constexpr,
):
    v = tl.load(sources[n] + offs, mask, other=0).to(tl.float32)
    if HAS_ADDEND and n == N - 1:
        v = v + tl.load(addend + offs, mask, other=0).to(tl.float32)
    return v


@triton.jit
def _norm_tile(
    params, qs, c, mask, Q: tl.constexpr, QP: tl.constexpr, BLOCK: tl.constexpr
):
    """(QP, BLOCK) tile of per-query norm parameters (tuple of Q tensors)."""
    tile = tl.zeros((QP, BLOCK), tl.float32)
    for q in tl.static_range(Q):
        row = tl.load(params[q] + c, mask, other=0).to(tl.float32)
        tile = tl.where(qs[:, None] == q, row[None, :], tile)
    return tile


@triton.jit
def _attn_res_kernel(
    sources,
    addend,
    queries,
    out,
    norm_w,
    norm_pw,
    normed,
    norm_eps,
    tokens,
    H: tl.constexpr,
    N: tl.constexpr,
    NP: tl.constexpr,
    Q: tl.constexpr,
    QP: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    NORMALIZE: tl.constexpr,
    NORM: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_PW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    qs = tl.arange(0, QP)
    q_mask = qs < Q
    ns = tl.arange(0, NP)
    logits = tl.full((QP, NP), float("-inf"), tl.float32)
    invs = tl.zeros((NP,), tl.float32)
    for n in tl.static_range(N):
        ss = tl.zeros((BLOCK,), tl.float32)
        dots = tl.zeros((QP, BLOCK), tl.float32)
        for k0 in range(0, H, BLOCK):
            c = k0 + cols
            mask = c < H
            v = _source_chunk(sources, addend, n, N, row * H + c, mask, HAS_ADDEND)
            ss += v * v
            qv = tl.load(
                queries + qs[:, None] * H + c[None, :],
                q_mask[:, None] & mask[None, :],
                other=0,
            )
            dots += qv * v[None, :]
        inv = tl.math.rsqrt(tl.sum(ss, 0) / H + 1e-6)
        logits = tl.where(ns[None, :] == n, (tl.sum(dots, 1) * inv)[:, None], logits)
        invs = tl.where(ns == n, inv, invs)
    probs = tl.exp(logits - tl.max(logits, 1)[:, None])
    probs = probs / tl.sum(probs, 1)[:, None]
    norm_ss = tl.zeros((QP,), tl.float32)
    for k0 in range(0, H, BLOCK):
        c = k0 + cols
        mask = c < H
        acc = tl.zeros((QP, BLOCK), tl.float32)
        for n in tl.static_range(N):
            v = _source_chunk(sources, addend, n, N, row * H + c, mask, HAS_ADDEND)
            if NORMALIZE:
                v = v * tl.sum(tl.where(ns == n, invs, 0.0), 0)
            p = tl.sum(tl.where(ns[None, :] == n, probs, 0.0), 1)
            acc += p[:, None] * v[None, :]
        tl.store(
            out + (qs[:, None] * tokens + row) * H + c[None, :],
            acc,
            q_mask[:, None] & mask[None, :],
        )
        if NORM:
            u = acc
            if HAS_PW:
                u = u * _norm_tile(norm_pw, qs, c, mask, Q, QP, BLOCK)
            norm_ss += tl.sum(u * u, 1)
    if NORM:
        # The following RMSNorm of each query's output, reading back this row's FP32.
        norm_inv = tl.math.rsqrt(norm_ss / H + norm_eps)
        for k0 in range(0, H, BLOCK):
            c = k0 + cols
            mask = c < H
            m2 = q_mask[:, None] & mask[None, :]
            offs = (qs[:, None] * tokens + row) * H + c[None, :]
            y = tl.load(out + offs, m2, other=0)
            if HAS_PW:
                y = y * _norm_tile(norm_pw, qs, c, mask, Q, QP, BLOCK)
            y = y * norm_inv[:, None]
            if HAS_W:
                y = y * _norm_tile(norm_w, qs, c, mask, Q, QP, BLOCK)
            tl.store(normed + offs, y.to(normed.dtype.element_ty), m2)


# One row counter per 256 bytes, so rows do not contend for a cache line.
_COUNTER_STRIDE = tl.constexpr(32)


@triton.jit
def _row_barrier(Counter, C: tl.constexpr):
    """Every one of the row's C CTAs arrives before any leaves. The counter only
    grows, by C per barrier, so it needs no reset between launches."""
    # Each release writes back and each acquire invalidates the whole XCD L2, so
    # there is one of each per CTA per barrier: a release arrival, relaxed polls
    # (coherent loads, not read-modify-writes), then one acquire on the line's
    # spare word (on the counter itself LLVM folds it into every poll). A
    # volatile load of this uniform address would go through the scalar cache
    # and never see the update.
    arrived = tl.atomic_add(Counter, 1, sem="release", scope="gpu")
    target = (arrived // C + 1) * C
    seen = arrived + 1
    while seen < target:
        seen = tl.atomic_add(Counter, 0, sem="relaxed", scope="gpu")
    tl.atomic_add(Counter + 1, 0, sem="acquire", scope="gpu")


@triton.jit
def _attn_res_split_kernel(
    sources,
    addend,
    queries,
    out,
    norm_w,
    norm_pw,
    normed,
    norm_eps,
    partials,
    counters,
    tokens,
    H: tl.constexpr,
    N: tl.constexpr,
    NP: tl.constexpr,
    Q: tl.constexpr,
    QP: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    NORMALIZE: tl.constexpr,
    NORM: tl.constexpr,
    HAS_W: tl.constexpr,
    HAS_PW: tl.constexpr,
    CHUNK: tl.constexpr,
    C: tl.constexpr,
    CP: tl.constexpr,
):
    # attn_res for one CHUNK of one row per CTA: the row's C CTAs exchange their
    # partial sums of squares and query dots (then output sums of squares) through
    # partials, each summing all C in chunk order, so every CTA (and every run)
    # gets the same totals. One wave per CTA, so each release covers all stores.
    chunk = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64)
    c = chunk * CHUNK + tl.arange(0, CHUNK)
    ns = tl.arange(0, NP)
    qs = tl.arange(0, QP)
    q_mask = qs < Q
    K: tl.constexpr = NP * (1 + QP) + QP
    tile = tl.zeros((NP, CHUNK), tl.float32)
    for n in tl.static_range(N):
        v = _source_chunk(sources, addend, n, N, row * H + c, c < H, HAS_ADDEND)
        tile = tl.where(ns[:, None] == n, v[None, :], tile)
    own = partials + (row * C + chunk) * K
    tl.store(own + ns, tl.sum(tile * tile, 1))
    for q in tl.static_range(Q):
        qv = tl.load(queries + q * H + c, c < H, other=0)
        tl.store(own + NP * (1 + q) + ns, tl.sum(tile * qv[None, :], 1))
    _row_barrier(counters + row * _COUNTER_STRIDE, C)
    cs = tl.arange(0, CP)
    rows = partials + (row * C + cs[:, None]) * K
    c_mask = cs[:, None] < C
    ss = tl.sum(tl.load(rows + ns[None, :], c_mask, other=0), 0)
    inv = tl.math.rsqrt(ss / H + 1e-6)
    values = tile
    if NORMALIZE:
        values = tile * inv[:, None]
    acc = tl.zeros((QP, CHUNK), tl.float32)
    for q in tl.static_range(Q):
        dots = tl.sum(
            tl.load(rows + NP * (1 + q) + ns[None, :], c_mask, other=0),
            0,
        )
        logits = tl.where(ns < N, dots * inv, float("-inf"))
        probs = tl.exp(logits - tl.max(logits, 0))
        probs = probs / tl.sum(probs, 0)
        acc = tl.where(
            qs[:, None] == q, tl.sum(probs[:, None] * values, 0)[None, :], acc
        )
    m2 = q_mask[:, None] & (c < H)[None, :]
    offs = (qs[:, None] * tokens + row) * H + c[None, :]
    if NORM:
        u = acc
        if HAS_PW:
            u = u * _norm_tile(norm_pw, qs, c, c < H, Q, QP, CHUNK)
        tl.store(own + NP * (1 + QP) + qs, tl.sum(u * u, 1))
        _row_barrier(counters + row * _COUNTER_STRIDE, C)
        norm_ss = tl.sum(
            tl.load(rows + NP * (1 + QP) + qs[None, :], c_mask, other=0),
            0,
        )
        y = u * tl.math.rsqrt(norm_ss / H + norm_eps)[:, None]
        if HAS_W:
            y = y * _norm_tile(norm_w, qs, c, c < H, Q, QP, CHUNK)
        tl.store(normed + offs, y.to(normed.dtype.element_ty), m2)
    else:
        tl.store(out + offs, acc, m2)


# Rows the split kernel takes (each row's C CTAs must be resident together).
ATTN_RES_SPLIT_MAX_TOKENS = 64
_SPLIT_CHUNKS = (512, 1024)
_ROW_COUNTERS: dict = {}


def _split_chunk(tokens, n, n_queries):
    """Chunk width for the split kernel, or 0 for the one-CTA kernel. Swept on
    MI350X (idle, hidden 4608, fused RMSNorm): the split wins with more sources
    and queries, the one-CTA kernel with more rows (each split barrier writes back
    and invalidates the XCD L2 once per CTA); wider chunks mean fewer CTAs."""
    if tokens <= 8:
        return 512 if n_queries > 1 or n >= 5 else 0
    min_sources = (2, 3, 6) if n_queries > 1 else (7, 9, n + 1)
    for bound, need in zip((16, 32, ATTN_RES_SPLIT_MAX_TOKENS), min_sources):
        if tokens <= bound:
            return 1024 if n >= need else 0
    return 0


def _row_counters(device, hidden, chunk):
    """Persistent row counters per chunk count, allocated on the first call outside
    graph capture for every chunk width _split_chunk picks (graphs capture the
    split kernel only if they exist)."""
    key = (device, triton.cdiv(hidden, chunk))
    if key not in _ROW_COUNTERS and not torch.cuda.is_current_stream_capturing():
        for width in _SPLIT_CHUNKS:
            _ROW_COUNTERS.setdefault(
                (device, triton.cdiv(hidden, width)),
                torch.zeros(
                    ATTN_RES_SPLIT_MAX_TOKENS * _COUNTER_STRIDE.value,
                    dtype=torch.int64,
                    device=device,
                ),
            )
    return _ROW_COUNTERS.get(key)


def attn_res(
    sources,
    partial_addend,
    queries,
    *,
    normalize_values,
    norms=None,
    out=None,
    split=True,
    split_chunk=None,
):
    """Block attention residual over sources (each (tokens, hidden), the last one plus
    partial_addend) for each row of queries (n_queries, hidden; already scaled by
    hidden**-0.5). Returns one FP32 (tokens, hidden) tensor per query, or with norms
    (one (weight, pre_weight, eps, dtype) per query) those tensors RMS-normalized in
    dtype, written into out (n_queries, tokens, hidden) when given."""
    n, (tokens, hidden) = len(sources), sources[0].shape
    if n > ATTN_RES_MAX_SOURCES:
        raise ValueError(f"attn_res supports at most {ATTN_RES_MAX_SOURCES} sources")
    sources = tuple(s if s.is_contiguous() else s.contiguous() for s in sources)
    if partial_addend is not None and not partial_addend.is_contiguous():
        partial_addend = partial_addend.contiguous()
    queries = queries.float().contiguous()
    n_queries = queries.shape[0]
    device = sources[0].device
    scratch = torch.empty(
        (n_queries, tokens, hidden), dtype=torch.float32, device=device
    )
    norm_w = norm_pw = normed = None
    eps = 0.0
    if norms is not None:
        assert len(norms) == n_queries
        weights, pre_weights, epss, dtypes = zip(*norms)
        assert len(set(epss)) == 1 and len(set(dtypes)) == 1
        assert len({w is None for w in weights}) == 1
        assert len({w is None for w in pre_weights}) == 1
        norm_w = None if weights[0] is None else tuple(weights)
        norm_pw = None if pre_weights[0] is None else tuple(pre_weights)
        eps = epss[0]
        normed = (
            out
            if out is not None
            else torch.empty(
                (n_queries, tokens, hidden), dtype=dtypes[0], device=device
            )
        )
        assert normed.shape == (n_queries, tokens, hidden) and normed.is_contiguous()
    if split and not split_chunk:
        split_chunk = _split_chunk(tokens, n, n_queries)
    if split and split_chunk and 0 < tokens <= ATTN_RES_SPLIT_MAX_TOKENS:
        chunks = triton.cdiv(hidden, split_chunk)
        counters = _row_counters(device, hidden, split_chunk)
    else:
        counters = None
    if counters is not None:
        np_, qp = max(2, triton.next_power_of_2(n)), triton.next_power_of_2(n_queries)
        partials = torch.empty(
            (tokens, chunks, np_ * (1 + qp) + qp), dtype=torch.float32, device=device
        )
        _attn_res_split_kernel[(chunks, tokens)](
            sources,
            partial_addend,
            queries,
            scratch,
            norm_w,
            norm_pw,
            normed,
            eps,
            partials,
            counters,
            tokens,
            hidden,
            n,
            np_,
            n_queries,
            qp,
            partial_addend is not None,
            normalize_values,
            norms is not None,
            norm_w is not None,
            norm_pw is not None,
            split_chunk,
            chunks,
            triton.next_power_of_2(chunks),
            num_warps=1,
        )
        return tuple((normed if norms is not None else scratch).unbind(0))
    # Swept on MI350X (idle, hidden 4608): small batches with few sources take the
    # whole hidden row in one 8192-wide chunk; otherwise 1024-wide chunks.
    if tokens <= 64 and (n <= 5 or (n <= 9 and n_queries == 1)):
        block, num_warps = 8192, 16
    else:
        block, num_warps = 1024, 4
    if tokens:
        _attn_res_kernel[(tokens,)](
            sources,
            partial_addend,
            queries,
            scratch,
            norm_w,
            norm_pw,
            normed,
            eps,
            tokens,
            hidden,
            n,
            max(2, triton.next_power_of_2(n)),
            n_queries,
            triton.next_power_of_2(n_queries),
            partial_addend is not None,
            normalize_values,
            norms is not None,
            norm_w is not None,
            norm_pw is not None,
            block,
            num_warps=num_warps,
        )
    return tuple((normed if norms is not None else scratch).unbind(0))
