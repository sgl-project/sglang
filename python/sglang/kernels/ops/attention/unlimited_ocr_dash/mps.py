"""DASH routing and sparse attend for Apple MPS.

The dispatch wrappers in ``dispatch`` route here for MPS tensors. Routing
(decode and prefill), sparse attend (decode and prefill), page summaries,
decode metadata and Page2D RoPE run Metal kernels (``metal.py``) and fall back
to the torch versions below for shapes those kernels do not cover. Semantics
follow the Triton kernels. Reductions run in fp32 because MPS has no fp64, and
the torch entmax threshold uses the exact sort form instead of bisection; both
agree to fp32 rounding.
"""

from __future__ import annotations

import torch

from sglang.kernels.ops.attention.unlimited_ocr_dash import metal

# Same finite sentinels as the Triton router and training.
SCORE_NEG = -1.0e9
ENTMAX_NEG = -1.0e4
ACT_CLAMP = 1.0e4
# Query rows per scoring / dense-attend chunk; bounds the [rows, keys] scratch.
_ROW_CHUNK = 512
_flush_event = None


def _flush() -> None:
    """Commit the queued MPS work so the GPU runs it while the CPU encodes on.

    Eager MPS otherwise keeps encoding into one command buffer until the next
    sync, and the GPU starts a decode step only when sampling reads the tokens.
    Recording an event commits the buffer; the decode attend does it once per
    layer.
    """
    global _flush_event
    if _flush_event is None:
        _flush_event = torch.mps.Event()
    _flush_event.record()


def _summary_keys(keys, valid, head_cls, inv_sqrt_d):
    """[P, bs, d] index keys -> [P, d] head_cls summaries (masked softmax)."""
    aw = (keys * head_cls).sum(-1) * inv_sqrt_d
    aw = aw.clamp(-ACT_CLAMP, ACT_CLAMP)
    row_max = torch.where(valid, aw, float("-inf")).amax(dim=-1, keepdim=True)
    row_max = torch.where(valid.any(dim=-1, keepdim=True), row_max, 0.0)
    e = torch.exp(aw - row_max) * valid
    w = e / e.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    k_bar = (w.unsqueeze(-1) * keys).sum(dim=1)
    return torch.nan_to_num(k_bar, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def kbar_refresh(*, kv_cache, k_bar, head_cls, plan) -> None:
    """Rewrite the ``plan`` pages' summaries in ``k_bar`` ([pages, d] fp32)."""
    if not metal.kbar_refresh(kv_cache, k_bar, head_cls, *plan):
        kbar_refresh_torch(kv_cache=kv_cache, k_bar=k_bar, head_cls=head_cls, plan=plan)


def kbar_refresh_torch(*, kv_cache, k_bar, head_cls, plan) -> None:
    pages, tok = plan
    pages = pages.to(torch.int64)
    block_size, d = kv_cache.shape[1], k_bar.shape[-1]
    keys = kv_cache[pages].float()
    valid = torch.arange(block_size, device=tok.device)[None, :] < tok[:, None]
    summary = _summary_keys(keys, valid, head_cls.float().view(-1), d**-0.5)
    # Entries without tokens are not pages of a request; page 0 is the
    # allocator's padding page, so they write there instead.
    dest = torch.where(tok > 0, pages, 0)
    k_bar.index_copy_(0, dest, summary)


def _decode_rows(seq_lens, nd: int, dql: int):
    """Request index and absolute position of each of ``nd`` decode queries."""
    row = torch.arange(nd, device=seq_lens.device)
    req = row // dql
    q_pos = seq_lens.to(torch.int64)[req] - dql + row % dql
    return req, q_pos


@torch.no_grad()
def win_bounds_decode(
    *,
    allow,
    seq_lens,
    lo,
    hi,
    fit,
    max_block: int,
    block_size: int,
    local_blocks: int,
    decode_query_len: int,
    tile: int,
) -> None:
    """Remote page span ``[lo, hi)`` of each decode query and its narrow-tile fit."""
    nd = int(lo.numel())
    if nd <= 0:
        return
    W = int(max_block)
    req, q_pos = _decode_rows(seq_lens, nd, max(1, int(decode_query_len)))
    seq = seq_lens.to(torch.int64)[req]
    q_blk = q_pos // block_size
    a = allow[:nd, :W] != 0
    off = torch.arange(a.shape[1], device=a.device)
    keep = (
        a
        & (off[None, :] <= (q_blk - local_blocks)[:, None])
        & (off[None, :] * block_size < seq[:, None])
    )
    any_r = keep.any(dim=1)
    lo_v = torch.where(keep, off, W).amin(dim=1, keepdim=False)
    hi_v = torch.where(keep, off, -1).amax(dim=1)
    w_seq = ((seq + block_size - 1) // block_size).clamp(min=1, max=W)
    no_remote = (seq <= 0) | (q_blk - local_blocks < 0)
    lo_v = torch.where(any_r, lo_v, 0)
    hi_v = torch.where(any_r, hi_v + 1, torch.where(no_remote, 0, w_seq))
    lo.copy_(lo_v)
    hi.copy_(hi_v)
    fit.copy_((hi_v - lo_v) <= tile)


def _block_scores(idx_q, k_bar, pages, keep):
    """Clamped ``q . k_bar / sqrt(d)`` per (row, block); dropped blocks get SCORE_NEG."""
    kb = k_bar[pages.to(torch.int64)]
    z = torch.matmul(kb, idx_q.float().unsqueeze(-1)).squeeze(-1)
    z = (z * idx_q.shape[-1] ** -0.5).clamp(-ACT_CLAMP, ACT_CLAMP)
    return torch.where(keep, z, SCORE_NEG)


def _remote_keep(q_pos, seq, W: int, block_size: int, local_blocks: int):
    """Blocks strictly before the local window and inside the sequence."""
    off = torch.arange(W, device=q_pos.device)
    q_blk = q_pos // block_size
    return (off[None, :] <= (q_blk - local_blocks)[:, None]) & (
        off[None, :] * block_size < seq[:, None]
    )


def _entmax_select(
    scores, img, out_idx, out_bias, *, idx_scale, prior_floor, sigma, budget, M
) -> None:
    """1.5-entmax prior over remote candidates, hard top-``budget``, packed output.

    Writes the kept block ids in ascending order with their centered log-prior
    bias into ``out_idx`` / ``out_bias`` ``[:, :M]``; unused slots get -1 / 0.
    """
    n, W = scores.shape
    remote = scores > SCORE_NEG * 0.5
    cand = remote
    if img is not None:
        cand_img = remote & (img[:, :W] != 0)
        # The page mask leaving nothing drops the constraint for that query.
        cand = torch.where(cand_img.any(dim=1, keepdim=True), cand_img, remote)
    has_any = cand.any(dim=1, keepdim=True)
    x = torch.nan_to_num(torch.where(cand, scores * idx_scale, ENTMAX_NEG), ENTMAX_NEG)
    xm = torch.where(cand, x, float("-inf")).amax(dim=1, keepdim=True)
    z = (x - torch.where(has_any, xm, 0.0)) * 0.5

    # Exact threshold: tau is the k-th sorted-prefix root over the support.
    zs = torch.where(cand, z, float("-inf")).sort(dim=1, descending=True).values
    finite = zs > float("-inf")
    zs0 = torch.where(finite, zs, 0.0)
    k = torch.arange(1, W + 1, device=z.device, dtype=z.dtype)
    mean = zs0.cumsum(dim=1) / k
    msq = (zs0 * zs0).cumsum(dim=1) / k
    delta = ((1.0 - k * (msq - mean * mean)) / k).clamp_min(0.0)
    tau_k = mean - delta.sqrt()
    support = ((tau_k <= zs) & finite).sum(dim=1, keepdim=True).clamp_min(1)
    tau = tau_k.gather(1, support - 1)
    prior = (z - tau).clamp_min(0.0).square()

    # Hard top-K keeps candidates strictly above the (K+1)-th largest score, so
    # ties at the boundary drop out exactly as in the Triton count bisection.
    active = cand
    if W > budget:
        over = cand.sum(dim=1, keepdim=True) > budget
        thr = torch.where(over, zs[:, budget : budget + 1], float("-inf"))
        active = cand & (z > thr)

    logp = torch.log(prior.clamp_min(prior_floor))
    cnt = active.sum(dim=1, keepdim=True).clamp_min(1)
    mu = torch.where(active, logp, 0.0).sum(dim=1, keepdim=True) / cnt
    bias = torch.where(active, (logp - mu) / sigma, 0.0)

    pos = active.to(torch.int32).cumsum(dim=1) - 1
    ok = active & (pos < M)
    # Column M collects the dropped entries and is cut off below.
    col = torch.where(ok, pos, M).to(torch.int64)
    blk = torch.arange(W, device=z.device, dtype=torch.int32).expand(n, W)
    idx = torch.full((n, M + 1), -1, dtype=torch.int32, device=z.device)
    idx.scatter_(1, col, torch.where(ok, blk, -1))
    val = torch.zeros((n, M + 1), dtype=torch.float32, device=z.device)
    val.scatter_(1, col, torch.where(ok, bias, 0.0))
    out_idx[:n, :M].copy_(idx[:, :M])
    out_bias[:n, :M].copy_(val[:, :M])


@torch.no_grad()
def route_decode(
    *,
    idx_q,
    k_bar,
    block_table,
    seq_lens,
    image_allow,
    out_idx,
    out_bias,
    max_block: int,
    block_size: int,
    local_blocks: int,
    decode_query_len: int,
    idx_scale: float,
    prior_floor: float,
    sigma: float,
    budget: int,
    M: int,
) -> int:
    """Route decode queries over the full block-table width; returns ``M``."""
    nd = int(idx_q.shape[0])
    W = min(int(max_block), int(block_table.shape[1]))
    if metal.route(
        idx_q,
        k_bar,
        block_table,
        seq_lens,
        image_allow,
        out_idx,
        out_bias,
        width=W,
        block_size=block_size,
        local_blocks=local_blocks,
        decode_query_len=max(1, int(decode_query_len)),
        idx_scale=idx_scale,
        prior_floor=prior_floor,
        sigma=sigma,
        budget=budget,
        M=M,
    ):
        return M
    return route_decode_torch(
        idx_q=idx_q,
        k_bar=k_bar,
        block_table=block_table,
        seq_lens=seq_lens,
        image_allow=image_allow,
        out_idx=out_idx,
        out_bias=out_bias,
        max_block=max_block,
        block_size=block_size,
        local_blocks=local_blocks,
        decode_query_len=decode_query_len,
        idx_scale=idx_scale,
        prior_floor=prior_floor,
        sigma=sigma,
        budget=budget,
        M=M,
    )


def route_decode_torch(
    *,
    idx_q,
    k_bar,
    block_table,
    seq_lens,
    image_allow,
    out_idx,
    out_bias,
    max_block: int,
    block_size: int,
    local_blocks: int,
    decode_query_len: int,
    idx_scale: float,
    prior_floor: float,
    sigma: float,
    budget: int,
    M: int,
) -> int:
    nd = int(idx_q.shape[0])
    W = min(int(max_block), int(block_table.shape[1]))
    req, q_pos = _decode_rows(seq_lens, nd, max(1, int(decode_query_len)))
    seq = seq_lens.to(torch.int64)[req]
    keep = _remote_keep(q_pos, seq, W, block_size, local_blocks)
    scores = _block_scores(idx_q, k_bar, block_table[req, :W], keep)
    _entmax_select(
        scores,
        image_allow,
        out_idx,
        out_bias,
        idx_scale=idx_scale,
        prior_floor=prior_floor,
        sigma=sigma,
        budget=budget,
        M=M,
    )
    return M


@torch.no_grad()
def route_prefill(
    *,
    idx_q,
    k_bar,
    block_table,
    seq_lens,
    cu_seqlens_q,
    context_lens,
    row2req,
    image_allow,
    out_idx,
    out_bias,
    max_block: int,
    block_size: int,
    local_blocks: int,
    idx_scale: float,
    prior_floor: float,
    sigma: float,
    budget: int,
    M: int,
) -> int:
    """Route prefill queries; returns ``M``."""
    W = min(int(max_block), int(block_table.shape[1]))
    if metal.route(
        idx_q,
        k_bar,
        block_table,
        seq_lens,
        image_allow,
        out_idx,
        out_bias,
        width=W,
        block_size=block_size,
        local_blocks=local_blocks,
        idx_scale=idx_scale,
        prior_floor=prior_floor,
        sigma=sigma,
        budget=budget,
        M=M,
        rows=(row2req, cu_seqlens_q, context_lens),
    ):
        return M
    return route_prefill_torch(
        idx_q=idx_q,
        k_bar=k_bar,
        block_table=block_table,
        seq_lens=seq_lens,
        cu_seqlens_q=cu_seqlens_q,
        context_lens=context_lens,
        row2req=row2req,
        image_allow=image_allow,
        out_idx=out_idx,
        out_bias=out_bias,
        max_block=max_block,
        block_size=block_size,
        local_blocks=local_blocks,
        idx_scale=idx_scale,
        prior_floor=prior_floor,
        sigma=sigma,
        budget=budget,
        M=M,
    )


def route_prefill_torch(
    *,
    idx_q,
    k_bar,
    block_table,
    seq_lens,
    cu_seqlens_q,
    context_lens,
    row2req,
    image_allow,
    out_idx,
    out_bias,
    max_block: int,
    block_size: int,
    local_blocks: int,
    idx_scale: float,
    prior_floor: float,
    sigma: float,
    budget: int,
    M: int,
) -> int:
    """Torch prefill routing in row chunks; returns ``M``."""
    total_q = int(idx_q.shape[0])
    W = min(int(max_block), int(block_table.shape[1]))
    seq_all = seq_lens.to(torch.int64)
    cu = cu_seqlens_q.to(torch.int64)
    ctx = context_lens.to(torch.int64)
    for off in range(0, total_q, _ROW_CHUNK):
        rows = torch.arange(off, min(off + _ROW_CHUNK, total_q), device=idx_q.device)
        req = row2req[off : off + rows.numel()].to(torch.int64)
        q_pos = ctx[req] + rows - cu[req]
        keep = _remote_keep(q_pos, seq_all[req], W, block_size, local_blocks)
        scores = _block_scores(idx_q[rows], k_bar, block_table[req, :W], keep)
        _entmax_select(
            scores,
            None if image_allow is None else image_allow[off : off + rows.numel()],
            out_idx[off:],
            out_bias[off:],
            idx_scale=idx_scale,
            prior_floor=prior_floor,
            sigma=sigma,
            budget=budget,
            M=M,
        )
    return M


def _attend(qg, keys, vals, logit_bias):
    """Softmax attention in fp32; rows with no visible key produce zeros.

    ``qg`` [Hkv, n, G, D], ``keys`` / ``vals`` [Hkv, n, S, D] or [Hkv, S, D],
    ``logit_bias`` [Hkv, n, S] additive (``-inf`` hides a key). Everything is
    folded into 3-D ``bmm``: MPS runs broadcast 4-D matmuls far slower.
    """
    hkv, n, group, dim = qg.shape
    if keys.dim() == 3:
        s = torch.bmm(qg.reshape(hkv, n * group, dim), keys.transpose(1, 2))
        s = s.view(hkv, n, group, -1) + logit_bias.unsqueeze(2)
        p = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)
        out = torch.bmm(p.view(hkv, n * group, -1), vals)
        return out.view(hkv, n, group, dim)
    k = keys.flatten(0, 1)
    s = torch.bmm(qg.reshape(hkv * n, group, dim), k.transpose(1, 2))
    s = s.view(hkv, n, group, -1) + logit_bias.unsqueeze(2)
    p = torch.nan_to_num(torch.softmax(s, dim=-1), nan=0.0)
    out = torch.bmm(p.view(hkv * n, group, -1), vals.flatten(0, 1))
    return out.view(hkv, n, group, dim)


@torch.no_grad()
def dash_sparse_attn_decode(
    q,
    k_cache,
    v_cache,
    topk_idx,
    blk_bias,
    block_table,
    seq_lens,
    num_kv_heads: int,
    sm_scale: float,
    output,
    decode_query_len: int,
    local_blocks: int = 0,
    sparse_block_size: int = 32,
) -> None:
    """Gather the selected remote blocks and the local window, then attend.

    Remote keys stop below the local window ``[q_pos - local*bs + 1, q_pos]``;
    the window is always attended without bias.
    """
    # The Metal kernel reads one slot row per query; the backend's selection is
    # a stride-0 expand over kv heads.
    shared = topk_idx.stride(0) == 0 and (blk_bias is None or blk_bias.stride(0) == 0)
    if (shared or num_kv_heads == 1) and metal.sparse_attend(
        q,
        k_cache,
        v_cache,
        topk_idx[0],
        None if blk_bias is None else blk_bias[0],
        block_table,
        seq_lens,
        output,
        num_kv_heads=num_kv_heads,
        scale=sm_scale,
        decode_query_len=max(1, int(decode_query_len)),
        local_blocks=local_blocks,
        block_size=sparse_block_size,
    ):
        _flush()
        return
    dash_sparse_attn_decode_torch(
        q,
        k_cache,
        v_cache,
        topk_idx,
        blk_bias,
        block_table,
        seq_lens,
        num_kv_heads,
        sm_scale,
        output,
        decode_query_len,
        local_blocks=local_blocks,
        sparse_block_size=sparse_block_size,
    )
    _flush()


def dash_sparse_attn_decode_torch(
    q,
    k_cache,
    v_cache,
    topk_idx,
    blk_bias,
    block_table,
    seq_lens,
    num_kv_heads: int,
    sm_scale: float,
    output,
    decode_query_len: int,
    local_blocks: int = 0,
    sparse_block_size: int = 32,
) -> None:
    nq, num_heads, head_dim = q.shape
    bs = int(sparse_block_size)
    group = num_heads // num_kv_heads
    dev = q.device
    req, q_pos = _decode_rows(seq_lens, nq, max(1, int(decode_query_len)))
    win_start = (q_pos - local_blocks * bs + 1).clamp_min(0)
    tok = torch.arange(bs, device=dev)
    heads = torch.arange(num_kv_heads, device=dev)
    qg = (q.float() * sm_scale).view(nq, num_kv_heads, group, head_dim).transpose(0, 1)

    sel = topk_idx.to(torch.int64)  # [Hkv, nq, T]
    blk = sel.clamp_min(0)
    pages = block_table.to(torch.int64)[req[None, :, None], blk]
    # [Hkv, nq, T, bs, D]: each kv head reads its own selection.
    k_r = k_cache[pages, :, heads[:, None, None]].float()
    v_r = v_cache[pages, :, heads[:, None, None]].float()
    pos = blk[..., None] * bs + tok
    vis = (sel >= 0)[..., None] & (pos < win_start[None, :, None, None])
    bias_r = 0.0 if blk_bias is None else blk_bias.float()[..., None]
    logit_r = torch.where(vis, bias_r, float("-inf")).flatten(2)

    parts_k = [k_r.flatten(2, 3)]
    parts_v = [v_r.flatten(2, 3)]
    parts_b = [logit_r]
    if local_blocks > 0:
        lblk = q_pos[:, None] // bs - torch.arange(local_blocks + 1, device=dev)
        lpages = block_table.to(torch.int64)[req[:, None], lblk.clamp_min(0)]
        # [nq, L+1, bs, Hkv, D] -> [Hkv, nq, (L+1)*bs, D]
        k_l = k_cache[lpages].float().permute(3, 0, 1, 2, 4).flatten(2, 3)
        v_l = v_cache[lpages].float().permute(3, 0, 1, 2, 4).flatten(2, 3)
        lpos = lblk[..., None] * bs + tok
        lvis = (
            (lblk >= 0)[..., None]
            & (lpos >= win_start[:, None, None])
            & (lpos <= q_pos[:, None, None])
        )
        logit_l = torch.where(lvis, 0.0, float("-inf")).flatten(1)
        parts_k.append(k_l)
        parts_v.append(v_l)
        parts_b.append(logit_l.expand(num_kv_heads, -1, -1))
    out = _attend(
        qg, torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2), torch.cat(parts_b, 2)
    )
    output.copy_(out.transpose(0, 1).reshape(nq, num_heads, head_dim))


@torch.no_grad()
def dash_sparse_attn(
    q,
    k_cache,
    v_cache,
    topk_idx,
    blk_bias,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    num_kv_heads: int,
    sm_scale: float,
    output,
    local_blocks: int = 0,
    sparse_block_size: int = 32,
) -> None:
    """Bias-aware block-sparse prefill attention (one query row per token).

    A key is visible when it lies in the local window, or lies below it inside
    a selected remote block (which then adds its bias).
    """
    shared = topk_idx.stride(0) == 0 and (blk_bias is None or blk_bias.stride(0) == 0)
    if shared or num_kv_heads == 1:
        cu = cu_seqlens_q.to(torch.int64)
        num_reqs = cu.numel() - 1
        # output_size keeps repeat_interleave free of a host sync.
        row2req = torch.repeat_interleave(
            torch.arange(num_reqs, device=q.device, dtype=torch.int32),
            cu[1:] - cu[:-1],
            output_size=q.shape[0],
        )
        if metal.sparse_attend(
            q,
            k_cache,
            v_cache,
            topk_idx[0],
            None if blk_bias is None else blk_bias[0],
            block_table,
            seq_lens,
            output,
            num_kv_heads=num_kv_heads,
            scale=sm_scale,
            local_blocks=local_blocks,
            block_size=sparse_block_size,
            rows=(row2req, cu, prefix_lens),
        ):
            _flush()
            return
    dash_sparse_attn_torch(
        q,
        k_cache,
        v_cache,
        topk_idx,
        blk_bias,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        num_kv_heads,
        sm_scale,
        output,
        local_blocks=local_blocks,
        sparse_block_size=sparse_block_size,
    )


def dash_sparse_attn_torch(
    q,
    k_cache,
    v_cache,
    topk_idx,
    blk_bias,
    block_table,
    cu_seqlens_q,
    seq_lens,
    prefix_lens,
    num_kv_heads: int,
    sm_scale: float,
    output,
    local_blocks: int = 0,
    sparse_block_size: int = 32,
) -> None:
    """Dense attention over each request's keys under the DASH visibility mask."""
    _, num_heads, head_dim = q.shape
    bs = int(sparse_block_size)
    group = num_heads // num_kv_heads
    dev = q.device
    num_reqs = int(seq_lens.shape[0])
    # One host read per layer for the per-request loop bounds.
    host = torch.cat(
        [
            cu_seqlens_q.to(torch.int64),
            seq_lens.to(torch.int64),
            prefix_lens.to(torch.int64),
        ]
    ).tolist()
    cu, seq, ctx = (
        host[: num_reqs + 1],
        host[num_reqs + 1 : 2 * num_reqs + 1],
        host[2 * num_reqs + 1 :],
    )
    lw = local_blocks * bs
    for r in range(num_reqs):
        q0, q1, n_keys = cu[r], cu[r + 1], seq[r]
        if q1 <= q0 or n_keys <= 0:
            continue
        n_blk = -(-n_keys // bs)
        pages = block_table[r, :n_blk].to(torch.int64)
        # [Hkv, n_keys, D]
        keys = k_cache[pages].flatten(0, 1)[:n_keys].float().transpose(0, 1)
        vals = v_cache[pages].flatten(0, 1)[:n_keys].float().transpose(0, 1)
        kpos = torch.arange(n_keys, device=dev)
        for c0 in range(q0, q1, _ROW_CHUNK):
            c1 = min(c0 + _ROW_CHUNK, q1)
            n = c1 - c0
            q_pos = ctx[r] + (c0 - q0) + torch.arange(n, device=dev)
            sel = topk_idx[:, c0:c1].to(torch.int64)
            col = torch.where(sel >= 0, sel, n_blk).clamp_max(n_blk)
            src = (
                torch.zeros_like(sel, dtype=torch.float32)
                if blk_bias is None
                else blk_bias[:, c0:c1].float()
            )
            blk_logit = torch.full(
                (num_kv_heads, n, n_blk + 1), float("-inf"), device=dev
            )
            blk_logit.scatter_(2, col, src)
            tok_logit = blk_logit[..., :n_blk].repeat_interleave(bs, dim=2)[
                ..., :n_keys
            ]
            remote = kpos[None, :] <= (q_pos - lw)[:, None]
            local = (kpos[None, :] > (q_pos - lw)[:, None]) & (
                kpos[None, :] <= q_pos[:, None]
            )
            logit = torch.where(
                local, 0.0, torch.where(remote, tok_logit, float("-inf"))
            )
            qg = (q[c0:c1].float() * sm_scale).view(n, num_kv_heads, group, head_dim)
            out = _attend(qg.transpose(0, 1), keys, vals, logit)
            output[c0:c1].copy_(out.transpose(0, 1).reshape(n, num_heads, head_dim))
