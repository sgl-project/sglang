"""Pinned Qwen Sparse Attention (QSA) program shared by rollout and trainer.

The same arithmetic serves prefill, chunked prefill, decode and trainer scoring:
- index Q/K projection on the shared BF16 BI GEMM, per-row offset RMSNorm;
- RoPE from one FP32 table built on the device, BF16(x1*c - x2*s), each rotary lane at the position of its
  MRoPE axis (a text token has one position on every axis);
- compressed key = BF16(sum of the r raw keys in order, FP32, times 1/r), then norm + RoPE
  at the group's first position; only complete groups exist;
- block score = sum_h relu(q_h . k_b) / sqrt(D), FP32, one fixed tile shape;
- top-k by score, ties to the lower block id (FlashInfer deterministic top-k); blocks ascending;
- attention over the ascending token list in fixed 64-entry tiles, FP32 online softmax.
K/V are read through per-row slot lists, so serving reads its pool and the trainer
its packed tensors with the same kernel.
"""

import torch
import triton
import triton.language as tl

SCORE_BLOCK = 64
ATTN_BLOCK = 64
# Fixed split-KV segment; a row's math depends on it, never on T.
ATTN_SEGMENT = 512
_ATTN_PARTIAL_BYTES = 256 << 20
# Arbitrary cap on the per-chunk FP32 score buffer; rows are independent.
_SELECT_BYTES = 256 << 20


def index_projection(hidden_states, weight):
    from sglang.srt.batch_invariant_ops import matmul_persistent

    return matmul_persistent(hidden_states.contiguous(), weight.t())


def rope_positions(positions):
    """Model RoPE positions as the kernels read them: [T] (one position on every axis) or MRoPE [3, T] ->
    int64 [T, 3]."""
    positions = positions.to(torch.int64)
    if positions.ndim == 1:
        return positions.unsqueeze(1).expand(-1, 3).contiguous()
    return positions.transpose(0, 1).contiguous()


def lane_axes(rotary_emb, device):
    """int32 [rotary_dim / 2]: the position axis each rotary lane reads; axis 0 for a RoPE without MRoPE sections."""
    from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding

    if isinstance(rotary_emb, MRotaryEmbedding) and rotary_emb.axis_map is not None:
        return rotary_emb.axis_map.to(device=device, dtype=torch.int32).contiguous()
    return torch.zeros(rotary_emb.rotary_dim // 2, dtype=torch.int32, device=device)


_ROPE_TABLES = {}
# Default table length covers Qwen3.8-Next's max_position_embeddings; grows outside capture.
_ROPE_MIN_POSITIONS = 262144


def rope_table(*, base, rotary_dim, device, positions=_ROPE_MIN_POSITIONS):
    """FP32 (cos, sin) [positions, rotary_dim / 2]; entries do not depend on the table length."""
    key = (float(base), int(rotary_dim), str(device))
    table = _ROPE_TABLES.get(key)
    if table is None or table[0].shape[0] < positions:
        n = max(positions, _ROPE_MIN_POSITIONS)
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
                / rotary_dim
            )
        )
        freqs = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(
            1
        ) * inv_freq.unsqueeze(0)
        table = _ROPE_TABLES[key] = (freqs.cos().contiguous(), freqs.sin().contiguous())
    return table


@triton.jit
def _norm_rope_row(
    x,
    xp,
    w_ptr,
    offs,
    partner,
    cos_ptr,
    sin_ptr,
    pos_ptr,
    axis_ptr,
    D: tl.constexpr,
    RD: tl.constexpr,
    EPS: tl.constexpr,
):
    """BF16(offset RMSNorm) then neox RoPE on the leading RD dims; ``xp`` is x at ``partner``. ``pos_ptr`` is the
    row's int64 [3] position and ``axis_ptr`` the int32 [RD / 2] axis of each lane."""
    HALF: tl.constexpr = RD // 2
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + EPS)
    y = (
        (x * rstd * (1.0 + tl.load(w_ptr + offs).to(tl.float32)))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    yp = (
        (xp * rstd * (1.0 + tl.load(w_ptr + partner).to(tl.float32)))
        .to(tl.bfloat16)
        .to(tl.float32)
    )
    freq = tl.where(offs < HALF, offs, offs - HALF)
    rot = offs < RD
    pos = tl.load(
        pos_ptr + tl.load(axis_ptr + freq, mask=rot, other=0), mask=rot, other=0
    )
    c = tl.load(cos_ptr + pos * HALF + freq, mask=rot, other=1.0)
    s = tl.load(sin_ptr + pos * HALF + freq, mask=rot, other=0.0)
    signed = tl.where(offs < HALF, -yp, yp)
    return tl.where(rot, y * c + signed * s, y)


@triton.jit
def _partner(offs, RD: tl.constexpr):
    return tl.where(
        offs < RD // 2, offs + RD // 2, tl.where(offs < RD, offs - RD // 2, offs)
    )


@triton.jit
def _norm_rope_kernel(
    x_ptr,
    w_ptr,
    pos_ptr,
    axis_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    H: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    EPS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, D)
    partner = _partner(offs, RD)
    x = tl.load(x_ptr + row * D + offs).to(tl.float32)
    xp = tl.load(x_ptr + row * D + partner).to(tl.float32)
    out = _norm_rope_row(
        x,
        xp,
        w_ptr,
        offs,
        partner,
        cos_ptr,
        sin_ptr,
        pos_ptr + (row // H) * 3,
        axis_ptr,
        D,
        RD,
        EPS,
    )
    tl.store(out_ptr + row * D + offs, out.to(tl.bfloat16))


@triton.jit
def _compress_kernel(
    src_ptr,
    locs_ptr,
    pos_ptr,
    axis_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    R: tl.constexpr,
    D: tl.constexpr,
    RD: tl.constexpr,
    EPS: tl.constexpr,
):
    """Mean of R raw keys (FP32, oldest first, x 1/R, BF16), then _norm_rope_row at the first position."""
    g = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, D)
    partner = _partner(offs, RD)
    acc = tl.zeros([D], dtype=tl.float32)
    accp = tl.zeros([D], dtype=tl.float32)
    for r in tl.static_range(R):
        loc = tl.load(locs_ptr + g * R + r).to(tl.int64)
        acc += tl.load(src_ptr + loc * D + offs).to(tl.float32)
        accp += tl.load(src_ptr + loc * D + partner).to(tl.float32)
    x = (acc * (1.0 / R)).to(tl.bfloat16).to(tl.float32)
    xp = (accp * (1.0 / R)).to(tl.bfloat16).to(tl.float32)
    out = _norm_rope_row(
        x,
        xp,
        w_ptr,
        offs,
        partner,
        cos_ptr,
        sin_ptr,
        pos_ptr + g * 3,
        axis_ptr,
        D,
        RD,
        EPS,
    )
    tl.store(out_ptr + g * D + offs, out.to(tl.bfloat16))


def norm_rope(x, positions, *, weight, eps, rotary_dim, base, axes):
    """Indexer queries: BF16 x [N, H, D], int64 positions [N, 3], int32 lane axes [rotary_dim / 2] -> BF16
    [N, H, D]."""
    n, h, d = x.shape
    out = torch.empty_like(x)
    if n:
        cos, sin = rope_table(base=base, rotary_dim=rotary_dim, device=x.device)
        _norm_rope_kernel[(n * h,)](
            x.contiguous(),
            weight.contiguous(),
            _position_rows(positions, n),
            axes,
            cos,
            sin,
            out,
            H=h,
            D=d,
            RD=rotary_dim,
            EPS=eps,
        )
    return out


def _position_rows(positions, rows):
    if positions.shape != (rows, 3):
        raise ValueError(
            f"[top] QSA RoPE positions must be [{rows}, 3], got {tuple(positions.shape)}"
        )
    return positions.to(torch.int64).contiguous()


def compress_norm_rope(src, locs, positions, *, weight, eps, rotary_dim, base, axes):
    """Stands in for the stock QSA key compression (average_pool_qsa_keys + normalize_compressed_keys, or the fused
    qsa_index_k_compress_store when the pool carries its rope buffer): which program runs depends on the serving pool,
    and the fused kernel writes the pool, which the trainer does not have.

    Compressed keys from raw key rows ``src`` [S, D] at ``locs`` [G, r] (oldest first), each rotated at its
    group's position [G, 3] -> BF16 [G, D]."""
    g, r = locs.shape
    d = src.shape[-1]
    out = torch.empty(g, d, dtype=torch.bfloat16, device=src.device)
    if g:
        cos, sin = rope_table(base=base, rotary_dim=rotary_dim, device=src.device)
        _compress_kernel[(g,)](
            src.reshape(-1, d).contiguous(),
            locs.to(torch.int64).contiguous(),
            _position_rows(positions, g),
            axes,
            weight.contiguous(),
            cos,
            sin,
            out,
            R=r,
            D=d,
            RD=rotary_dim,
            EPS=eps,
        )
    return out


@triton.jit(do_not_specialize=["width"])
def _score_kernel(
    q_ptr,  # BF16 [rows, H, D]
    k_ptr,  # BF16 compressed keys: [N, D] or [pages, CPS, D]
    start_ptr,  # int32 [rows] (contiguous) or unused
    table_ptr,  # int32 [rows, table_width] (paged) or unused
    count_ptr,  # int32 [rows], visible blocks
    out_ptr,  # FP32 [rows, width]
    width,
    table_width,
    H: tl.constexpr,
    D: tl.constexpr,
    PAGED: tl.constexpr,
    CPS: tl.constexpr,
    SQRT_D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    offs_b = tile * BLOCK + tl.arange(0, BLOCK)
    offs_d = tl.arange(0, D)
    count = tl.load(count_ptr + row)
    valid = offs_b < count
    if PAGED:
        page = tl.load(
            table_ptr + row * table_width + offs_b // CPS, mask=valid, other=0
        ).to(tl.int64)
        loc = page * CPS + offs_b % CPS
    else:
        loc = tl.load(start_ptr + row).to(tl.int64) + offs_b
    k = tl.load(
        k_ptr + loc[:, None] * D + offs_d[None, :], mask=valid[:, None], other=0.0
    ).to(tl.float32)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for h in tl.static_range(H):
        qh = tl.load(q_ptr + (row * H + h) * D + offs_d).to(tl.float32)
        acc += tl.maximum(tl.sum(k * qh[None, :], axis=1), 0.0)
    # Scores are >= 0 with +0.0 forced, so equal scores have equal bits and tie in the selection.
    # Blocks at and past count are never read.
    logit = tl.where(acc > 0, acc / SQRT_D, 0.0)
    tl.store(out_ptr + row * width + offs_b, logit, mask=offs_b < width)


def _score_logits(q, keys, counts, width, *, starts=None, table=None, cps=1):
    """FP32 block scores [rows, width]; row r's visible blocks are [:counts[r]]."""
    rows, H, D = q.shape
    out = torch.empty(rows, width, dtype=torch.float32, device=q.device)
    if rows == 0 or width == 0:
        return out
    paged = table is not None
    counts = counts.to(torch.int32).contiguous()
    if paged:
        table = table.to(torch.int32).contiguous()
    else:
        starts = starts.to(torch.int32).contiguous()
    _score_kernel[(rows, triton.cdiv(width, SCORE_BLOCK))](
        q.contiguous(),
        keys.contiguous(),
        counts if paged else starts,
        table if paged else counts,
        counts,
        out,
        width,
        table.shape[1] if paged else 0,
        H=H,
        D=D,
        PAGED=paged,
        CPS=cps,
        SQRT_D=float(D) ** 0.5,
        BLOCK=SCORE_BLOCK,
    )
    return out


def select_blocks(logits, counts, block_topk):
    """Top-k of each row's visible blocks logits[r, :counts[r]] by score, ties to the lower block id;
    ascending ids, -1 padded. A row with at most block_topk visible blocks selects all of them.

    CUDA runs FlashInfer's deterministic top-k directly, so the trainer, which does not set SGLang's
    batch-invariant flag, selects with the same kernel as serving.
    """
    counts = counts.to(torch.int32).clamp(max=logits.shape[1])
    if logits.is_cuda:
        from sglang.srt.true_on_policy.dsa.ordered_topk import ordered_dsa_topk

        return ordered_dsa_topk(logits, counts, block_topk)
    return select_blocks_reference(logits, counts, block_topk)


def select_blocks_reference(logits, counts, block_topk):
    """Torch reference of select_blocks: a stable descending sort keeps the lower id first among ties."""
    rows = logits.shape[0]
    out = torch.full((rows, block_topk), -1, dtype=torch.int32, device=logits.device)
    for row in range(rows):
        count = int(counts[row])
        picked = torch.sort(logits[row, :count], descending=True, stable=True)
        picked = picked.indices[:block_topk].sort().values
        out[row, : picked.numel()] = picked.to(torch.int32)
    return out


def select_contiguous(q, keys, starts, counts, block_topk):
    """Prefill/trainer: keys [N, D] packed per sequence, row i sees keys[starts[i]:][:counts[i]]."""
    out = torch.empty((q.shape[0], block_topk), dtype=torch.int32, device=q.device)
    max_width = int(counts.max().item()) if q.shape[0] else 0
    chunk = max(1, _SELECT_BYTES // (4 * max(max_width, 1)))
    for lo in range(0, q.shape[0], chunk):
        hi = min(lo + chunk, q.shape[0])
        width = int(counts[lo:hi].max().item())
        out[lo:hi] = select_blocks(
            _score_logits(q[lo:hi], keys, counts[lo:hi], width, starts=starts[lo:hi]),
            counts[lo:hi],
            block_topk,
        )
    return out


def select_paged(q, cache, page_table, counts, width, block_topk):
    """Stands in for the stock QSA block scoring (qsa_mqa_decode in decode, qsa_mqa_prefill in prefill): two kernels,
    so a token's block scores would depend on whether it was prefilled or decoded. The top-k itself is stock FlashInfer
    deterministic top-k (ordered_dsa_topk).

    Decode: cache [pages, CPS, D], page_table [rows, pages_per_row]; graph-safe."""
    return select_blocks(
        _score_logits(q, cache, counts, width, table=page_table, cps=cache.shape[1]),
        counts,
        block_topk,
    )


@triton.jit(do_not_specialize=["T", "W"])
def _sparse_attn_partial_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    slot_ptr,
    count_ptr,
    table_ptr,
    req_ptr,
    table_stride,
    m_ptr,  # FP32 [T, HQ, S]
    l_ptr,  # FP32 [T, HQ, S]
    acc_ptr,  # FP32 [T, HQ, S, D], unnormalized
    T,
    W,
    scale,
    S: tl.constexpr,
    HQ: tl.constexpr,
    HKV: tl.constexpr,
    D: tl.constexpr,
    GROUP_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
    SEGMENT: tl.constexpr,
    MAP: tl.constexpr,
):
    """Online softmax over slots [seg*SEGMENT, ...) of one query row and KV head."""
    t = tl.program_id(0).to(tl.int64)
    kvh = tl.program_id(1)
    seg = tl.program_id(2)
    count = tl.load(count_ptr + t)
    lo = seg * SEGMENT
    if lo < count:
        hi = tl.minimum(count, lo + SEGMENT)
        GROUP: tl.constexpr = HQ // HKV
        offs_g = tl.arange(0, GROUP_PAD)
        offs_d = tl.arange(0, D)
        g_mask = offs_g < GROUP
        heads = kvh * GROUP + offs_g
        q = tl.load(
            q_ptr + (t * HQ + heads[:, None]) * D + offs_d[None, :],
            mask=g_mask[:, None],
            other=0.0,
        )
        m = tl.full([GROUP_PAD], float("-inf"), dtype=tl.float32)
        l = tl.zeros([GROUP_PAD], dtype=tl.float32)
        acc = tl.zeros([GROUP_PAD, D], dtype=tl.float32)
        for start in tl.range(lo, hi, BLOCK):
            offs_n = start + tl.arange(0, BLOCK)
            valid = offs_n < hi
            slot = tl.load(slot_ptr + t * W + offs_n, mask=valid, other=0).to(tl.int64)
            if MAP:
                req = tl.load(req_ptr + t)
                slot = tl.load(
                    table_ptr + req * table_stride + slot, mask=valid, other=0
                ).to(tl.int64)
            k = tl.load(
                k_ptr + (slot[:, None] * HKV + kvh) * D + offs_d[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            v = tl.load(
                v_ptr + (slot[:, None] * HKV + kvh) * D + offs_d[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            s = tl.dot(q, tl.trans(k)) * scale
            s = tl.where(valid[None, :], s, float("-inf"))
            m_new = tl.maximum(m, tl.max(s, axis=1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m - m_new)
            l = l * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m = m_new
        row = (t * HQ + heads) * S + seg
        tl.store(m_ptr + row, m, mask=g_mask)
        tl.store(l_ptr + row, l, mask=g_mask)
        tl.store(
            acc_ptr + row[:, None] * D + offs_d[None, :], acc, mask=g_mask[:, None]
        )


@triton.jit(do_not_specialize=["T"])
def _sparse_attn_combine_kernel(
    m_ptr,
    l_ptr,
    acc_ptr,
    count_ptr,
    out_ptr,  # BF16 [T, HQ, D]
    o32_ptr,  # FP32 [T, HQ, D]
    lse_ptr,  # FP32 [HQ, T]
    T,
    S: tl.constexpr,
    HQ: tl.constexpr,
    D: tl.constexpr,
    HEADS: tl.constexpr,
    SEGMENT: tl.constexpr,
    AUX: tl.constexpr,
):
    """Merge the non-empty segments of HEADS query heads in segment order."""
    t = tl.program_id(0).to(tl.int64)
    heads = tl.program_id(1) * HEADS + tl.arange(0, HEADS)
    offs_d = tl.arange(0, D)
    h_mask = heads < HQ
    used = tl.cdiv(tl.load(count_ptr + t), SEGMENT)
    base = (t * HQ + heads) * S
    big_m = tl.full([HEADS], float("-inf"), dtype=tl.float32)
    for seg in range(0, used):
        big_m = tl.maximum(
            big_m, tl.load(m_ptr + base + seg, mask=h_mask, other=float("-inf"))
        )
    l = tl.zeros([HEADS], dtype=tl.float32)
    acc = tl.zeros([HEADS, D], dtype=tl.float32)
    for seg in range(0, used):
        w = tl.exp(
            tl.load(m_ptr + base + seg, mask=h_mask, other=float("-inf")) - big_m
        )
        l += tl.load(l_ptr + base + seg, mask=h_mask, other=0.0) * w
        a = tl.load(
            acc_ptr + (base + seg)[:, None] * D + offs_d[None, :],
            mask=h_mask[:, None],
            other=0.0,
        )
        acc += a * w[:, None]
    l_safe = tl.where(l > 0, l, 1.0)
    o = acc / l_safe[:, None]
    ptrs = (t * HQ + heads[:, None]) * D + offs_d[None, :]
    tl.store(out_ptr + ptrs, o.to(out_ptr.dtype.element_ty), mask=h_mask[:, None])
    if AUX:
        tl.store(o32_ptr + ptrs, o, mask=h_mask[:, None])
        tl.store(
            lse_ptr + heads * T + t,
            tl.where(l > 0, big_m + tl.log(l_safe), float("-inf")),
            mask=h_mask,
        )


def sparse_attention(q, k, v, slots, scale, *, table=None, table_rows=None, aux=True):
    """Stands in for the stock QSA sparse attention (the trtllm-gen sparse decode, else qsa_sparse_attention): the
    trtllm-gen kernel splits each row's KV across CTAs by a plan sized to the batch (its multi-CTA KV counter is
    keyed by batch size).

    (BF16 out [T, HQ, D], FP32 out, FP32 LSE [HQ, T]); FP32 out/LSE are None without ``aux``.

    ``slots`` index rows of k/v, or with ``table`` are logical positions looked up in
    ``table[table_rows[t]]`` (the serving req_to_token map).
    """
    T, HQ, D = q.shape
    HKV = k.shape[1]
    if HQ % HKV or k.shape[2] != D or v.shape != k.shape or slots.shape[0] != T:
        raise ValueError(
            "TOP QSA attention requires q [T,HQ,D], k/v [S,HKV,D], slots [T,W]"
        )
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
    ):
        raise ValueError("TOP QSA attention requires BF16 q and K/V")
    if not (k.is_contiguous() and v.is_contiguous()):
        raise ValueError("TOP QSA attention addresses K/V as dense [S, HKV, D]")
    slots = slots.to(torch.int32).contiguous()
    counts = (slots >= 0).sum(dim=1, dtype=torch.int32)
    out = torch.empty_like(q)
    o32 = torch.empty(T, HQ, D, dtype=torch.float32, device=q.device) if aux else out
    lse = torch.empty(HQ, T, dtype=torch.float32, device=q.device) if aux else out
    if T:
        _sparse_attention_split(
            q.contiguous(),
            k,
            v,
            slots,
            counts,
            scale,
            table=table,
            table_rows=table_rows,
            out=out,
            o32=o32,
            lse=lse,
            aux=aux,
        )
    return out, (o32 if aux else None), (lse if aux else None)


def _sparse_attention_split(
    q, k, v, slots, counts, scale, *, table, table_rows, out, o32, lse, aux
):
    """Fixed ATTN_SEGMENT split-KV; row chunks bound the FP32 partial buffer, not the math."""
    T, HQ, D = q.shape
    HKV = k.shape[1]
    W = slots.shape[1]
    S = max(1, triton.cdiv(W, ATTN_SEGMENT))
    mapped = table is not None
    rows = table_rows.to(torch.int64).contiguous() if mapped else counts
    chunk = max(1, _ATTN_PARTIAL_BYTES // (HQ * S * (D + 2) * 4))
    heads = 16
    for lo in range(0, T, chunk):
        hi = min(T, lo + chunk)
        n = hi - lo
        m = torch.empty(n, HQ, S, dtype=torch.float32, device=q.device)
        l = torch.empty_like(m)
        acc = torch.empty(n, HQ, S, D, dtype=torch.float32, device=q.device)
        _sparse_attn_partial_kernel[(n, HKV, S)](
            q[lo:hi],
            k,
            v,
            slots[lo:hi],
            counts[lo:hi],
            table if mapped else counts,
            rows[lo:hi] if mapped else counts,
            table.stride(0) if mapped else 0,
            m,
            l,
            acc,
            n,
            W,
            scale,
            S=S,
            HQ=HQ,
            HKV=HKV,
            D=D,
            GROUP_PAD=max(16, triton.next_power_of_2(HQ // HKV)),
            BLOCK=ATTN_BLOCK,
            SEGMENT=ATTN_SEGMENT,
            MAP=mapped,
        )
        lse_view = lse[:, lo:hi] if aux else counts
        _sparse_attn_combine_kernel[(n, triton.cdiv(HQ, heads))](
            m,
            l,
            acc,
            counts[lo:hi],
            out[lo:hi],
            o32[lo:hi] if aux else counts,
            lse_view,
            n,
            S=S,
            HQ=HQ,
            D=D,
            HEADS=heads,
            SEGMENT=ATTN_SEGMENT,
            AUX=aux,
        )
