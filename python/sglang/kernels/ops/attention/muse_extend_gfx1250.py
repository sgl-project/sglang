"""Muse Spark extend (prefill) attention for gfx1250, Triton: FP8 Q (per-tensor scale), paged FP8 KV
(page 16), head dim 64, 8 query heads per KV head, causal full layers and the 2047-key sliding window.

Targeted port of the gfx950 Gluon kernel (muse_extend_gluon.py): one program per (KV head, BLOCK_Q-token query
block); the BLOCK_Q tokens x GQA heads that share a KV head form one BM-row Q tile, so every K / V tile is read once
for all 8 heads. S = Q K^T and O += P V run as FP8 dots (WMMA on gfx1250); P = 448 * exp2(s * c - m) keeps the
probabilities in the FP8 E4M3 range (numerator and denominator scale alike). Key tiles are split into the masked
sliding head (keys some queries of the block no longer see), the unmasked interior and the masked causal tail, as
in the Gluon kernel; the interior loop is software pipelined by Triton (num_stages).
"""

import torch
import triton
import triton.language as tl

LOG2E = 1.4426950408889634
LOG2_448 = 8.807354922057604


@triton.jit
def _find_seq(cu_q, q_block, num_seqs, BLOCK_Q: tl.constexpr):
    # request owning q-block q_block in the grid's upper-bound numbering (cu_q[s] // BLOCK_Q + s)
    lo = 0
    hi = num_seqs
    while lo < hi:
        mid = (lo + hi) // 2
        if tl.load(cu_q + mid) // BLOCK_Q + mid <= q_block:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


@triton.jit
def _tile(
    q,
    m_i,
    l_i,
    acc,
    j,
    K,
    V,
    bt,
    kvh,
    seq_len,
    lo,
    q_pos,
    c,
    k_s0,
    k_s1,
    k_s2,
    v_s0,
    v_s1,
    v_s2,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    TILE: tl.constexpr,
    MASKED: tl.constexpr,
    WINDOW: tl.constexpr,
    FP8_DOT: tl.constexpr,
):
    kpos = j * TILE + tl.arange(0, TILE)
    d = tl.arange(0, D)
    if MASKED:
        live = (kpos < seq_len) & (kpos >= lo)
        page = tl.load(bt + kpos // PAGE, mask=live, other=0).to(tl.int64)
    else:
        page = tl.load(bt + kpos // PAGE).to(tl.int64)
    off = kpos % PAGE
    kp = K + page[:, None] * k_s0 + off[:, None] * k_s1 + kvh * k_s2 + d[None, :]
    vp = V + page[:, None] * v_s0 + off[:, None] * v_s1 + kvh * v_s2 + d[None, :]
    if MASKED:
        k = tl.load(kp, mask=live[:, None], other=0.0)
        v = tl.load(vp, mask=live[:, None], other=0.0)
    else:
        k = tl.load(kp)
        v = tl.load(vp)
    if FP8_DOT:
        s = tl.dot(q, tl.trans(k))
    else:
        s = tl.dot(q.to(tl.bfloat16), tl.trans(k.to(tl.bfloat16)))
    if MASKED:
        ok = (kpos[None, :] <= q_pos[:, None]) & (kpos < seq_len)[None, :]
        if WINDOW > 0:
            ok = ok & (kpos[None, :] > q_pos[:, None] - WINDOW)
        s = tl.where(ok, s, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(s, axis=1) * c)
    p = tl.exp2(s * c - (m_new - 8.807354922057604)[:, None])
    alpha = tl.exp2(m_i - m_new)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None]
    if FP8_DOT:
        acc = tl.dot(p.to(tl.float8e4nv), v, acc)
    else:
        acc = tl.dot(p.to(tl.bfloat16), v.to(tl.bfloat16), acc)
    return m_new, l_i, acc


@triton.jit
def _muse_extend_gfx1250_kernel(
    OUT,
    Q,
    K,
    V,
    BT,
    SEQ_LENS,
    CU_Q,
    QD,
    KD,
    VD,
    scale,
    num_seqs,
    bt_stride,
    q_stride0,
    o_stride0,
    k_s0,
    k_s1,
    k_s2,
    v_s0,
    v_s1,
    v_s2,
    GQA: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    TILE: tl.constexpr,
    WINDOW: tl.constexpr,
    FP8_DOT: tl.constexpr,
    STAGES: tl.constexpr,
):
    BM: tl.constexpr = BLOCK_Q * GQA
    kvh = tl.program_id(0)
    qb = tl.program_id(1)
    seq = _find_seq(CU_Q, qb, num_seqs, BLOCK_Q)
    q_start = tl.load(CU_Q + seq)
    q_len = tl.load(CU_Q + seq + 1) - q_start
    local = qb - (q_start // BLOCK_Q + seq)
    if local * BLOCK_Q < q_len:
        seq_len = tl.load(SEQ_LENS + seq).to(tl.int32)
        ctx = seq_len - q_len
        c = scale * 1.4426950408889634 * tl.load(QD) * tl.load(KD)
        r = tl.arange(0, BM)
        tok = local * BLOCK_Q + r // GQA
        head = kvh * GQA + r % GQA
        d = tl.arange(0, D)
        q = tl.load(
            Q
            + (q_start + tok).to(tl.int64)[:, None] * q_stride0
            + head[:, None] * D
            + d[None, :],
            mask=(tok < q_len)[:, None],
            other=0.0,
        )
        q_pos = ctx + tok
        bt = BT + seq * bt_stride
        m_i = tl.full([BM], -1.0e30 if WINDOW > 0 else float("-inf"), tl.float32)
        l_i = tl.zeros([BM], tl.float32)
        acc = tl.zeros([BM, D], tl.float32)
        # tiles below `interior` are seen in full by every query of the block
        interior = (ctx + local * BLOCK_Q + 1) // TILE
        n_tiles = tl.cdiv(ctx + tl.minimum(local * BLOCK_Q + BLOCK_Q, q_len), TILE)
        if WINDOW > 0:
            lo = tl.maximum(ctx + local * BLOCK_Q - WINDOW + 1, 0)
            j0 = lo // TILE
            q_hi = ctx + tl.minimum(local * BLOCK_Q + BLOCK_Q, q_len) - 1
            full_lo = tl.minimum(
                tl.cdiv(tl.maximum(q_hi - WINDOW + 1, 0), TILE), interior
            )
            full_lo = tl.maximum(full_lo, j0)
            for j in range(j0, full_lo):
                m_i, l_i, acc = _tile(
                    q,
                    m_i,
                    l_i,
                    acc,
                    j,
                    K,
                    V,
                    bt,
                    kvh,
                    seq_len,
                    lo,
                    q_pos,
                    c,
                    k_s0,
                    k_s1,
                    k_s2,
                    v_s0,
                    v_s1,
                    v_s2,
                    D,
                    PAGE,
                    TILE,
                    True,
                    WINDOW,
                    FP8_DOT,
                )
        else:
            lo = 0
            full_lo = 0
        for j in tl.range(full_lo, interior, num_stages=STAGES):
            m_i, l_i, acc = _tile(
                q,
                m_i,
                l_i,
                acc,
                j,
                K,
                V,
                bt,
                kvh,
                seq_len,
                lo,
                q_pos,
                c,
                k_s0,
                k_s1,
                k_s2,
                v_s0,
                v_s1,
                v_s2,
                D,
                PAGE,
                TILE,
                False,
                WINDOW,
                FP8_DOT,
            )
        for j in range(tl.maximum(interior, full_lo), n_tiles):
            m_i, l_i, acc = _tile(
                q,
                m_i,
                l_i,
                acc,
                j,
                K,
                V,
                bt,
                kvh,
                seq_len,
                lo,
                q_pos,
                c,
                k_s0,
                k_s1,
                k_s2,
                v_s0,
                v_s1,
                v_s2,
                D,
                PAGE,
                TILE,
                True,
                WINDOW,
                FP8_DOT,
            )
        o = acc * (tl.load(VD) / l_i)[:, None]
        tl.store(
            OUT
            + (q_start + tok).to(tl.int64)[:, None] * o_stride0
            + head[:, None] * D
            + d[None, :],
            o.to(OUT.dtype.element_ty),
            mask=(tok < q_len)[:, None],
        )


# (BLOCK_Q, TILE, num_warps, STAGES, waves_per_eu, FP8_DOT)
CONFIG = dict(BLOCK_Q=16, TILE=64, num_warps=4, STAGES=3, waves_per_eu=2, FP8_DOT=True)
CONFIG_WINDOW = dict(STAGES=2)


@triton.jit
def _q_absmax_kernel(
    Q, AMAX, n_rows, stride_q, W: tl.constexpr, BW: tl.constexpr, BLOCK_R: tl.constexpr
):
    r = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.arange(0, BW)
    q = tl.load(
        Q + r[:, None] * stride_q + c[None, :],
        mask=(r < n_rows)[:, None] & (c < W)[None, :],
        other=0.0,
    )
    tl.atomic_max(AMAX, tl.max(tl.max(tl.abs(q.to(tl.float32)), axis=1), axis=0))


@triton.jit
def _q_fp8_kernel(
    Q,
    AMAX,
    OUT,
    DESCALE,
    n_rows,
    stride_q,
    W: tl.constexpr,
    BW: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    # torch divides by a Python scalar as a multiply by its fp32 reciprocal
    descale = tl.maximum(tl.load(AMAX) * (1.0 / 448.0), 1e-12)
    if pid == 0:
        tl.store(DESCALE, descale)
    r = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    c = tl.arange(0, BW)
    mask = (r < n_rows)[:, None] & (c < W)[None, :]
    q = tl.load(Q + r[:, None] * stride_q + c[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    q = tl.math.div_rn(q, descale).to(tl.float8e4nv, fp_downcast_rounding="rtne")
    tl.store(OUT + r[:, None] * W + c[None, :], q, mask=mask)


def quantize_q_fp8(q: torch.Tensor, heads: int, block_rows: int = 8):
    """Per-tensor FP8 E4M3 of the extend query ``q`` [tokens, heads * head_dim] (rows may be strided): returns
    ([tokens, heads, head_dim] FP8, descale [1] fp32), equal to ``(q.abs().amax().float() / 448).clamp_min(1e-12)``
    and ``(q.float() / descale).to(fp8)`` in two passes instead of six torch kernels and a contiguous copy."""
    n, w = q.shape
    amax = torch.zeros(1, dtype=torch.float32, device=q.device)
    descale = torch.empty(1, dtype=torch.float32, device=q.device)
    out = torch.empty((n, w), dtype=torch.float8_e4m3fn, device=q.device)
    grid = (triton.cdiv(n, block_rows),)
    bw = triton.next_power_of_2(w)
    _q_absmax_kernel[grid](q, amax, n, q.stride(0), W=w, BW=bw, BLOCK_R=block_rows)
    _q_fp8_kernel[grid](
        q, amax, out, descale, n, q.stride(0), W=w, BW=bw, BLOCK_R=block_rows
    )
    return out.view(n, heads, w // heads), descale


def muse_extend_gfx1250_applies(q: torch.Tensor, k_cache: torch.Tensor) -> bool:
    """FP8 Q [tokens, 8 * KV heads, 64] over a paged FP8 KV cache [pages, page, KV heads, 64]."""
    return (
        q.dim() == 3
        and k_cache.dim() == 4
        and q.shape[2] == k_cache.shape[3] == 64
        and k_cache.shape[1] in (16, 32, 64, 128)
        and q.shape[1] == 8 * k_cache.shape[2]
        and q.dtype == k_cache.dtype == torch.float8_e4m3fn
    )


def muse_extend_attention_gfx1250(
    q,
    k_cache,
    v_cache,
    out,
    cu_seqlens_q,
    seq_lens,
    block_table,
    q_descale,
    k_descale,
    v_descale,
    scale,
    window=0,
    cfg=None,
):
    """Same contract as muse_extend_gluon.muse_extend_attention: ``out [tokens, heads, 64]`` (BF16) from FP8 ``q``
    and the paged FP8 cache; ``window`` > 0 = sliding (key visible when q - window < key <= q)."""
    cfg = dict(
        CONFIG, **(CONFIG_WINDOW if window and window > 0 else {}), **(cfg or {})
    )
    n = q.shape[0]
    hk = k_cache.shape[2]
    bs = seq_lens.numel()
    bq = cfg["BLOCK_Q"]
    _muse_extend_gfx1250_kernel[(hk, n // bq + bs)](
        out,
        q,
        k_cache,
        v_cache,
        block_table,
        seq_lens,
        cu_seqlens_q,
        q_descale,
        k_descale,
        v_descale,
        scale,
        bs,
        block_table.stride(0),
        q.stride(0),
        out.stride(0),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        GQA=q.shape[1] // hk,
        D=q.shape[2],
        PAGE=k_cache.shape[1],
        BLOCK_Q=bq,
        TILE=cfg["TILE"],
        WINDOW=window,
        FP8_DOT=cfg["FP8_DOT"],
        STAGES=cfg["STAGES"],
        num_warps=cfg["num_warps"],
        waves_per_eu=cfg["waves_per_eu"],
    )
    return out
