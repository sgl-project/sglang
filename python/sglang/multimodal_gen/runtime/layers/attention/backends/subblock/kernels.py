# SPDX-License-Identifier: Apache-2.0
"""Fused sub-block score kernel: bf16 GEMM + segmented log-sum-exp in one pass.

    score[i, j] = log2 sum_{b < n_k} 2 ** ( qbar_i . kbar_{j,b} )

The PyTorch router materialises ``[B, H, Gq, Gk*n_k]`` fp32 (254 MB at S=96k, n_k=4),
reduces it, and throws it away -- the GEMM is 4% of its time, the rest is that tensor's
round trip to HBM. Here the reduction happens in registers before anything is written, so
only ``[B, H, Gq, Gk]`` ever reaches memory: n_k times less traffic.

exp2/log2 are used internally (they are the hardware instructions; the caller folds
``softmax_scale * log2(e)`` into Q), and the result is converted back to natural-log units
so it matches the reference implementation exactly, not just up to ranking.

Padding: sub-cells are ordered, and validity is monotone, so a single ``n_valid`` scalar
(the number of key sub-cells holding at least one real token) is enough -- everything at or
past it is forced to -inf so it can never win a slot.
"""

import math

import torch
import triton
import triton.language as tl

_NEG = tl.constexpr(-1.0e30)  # Triton only lets @jit read constexpr globals
_LN2 = tl.constexpr(0.6931471805599453)


@triton.jit
def _score_kernel(
    Q,
    K,
    O,
    stride_qm,
    stride_ql,
    stride_kn,
    stride_kl,
    stride_om,
    stride_on,
    stride_ol,
    M,
    M_VALID,
    N_VALID,
    NOUT,
    MOUT,
    BLK_M: tl.constexpr,
    BLK_N: tl.constexpr,
    NK: tl.constexpr,
    NQR: tl.constexpr,
    D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_l = tl.program_id(2)

    offs_m = pid_m * BLK_M + tl.arange(0, BLK_M)
    offs_n = pid_n * BLK_N + tl.arange(0, BLK_N)
    offs_d = tl.arange(0, D)

    q = tl.load(
        Q + pid_l * stride_ql + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=offs_m[:, None] < M,
        other=0.0,
    )
    k = tl.load(
        K + pid_l * stride_kl + offs_n[:, None] * stride_kn + offs_d[None, :],
        mask=offs_n[:, None] < N_VALID,
        other=0.0,
    )
    acc = tl.dot(q, tl.trans(k), out_dtype=tl.float32)  # [BLK_M, BLK_N]

    # a sub-cell past the last real one must not contribute to its group's log-sum-exp
    acc = tl.where(offs_n[None, :] < N_VALID, acc, _NEG)
    # same on the query side: with NQ > 1 the last query block can own sub-cells that
    # are entirely padding, and those pool to zero -- an exp2(0) = 1 term that would
    # otherwise be folded into the block's score.
    acc = tl.where(offs_m[:, None] < M_VALID, acc, _NEG)

    acc = tl.reshape(acc, (BLK_M, BLK_N // NK, NK))
    m = tl.max(acc, axis=2)
    s = tl.sum(tl.exp2(acc - m[:, :, None]), axis=2)
    lse = m + tl.log2(s)
    lse = tl.where(m > _NEG / 2, lse, _NEG)  # whole group was padding

    # Fold the NQR query sub-cells of a query block together. Log-sum-exp is
    # associative, so reducing NK then NQ is the same one log-sum-exp over all
    # NQ*NK sub-block pairs -- and two stages keeps both reductions on an axis
    # that is already contiguous in registers.
    if NQR > 1:
        lse = tl.reshape(lse, (BLK_M // NQR, NQR, BLK_N // NK))
        m2 = tl.max(lse, axis=1)
        s2 = tl.sum(tl.exp2(lse - m2[:, None, :]), axis=1)
        lse = tl.where(m2 > _NEG / 2, m2 + tl.log2(s2), _NEG)

    # exp2/log2 internally (they map to the hardware instructions), then back to natural
    # log units so the fused and reference backends return the same numbers, not just the
    # same ranking. One multiply in registers.
    out = lse * _LN2
    out = out.to(O.dtype.element_ty)  # bf16 halves what the selection step has to read

    offs_o = pid_n * (BLK_N // NK) + tl.arange(0, BLK_N // NK)
    offs_q = pid_m * (BLK_M // NQR) + tl.arange(0, BLK_M // NQR)
    tl.store(
        O
        + pid_l * stride_ol
        + offs_q[:, None] * stride_om
        + offs_o[None, :] * stride_on,
        out,
        mask=(offs_q[:, None] < MOUT) & (offs_o[None, :] < NOUT),
    )


BLK_M = BLK_N = 128  # score tile; must hold whole blocks, so a multiple of n_q and n_k


def fused_scores(qp, kp, out, *, n_k, n_valid, n_q, m_valid):
    """qp: [L, Gq*n_q, D] bf16 (already carrying softmax_scale*log2e),
    kp: [L, Gk*n_k, D] bf16 -> out: [L, Gq, Gk], natural-log scores.

    The n_q query sub-cells of a query block are folded together by log-sum-exp.

    ``n_valid`` / ``m_valid`` are the counts of key / query sub-cells holding at
    least one real token; the rest pooled to zero and must not contribute.
    """
    L, M, D = qp.shape
    N = kp.shape[1]
    Mout, Nout = out.shape[1], out.shape[2]
    grid = (triton.cdiv(M, BLK_M), triton.cdiv(N, BLK_N), L)
    _score_kernel[grid](
        qp,
        kp,
        out,
        qp.stride(1),
        qp.stride(0),
        kp.stride(1),
        kp.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(0),
        M,
        m_valid,
        n_valid,
        Nout,
        Mout,
        BLK_M=BLK_M,
        BLK_N=BLK_N,
        NK=n_k,
        NQR=n_q,
        D=D,
        num_warps=4,
        num_stages=3,
    )
    return out


@triton.jit
def _pool_kernel(
    X,
    Y,
    stride_xb,
    stride_xt,
    stride_xh,
    stride_yl,
    stride_yn,
    S,
    H,
    SUB: tl.constexpr,
    D: tl.constexpr,
    SCALE,
):
    """[B, S, H, D] -> [B*H, n_cells, D]: masked mean of every SUB consecutive tokens.

    Fused so the activation is read once and the pooled result is written straight in
    bf16; the PyTorch version needs an fp32 temporary plus a transpose.
    """
    cell = tl.program_id(0)
    l = tl.program_id(1)
    b = l // H
    h = l % H
    offs_t = cell * SUB + tl.arange(0, SUB)
    offs_d = tl.arange(0, D)
    mask = offs_t < S
    x = tl.load(
        X
        + b * stride_xb
        + offs_t[:, None] * stride_xt
        + h * stride_xh
        + offs_d[None, :],
        mask=mask[:, None],
        other=0.0,
    ).to(tl.float32)
    cnt = tl.sum(mask.to(tl.float32), axis=0)
    acc = tl.sum(x, axis=0) / tl.maximum(cnt, 1.0) * SCALE
    tl.store(Y + l * stride_yl + cell * stride_yn + offs_d, acc.to(tl.bfloat16))


def fused_pool(x, n_cells, sub, out, scale=1.0):
    """x: [B, S, H, D] bf16 -> out: [B*H, n_cells, D] bf16"""
    B, S, H, D = x.shape
    _pool_kernel[(n_cells, B * H)](
        x,
        out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        out.stride(0),
        out.stride(1),
        S,
        H,
        SUB=sub,
        D=D,
        SCALE=scale,
        # one warp, not four: the tile is only SUB x 128, so extra warps buy no parallelism
        # and cost scheduling. Measured 0.272 -> 0.074 ms at S=96k, 1.27 -> 4.7 TB/s.
        num_warps=1,
    )
    return out


@triton.jit
def _topk_kernel(S, OUT, G, K, BLK: tl.constexpr, ITERS: tl.constexpr):
    """Exact-enough per-row top-K in a single pass over global memory.

    A score row is only Gk values -- 3 KB in bf16 at S=96k -- so it fits in registers. Load
    it once, then do everything on chip. torch.topk instead makes several passes over the
    whole matrix, which is why it costs half the router.

    The threshold search interpolates on the count rather than halving the interval: the
    count-vs-threshold curve is the row's empirical CDF, so a secant step lands far closer
    than a bisection step. Full-row reductions are what this kernel pays for, so fewer
    steps is the whole game.

    The invariant is count(s >= lo) >= K, so the compaction can only over-fill, never
    under-fill; `pos < K` truncates the boundary group. At ITERS=12 about 1.6% of rows end
    up with a different set than an exact top-K, but only among blocks that tie at the
    threshold -- the total selected score differs by ~1e-6 relative, which is nothing.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLK)
    m = offs < G
    s = tl.load(S + row * G + offs, mask=m, other=-float("inf")).to(tl.float32)
    lo = tl.min(tl.where(m, s, float("inf")))
    hi = tl.max(tl.where(m, s, -float("inf"))) + 1.0
    clo = tl.sum(m.to(tl.int32), axis=0).to(tl.float32)
    chi = 0.0
    for _ in tl.static_range(ITERS):
        den = clo - chi
        t = (clo - K) / tl.where(den > 0.5, den, 1.0)
        t = tl.minimum(tl.maximum(t, 0.05), 0.95)  # keep the step inside the bracket
        mid = lo + (hi - lo) * t
        cnt = tl.sum(((s >= mid) & m).to(tl.int32), axis=0).to(tl.float32)
        take = cnt >= K
        lo = tl.where(take, mid, lo)
        clo = tl.where(take, cnt, clo)
        hi = tl.where(take, hi, mid)
        chi = tl.where(take, chi, cnt)
    sel = (s >= lo) & m
    pos = tl.cumsum(sel.to(tl.int32), axis=0) - 1
    tl.store(OUT + row * K + pos, offs.to(tl.int32), mask=sel & (pos < K))


def topk_iters(G, k):
    """Threshold-search steps needed to match ``torch.topk``.

    The search interpolates on the *count* above a trial threshold, which assumes the count
    is linear in the threshold. That holds near the median but not in the tail, where the
    score density decays roughly exponentially and the secant undershoots -- so the further
    into the tail k sits, the more steps are needed. Measured on 21 real files as the
    fraction of query rows selecting a different block set than exact top-k:

        log2(G/k)   1.0    2.1    3.3    4.3    5.6      (sparsity .50 .76 .90 .95 .98)
        16 steps  0.09%  0.14%  1.62%  5.17%  1.88%
        24 steps  0.02%  0.01%  0.14%  0.61%  0.21%
        32 steps  0.02%  0.01%  0.02%  0.19%  0.05%

    A flat 16 is fine at the usual operating points and silently wrong past sparsity 0.9
    (+3.2% relative L2 at 0.90, +7.1% at 0.95). These cutoffs hold every regime under 0.2%.
    Interpolating on log(count) instead linearises the tail and does fix sparsity >= 0.95,
    but it is far worse where the tail model does not apply (27% differing rows at sparsity
    0.5), so the step count is the robust knob, not the model.
    """
    L = math.log2(max(G, 1) / max(k, 1))
    return 16 if L <= 2.5 else 24 if L <= 3.5 else 32


def fused_topk(scores2d, k):
    """scores2d: [rows, G] contiguous -> [rows, k] int32 column ids (unsorted)."""
    rows, G = scores2d.shape
    out = torch.empty(rows, k, dtype=torch.int32, device=scores2d.device)
    _topk_kernel[(rows,)](
        scores2d,
        out,
        G,
        k,
        BLK=triton.next_power_of_2(G),
        ITERS=topk_iters(G, k),
        num_warps=4 if G >= 1024 else 2,  # short rows do not fill four warps
    )
    return out
