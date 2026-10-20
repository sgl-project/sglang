"""Split-KV attention for a DFlash draft block: 8 query rows x 8 heads x 1 KV head, head_dim 128, non-causal.

Each request's block attends to its FP8 prefix cache (paged through ``kv_indptr`` / ``kv_indices``, left window
``window_left``) and to the block's own BF16 K/V. The prefix is split across workgroups; FP32 partials are merged
by a second kernel. Prefix Q.K stays on FP8 arithmetic; with ``pv_bf16`` the prefix P.V runs on BF16
probabilities (more precise than the FP8 P.V of the generic Triton extend kernel).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _partials(
    Q,
    KE,
    VE,
    KB,
    VB,
    CUQ,
    CUK,
    INDICES,
    PART,
    MAXIMUM,
    DENOM,
    QS0: tl.constexpr,
    QS1: tl.constexpr,
    KES: tl.constexpr,
    VES: tl.constexpr,
    KBS: tl.constexpr,
    VBS: tl.constexpr,
    KS,
    VS,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    PV_BF16: tl.constexpr,
    WINDOW: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    qb, seq, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qstart = tl.load(CUQ + seq)
    qlen = tl.load(CUQ + seq + 1) - qstart
    kstart = tl.load(CUK + seq)
    prefix = tl.load(CUK + seq + 1) - kstart
    r = tl.arange(0, BM)
    d = tl.arange(0, 128)
    col = tl.arange(0, BN)
    qr = qb * (BM // 8) + r // 8
    qh = r % 8
    valid_q = qr < qlen
    qrow = qstart + qr
    q = tl.load(
        Q + qrow[:, None] * QS0 + qh[:, None] * QS1 + d[None, :],
        valid_q[:, None],
        other=0.0,
    )
    # Keep only tiles that can intersect any row's inclusive2048-token window.
    first = 0
    if WINDOW > 0:
        first = tl.maximum(0, prefix + qb * (BM // 8) - WINDOW) // BN
    stop = tl.cdiv(prefix, BN)
    per = tl.cdiv(tl.maximum(stop - first, 0), SPLITS)
    begin = first + split * per
    end = tl.minimum(stop, begin + per)
    maximum = tl.full((BM,), float("-inf"), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    accum = tl.zeros((BM, 128), tl.float32)
    for block in range(begin, end):
        pos = block * BN + col
        valid = pos < prefix
        loc = tl.load(INDICES + kstart + pos, valid, other=0).to(tl.int64)
        k = tl.load(KB + loc[None, :] * KBS + d[:, None], valid[None, :], other=0.0)
        v = tl.load(VB + loc[:, None] * VBS + d[None, :], valid[:, None], other=0.0)
        score = tl.dot(q.to(k.dtype), k) * (SCALE * 1.4426950408889634) * KS
        allowed = valid_q[:, None] & valid[None, :]
        if WINDOW > 0:
            allowed &= prefix + qr[:, None] <= pos[None, :] + WINDOW
        score = tl.where(allowed, score, float("-inf"))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        new_max = tl.where(new_max > float("-inf"), new_max, 0.0)
        probability = tl.exp2(score - new_max[:, None])
        alpha = tl.exp2(maximum - new_max)
        denom = denom * alpha + tl.sum(probability, 1)
        if PV_BF16:
            product = tl.dot(probability.to(tl.bfloat16), v.to(tl.bfloat16))
        else:
            product = tl.dot(probability.to(v.dtype), v)
        accum = accum * alpha[:, None] + product * VS
        maximum = new_max
    if split == 0:
        c = tl.arange(0, 32)
        k = tl.load(
            KE + (qstart + c[None, :]) * KES + d[:, None], c[None, :] < qlen, other=0.0
        )
        v = tl.load(
            VE + (qstart + c[:, None]) * VES + d[None, :], c[:, None] < qlen, other=0.0
        )
        score = tl.dot(q, k) * (SCALE * 1.4426950408889634)
        score = tl.where(valid_q[:, None] & (c[None, :] < qlen), score, float("-inf"))
        new_max = tl.maximum(maximum, tl.max(score, 1))
        new_max = tl.where(new_max > float("-inf"), new_max, 0.0)
        probability = tl.exp2(score - new_max[:, None])
        alpha = tl.exp2(maximum - new_max)
        denom = denom * alpha + tl.sum(probability, 1)
        accum = accum * alpha[:, None] + tl.dot(probability.to(v.dtype), v)
        maximum = new_max
    index = (qrow * 8 + qh) * SPLITS + split
    tl.store(MAXIMUM + index, maximum, valid_q)
    tl.store(DENOM + index, denom, valid_q)
    tl.store(PART + index[:, None] * 128 + d[None, :], accum, valid_q[:, None])


@triton.jit
def _reduce(PART, MAXIMUM, DENOM, OUT, SPLITS: tl.constexpr):
    row = tl.program_id(0)
    s = tl.arange(0, SPLITS)
    d = tl.arange(0, 128)
    ix = row * SPLITS + s
    maximum = tl.load(MAXIMUM + ix)
    denom = tl.load(DENOM + ix)
    overall = tl.max(tl.where(denom > 0, maximum, float("-inf")), 0)
    overall = tl.where(overall > float("-inf"), overall, 0.0)
    weight = tl.where(denom > 0, tl.exp2(maximum - overall), 0.0)
    total = tl.sum(denom * weight, 0)
    part = tl.load(PART + ix[:, None] * 128 + d[None, :])
    answer = tl.sum(part * weight[:, None], 0) / tl.maximum(total, 1e-20)
    tl.store(OUT + row * 128 + d, answer)


def dflash_block_splitkv(
    q,
    ke,
    ve,
    kb,
    vb,
    cuq,
    cuk,
    ids,
    *,
    kscale=1.0,
    vscale=1.0,
    softmax_scale=128**-0.5,
    splits=16,
    pv_bf16=False,
    out=None,
    window_left=2048,
):
    if not (
        q.dtype == ke.dtype == ve.dtype == torch.bfloat16
        and q.shape[1:] == (8, 128)
        and ke.shape == ve.shape == (q.shape[0], 1, 128)
        and kb.dtype == vb.dtype == torch.float8_e4m3fn
        and kb.ndim == vb.ndim == 3
        and kb.shape[1:] == vb.shape[1:] == (1, 128)
        and q.shape[0] == (cuq.numel() - 1) * 8
        and cuk.numel() == cuq.numel()
        and splits in (8, 16, 32)
        and all(t.stride(-1) == 1 for t in (q, ke, ve, kb, vb))
    ):
        raise ValueError(
            "DFlash split-KV requires flat 8Q/8heads/1KV/HD128 mixed BF16/FP8 tensors"
        )
    rows = q.shape[0] * 8
    part = torch.empty((rows, splits, 128), device=q.device, dtype=torch.float32)
    maximum = torch.empty((rows, splits), device=q.device, dtype=torch.float32)
    denom = torch.empty_like(maximum)
    if out is None:
        out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    if out.shape != q.shape or out.dtype != q.dtype or not out.is_contiguous():
        raise ValueError("Output must be contiguous BF16 with the query shape")
    _partials[(4, cuq.numel() - 1, splits)](
        q,
        ke,
        ve,
        kb,
        vb,
        cuq,
        cuk,
        ids,
        part,
        maximum,
        denom,
        q.stride(0),
        q.stride(1),
        ke.stride(0),
        ve.stride(0),
        kb.stride(0),
        vb.stride(0),
        kscale,
        vscale,
        softmax_scale,
        splits,
        pv_bf16,
        window_left,
        16,
        128,
        num_warps=4,
        num_stages=1,
        matrix_instr_nonkdim=16,
    )
    _reduce[(rows,)](part, maximum, denom, out, splits, num_warps=4)
    return out
