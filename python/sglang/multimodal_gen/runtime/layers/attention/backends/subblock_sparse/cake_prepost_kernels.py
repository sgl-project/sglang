# SPDX-License-Identifier: Apache-2.0
"""Fuse the layout, routing and output glue around prepared-Sage CAKE SSA.

The routing path deliberately rounds K's scale and scaled elements to BF16
before pooling, exactly like the unfused reference. It changes no estimator.
"""

import math

import torch
import triton
import triton.language as tl

from .kernels import fused_scores, fused_topk


@triton.jit
def _pack_cake_operands(
    Q,
    K,
    V,
    QS,
    KS,
    CQ,
    CK,
    CV,
    CQS,
    CKS,
    QB: tl.constexpr,
    QT: tl.constexpr,
    QH: tl.constexpr,
    KB: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    VB: tl.constexpr,
    VD: tl.constexpr,
    VH: tl.constexpr,
    QSB: tl.constexpr,
    QSH: tl.constexpr,
    KSB: tl.constexpr,
    KSH: tl.constexpr,
    S: tl.constexpr,
    VS: tl.constexpr,
    H: tl.constexpr,
    B: tl.constexpr,
    QG: tl.constexpr,
    KG: tl.constexpr,
    TILE: tl.constexpr,
):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    d = i % 128
    t = i // 128 % S
    h = i // (128 * S) % H
    b = i // (128 * S * H)
    valid = i < B * H * S * 128
    q = tl.load(Q + b * QB + t * QT + h * QH + d, valid, 0)
    k = tl.load(K + b * KB + t * KT + h * KH + d, valid, 0)
    tl.store(CQ + i, q, valid)
    tl.store(CK + i, k, valid)
    t = i % VS
    d = i // VS % 128
    h = i // (VS * 128) % H
    b = i // (VS * 128 * H)
    valid = i < B * H * 128 * VS
    v = tl.load(V + b * VB + d * VD + h * VH + t, valid, 0)
    tl.store(CV + i, v, valid)
    g = i % QG
    h = i // QG % H
    b = i // (QG * H)
    scale = tl.load(QS + b * QSB + h * QSH + g, i < B * H * QG, 0)
    tl.store(CQS + i, scale, i < B * H * QG)
    g = i % KG
    h = i // KG % H
    b = i // (KG * H)
    scale = tl.load(KS + b * KSB + h * KSH + g, i < B * H * KG, 0)
    tl.store(CKS + i, scale, i < B * H * KG)


def pack_cake_operands(q, k, v, qs, ks, used):
    b, _, h, d = q.shape
    vs = triton.cdiv(used, 64) * 64
    qg = triton.cdiv(used, 128) * 4
    kg = triton.cdiv(used, 64)
    cq = torch.empty((b, h, used, d), device=q.device, dtype=q.dtype)
    ck = torch.empty_like(cq)
    cv = torch.empty((b, h, d, vs), device=v.device, dtype=v.dtype)
    cqs = torch.empty((b, h, qg), device=qs.device, dtype=qs.dtype)
    cks = torch.empty((b, h, kg), device=ks.device, dtype=ks.dtype)
    _pack_cake_operands[(triton.cdiv(cv.numel(), 1024),)](
        q,
        k,
        v.view(torch.uint8),
        qs,
        ks,
        cq,
        ck,
        cv.view(torch.uint8),
        cqs,
        cks,
        *q.stride()[:3],
        *k.stride()[:3],
        *v.stride()[:3],
        *qs.stride()[:2],
        *ks.stride()[:2],
        used,
        vs,
        h,
        b,
        qg,
        kg,
        1024,
    )
    return cq, ck, cv, cqs, cks


@triton.jit
def _pool_prequantized(
    Q,
    K,
    KS,
    PQ,
    PK,
    QB: tl.constexpr,
    QT: tl.constexpr,
    QH: tl.constexpr,
    KB: tl.constexpr,
    KT: tl.constexpr,
    KH: tl.constexpr,
    KSB: tl.constexpr,
    KSH: tl.constexpr,
    S: tl.constexpr,
    H: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    SUBQ: tl.constexpr,
    SUBK: tl.constexpr,
    SCALE,
):
    cell = tl.program_id(0)
    bh = tl.program_id(1)
    b, h = bh // H, bh % H
    d = tl.arange(0, 128)
    if cell < NQ:
        t = cell * SUBQ + tl.arange(0, SUBQ)
        x = tl.load(
            Q + b * QB + t[:, None] * QT + h * QH + d[None, :], t[:, None] < S, 0
        ).to(tl.float32)
        count = tl.sum((t < S).to(tl.float32), 0)
        value = tl.sum(x, 0) / tl.maximum(count, 1.0) * SCALE
        tl.store(PQ + (bh * NQ + cell) * 128 + d, value.to(tl.bfloat16))
    if cell < NK:
        t = cell * SUBK + tl.arange(0, SUBK)
        x = tl.load(
            K + b * KB + t[:, None] * KT + h * KH + d[None, :], t[:, None] < S, 0
        ).to(tl.float32)
        scale = tl.load(KS + b * KSB + h * KSH + t // 64, t < S, 0)
        # Preserve both BF16 roundings of k.int8.to(bf16) * scale.to(bf16).
        scale = scale.to(tl.bfloat16).to(tl.float32)
        x = (x * scale[:, None]).to(tl.bfloat16).to(tl.float32)
        count = tl.sum((t < S).to(tl.float32), 0)
        value = tl.sum(x, 0) / tl.maximum(count, 1.0)
        tl.store(PK + (bh * NK + cell) * 128 + d, value.to(tl.bfloat16))


def route_prequantized(q, k, ks, used, n_q, n_k, sparsity, softmax_scale):
    b, _, h, d = q.shape
    blocks = triton.cdiv(used, 64)
    nq, nk = blocks * n_q, blocks * n_k
    pq = torch.empty((b * h, nq, d), device=q.device, dtype=torch.bfloat16)
    pk = torch.empty((b * h, nk, d), device=q.device, dtype=torch.bfloat16)
    _pool_prequantized[(max(nq, nk), b * h)](
        q,
        k,
        ks,
        pq,
        pk,
        *q.stride()[:3],
        *k.stride()[:3],
        *ks.stride()[:2],
        used,
        h,
        nq,
        nk,
        64 // n_q,
        64 // n_k,
        softmax_scale * 1.4426950408889634,
        num_warps=1,
    )
    scores = torch.empty((b * h, blocks, blocks), device=q.device, dtype=torch.float32)
    fused_scores(
        pq,
        pk,
        scores,
        n_k=n_k,
        n_valid=triton.cdiv(used, 64 // n_k),
        n_q=n_q,
        m_valid=triton.cdiv(used, 64 // n_q),
    )
    topk = min(blocks, triton.cdiv(max(1, math.ceil((1 - sparsity) * blocks)), 8) * 8)
    index = fused_topk(scores.reshape(-1, blocks), topk).view(b, h, blocks, topk)
    return index, topk


@triton.jit
def _expand_mixed_rows(
    INDEX,
    MASK,
    OUTPUT,
    COUNTS,
    G: tl.constexpr,
    TOPK: tl.constexpr,
    MASK_STRIDE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0)
    selected = tl.load(MASK + row % G * MASK_STRIDE)
    col = tl.program_id(1) * TILE + tl.arange(0, TILE)
    sparse = tl.load(INDEX + row * TOPK + col, (col < TOPK) & selected, 0)
    value = tl.where(selected & (col < TOPK), sparse, col)
    tl.store(OUTPUT + row * G + col, value, col < G)
    if tl.program_id(1) == 0:
        tl.store(COUNTS + row, tl.where(selected, TOPK, G))


def mixed_row_indices(index, mask, blocks):
    b, h, g, topk = index.shape
    if mask.numel() != blocks:
        raise ValueError("H3 sparse query-block mask does not match the live sequence")
    mask = mask.to(device=index.device, dtype=torch.bool).view(-1)
    dense = torch.empty((b, h, g, blocks), device=index.device, dtype=torch.int32)
    counts = torch.empty((b, h, g), device=index.device, dtype=torch.int32)
    _expand_mixed_rows[(b * h * g, triton.cdiv(blocks, 256))](
        index,
        mask,
        dense,
        counts,
        blocks,
        topk,
        mask.stride(0),
        256,
    )
    return dense, counts


@triton.jit
def _write_cake_output(
    SRC,
    DST,
    S: tl.constexpr,
    TOTAL: tl.constexpr,
    H: tl.constexpr,
    B: tl.constexpr,
    TILE: tl.constexpr,
):
    i = tl.program_id(0) * TILE + tl.arange(0, TILE)
    d = i % 128
    h = i // 128 % H
    t = i // (128 * H) % TOTAL
    b = i // (128 * H * TOTAL)
    value = tl.load(
        SRC + ((b * H + h) * S + t) * 128 + d, (i < B * TOTAL * H * 128) & (t < S), 0
    )
    tl.store(DST + i, value, i < B * TOTAL * H * 128)


def write_cake_output(attn, out, used, total):
    b, _, h, _ = out.shape
    _write_cake_output[(triton.cdiv(out.numel(), 1024),)](
        attn,
        out,
        used,
        total,
        h,
        b,
        1024,
    )
