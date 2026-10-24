# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Vendored from ROCm/aiter#5882 (https://github.com/ROCm/aiter/pull/5882),
# commit 03359c43df (03359c43df40abf91ac38a2040c9b8e284d92297),
# file aiter/ops/flydsl/mla_v4_grouped.py.
# Unchanged except for import paths and sglang formatting (isort, ruff format).
# TODO(aiter#5882): remove once ROCm/aiter#5882 is in the aiter pin

"""Grouped DSpark verify streams for the FlyDSL v4 decode.

The num_draft tokens of a request (positions p-nd+1 .. p) share one stream at an HCA /
SWA-only layer: [compressed tail of the last draft (C entries)][SWA ring slots of
positions p-n+1 .. p], n = min(p+1, win+nd-1) (the sglang#41120 layout). q_kv_bounds
gives each draft its exact key set in it: [0, C_q) and its own window.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

__all__ = ["flydsl_mla_v4_grouped_verify_meta"]

GROUPED_MAX_MSQ = 8  # drafts per request


@triton.jit
def _grouped_meta_kernel(
    idx_ptr,
    kv_indptr_ptr,
    qo_indptr_ptr,
    bnd_ptr,
    slot_ptr,
    pos_ptr,
    tail_len_ptr,
    tail_page_ptr,
    G,
    ring_stride,
    swa_pages,
    tail_stride,
    ND: tl.constexpr,
    WIN: tl.constexpr,
    TAIL_B: tl.constexpr,
    WIN_B: tl.constexpr,
    GB: tl.constexpr,
    HAS_TAIL: tl.constexpr,
    MAXND: tl.constexpr,
):
    g = tl.program_id(0)
    # start of this group's stream: the previous groups' lengths
    base = 0
    for off in tl.range(0, G, GB):
        gg = off + tl.arange(0, GB)
        m = gg < g
        last = gg * ND + ND - 1
        p = tl.load(pos_ptr + last, mask=m, other=0).to(tl.int32)
        if HAS_TAIL:
            c = tl.load(tail_len_ptr + last, mask=m, other=0).to(tl.int32)
        else:
            c = tl.zeros([GB], tl.int32)
        n = tl.minimum(p + 1, WIN + ND - 1)
        base += tl.sum(tl.where(m, c + n, 0))
    first = g * ND
    last = first + ND - 1
    p = tl.load(pos_ptr + last).to(tl.int32)
    if HAS_TAIL:
        c = tl.load(tail_len_ptr + last).to(tl.int32)
    else:
        c = 0
    n = tl.minimum(p + 1, WIN + ND - 1)
    tl.store(kv_indptr_ptr + g + 1, base + c + n)
    tl.store(qo_indptr_ptr + g + 1, first + ND)
    if g == 0:
        tl.store(kv_indptr_ptr, 0)
        tl.store(qo_indptr_ptr, 0)
    if HAS_TAIL:
        for off in tl.range(0, c, TAIL_B):
            jt = off + tl.arange(0, TAIL_B)
            mt = jt < c
            pi = tl.load(
                tail_page_ptr + last.to(tl.int64) * tail_stride + jt, mask=mt, other=-1
            )
            tl.store(
                idx_ptr + base + jt, tl.where(pi >= 0, pi + swa_pages, -1), mask=mt
            )
    slot = tl.load(slot_ptr + last).to(tl.int64)
    i = tl.arange(0, WIN_B)
    ap = (p - n + 1 + i).to(tl.int64)
    tl.store(
        idx_ptr + base + c + i,
        (slot * ring_stride + ap % ring_stride).to(tl.int32),
        mask=i < n,
    )
    # per-row bounds, stream relative
    q = tl.arange(0, MAXND)
    mq = q < ND
    t = first + q
    pq = tl.load(pos_ptr + t, mask=mq, other=0).to(tl.int32)
    if HAS_TAIL:
        cq = tl.load(tail_len_ptr + t, mask=mq, other=0).to(tl.int32)
    else:
        cq = tl.zeros([MAXND], tl.int32)
    s0 = p - n + 1
    tb = bnd_ptr + t.to(tl.int64) * 4
    tl.store(tb + 0, tl.zeros([MAXND], tl.int32), mask=mq)
    tl.store(tb + 1, cq, mask=mq)
    tl.store(tb + 2, c + tl.maximum(0, pq - WIN + 1 - s0), mask=mq)
    tl.store(tb + 3, c + pq - s0 + 1, mask=mq)


def flydsl_mla_v4_grouped_verify_meta(
    state_slot: torch.Tensor,
    positions: torch.Tensor,
    tail_len: torch.Tensor | None,
    tail_page_indices: torch.Tensor | None,
    *,
    win: int,
    ring_stride: int,
    swa_pages: int,
    num_draft: int,
):
    """(kv_indices, kv_indptr, qo_indptr, q_kv_bounds) of the grouped streams, one
    graph-capturable launch; N = requests * num_draft tokens, request-major.

    state_slot / positions [N]; tail_len [N] / tail_page_indices [N, Wc] int32: the
    committed compressed entries per token (None: SWA-only). kv_indices is sized for
    the worst case. Use with max_seqlen_q = num_draft.
    """
    assert 1 <= num_draft <= GROUPED_MAX_MSQ
    dev = positions.device
    N = positions.shape[0]
    assert N % num_draft == 0, (N, num_draft)
    G = N // num_draft
    has_tail = tail_len is not None and tail_page_indices is not None
    Wc = tail_page_indices.shape[1] if has_tail else 0
    idx = torch.empty(
        max(G * (Wc + win + num_draft - 1), 1), dtype=torch.int32, device=dev
    )
    kv_indptr = torch.empty(G + 1, dtype=torch.int32, device=dev)
    qo_indptr = torch.empty(G + 1, dtype=torch.int32, device=dev)
    bnd = torch.empty((N, 4), dtype=torch.int32, device=dev)
    if G == 0:
        kv_indptr.zero_()
        qo_indptr.zero_()
        return idx, kv_indptr, qo_indptr, bnd
    assert state_slot.is_contiguous() and positions.is_contiguous()
    if has_tail:
        assert tail_len.is_contiguous() and tail_page_indices.stride(1) == 1
    tp = tail_page_indices if has_tail else positions
    _grouped_meta_kernel[(G,)](
        idx,
        kv_indptr,
        qo_indptr,
        bnd,
        state_slot,
        positions,
        tail_len if has_tail else positions,
        tp,
        G,
        ring_stride,
        swa_pages,
        tp.stride(0) if has_tail else 0,
        ND=num_draft,
        WIN=win,
        TAIL_B=min(1024, triton.next_power_of_2(max(Wc, 1))),
        WIN_B=triton.next_power_of_2(win + num_draft - 1),
        GB=min(1024, triton.next_power_of_2(G)),
        HAS_TAIL=has_tail,
        MAXND=GROUPED_MAX_MSQ,
        num_warps=4,
    )
    return idx, kv_indptr, qo_indptr, bnd
