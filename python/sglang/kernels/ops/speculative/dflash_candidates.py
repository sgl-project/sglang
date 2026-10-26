"""DFlash candidate top-k across a vocab-sharded head: two launches before the all-gather, one after.

``topk_packed`` takes each rank's local top-k (per-chunk top-k, then the row's top-k of those) and writes the
values and global ids as one FP32 row ``[k values | k ids]`` (ids are exact below 2**24), so the ranks exchange
them in a single all-gather; ``merge_candidates`` picks the global top-k from the gathered rows and applies the
draft's unary-logit transform (``tanh(v * multiplier / softcap) * softcap``). Ties go to the lower id.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

MAX_PACKED_ID = 1 << 24


@triton.jit
def _merge_kernel(
    G,
    IDS,
    VALS,
    multiplier,
    softcap,
    K: tl.constexpr,
    TP: tl.constexpr,
    KP: tl.constexpr,
    SOFTCAP: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.arange(0, TP * KP)
    j = c % KP
    valid = j < K
    src = G + row * (TP * 2 * K) + (c // KP) * (2 * K) + j
    v = tl.load(src, mask=valid, other=float("-inf"))
    ids = tl.load(src + K, mask=valid, other=0.0)
    v = tl.where(valid, v, float("-inf"))
    for t in tl.static_range(K):
        m = tl.max(v, axis=0)
        # Ties go to the lowest rank, then the earlier local rank.
        pos = tl.min(tl.where(valid & (v == m), c, TP * KP), axis=0)
        sel = c == pos
        cid = tl.sum(tl.where(sel, ids, 0.0), axis=0)
        x = m * multiplier
        if SOFTCAP:
            x = libdevice.tanh(x / softcap) * softcap
        tl.store(IDS + row * K + t, cid.to(tl.int64))
        tl.store(VALS + row * K + t, x)
        v = tl.where(sel, float("-inf"), v)


@triton.jit
def _chunk_topk_kernel(
    L, CV, CI, n_cols, row_stride, K: tl.constexpr, CH: tl.constexpr, KP: tl.constexpr
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    c = chunk * CH + tl.arange(0, CH)
    v = tl.load(
        L + row.to(tl.int64) * row_stride + c, mask=c < n_cols, other=float("-inf")
    ).to(tl.float32)
    out = (row * tl.num_programs(1) + chunk) * KP
    for t in tl.static_range(K):
        m = tl.max(v, axis=0)
        pos = tl.min(tl.where(v == m, c, n_cols), axis=0)
        tl.store(CV + out + t, m)
        tl.store(CI + out + t, pos)
        v = tl.where(c == pos, float("-inf"), v)


@triton.jit
def _row_topk_pack_kernel(
    CV, CI, P, start, n_cand, K: tl.constexpr, CP: tl.constexpr, KP: tl.constexpr
):
    row = tl.program_id(0)
    c = tl.arange(0, CP)
    # Chunks wrote KP slots each; only the first K of every chunk hold candidates.
    valid = (c < n_cand) & (c % KP < K)
    v = tl.load(CV + row * n_cand + c, mask=valid, other=float("-inf"))
    ids = tl.load(CI + row * n_cand + c, mask=valid, other=0)
    big = tl.full((CP,), 2147483647, tl.int32)
    for t in tl.static_range(K):
        m = tl.max(v, axis=0)
        # The lowest id among equal values: chunks hold disjoint, increasing id ranges.
        hit = valid & (v == m)
        cid = tl.min(tl.where(hit, ids, big), axis=0)
        tl.store(P + row * 2 * K + t, m)
        tl.store(P + row * 2 * K + K + t, (cid.to(tl.int64) + start).to(tl.float32))
        v = tl.where(hit & (ids == cid), float("-inf"), v)


def topk_packed(logits: torch.Tensor, k: int, start: int) -> torch.Tensor:
    """``[N, 2k]`` FP32: the top-k values of ``logits [N, V]`` in descending order, then their ids plus ``start``
    (the shard's first id)."""
    n, n_cols = logits.shape
    kp = triton.next_power_of_2(k)
    # One-warp programs over 1024 columns, then two warps per row: measured on MI355X at the TP4 Muse Spark
    # shard (50512 columns, k 16; 23 vs 51 us for torch.topk at 64 rows).
    ch = 1024
    n_chunks = triton.cdiv(n_cols, ch)
    cand_v = torch.empty((n, n_chunks * kp), device=logits.device, dtype=torch.float32)
    cand_i = torch.empty((n, n_chunks * kp), device=logits.device, dtype=torch.int32)
    _chunk_topk_kernel[(n, n_chunks)](
        logits, cand_v, cand_i, n_cols, logits.stride(0), k, ch, kp, num_warps=1
    )
    packed = torch.empty((n, 2 * k), device=logits.device, dtype=torch.float32)
    _row_topk_pack_kernel[(n,)](
        cand_v,
        cand_i,
        packed,
        start,
        n_chunks * kp,
        k,
        triton.next_power_of_2(n_chunks * kp),
        kp,
        num_warps=2,
    )
    return packed


def merge_candidates(
    gathered: torch.Tensor,
    k: int,
    *,
    multiplier: float,
    softcap,
):
    """Global top-k of the all-gathered packed rows ``[N, tp * 2k]``: int64 ids and FP32 transformed values, both
    ``[N, k]`` in descending value order."""
    n = gathered.shape[0]
    tp = gathered.shape[1] // (2 * k)
    ids = torch.empty((n, k), device=gathered.device, dtype=torch.int64)
    vals = torch.empty((n, k), device=gathered.device, dtype=torch.float32)
    _merge_kernel[(n,)](
        gathered,
        ids,
        vals,
        float(multiplier),
        1.0 if softcap is None else float(softcap),
        k,
        tp,
        triton.next_power_of_2(k),
        softcap is not None,
        num_warps=1,
    )
    return ids, vals
