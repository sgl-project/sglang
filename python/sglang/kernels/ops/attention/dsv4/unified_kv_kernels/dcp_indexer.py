"""Torch-only helpers for DCP sharding of the DeepSeek-V4 C4 indexer.

Split out of srt/layers/attention/dsv4/indexer.py so the pure-tensor logic can
be imported (and GPU-tested) without pulling in the full sglang attention stack
-- the dsv4 kernels package __init__ imports compiled extensions.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

_local_arange_cache: Dict[str, torch.Tensor] = {}

# ---------------------------------------------------------------------------
# DCP: shard the indexer CANDIDATE axis (scoring), not just the selection.
#
# #29185 shards only the top-k *selection*: every rank still scores the whole
# context, so the O(c4_len) HBM traffic of the indexer scan (the single largest
# read term in a DSV4 decode step) is still paid dcp_size times over.
#
# Scoring is driven entirely by (page_table, seq_lens) -- see
# _aiter_fp8_paged_mqa_logits -- so sharding the candidate axis is a pure
# metadata operation: hand each rank a page-strided slice of the page table.
# Page granularity is c4_page_size (= page_size//4 = 64 c4 rows = 256 raw
# tokens), i.e. the slice lands on COMPRESSED rows and cannot break the c4
# recurrence.
#
# Because topk_transform resolves a candidate to its physical page by indexing
# through the page table, mapping local -> global coordinates is free: we do the
# indirection with the LOCAL page table BEFORE the all-gather, so what goes on
# the wire is already global physical page indices.
# ---------------------------------------------------------------------------


# Persistent buffers: the sharded path must be CUDA-graph capturable, so the
# local page table and local sequence lengths live in preallocated tensors that
# are filled in place (never reallocated) once a shape has been seen.
_dcp_shard_cache: Dict[tuple, torch.Tensor] = {}


def _dcp_cached(key: tuple, factory) -> torch.Tensor:
    t = _dcp_shard_cache.get(key)
    if t is None:
        t = factory()
        _dcp_shard_cache[key] = t
    return t


def dcp_candidate_shard(
    page_table: torch.Tensor,
    c4_seq_lens: torch.Tensor,
    c4_page_size: int,
    dcp_size: int,
    dcp_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Page-strided slice of the indexer candidate axis for this DCP rank.

    Rank ``r`` owns global c4 pages ``{r, r+N, r+2N, ...}``. Returns
    ``(local_page_table, local_c4_seq_lens, local_max_c4_len)``.

    Validity stays a prefix in local coordinates: the global sequence is a
    prefix, and pages are handed out in increasing order, so a rank's local
    pages are "all full, plus at most one partial tail" -- exactly the shape the
    ``positions < seq_lens`` masks in the scorer and in topk_transform assume.

        P        = ceil(S / c4_page_size)                     # global pages
        n_local  = ceil((P - r) / N)                          # clamped at 0
        owns_last= ((P - 1) % N == r) and P > 0
        local_S  = n_local*page - (page - tail) if owns_last else n_local*page

    NOTE: ``page_table`` is shared with the main attention metadata
    (``indexer_metadata.page_table is core_metadata.page_table``), so it must
    never be sliced in place -- we materialise into our own buffer.
    """
    B, p_max = page_table.shape
    device = page_table.device
    n_local = max((p_max - dcp_rank + dcp_size - 1) // dcp_size, 1)

    # Built as arange(n_local)*N + r then clamped, NOT as
    # arange(r, p_max, N): when p_max <= r (a batch whose whole context fits in
    # fewer pages than there are ranks) the latter is EMPTY while n_local is
    # floored at 1, and index_select would raise on the shape mismatch. The
    # clamped form always yields exactly n_local valid indices; the rows those
    # ranks read are then masked off by local_seq_len == 0.
    idx = _dcp_cached(
        ("pages", device, p_max, dcp_size, dcp_rank, n_local),
        lambda: torch.clamp(
            torch.arange(n_local, device=device) * dcp_size + dcp_rank,
            max=max(p_max - 1, 0),
        ),
    )
    local_pt = _dcp_cached(
        ("pt", device, B, n_local, page_table.dtype),
        lambda: torch.empty((B, n_local), dtype=page_table.dtype, device=device),
    )
    torch.index_select(page_table, 1, idx, out=local_pt)

    sl = c4_seq_lens.view(B, -1)[:, 0] if c4_seq_lens.dim() > 1 else c4_seq_lens.view(B)
    S = sl.to(torch.int64)
    P = (S + c4_page_size - 1) // c4_page_size
    nl = torch.clamp((P - dcp_rank + dcp_size - 1) // dcp_size, min=0)
    tail = S - (P - 1) * c4_page_size  # only meaningful where P > 0
    owns_last = ((P - 1) % dcp_size == dcp_rank) & (P > 0)
    local_S = nl * c4_page_size - torch.where(
        owns_last, c4_page_size - tail, torch.zeros_like(tail)
    )
    local_S = torch.clamp(local_S, min=0)

    local_sl = _dcp_cached(
        ("sl", device, B, c4_seq_lens.dtype),
        lambda: torch.empty((B,), dtype=c4_seq_lens.dtype, device=device),
    )
    local_sl.copy_(local_S.to(c4_seq_lens.dtype))
    return local_pt, local_sl, n_local * c4_page_size


def topk_transform_512_dcp_sharded(
    scores: torch.Tensor,
    local_seq_lens: torch.Tensor,
    local_page_table: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    dcp_group,
    dcp_size: int,
    dcp_rank: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Merge top-k when ``scores`` are already this rank's candidate shard.

    Unlike ``topk_transform_512_dcp`` (which slices a full score row), here the
    scores only cover the rank's own pages, so there is no local->global index
    arithmetic on the score axis. Instead we resolve each local candidate to its
    physical page through ``local_page_table`` *before* the all-gather, so the
    gathered payload is already in global coordinates and no rank ever has to
    interpret another rank's local indices.

    Equivalence: a globally top-k candidate is necessarily top-k within its own
    shard, so the union of per-shard top-k always contains the exact global
    top-k. Output page order may differ from the single-rank kernel; the
    downstream sparse attention is order-invariant.
    """
    TOPK = out_page_indices.shape[1]
    B, N = scores.shape
    device = scores.device
    neg_inf = float("-inf")

    page_bits = (page_size - 1).bit_length() if page_size > 1 else 0
    page_mask = page_size - 1

    cache = _local_arange_cache
    key_seq = f"arange_{N}_{device}"
    if key_seq not in cache:
        cache[key_seq] = torch.arange(N, device=device)
    positions = cache[key_seq].unsqueeze(0)
    valid = positions < local_seq_lens.view(B, 1)
    masked = torch.where(valid, scores, torch.full_like(scores, neg_inf))

    k = min(TOPK, N)
    local_scores, local_idx = torch.topk(masked, k, dim=1, largest=True, sorted=False)

    # Local candidate -> global physical page index, using the LOCAL page table.
    lidx = local_idx.to(torch.int64)
    page_idx = lidx >> page_bits
    offset = lidx & page_mask
    physical = torch.gather(local_page_table.to(torch.int64), 1, page_idx)
    local_pages = ((physical << page_bits) | offset).to(torch.int32)
    live = local_scores != neg_inf
    local_pages = torch.where(live, local_pages, torch.full_like(local_pages, -1))

    gather_raw = out_raw_indices is not None
    if gather_raw:
        # Global raw c4 token index of a local candidate:
        #   (dcp_rank + local_page*dcp_size) * page_size + offset
        g_page = dcp_rank + page_idx * dcp_size
        local_raw = ((g_page << page_bits) | offset).to(torch.int32)
        local_raw = torch.where(live, local_raw, torch.full_like(local_raw, -1))

    if k < TOPK:
        pad = TOPK - k
        local_scores = F.pad(local_scores, (0, pad), value=neg_inf)
        local_pages = F.pad(local_pages, (0, pad), value=-1)
        if gather_raw:
            local_raw = F.pad(local_raw, (0, pad), value=-1)

    # Ship every channel in ONE all-gather. These payloads are a few KB, and at
    # that size a collective costs the same whether it carries 4 B or 12 KB
    # (measured: 28.3 us vs 31.7 us on 8x MI355X), so the cost is per call, not
    # per byte. Channel-major [C, B, TOPK] gathered on the last dim keeps every
    # channel contiguous afterwards, so no repacking is needed on the way out.
    channels = 3 if gather_raw else 2
    pack_key = f"pack_{channels}_{B}_{TOPK}_{device}"
    if pack_key not in cache:
        cache[pack_key] = torch.empty(
            (channels, B, TOPK), dtype=torch.float32, device=device
        )
    packed = cache[pack_key]
    # int32 payloads ride along bit-for-bit; all-gather only copies bytes.
    packed[0] = local_scores
    packed[1] = local_pages.view(torch.float32)
    if gather_raw:
        packed[2] = local_raw.view(torch.float32)

    gathered = dcp_group.all_gather(packed, dim=2)
    g_scores = gathered[0]
    g_pages = gathered[1].view(torch.int32)

    m_scores, m_pos = torch.topk(g_scores, TOPK, dim=1, largest=True, sorted=False)
    final_valid = m_scores != neg_inf

    final_pages = torch.gather(g_pages, 1, m_pos)
    final_pages = torch.where(
        final_valid, final_pages, torch.full_like(final_pages, -1)
    )
    out_page_indices.copy_(final_pages)

    if gather_raw:
        g_raw = gathered[2].view(torch.int32)
        final_raw = torch.gather(g_raw, 1, m_pos)
        final_raw = torch.where(final_valid, final_raw, torch.full_like(final_raw, -1))
        out_raw_indices.copy_(final_raw.to(out_raw_indices.dtype))
