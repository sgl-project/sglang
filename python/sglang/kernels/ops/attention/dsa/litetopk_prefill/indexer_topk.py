"""Prefill DSA indexer top-k: each candidate indexer paired with the top-k kernel of its slab dtype.

The fp8 indexer emits fp32 candidate scores and pairs with the fp32 top-k; the fp4 indexer emits
bf16 scores and pairs with the bf16 top-k. Both return the same indices as the dense path's ragged
top-k transform (``dsv4.topk.topk_transform_ragged_v2``) for the same rows, so a caller can switch
between them.
"""

from __future__ import annotations

from typing import Optional

import torch

from .fp4_indexer import fp4_indexer_candidates
from .fp8_indexer import fp8_indexer_candidates
from .topk_candidates import topk_candidates


def fp8_indexer_topk(
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk: int,
    *,
    out_offsets: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Top-``topk`` keys of every row for the fp8 indexer (operands as in
    :func:`fp8_indexer_candidates`): ``out[r]`` holds ``j - ks[r] + out_offsets[r]`` for the
    ``topk`` highest-scoring keys ``j`` in ``[ks[r], ke[r])``, unordered, ``-1`` past
    ``min(topk, ke[r] - ks[r])``. ``out`` is int32 ``[rows, topk]``, ``topk`` in {512, 1024, 2048}.
    """
    if out is None:
        out = torch.empty(
            (q_fp8.shape[0], topk), dtype=torch.int32, device=q_fp8.device
        )
    if q_fp8.shape[0] == 0:  # e.g. a dummy run: nothing to launch
        return out
    cands = fp8_indexer_candidates(q_fp8, weights, kv_fp8, kv_scale, ks, ke, topk)
    topk_candidates(
        cands.scores,
        cands.indices,
        cands.counts,
        cands.thresholds,
        out,
        out_offsets=out_offsets,
    )
    return out


def fp4_indexer_topk(
    q_fp4: torch.Tensor,
    q_sf: torch.Tensor,
    weights: torch.Tensor,
    kv_fp4: torch.Tensor,
    kv_sf: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    topk: int,
    *,
    out_offsets: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Top-``topk`` keys of every row for the fp4 indexer (operands as in
    :func:`fp4_indexer_candidates`), over its bf16 scores: ``out[r]`` holds
    ``j - ks[r] + out_offsets[r]`` for the ``topk`` highest-scoring keys ``j`` in ``[ks[r], ke[r])``,
    unordered, ``-1`` past ``min(topk, ke[r] - ks[r])``. ``out`` is int32 ``[rows, topk]``, ``topk``
    in {512, 1024, 2048}.
    """
    if out is None:
        out = torch.empty((q_sf.shape[0], topk), dtype=torch.int32, device=q_fp4.device)
    if q_sf.shape[0] == 0:  # e.g. a dummy run: nothing to launch
        return out
    cands = fp4_indexer_candidates(q_fp4, q_sf, weights, kv_fp4, kv_sf, ks, ke, topk)
    topk_candidates(
        cands.scores,
        cands.indices,
        cands.counts,
        cands.thresholds,
        out,
        out_offsets=out_offsets,
    )
    return out
