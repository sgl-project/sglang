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
from .fp8_indexer import FP8_INDEXER_NUM_HEADS, fp8_indexer_candidates
from .topk_candidates import (
    SUPPORTED_TOPK,
    is_topk_candidates_supported,
    topk_candidates,
)


def is_fp8_indexer_topk_supported(
    num_heads: int, topk: int, num_forced_tokens: int = 0
) -> bool:
    """Whether :func:`fp8_indexer_topk` takes ``num_heads`` index heads (up to 64; fewer than a kernel width
    are zero-padded) with ``topk`` and ``num_forced_tokens`` forced keys per row."""
    return (
        0 < num_heads <= max(FP8_INDEXER_NUM_HEADS)
        and topk in SUPPORTED_TOPK
        and (
            num_forced_tokens == 0
            or is_topk_candidates_supported(topk - num_forced_tokens, torch.float32)
        )
    )


def _pad_heads(
    q_fp8: torch.Tensor, weights: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # Zero heads add nothing to sum_h w_h relu(q_h . k): the scores are those of the real heads.
    heads = q_fp8.shape[1]
    width = min(h for h in FP8_INDEXER_NUM_HEADS if h >= heads)
    if width == heads:
        return q_fp8, weights
    pad = width - heads
    return (
        torch.nn.functional.pad(q_fp8.view(torch.uint8), (0, 0, 0, pad)).view(
            q_fp8.dtype
        ),
        torch.nn.functional.pad(weights, (0, pad)),
    )


def _forced_columns(
    *,
    lengths: torch.Tensor,
    offsets: torch.Tensor,
    num_init_tokens: int,
    num_forced_tokens: int,
) -> torch.Tensor:
    """[rows, num_forced_tokens] int32: keys [0, init) and [len - local, len) of rows with at least
    num_forced_tokens keys; a shorter row lists its keys [0, len) and pads with -1. Plus ``offsets``."""
    cols = torch.arange(num_forced_tokens, dtype=torch.int32, device=lengths.device)
    lens = lengths[:, None]
    short = lens < num_forced_tokens
    key = torch.where(
        short | (cols < num_init_tokens), cols, lens - num_forced_tokens + cols
    )
    return torch.where(short & (cols >= lens), -1, key + offsets[:, None])


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
    num_init_tokens: int = 0,
    num_local_tokens: int = 0,
    num_ctas: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Top-``topk`` keys of every row for the fp8 indexer (operands as in
    :func:`fp8_indexer_candidates`): ``out[r]`` holds ``j - ks[r] + out_offsets[r]`` for the
    ``topk`` highest-scoring keys ``j`` in ``[ks[r], ke[r])``, unordered, ``-1`` past
    ``min(topk, ke[r] - ks[r])``. ``out`` is int32 ``[rows, topk]``, ``topk`` in {512, 1024, 2048}.
    With forced tokens (``Indexer._mask_init_and_local_tokens``) a row of more than ``topk`` keys always keeps
    its first ``num_init_tokens`` and last ``num_local_tokens`` keys (the first columns) and the top
    ``topk - init - local`` of the keys between; a shorter row keeps every key. Up to 64 heads; fewer than 32
    (or than 64) are zero-padded. See :func:`is_fp8_indexer_topk_supported`.
    ``num_ctas`` caps the persistent indexer grid (0: one CTA per SM). Callers keep ``kv_fp8.shape[0]`` at most
    ``FP8_INDEXER_MAX_SPAN``: rows of a block spanning more keys may get no sample and a clipped slab.
    """
    if out is None:
        out = torch.empty(
            (q_fp8.shape[0], topk), dtype=torch.int32, device=q_fp8.device
        )
    if q_fp8.shape[0] == 0:  # e.g. a dummy run: nothing to launch
        return out
    q_fp8, weights = _pad_heads(q_fp8, weights)
    num_forced = num_init_tokens + num_local_tokens
    operands = dict(q_fp8=q_fp8, weights=weights, kv_fp8=kv_fp8, kv_scale=kv_scale)
    if num_forced == 0:
        _fp8_topk_of_ranges(
            **operands,
            ks=ks,
            ke=ke,
            out=out,
            out_offsets=out_offsets,
            num_ctas=num_ctas,
        )
        return out
    # Score only the keys between the forced ones: their top-(k - forced) lands after the forced columns
    # (empty ranges, i.e. rows of at most `num_forced` keys, emit -1 there).
    if out_offsets is None:
        out_offsets = torch.zeros_like(ks)
    inner_ks = ks + num_init_tokens
    _fp8_topk_of_ranges(
        **operands,
        ks=inner_ks,
        ke=torch.maximum(ke - num_local_tokens, inner_ks),
        out=out[:, num_forced:],
        out_offsets=out_offsets + num_init_tokens,
        num_ctas=num_ctas,
    )
    out[:, :num_forced] = _forced_columns(
        lengths=ke - ks,
        offsets=out_offsets,
        num_init_tokens=num_init_tokens,
        num_forced_tokens=num_forced,
    )
    return out


def _fp8_topk_of_ranges(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_fp8: torch.Tensor,
    kv_scale: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    out: torch.Tensor,
    out_offsets: Optional[torch.Tensor],
    num_ctas: int,
) -> None:
    cands = fp8_indexer_candidates(
        q_fp8, weights, kv_fp8, kv_scale, ks, ke, out.shape[1], num_ctas=num_ctas
    )
    topk_candidates(
        cands.scores,
        cands.indices,
        cands.counts,
        cands.thresholds,
        out,
        out_offsets=out_offsets,
    )


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
    in {512, 1024, 2048}. Callers keep ``kv_fp4.shape[0]`` at most ``FP4_INDEXER_MAX_SPAN[heads]``, as for
    :func:`fp8_indexer_topk`.
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
