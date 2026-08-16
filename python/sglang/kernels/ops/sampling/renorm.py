"""Portable top-k / top-p probability renormalization.

Matches the threshold semantics of the FlashInfer-derived AOT kernels: locate the
pivot entry that the requested budget reaches, keep every entry ``>= pivot`` (so
ties at the pivot are all retained), and renormalize. A rank-based cutoff would
instead keep exactly k entries and break ties by sort order, which diverges from
CUDA whenever probabilities tie at the boundary.

The pivot is found with ``torch.topk`` rather than a full ``torch.sort``. Both are
exact -- ``topk(..., sorted=True)`` is the true descending prefix -- but selection
is O(V) where sorting a 100K+ vocabulary is O(V log V) and moves int64 indices.

Pivot selection is shared with the Triton path in :mod:`.renorm_triton`; only the
apply-and-renormalize step differs.
"""

from typing import Union

import torch

# Nucleus size beyond which top-p falls back to a full sort. Real decode
# distributions need a handful of entries; flat ones (high temperature, early
# generation) can need far more, so the fallback must stay correct, not fast.
#
# Sized against the selection cost rather than the expected nucleus: on a 151936
# vocabulary topk is flat from 1024 to 4096 (1.84ms to 1.92ms over 1536 rows) and
# only starts paying beyond that (2.78ms at 8192). Widening the prefix to the end
# of that plateau is close to free and keeps rows off the 13.4ms sort.
_TOP_P_PREFIX = 4096


def per_row_threshold(
    value: Union[torch.Tensor, int, float],
    *,
    probs: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Broadcast a scalar or per-row threshold to one value per row."""
    if isinstance(value, torch.Tensor):
        value = value.to(device=probs.device, dtype=dtype).reshape(-1)
        assert value.numel() in (1, probs.shape[0])
        if value.numel() == 1:
            value = value.expand(probs.shape[0])
        return value
    return torch.full((probs.shape[0],), value, dtype=dtype, device=probs.device)


def top_k_pivots(probs: torch.Tensor, top_ks: torch.Tensor) -> torch.Tensor:
    """Value of the k-th largest entry in each row."""
    k_max = int(top_ks.max().item())
    values, _ = torch.topk(probs, k_max, dim=-1, sorted=True)
    return values.gather(1, (top_ks - 1).unsqueeze(1)).squeeze(1)


def top_p_pivots(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """Pivot of the nucleus: the least likely entry that is still kept.

    Selected by discarded mass rather than by ``cumsum >= top_p``, because a row of
    float32 probabilities sums to slightly under one. Testing the kept prefix against
    the row's own total keeps ``top_p=1`` a no-op instead of truncating the tail on
    a peaked row, where the leading terms round up to one on their own.
    """
    vocab_size = probs.shape[1]
    budget = probs.sum(dim=-1) - (1.0 - top_ps)
    prefix = min(_TOP_P_PREFIX, vocab_size)

    values, _ = torch.topk(probs, prefix, dim=-1, sorted=True)
    within = (values.cumsum(dim=-1) - values) <= budget.unsqueeze(1)
    position = (within.sum(dim=-1) - 1).clamp(min=0)
    pivots = values.gather(1, position.unsqueeze(1)).squeeze(1)

    overflow = within[:, -1]
    if prefix < vocab_size and bool(overflow.any()):
        rows = overflow.nonzero(as_tuple=True)[0]
        pivots[rows] = _top_p_pivots_sorted(probs[rows], top_ps[rows])
    return pivots


def _top_p_pivots_sorted(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """Same pivot, resolved by a full ascending sort.

    Accumulating the discarded tail from the smallest entry upwards is what the
    in-tree Triton implementation does, and on a flat row over a 100K vocabulary the
    summation order is worth several entries at the boundary.
    """
    ascending, _ = torch.sort(probs, dim=-1)
    cutoff = torch.searchsorted(
        ascending.cumsum(dim=-1).contiguous(),
        (1.0 - top_ps).unsqueeze(1).contiguous(),
        right=False,
    ).squeeze(1)
    cutoff = cutoff.clamp(max=probs.shape[1] - 1)
    return ascending.gather(1, cutoff.unsqueeze(1)).squeeze(1)


def _apply_pivot(probs: torch.Tensor, pivots: torch.Tensor) -> torch.Tensor:
    kept = torch.where(probs >= pivots.unsqueeze(1), probs, torch.zeros_like(probs))
    normalizer = kept.sum(dim=-1, keepdim=True)
    return torch.where(normalizer > 0, kept / normalizer, torch.zeros_like(kept))


def top_k_renorm_probs_torch(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
) -> torch.Tensor:
    """Keep every entry at least as likely as the k-th largest, then renormalize."""
    assert probs.ndim == 2
    probs = probs.float()
    batch_size, vocab_size = probs.shape
    if batch_size == 0:
        return probs.clone()
    assert vocab_size > 0

    top_ks = per_row_threshold(top_k, probs=probs, dtype=torch.int64).clamp(
        1, vocab_size
    )
    return _apply_pivot(probs, top_k_pivots(probs, top_ks))


def top_p_renorm_probs_torch(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
) -> torch.Tensor:
    """Keep the nucleus -- every entry at least as likely as its pivot -- and
    renormalize."""
    assert probs.ndim == 2
    probs = probs.float()
    batch_size, vocab_size = probs.shape
    if batch_size == 0:
        return probs.clone()
    assert vocab_size > 0

    top_ps = per_row_threshold(top_p, probs=probs, dtype=torch.float32).clamp(0.0, 1.0)
    return _apply_pivot(probs, top_p_pivots(probs, top_ps))
