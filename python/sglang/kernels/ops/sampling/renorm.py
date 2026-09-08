"""Pivot selection for top-p probability renormalization.

Top-p keeps the smallest set of entries whose mass reaches ``p``. Everything the
renormalization needs from that definition is the *pivot* -- the probability of the
smallest kept entry -- after which masking and rescaling are pure bandwidth work,
done in Triton (see :mod:`.renorm_triton`).

There are two ways to find the pivot, and they fail in opposite directions.

:func:`_top_p_pivots_sorted` sorts each row and walks its CDF. It is exact for any
input and never reads device memory from the host, but it pays a full ``O(V log V)``
sort over a 100K+ vocabulary, and that cost grows with the number of rows.

:func:`_top_p_pivots_prefix` searches a bounded top-k prefix instead. The nucleus of
a real decode distribution is a handful of entries, so the prefix almost always
contains the pivot; the rows where it does not are resolved by sorting only those
rows. Finding out whether any such row exists means reading one device tensor from
the host, which drains the queue -- a cost that is fixed rather than proportional to
the batch.

So the choice is a batch-proportional sort against a fixed stall, and it turns on the
row count alone, which the host already knows without synchronizing.
:func:`top_p_pivots` dispatches on it.
"""

from __future__ import annotations

import torch

from sglang.srt.environ import envs

# Nucleus size beyond which the prefix search cannot answer and has to fall back to a
# sort. Real decode distributions need a handful of entries; flat ones (high
# temperature, early generation) can need far more, so the fallback must stay
# correct, not fast.
_TOP_P_PREFIX = 4096


def _top_p_pivots_sorted(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """Exact pivot for every row, from a full ascending sort.

    Matches FlashInfer's threshold semantics: discard the smallest entries whose
    cumulative mass stays below ``1 - p`` and retain all ties at the pivot.
    """
    vocab_size = probs.shape[1]
    sorted_probs = torch.sort(probs, dim=-1).values
    cdf = torch.cumsum(sorted_probs, dim=-1)
    cutoff = torch.searchsorted(cdf, (1.0 - top_ps).unsqueeze(1), right=False).squeeze(
        1
    )
    cutoff.clamp_(max=vocab_size - 1)
    return sorted_probs.gather(1, cutoff.unsqueeze(1)).squeeze(1)


def _top_p_pivots_prefix(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """Pivot from a bounded top-k prefix, falling back to a sort for rows it misses.

    Walking the prefix downward from the largest entry is the mirror image of walking
    the CDF upward from the smallest: an entry is kept exactly while the mass above it
    still leaves at least ``1 - p`` behind. Budgeting against the row's own total
    rather than against ``1.0`` keeps ``top_p=1`` a no-op instead of truncating the
    tail of a peaked row, whose leading terms round up to one on their own.

    A row whose nucleus runs past the prefix has its pivot outside the prefix, and only
    a sort can find it. Those rows are flagged by the last prefix entry still being
    kept, and re-resolved exactly; the host read that detects them is what
    :func:`top_p_pivots` is weighing against a sort.
    """
    vocab_size = probs.shape[1]
    prefix = min(_TOP_P_PREFIX, vocab_size)

    budget = probs.sum(dim=-1) - (1.0 - top_ps)
    values = torch.topk(probs, prefix, dim=-1).values
    within = values.cumsum(dim=-1) <= budget.unsqueeze(1)
    position = within.sum(dim=-1).clamp(max=prefix - 1)
    pivots = values.gather(1, position.unsqueeze(1)).squeeze(1)

    overflow = within[:, -1]
    if prefix < vocab_size and bool(overflow.any()):
        rows = overflow.nonzero(as_tuple=True)[0]
        pivots[rows] = _top_p_pivots_sorted(probs[rows], top_ps[rows])
    return pivots


def top_p_pivots(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """Per-row top-p pivot for ``probs``, one value per row.

    Both paths implement the same threshold; they differ only in cost. The prefix
    path is worth its host synchronization once the sort it replaces is large enough
    to outweigh a fixed queue drain. The crossover was measured on MI355X with a
    ~151K vocabulary under speculative decoding, where the verify batch is
    ``requests x draft tokens``: below it the sort wins on both throughput and TPOT,
    above it the sort starts costing TPOT.

    Descending and ascending accumulation round differently, so on a row flat enough
    that thousands of entries sit within a few ULPs of each other the two paths can
    land on adjacent entries. Ascending is the better-conditioned order, which is why
    it stays the default for the batch sizes where it is affordable.
    """
    if probs.shape[0] >= envs.SGLANG_OPT_TOP_P_PREFIX_MIN_ROWS.get():
        return _top_p_pivots_prefix(probs, top_ps)
    return _top_p_pivots_sorted(probs, top_ps)


__all__ = ["top_p_pivots"]
