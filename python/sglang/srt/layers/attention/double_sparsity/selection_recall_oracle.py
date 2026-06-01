"""Selection-recall oracle — score-only diagnostic for the DS long-context recall gap.

This is a **debug instrument**, off the production hot path. Given the live
all-reduced token-score tensor — the tensor produced *after*
:func:`selection_kernel.all_reduce_token_scores` and consumed *before*
:func:`selection_kernel.select_topk_sequence_order` — and the harness-provided
needle token span, it answers: where does the needle rank among the DS
selection scores, and would a top-K of size ``K`` have selected every needle
token?

It computes scores **only** (no decode): ``recall@K`` here is a property of the
score ranking, NOT a decode result. For ``K`` greater than the kernel-locked
budget (``index_topk`` = 2048 on V3.2) this score-only measure is the only
meaningful one until an opt-in lifted-budget decode path exists — so callers
MUST treat ``K > index_topk`` curves as oracle diagnostics, never as a decode
result.

All ranking honors the exact deterministic tie-break the DS selector uses:
**(score DESCENDING, then logical position ASCENDING)** — the single ordering
shared by :func:`selection_kernel._topk_by_score_then_pos` and
:func:`selection_kernel.select_topk_sequence_order`. A token ``t`` outranks a
needle token ``p`` iff ``score[t] > score[p]`` OR
(``score[t] == score[p]`` AND ``t < p``). The rank of ``p`` is the number of
tokens that outrank it; ``p`` is in the top-K iff ``rank(p) < K``.

The multi-token needle pass/fail rule is **worst-rank / all-needle-tokens-in-top-K**:
the needle counts as recalled at ``K`` iff EVERY needle token has ``rank < K``
(``needle_worst_rank < K``). A per-best-token (``min`` rank) value is only a
summary, never the pass/fail criterion.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import torch

# Default K grid for the score-only recall@K curve (AC-1).
DEFAULT_RECALL_K_VALUES: tuple = (512, 1024, 2048, 4096, 8192)

# Mirror of selection_kernel.SELECTED_PAD_VALUE — the padding sentinel in the
# selector's ``selected_indices`` output. Kept as a local constant so this
# diagnostic module has no import-time dependency on the hot-path kernel.
_SELECTED_PAD_VALUE = -1


def _normalize_needle_positions(
    needle_positions: torch.Tensor,
    bs: int,
    max_tokens: int,
) -> torch.Tensor:
    """Coerce ``needle_positions`` to an int64 ``[bs, n_needle]`` tensor.

    Accepts a shared 1-D span ``[n_needle]`` (broadcast to every batch row) or a
    per-row 2-D span ``[bs, n_needle]``. Raises on an empty span (the oracle
    must never guess the needle) or out-of-range positions.
    """

    if not isinstance(needle_positions, torch.Tensor):
        needle_positions = torch.as_tensor(needle_positions, dtype=torch.int64)
    pos = needle_positions.to(torch.int64)
    if pos.dim() == 1:
        pos = pos.unsqueeze(0).expand(bs, -1)
    elif pos.dim() == 2:
        if pos.shape[0] != bs:
            raise ValueError(
                f"needle_positions batch dim {int(pos.shape[0])} does not match "
                f"token_scores batch dim {bs}."
            )
    else:
        raise ValueError(
            f"needle_positions must be 1-D [n_needle] or 2-D [bs, n_needle], got "
            f"shape {tuple(needle_positions.shape)}."
        )
    if pos.shape[-1] == 0:
        raise ValueError(
            "needle_positions is empty: the recall oracle requires a "
            "harness-provided needle span and must not guess the needle."
        )
    if int(pos.min()) < 0 or int(pos.max()) >= max_tokens:
        raise ValueError(
            f"needle_positions out of range: got [{int(pos.min())}, {int(pos.max())}] "
            f"for token_scores with max_tokens={max_tokens}."
        )
    return pos.contiguous()


def needle_ranks(
    token_scores: torch.Tensor,
    needle_positions: torch.Tensor,
) -> torch.Tensor:
    """Rank of each needle token under the DS selector's tie-break.

    ``token_scores``: ``[bs, max_tokens]`` fp32, in logical-position order, with
    invalid/unwritten tokens set to ``-inf`` (exactly as consumed by
    :func:`select_topk_sequence_order`).

    Returns ``ranks[bs, n_needle]`` int64 — the number of tokens that strictly
    outrank each needle token under (score DESC, position ASC). A needle token
    is in the top-K iff its rank is ``< K``.
    """

    if token_scores.dim() != 2:
        raise ValueError(
            f"token_scores must be 2-D [bs, max_tokens], got {tuple(token_scores.shape)}."
        )
    bs, max_tokens = token_scores.shape
    pos = _normalize_needle_positions(needle_positions, bs, max_tokens)
    scores = token_scores.to(torch.float32)

    # Score of each needle token: [bs, n_needle]
    needle_scores = torch.gather(scores, 1, pos)

    # For each (row, needle j), count tokens t with score[t] > score[p_j]
    # (strictly greater) plus tokens with equal score and a strictly lower
    # logical position. p_j itself is never counted (strict > excludes it; the
    # equal-score test requires position < p_j, which p_j fails). This is the
    # exact (score DESC, position ASC) order the selector uses.
    positions = torch.arange(max_tokens, device=scores.device, dtype=torch.int64)

    s_t = scores.unsqueeze(2)                  # [bs, T, 1]
    s_p = needle_scores.unsqueeze(1)           # [bs, 1, n]
    greater = s_t > s_p                        # [bs, T, n]
    equal = s_t == s_p                         # [bs, T, n]
    lower_pos = positions.view(1, -1, 1) < pos.unsqueeze(1)  # [bs, T, n]
    outranks = greater | (equal & lower_pos)   # [bs, T, n]
    ranks = outranks.sum(dim=1)                # [bs, n]
    return ranks.to(torch.int64)


def needle_worst_rank(
    token_scores: torch.Tensor,
    needle_positions: torch.Tensor,
) -> torch.Tensor:
    """Worst (largest) rank over the needle span — the multi-token pass/fail basis.

    Returns ``[bs]`` int64. ``needle_all_tokens_in_topK(K)`` is exactly
    ``needle_worst_rank < K``.
    """
    return needle_ranks(token_scores, needle_positions).amax(dim=1)


def needle_best_rank(
    token_scores: torch.Tensor,
    needle_positions: torch.Tensor,
) -> torch.Tensor:
    """Best (smallest) rank over the needle span — a SUMMARY only, never pass/fail.

    Returns ``[bs]`` int64.
    """
    return needle_ranks(token_scores, needle_positions).amin(dim=1)


def needle_all_tokens_in_topk(
    token_scores: torch.Tensor,
    needle_positions: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """``True`` per row iff ALL needle tokens land in the score-only top-``k``.

    Returns ``[bs]`` bool. This is the authoritative multi-token recall rule.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")
    return needle_worst_rank(token_scores, needle_positions) < k


def score_only_recall_at_k(
    token_scores: torch.Tensor,
    needle_positions: torch.Tensor,
    k_values: Iterable[int] = DEFAULT_RECALL_K_VALUES,
) -> Dict[int, torch.Tensor]:
    """Score-only recall@K curve, using the all-needle-tokens-in-top-K rule.

    Computes the worst rank once and thresholds it against each ``K`` (cheap and
    consistent across the whole curve). Returns ``{K: [bs] bool}``.

    NOTE: this is a SCORE ranking property, not a decode result. ``K`` larger
    than the locked decode budget (``index_topk``) is a legitimate oracle
    diagnostic; callers must not present such a ``K`` as a decode outcome.
    """
    worst = needle_worst_rank(token_scores, needle_positions)
    out: Dict[int, torch.Tensor] = {}
    for k in k_values:
        k_int = int(k)
        if k_int <= 0:
            raise ValueError(f"k_values must be positive, got {k}.")
        out[k_int] = worst < k_int
    return out


def selected_contains_needle(
    selected_indices: torch.Tensor,
    needle_positions: torch.Tensor,
    pad_value: int = _SELECTED_PAD_VALUE,
) -> torch.Tensor:
    """``True`` per row iff every needle token appears in ``selected_indices``.

    ``selected_indices``: ``[bs, max_top_k]`` int (the selector's output of
    ascending logical positions, ``pad_value``-padded). This is computed from
    the ACTUAL selected set, and — for ``K == index_topk`` — must equal
    ``needle_all_tokens_in_topk(..., index_topk)`` (the AC-1 invariant the unit
    test pins).

    Returns ``[bs]`` bool.
    """
    if selected_indices.dim() != 2:
        raise ValueError(
            f"selected_indices must be 2-D [bs, max_top_k], got {tuple(selected_indices.shape)}."
        )
    bs = selected_indices.shape[0]
    sel = selected_indices.to(torch.int64)
    # Per-row needle span (allow shared 1-D), validated only for batch shape.
    if not isinstance(needle_positions, torch.Tensor):
        needle_positions = torch.as_tensor(needle_positions, dtype=torch.int64)
    pos = needle_positions.to(torch.int64)
    if pos.dim() == 1:
        pos = pos.unsqueeze(0).expand(bs, -1)
    if pos.shape[-1] == 0:
        raise ValueError(
            "needle_positions is empty: selected_contains_needle requires a "
            "harness-provided needle span."
        )
    # [bs, n_needle, max_top_k] membership; ignore pad columns implicitly since
    # real needle positions are >= 0 and never equal the pad sentinel.
    present = (sel.unsqueeze(1) == pos.unsqueeze(2)) & (sel.unsqueeze(1) != pad_value)
    each_needle_present = present.any(dim=2)  # [bs, n_needle]
    return each_needle_present.all(dim=1)


def needle_positions_from_spans(
    spans: Iterable[Iterable[int]],
) -> List[torch.Tensor]:
    """Build per-row needle position tensors from harness-provided token spans.

    Convenience for the NIAH harness: ``spans`` is one inclusive/explicit list
    of needle logical token positions per request. Returns a list of int64
    tensors (callers pad/stack as needed for a batched oracle call).
    """
    out: List[torch.Tensor] = []
    for span in spans:
        t = torch.as_tensor(list(span), dtype=torch.int64)
        if t.numel() == 0:
            raise ValueError("each needle span must be non-empty (no guessing).")
        out.append(t)
    return out
