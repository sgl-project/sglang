"""Dense causal prefill for QSA batches whose block selection is exhaustive.

The indexer scores ``compress_ratio``-token blocks and keeps ``block_topk`` of
them per query row, and attention then runs over the selected blocks plus the
row's causal remainder. A query at logical position ``p`` has
``(p + 1) // compress_ratio`` complete blocks behind it, so once a request is
no longer than ``block_topk * compress_ratio`` tokens every candidate block
fits the budget and ``qsa_fast_topk`` keeps all of them: it returns
``min(candidates, topk)`` entries. The attended key set is then every complete
block plus the ``0 .. compress_ratio - 1`` token remainder, which is exactly
the full causal prefix, so the sparse result equals the dense causal result and
only the summation order differs.

Dense causal attention is cheaper on that input: it drops the indexer-selected
gather, drops the sparse kernel's fixed ``budget``-wide inner loop, and lets
the causal mask skip the upper triangle the sparse kernel still walks once a
row is fully selected.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from sglang.srt.layers.attention.qsa.config import QSAProfile


def exhaustive_prefill_budget(profile: Optional[QSAProfile]) -> int:
    """Longest request whose candidate blocks always fit the indexer budget.

    Zero disables the dense path.
    """

    if profile is None:
        return 0
    return profile.block_topk * profile.compress_ratio


def is_prefill_selection_exhaustive(
    *,
    sequence_lens: Sequence[int],
    extend_lens: Sequence[int],
    query_rows: int,
    budget: int,
) -> bool:
    """Whether every row's selection provably covers its full causal prefix.

    Decided from host metadata alone, so the hot path adds no device read. The
    whole batch is rejected unless every request qualifies: a half-dense batch
    would need per-request kernels and is not worth the dispatch.
    """

    if budget <= 0 or query_rows <= 0:
        return False
    if len(sequence_lens) != len(extend_lens):
        return False
    if sum(extend_lens) != query_rows:
        # DP padding or a speculative layout; the sparse path owns those.
        return False
    return all(
        0 <= extend <= total <= budget
        for total, extend in zip(sequence_lens, extend_lens)
    )


def dense_causal_prefill_reference(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    extend_lens: Sequence[int],
    sequence_lens: Sequence[int],
    scale: float,
) -> torch.Tensor:
    """Device-agnostic varlen dense causal GQA over packed full-sequence K/V.

    ``q`` holds the chunk's rows, ``k``/``v`` hold every request's whole
    sequence packed back to back, and each request's queries are aligned to the
    end of its keys, matching ``flash_attn_varlen_func(causal=True)``.
    """

    query_heads, kv_heads = q.shape[1], k.shape[1]
    repeats = query_heads // kv_heads
    outputs = []
    query_start = 0
    key_start = 0
    for extend, total in zip(extend_lens, sequence_lens):
        queries = q[query_start : query_start + extend]
        keys = k[key_start : key_start + total]
        values = v[key_start : key_start + total]
        query_start += extend
        key_start += total
        if extend == 0:
            continue
        if repeats > 1:
            keys = keys.repeat_interleave(repeats, dim=1)
            values = values.repeat_interleave(repeats, dim=1)
        scores = torch.einsum("qhd,khd->hqk", queries.float(), keys.float()) * scale
        visible = torch.arange(total, device=q.device).unsqueeze(0) <= (
            torch.arange(extend, device=q.device).unsqueeze(1) + (total - extend)
        )
        scores = scores.masked_fill(~visible.unsqueeze(0), float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
        outputs.append(
            torch.einsum("hqk,khd->qhd", probabilities, values.float()).to(q.dtype)
        )
    if not outputs:
        return q.new_zeros((0, query_heads, q.shape[2]))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)


def gather_packed_sequence_kv(
    *,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    token_slot_table: torch.Tensor,
    sequence_lens: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack each request's whole sequence back to back from the KV cache.

    Device-agnostic counterpart of ``pack_qsa_prefill_kv``, which owns the CUDA
    path; this one serves the reference backend.
    """

    slots = torch.cat(
        [token_slot_table[row, :length] for row, length in enumerate(sequence_lens)]
    ).long()
    return key_cache.index_select(0, slots), value_cache.index_select(0, slots)


def total_visible_tokens(
    *, sequence_lens: Sequence[int], extend_lens: Sequence[int]
) -> int:
    """Causal prefix sizes summed over the batch's query rows.

    A row's selected entries are distinct and causal, so a row contributes at
    most its own prefix size and the batch total is reachable only when every
    row is covered exactly. That makes the total a sound exhaustiveness check
    costing one device reduction and no host synchronization, unlike
    ``assert_prefill_selection_exhaustive``, which names the offending row but
    has to read the comparison back.
    """

    return sum(
        extend * (total - extend) + extend * (extend + 1) // 2
        for total, extend in zip(sequence_lens, extend_lens)
    )


def visible_prefix_counts(
    *,
    sequence_lens: Sequence[int],
    extend_lens: Sequence[int],
    device: torch.device,
) -> torch.Tensor:
    """Per-row causal prefix size. Builds on the host; debug use only."""

    counts = [
        total - extend + offset + 1
        for total, extend in zip(sequence_lens, extend_lens)
        for offset in range(extend)
    ]
    return torch.tensor(counts, dtype=torch.int32, device=device)


def assert_prefill_selection_exhaustive(
    *, token_indices: torch.Tensor, visible_counts: torch.Tensor
) -> None:
    """Raise unless every row selected exactly its causal prefix.

    A row's entries are distinct and causal: complete blocks do not overlap,
    and the remainder only covers the incomplete block, which is never a
    candidate. Matching the prefix size therefore proves the sets are equal.
    Synchronizes with the device, so this is a debug check, not a hot path one.
    """

    selected = (token_indices >= 0).sum(dim=1, dtype=torch.int32)
    expected = visible_counts.to(device=selected.device, dtype=selected.dtype)
    if selected.shape != expected.shape:
        raise RuntimeError(
            "QSA dense prefill assertion got "
            f"{tuple(selected.shape)} rows, expected {tuple(expected.shape)}"
        )
    mismatches = (selected != expected).nonzero()
    if mismatches.numel():
        row = int(mismatches[0])
        raise RuntimeError(
            "QSA selection is not exhaustive: row "
            f"{row} selected {int(selected[row])} of {int(expected[row])} "
            "causal tokens; the dense prefill guard let through a batch the "
            "indexer could not cover"
        )


__all__ = [
    "assert_prefill_selection_exhaustive",
    "dense_causal_prefill_reference",
    "exhaustive_prefill_budget",
    "gather_packed_sequence_kv",
    "is_prefill_selection_exhaustive",
    "total_visible_tokens",
    "visible_prefix_counts",
]
