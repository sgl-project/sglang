"""Request-local compact-index remap for the opt-in lifted-budget decode path.

The default Double Sparsity decode runs through the ``flashmla_kv`` kernel, whose
``indices.shape[-1] == dsa_index_topk`` assert caps the per-request selection at
the model's ``index_topk`` (2048). The opt-in lifted-budget path widens that
selection (``lifted_budget_top_k`` > ``index_topk``) and instead routes decode
through ``flash_mla_sparse_fwd`` (no cap). That kernel does NOT index the full
fp8 KV cache directly; for the fp8 op-point the selected slots are first
dequantized into a **compact** bf16 buffer via ``dequantize_k_cache_paged``, and
the kernel then attends that compact buffer by **request-local ordinal** indices.

This module owns the pure-tensor remap between the two index domains. Given each
request's selected **physical** KV-cache slots (in the selector's deterministic
score-desc / position-asc order, padded to a fixed width) plus per-request valid
counts, :func:`build_compact_decode_index` produces:

* ``page_table_1_flattened`` — the concatenation, across the batch, of every
  request's VALID physical slots in selection order. This is the tensor
  ``dequantize_k_cache_paged`` blindly loads, so it must never contain a ``-1``
  pad. Its length is the number of compact-buffer rows.
* ``compact_indices`` — per request, the **request-local** ordinal positions in
  the compact buffer (``request_base + selected_rank``, NOT the physical slot
  value), padded with ``-1`` for the lanes past the valid count.
  ``flash_mla_sparse_fwd`` masks ``-1`` (and any index ``>= s_kv``) lanes.

Two correctness properties this remap guarantees:

* **Prefix sharing is safe.** The same physical slot appearing in two requests'
  selections is dequantized once per request (it appears once in each request's
  span of ``page_table_1_flattened``), and each request's ``compact_indices``
  point only into its OWN span. No cross-request aliasing.
* **No within-row duplicate reaches the kernel.** ``flash_mla_sparse_fwd`` would
  double-attend a duplicated valid index, so a physical slot that appears more
  than once in a single request's selection is collapsed to its FIRST (highest
  selection-rank) occurrence; later occurrences are dropped (and counted, so a
  caller/test can assert the normal case drops none).

The remap is framework-light (plain ``torch`` ops, no host syncs beyond an
optional total-count read) so it is exercised directly in CPU unit tests; the
decode wiring that feeds it lives in the attention backend.
"""

from __future__ import annotations

from typing import NamedTuple

import torch


class CompactDecodeIndex(NamedTuple):
    """Result of :func:`build_compact_decode_index`.

    ``page_table_1_flattened``: ``int32 [total_valid]`` — valid physical slots,
        batch-major then selection-rank order; no ``-1`` pads. Feeds
        ``dequantize_k_cache_paged``.
    ``compact_indices``: ``int32 [bs, width]`` — request-local compact-buffer
        ordinals for valid lanes, ``-1`` for pad lanes. Feeds
        ``flash_mla_sparse_fwd`` (after an ``unsqueeze(1)`` for the ``h_kv=1``
        axis).
    ``valid_counts``: ``int32 [bs]`` — per-request compact row count (post-dedup).
    ``total_valid``: ``int`` — number of compact-buffer rows
        (``== page_table_1_flattened.numel()``).
    ``dropped_duplicates``: ``int`` — count of valid lanes dropped as within-row
        duplicates (0 in the normal case where the selector emits distinct
        positions).
    """

    page_table_1_flattened: torch.Tensor
    compact_indices: torch.Tensor
    valid_counts: torch.Tensor
    total_valid: int
    dropped_duplicates: int


def _within_row_duplicate_mask(
    physical: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """``True`` where a valid lane duplicates the physical slot of an EARLIER
    (lower-lane = higher selection-rank) valid lane in the same row.

    Implemented in ``O(bs * width log width)`` via a stable sort, so the kept
    occurrence is always the lowest original lane (the highest selection rank):
    a stable argsort on the slot values preserves original-lane order among
    equal values, so the first element of each equal-value run is the
    earliest-ranked lane. Invalid lanes are replaced by a per-lane-unique
    negative sentinel so they never collide with a real slot or each other (and
    are therefore never marked as a duplicate). No tensor aliases its own argsort
    input/output (see the topk-aliasing lesson).
    """
    bs, width = physical.shape
    device = physical.device
    lane = torch.arange(width, device=device).unsqueeze(0).expand(bs, width)
    # Invalid lanes -> unique-per-lane negative sentinel (real slots are >= 0).
    keyed = torch.where(valid, physical, -(lane.to(physical.dtype) + 2))

    order = torch.argsort(keyed, dim=1, stable=True)  # value-asc, lane-asc on ties
    sorted_vals = torch.gather(keyed, 1, order)
    dup_sorted = torch.zeros_like(valid)
    dup_sorted[:, 1:] = sorted_vals[:, 1:] == sorted_vals[:, :-1]

    dup = torch.zeros_like(valid)
    dup.scatter_(1, order, dup_sorted)
    return dup & valid


def build_compact_decode_index(
    selected_physical: torch.Tensor,
    valid_lengths: torch.Tensor,
    *,
    pad_value: int = -1,
) -> CompactDecodeIndex:
    """Build the request-local compact decode index (see module docstring).

    Args:
        selected_physical: ``int [bs, width]`` — per-request selected physical
            KV-cache slots in the selector's deterministic order. A lane is a pad
            if it is at or beyond ``valid_lengths`` for its row, or holds
            ``pad_value``.
        valid_lengths: ``int [bs]`` — per-request count of valid leading lanes.
        pad_value: the sentinel used for pad lanes in/out (default ``-1``).

    Returns:
        :class:`CompactDecodeIndex`.
    """
    if selected_physical.dim() != 2:
        raise ValueError(
            f"selected_physical must be [bs, width], got shape "
            f"{tuple(selected_physical.shape)}."
        )
    bs, width = selected_physical.shape
    device = selected_physical.device
    physical = selected_physical.to(torch.int64)
    vlen = valid_lengths.to(torch.int64).reshape(-1)
    if vlen.shape[0] != bs:
        raise ValueError(
            f"valid_lengths must have bs={bs} entries, got {vlen.shape[0]}."
        )

    lane = torch.arange(width, device=device).unsqueeze(0).expand(bs, width)
    # A lane is valid only within the declared prefix AND not the pad sentinel.
    valid = (lane < vlen.unsqueeze(1)) & (physical != pad_value)

    dup = _within_row_duplicate_mask(physical, valid)
    final_valid = valid & (~dup)

    valid_counts = final_valid.sum(dim=1)  # [bs]
    request_base = torch.zeros(bs, dtype=torch.int64, device=device)
    if bs > 1:
        request_base[1:] = torch.cumsum(valid_counts, dim=0)[:-1]

    # Request-local ordinal = exclusive cumsum of valid lanes along the row.
    valid_i64 = final_valid.to(torch.int64)
    rank_in_request = torch.cumsum(valid_i64, dim=1) - valid_i64
    compact = request_base.unsqueeze(1) + rank_in_request
    compact = torch.where(
        final_valid, compact, torch.full_like(compact, pad_value)
    )

    # Valid physical slots in batch-major, selection-rank order. Boolean masking
    # a [bs, width] tensor yields row-major (request-major) order, which matches
    # the request_base layout above.
    page_table_1_flattened = physical[final_valid]

    total_valid = int(page_table_1_flattened.numel())
    dropped_duplicates = int((valid & dup).sum().item())

    return CompactDecodeIndex(
        page_table_1_flattened=page_table_1_flattened.to(torch.int32),
        compact_indices=compact.to(torch.int32),
        valid_counts=valid_counts.to(torch.int32),
        total_valid=total_valid,
        dropped_duplicates=dropped_duplicates,
    )
