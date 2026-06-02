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


def build_lifted_compact_kv(
    kv_store: torch.Tensor,
    physical_slots: torch.Tensor,
    *,
    store_is_fp8: bool,
):
    """Materialize the compact KV buffer + compact indices for the lifted-budget
    decode kernel call.

    Given the per-request selected physical KV slots (``[bs, width]``, selector
    order, ``-1`` pads) and the paged KV store, this derives ``valid_lengths``
    from the pad sentinel, runs :func:`build_compact_decode_index`, and produces
    the dense compact KV buffer that ``flash_mla_sparse_fwd`` attends:

    * ``store_is_fp8``: the fp8 op-point. The store is the quantized
      ``[*, 656]`` layout, so the selected rows are dequantized with
      ``dequantize_k_cache_paged`` (which gathers by ``page_table_1_flattened``
      and returns ``[total_valid, 1, 576]`` bf16).
    * otherwise: a bf16 store (``[*, d]``); the selected rows are gathered
      directly into ``[total_valid, 1, d]`` (no dequant needed).

    Returns ``(compact_kv, compact_indices, valid_counts)`` where ``compact_kv``
    is ``[total_valid, 1, d]`` and ``compact_indices`` is ``[bs, width]`` int32
    (request-local ordinals, ``-1`` pads) ready for ``flash_mla_sparse_fwd``
    (after an ``unsqueeze(1)``). ``compact_kv`` has 0 rows iff no request selected
    any valid slot (a degenerate batch the caller must avoid at decode).
    """
    valid_lengths = (physical_slots >= 0).sum(dim=1)
    remap = build_compact_decode_index(physical_slots, valid_lengths)
    kv_flat = kv_store.reshape(-1, kv_store.shape[-1])
    ptf = remap.page_table_1_flattened
    if store_is_fp8:
        from sglang.srt.layers.attention.dsa.dequant_k_cache import (
            dequantize_k_cache_paged,
        )

        compact_kv = dequantize_k_cache_paged(kv_flat, ptf)
    else:
        compact_kv = kv_flat.index_select(0, ptf.to(torch.int64)).unsqueeze(1)
    return compact_kv, remap.compact_indices, remap.valid_counts


def build_lifted_compact_index_fixed(
    physical_slots: torch.Tensor,
    valid_lengths: torch.Tensor,
    *,
    out_page_table: torch.Tensor,
    out_compact_indices: torch.Tensor,
    out_valid_counts=None,
    pad_value: int = -1,
    safe_slot: int = 0,
) -> None:
    """Fixed-shape, graph-safe variant of :func:`build_compact_decode_index`.

    Unlike the eager builder (whose ``page_table_1_flattened`` has a dynamic
    ``total_valid`` length — uncapturable), this version keeps a **fixed**
    ``[bs, width]`` layout: EVERY lane gets a compact row at ordinal
    ``b*width + lane``, so the compact buffer is always ``[bs*width, 1, 576]``.
    Invalid / within-row-duplicate lanes are handled by writing a **safe
    in-bounds physical slot** into the dequant input (``out_page_table``, never
    ``-1`` — it is loaded) and ``-1`` into ``out_compact_indices`` (so
    ``flash_mla_sparse_fwd`` masks them). A request therefore attends exactly the
    same set of valid (post-dedup) slots as the eager builder — same attention
    result — but in a capturable fixed shape.

    Fully tensorized (no ``.item()``, no dynamic boolean-mask shapes); writes into
    the caller-owned scratch via ``copy_``. Intermediate tensors allocate from the
    CUDA-graph pool during capture and replay alloc-free (fixed shapes).

    Args:
        physical_slots: ``int [bs, width]`` selected physical KV slots, ``-1`` pads.
        valid_lengths: ``int [bs]`` per-request valid leading-lane count.
        out_page_table: ``int32 [bs*width]`` scratch (written): per-row physical slot.
        out_compact_indices: ``int32 [bs, width]`` scratch (written): request-local
            ordinal ``b*width+lane`` for valid lanes, ``-1`` for masked lanes.
        out_valid_counts: optional ``int32 [bs]`` scratch (written): post-dedup count.
        safe_slot: an in-bounds physical slot for masked lanes (default ``0``).
    """
    bs, width = physical_slots.shape
    device = physical_slots.device
    phys = physical_slots.to(torch.int64)
    vlen = valid_lengths.to(torch.int64).reshape(-1)
    lane = torch.arange(width, device=device).unsqueeze(0).expand(bs, width)
    valid = (lane < vlen.unsqueeze(1)) & (phys != pad_value)
    dup = _within_row_duplicate_mask(phys, valid)
    final_valid = valid & (~dup)

    base = (torch.arange(bs, device=device, dtype=torch.int64) * width).unsqueeze(1)
    compact = torch.where(
        final_valid, base + lane, torch.full_like(phys, pad_value)
    )
    out_compact_indices.copy_(compact)

    ptf = torch.where(final_valid, phys, torch.full_like(phys, safe_slot))
    out_page_table.copy_(ptf.reshape(-1))

    if out_valid_counts is not None:
        out_valid_counts.copy_(final_valid.sum(dim=1))


def build_lifted_compact_kv_fixed(
    kv_store: torch.Tensor,
    physical_slots: torch.Tensor,
    valid_lengths: torch.Tensor,
    *,
    out_page_table: torch.Tensor,
    out_compact_indices: torch.Tensor,
    out_compact_kv: torch.Tensor,
    store_is_fp8: bool,
    out_valid_counts=None,
) -> None:
    """Graph-safe, alloc-free materialization of the lifted compact KV + indices
    into caller-owned scratch (the fixed-shape counterpart of
    :func:`build_lifted_compact_kv`). Runs the fixed-shape index builder, then
    dequantizes (fp8) / gathers (bf16) the selected rows into
    ``out_compact_kv [bs*width, 1, dim]`` via the alloc-free `out=` path."""
    build_lifted_compact_index_fixed(
        physical_slots,
        valid_lengths,
        out_page_table=out_page_table,
        out_compact_indices=out_compact_indices,
        out_valid_counts=out_valid_counts,
    )
    kv_flat = kv_store.reshape(-1, kv_store.shape[-1])
    if store_is_fp8:
        from sglang.srt.layers.attention.dsa.dequant_k_cache import (
            dequantize_k_cache_paged_out,
        )

        dequantize_k_cache_paged_out(kv_flat, out_page_table, out_compact_kv)
    else:
        torch.index_select(
            kv_flat, 0, out_page_table.to(torch.int64), out=out_compact_kv[:, 0, :]
        )
