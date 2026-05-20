"""Double Sparsity page-table adapter.

Translates the DS selector's page-level output ``(selected_indices,
valid_lengths)`` into the token-level ``topk_indices`` shape that the
existing NSA pipeline consumes via :func:`transform_index_page_table_decode`.

The DS selector returns logical page IDs sorted ascending with ``-1``
padding. NSA's downstream transform expects token positions: this adapter
converts each logical page ID ``p`` to the first token position in that
page, ``p * page_size``. The transform then maps each token position to
its physical page ID via ``req_to_token``, producing the physical page
table that ``_forward_flashmla_kv`` consumes.

Strategy 2 from the refined plan ("expand-then-transform"): the adapter
emits ``topk_indices`` of the same shape NSA's indexer emits, so the
downstream consumer in ``forward_absorb_*`` and ``nsa_backend`` sees one
unified tensor shape from both selection paths. There is no typed-union
return, no isinstance dispatch, no bypass of the NSA transform pipeline.

Contract assertions raise typed exceptions for each named failure mode.
The validation in this module is host-side and triggers ``.item()`` sync,
so it is intended for use OFF the CUDA-graph captured path. The deferred
CUDA-graph allocation safety hardening (AC-8) will move shape/device
checks to ``init_forward_metadata`` and value-domain checks into a Triton
device assertion.
"""

from __future__ import annotations

from typing import Optional

import torch


class DSAdapterError(ValueError):
    """Base class for DS page-table adapter contract violations."""


class DSAdapterPageOutOfRange(DSAdapterError):
    """A selected logical page ID is negative or exceeds the sequence's logical page count."""


class DSAdapterValidLengthOverflow(DSAdapterError):
    """A row's ``valid_lengths`` value is negative or exceeds ``max_top_k``."""


class DSAdapterNonAscending(DSAdapterError):
    """A row's first ``valid_lengths`` entries are not strictly ascending."""


class DSAdapterPaddingViolation(DSAdapterError):
    """A row's padding slot (position past ``valid_lengths``) is not ``-1``."""


class DSAdapterDtypeMismatch(DSAdapterError):
    """Tensor dtype or rank does not match the adapter contract."""


class DSAdapterDeviceMismatch(DSAdapterError):
    """Inputs live on different devices."""


class DSAdapterBatchMismatch(DSAdapterError):
    """Batch sizes of ``selected_indices`` and ``valid_lengths`` disagree."""


def expand_ds_selection_to_topk_indices(
    selected_indices: torch.Tensor,
    valid_lengths: torch.Tensor,
    page_size: int,
    max_logical_pages: Optional[int] = None,
) -> torch.Tensor:
    """Translate DS selector page-level output to NSA-compatible token-level ``topk_indices``.

    The DS selector returns:
        ``selected_indices``: ``int32 [bs, max_top_k]`` logical page IDs,
            sorted strictly ascending in the first ``valid_lengths[b]``
            positions of each row, with ``-1`` padding past that point.
        ``valid_lengths``: ``int32 [bs]`` per-row unpadded length.

    NSA's ``transform_index_page_table_decode`` pipeline consumes:
        ``topk_indices``: ``int32 [bs, max_top_k]`` token positions in
            the sequence, with ``-1`` padding.

    For each in-range logical page ``p`` this emits ``p * page_size`` —
    the first token position in page ``p``. When that token position is
    looked up in ``req_to_token`` (inside the existing transform), it
    yields the physical page ID for that logical page. ``-1`` padding is
    preserved through the transform.

    Args:
        selected_indices: ``int32 [bs, max_top_k]`` from the DS selector.
        valid_lengths: ``int32 [bs]`` from the DS selector.
        page_size: positive integer page size (typically 64 on V3.2).
        max_logical_pages: optional upper bound. When provided, any
            in-range entry whose value is ``>= max_logical_pages`` raises
            :class:`DSAdapterPageOutOfRange`. If ``None``, only the
            lower bound (non-negative) is enforced.

    Returns:
        ``topk_indices`` of shape ``[bs, max_top_k]``, dtype ``int32``,
        on the same device as ``selected_indices``.

    Raises:
        DSAdapterDtypeMismatch: dtype or rank of an input is wrong.
        DSAdapterDeviceMismatch: inputs live on different devices.
        DSAdapterBatchMismatch: batch dimensions disagree.
        DSAdapterValidLengthOverflow: any ``valid_lengths`` entry is
            negative or ``> max_top_k``.
        DSAdapterNonAscending: a row's valid prefix is not strictly ascending.
        DSAdapterPaddingViolation: a row's padding past ``valid_lengths``
            contains a non-``-1`` entry.
        DSAdapterPageOutOfRange: an in-range entry is negative or
            ``>= max_logical_pages`` (when provided).
    """

    if selected_indices.dtype != torch.int32:
        raise DSAdapterDtypeMismatch(
            f"selected_indices must be int32, got {selected_indices.dtype}"
        )
    if valid_lengths.dtype != torch.int32:
        raise DSAdapterDtypeMismatch(
            f"valid_lengths must be int32, got {valid_lengths.dtype}"
        )
    if selected_indices.dim() != 2:
        raise DSAdapterDtypeMismatch(
            "selected_indices must be 2D [bs, max_top_k], got shape "
            f"{tuple(selected_indices.shape)}"
        )
    if valid_lengths.dim() != 1:
        raise DSAdapterDtypeMismatch(
            "valid_lengths must be 1D [bs], got shape "
            f"{tuple(valid_lengths.shape)}"
        )
    if selected_indices.device != valid_lengths.device:
        raise DSAdapterDeviceMismatch(
            f"selected_indices.device ({selected_indices.device}) != "
            f"valid_lengths.device ({valid_lengths.device})"
        )

    bs, max_top_k = selected_indices.shape
    if valid_lengths.shape[0] != bs:
        raise DSAdapterBatchMismatch(
            f"valid_lengths batch {valid_lengths.shape[0]} != "
            f"selected_indices batch {bs}"
        )

    if not isinstance(page_size, int) or page_size <= 0:
        raise ValueError(f"page_size must be a positive int, got {page_size!r}")

    if bs > 0:
        max_valid = int(valid_lengths.max().item())
        min_valid = int(valid_lengths.min().item())
        if min_valid < 0 or max_valid > max_top_k:
            raise DSAdapterValidLengthOverflow(
                f"valid_lengths range [{min_valid}, {max_valid}] outside "
                f"[0, max_top_k={max_top_k}]"
            )

    # Host-side per-row validation. This sync-trips so the path is NOT
    # CUDA-graph captured (AC-8 will move these checks to pre-capture
    # init_forward_metadata or to Triton device assertions in a follow-up).
    sel_cpu = selected_indices.detach().to("cpu")
    vl_cpu = valid_lengths.detach().to("cpu")

    for b in range(bs):
        vl = int(vl_cpu[b].item())
        row = sel_cpu[b]

        if vl < max_top_k:
            padding = row[vl:]
            if bool((padding != -1).any().item()):
                raise DSAdapterPaddingViolation(
                    f"row {b}: padding past valid_lengths[{b}]={vl} contains "
                    f"non-(-1) entries"
                )

        if vl == 0:
            continue

        valid = row[:vl]
        v_min = int(valid.min().item())
        if v_min < 0:
            raise DSAdapterPageOutOfRange(
                f"row {b}: in-range entry is negative (min={v_min})"
            )
        if max_logical_pages is not None:
            v_max = int(valid.max().item())
            if v_max >= max_logical_pages:
                raise DSAdapterPageOutOfRange(
                    f"row {b}: max in-range page ID {v_max} >= "
                    f"max_logical_pages {max_logical_pages}"
                )

        if vl >= 2:
            diffs = valid[1:] - valid[:-1]
            if bool((diffs <= 0).any().item()):
                raise DSAdapterNonAscending(
                    f"row {b}: valid prefix is not strictly ascending"
                )

    # Expansion is graph-safe (no host sync). The host-side validation
    # above is the part that prevents capture; once validation moves
    # pre-capture / device-side, this expansion can be captured as-is.
    minus_one = torch.full_like(selected_indices, -1)
    expanded = torch.where(
        selected_indices >= 0,
        selected_indices * int(page_size),
        minus_one,
    )
    return expanded.to(torch.int32)
