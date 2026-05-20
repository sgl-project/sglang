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

The adapter emits ``topk_indices`` of the same shape NSA's indexer
emits, so the downstream consumer in ``forward_absorb_*`` and
``nsa_backend`` sees one unified tensor shape from both selection
paths. There is no typed-union return, no isinstance dispatch, no
bypass of the NSA transform pipeline.

Contract assertions raise typed exceptions for each named failure mode.
This module exposes two paths:

* :func:`expand_ds_selection_to_topk_indices` — host-validated, performs
  cheap shape/dtype/device checks and a per-row value-domain audit
  (range, ascending, padding). Triggers a host sync via ``.item()`` and
  is therefore intended for use OFF the CUDA-graph captured path.

* :func:`expand_ds_selection_to_topk_indices_fast` — no host syncs, no
  allocations; writes into a caller-owned ``out`` buffer. Suitable for
  CUDA-graph captured execution; callers must arrange pre-capture
  shape/device checks and rely on Triton device assertions for any
  value-domain enforcement.
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
    req_to_token: Optional[torch.Tensor] = None,
    req_pool_indices: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    row_errors: Optional[dict] = None,
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
        max_logical_pages: optional global upper bound. When provided,
            any in-range entry whose value is ``>= max_logical_pages``
            raises :class:`DSAdapterPageOutOfRange`. Pass ``None`` to
            disable the global check and rely instead on per-row checks
            derived from ``seq_lens`` (see below).
        req_to_token: optional ``int32 [num_pools, max_seqlen_k]``. When
            provided alongside ``req_pool_indices``, its dtype/device are
            validated against ``selected_indices``; the second dim is
            used to derive each row's logical page limit when
            ``seq_lens`` is not given.
        req_pool_indices: optional ``int32 [bs]``. When provided, its
            dtype/device/batch are validated against ``selected_indices``.
        seq_lens: optional ``int32 [bs]``. When provided, each row's
            in-range entries are bounded by
            ``ceil(seq_lens[b] / page_size)`` — i.e., the logical page
            count of that request. Any in-range entry ``>=`` that
            per-row bound raises :class:`DSAdapterPageOutOfRange`.
        out: optional pre-allocated ``int32 [bs, max_top_k]`` output
            buffer. When provided, the function writes into it in-place
            (no allocation). Must match dtype, device, and shape.

    Returns:
        ``topk_indices`` of shape ``[bs, max_top_k]``, dtype ``int32``,
        on the same device as ``selected_indices``. Equal to ``out`` when
        ``out`` was provided.

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

    if req_to_token is not None:
        if req_to_token.dtype != torch.int32:
            raise DSAdapterDtypeMismatch(
                f"req_to_token must be int32, got {req_to_token.dtype}"
            )
        if req_to_token.device != selected_indices.device:
            raise DSAdapterDeviceMismatch(
                f"req_to_token.device ({req_to_token.device}) != "
                f"selected_indices.device ({selected_indices.device})"
            )
    if req_pool_indices is not None:
        if req_pool_indices.dtype != torch.int32:
            raise DSAdapterDtypeMismatch(
                f"req_pool_indices must be int32, got {req_pool_indices.dtype}"
            )
        if req_pool_indices.device != selected_indices.device:
            raise DSAdapterDeviceMismatch(
                f"req_pool_indices.device ({req_pool_indices.device}) != "
                f"selected_indices.device ({selected_indices.device})"
            )
        if req_pool_indices.shape[0] != bs:
            raise DSAdapterBatchMismatch(
                f"req_pool_indices batch {req_pool_indices.shape[0]} != "
                f"selected_indices batch {bs}"
            )
    if seq_lens is not None:
        if seq_lens.dtype != torch.int32:
            raise DSAdapterDtypeMismatch(
                f"seq_lens must be int32, got {seq_lens.dtype}"
            )
        if seq_lens.shape[0] != bs:
            raise DSAdapterBatchMismatch(
                f"seq_lens batch {seq_lens.shape[0]} != "
                f"selected_indices batch {bs}"
            )

    if out is not None:
        if out.dtype != torch.int32:
            raise DSAdapterDtypeMismatch(
                f"out must be int32, got {out.dtype}"
            )
        if out.device != selected_indices.device:
            raise DSAdapterDeviceMismatch(
                f"out.device ({out.device}) != "
                f"selected_indices.device ({selected_indices.device})"
            )
        if tuple(out.shape) != tuple(selected_indices.shape):
            raise DSAdapterBatchMismatch(
                f"out shape {tuple(out.shape)} != "
                f"selected_indices shape {tuple(selected_indices.shape)}"
            )

    if bs > 0:
        max_valid = int(valid_lengths.max().item())
        min_valid = int(valid_lengths.min().item())
        if min_valid < 0 or max_valid > max_top_k:
            raise DSAdapterValidLengthOverflow(
                f"valid_lengths range [{min_valid}, {max_valid}] outside "
                f"[0, max_top_k={max_top_k}]"
            )

    # Host-side per-row validation. This sync-trips so the path is NOT
    # CUDA-graph captured (the fast-path helper below is for capture).
    sel_cpu = selected_indices.detach().to("cpu")
    vl_cpu = valid_lengths.detach().to("cpu")
    seq_lens_cpu = (
        seq_lens.detach().to("cpu") if seq_lens is not None else None
    )

    # Per-row validation. When `row_errors` is provided, a row-level
    # contract violation records the typed exception class + message in
    # `row_errors[b]` and is sanitized to all `-1` so the row produces
    # an empty selection downstream; siblings keep running. When
    # `row_errors` is None, the first row-level violation raises (legacy
    # batch-fatal mode kept for unit tests that assert typed exceptions).
    def _record_or_raise(b: int, exc: DSAdapterError) -> bool:
        """Returns True when the row should be sanitized (continue),
        False when no error and the row passes."""
        if row_errors is not None:
            row_errors[b] = (type(exc).__name__, str(exc))
            return True
        raise exc

    sanitized_rows: list = []
    for b in range(bs):
        vl = int(vl_cpu[b].item())
        row = sel_cpu[b]

        if vl < max_top_k:
            padding = row[vl:]
            if bool((padding != -1).any().item()):
                _record_or_raise(
                    b,
                    DSAdapterPaddingViolation(
                        f"row {b}: padding past valid_lengths[{b}]={vl} contains "
                        f"non-(-1) entries"
                    ),
                )
                sanitized_rows.append(b)
                continue

        if vl == 0:
            continue

        valid = row[:vl]
        v_min = int(valid.min().item())
        if v_min < 0:
            _record_or_raise(
                b,
                DSAdapterPageOutOfRange(
                    f"row {b}: in-range entry is negative (min={v_min})"
                ),
            )
            sanitized_rows.append(b)
            continue
        if max_logical_pages is not None:
            v_max = int(valid.max().item())
            if v_max >= max_logical_pages:
                _record_or_raise(
                    b,
                    DSAdapterPageOutOfRange(
                        f"row {b}: max in-range page ID {v_max} >= "
                        f"max_logical_pages {max_logical_pages}"
                    ),
                )
                sanitized_rows.append(b)
                continue
        if seq_lens_cpu is not None:
            seq_len_b = int(seq_lens_cpu[b].item())
            row_max_pages = (seq_len_b + page_size - 1) // page_size
            if row_max_pages > 0:
                v_max = int(valid.max().item())
                if v_max >= row_max_pages:
                    _record_or_raise(
                        b,
                        DSAdapterPageOutOfRange(
                            f"row {b}: max in-range page ID {v_max} >= "
                            f"row's logical page count {row_max_pages} "
                            f"(seq_len={seq_len_b}, page_size={page_size})"
                        ),
                    )
                    sanitized_rows.append(b)
                    continue
            else:
                _record_or_raise(
                    b,
                    DSAdapterPageOutOfRange(
                        f"row {b}: seq_len={seq_len_b} but valid_lengths[{b}]={vl}"
                    ),
                )
                sanitized_rows.append(b)
                continue

        if vl >= 2:
            diffs = valid[1:] - valid[:-1]
            if bool((diffs <= 0).any().item()):
                _record_or_raise(
                    b,
                    DSAdapterNonAscending(
                        f"row {b}: valid prefix is not strictly ascending"
                    ),
                )
                sanitized_rows.append(b)
                continue

    # Sanitize failed rows to all -1 + valid_length 0 so the downstream
    # transform produces a no-op selection for those rows.
    if sanitized_rows:
        for b in sanitized_rows:
            selected_indices[b].fill_(-1)
            valid_lengths[b] = 0

    return expand_ds_selection_to_topk_indices_fast(
        selected_indices=selected_indices,
        page_size=page_size,
        out=out,
    )


def expand_ds_selection_to_topk_indices_fast(
    selected_indices: torch.Tensor,
    page_size: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Host-sync-free expansion without an extra bool-mask allocation.

    Computes ``selected_indices >= 0 ? selected_indices * page_size : -1``.
    Uses arithmetic that avoids allocating a transient bool mask:

        sign(selected) ∈ {-1 for pad, +1 for valid}
        out = ((selected_indices * page_size) * is_valid) + (-1 * (1 - is_valid))

    where ``is_valid = (selected_indices != -1).to(int32)``. The single
    multiplication / addition path writes into ``out`` in place. Callers
    must perform any required validation up-front via
    :func:`expand_ds_selection_to_topk_indices`.

    Args:
        selected_indices: ``int32 [bs, max_top_k]``.
        page_size: positive integer.
        out: optional pre-allocated ``int32 [bs, max_top_k]`` buffer
            owned by the caller (e.g. NSA / FlashMLA forward-metadata).
            When provided, the function writes into it in place and
            returns it.

    Returns:
        ``topk_indices`` tensor.
    """
    if out is None:
        out = torch.empty_like(selected_indices)
    # Three in-place ops, no Python-side mask tensor allocation:
    #   out = selected_indices
    #   out = where(out >= 0, out * page_size, -1)
    # We use the bit-pattern trick: int32 -1 is 0xFFFFFFFF (all ones).
    # `selected_indices.clamp(min=-1)` is a no-op for valid entries (>=0)
    # and leaves pad at -1. Multiplying by page_size sends -1 → -page_size
    # (not what we want); the simplest no-extra-alloc approach is:
    #   out.copy_(selected_indices)
    #   out.mul_(page_size).clamp_(min=-1)  # -1 * page_size → -page_size → -1
    # but clamp(min=-1) on -page_size returns -1 only if page_size > 1, which
    # we already require. Avoids allocating a bool mask.
    out.copy_(selected_indices)
    out.mul_(int(page_size))
    out.clamp_(min=-1)
    return out
