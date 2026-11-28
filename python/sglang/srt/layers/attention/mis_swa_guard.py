from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Optional


def format_swa_mis_overflow_error(
    *,
    max_logical_pos: int,
    window_left: int,
    prefix_len: int,
    max_item_span: int,
) -> str:
    """Format a user-facing error for SWA window overflow under --enable-mis."""
    return (
        "Multi-item scoring (--enable-mis) with sliding-window attention requires "
        f"max_logical_pos <= window_left ({max_logical_pos} > {window_left}; "
        f"prefix_len={prefix_len}, max_item_span={max_item_span}). "
        "Shorten the query/items or disable --enable-mis."
    )


def validate_swa_mis_query_items(
    *,
    query_len: int,
    item_lens: Sequence[int],
    window_left: Optional[int],
) -> None:
    """Reject --enable-mis inputs whose query + longest item exceeds window_left."""
    if window_left is None or not item_lens:
        return
    max_item_span = max(max(item_lens), 1)
    max_logical_pos = query_len + max_item_span - 1
    if max_logical_pos > window_left:
        raise ValueError(
            format_swa_mis_overflow_error(
                max_logical_pos=max_logical_pos,
                window_left=window_left,
                prefix_len=query_len,
                max_item_span=max_item_span,
            )
        )


def compute_max_logical_pos_from_delimiters(
    *,
    prefix_len: int,
    delimiters: Sequence[int] | Any,
    raw_seq_len: Optional[int] = None,
) -> tuple[int, int]:
    """Compute (max_logical_pos, max_item_span) without counting DP/TP padding."""
    delim_list = (
        delimiters.tolist()
        if hasattr(delimiters, "tolist")
        else [int(d) for d in delimiters]
    )
    if not delim_list:
        return max(prefix_len - 1, 0), 0

    inter_span = max(
        (delim_list[i + 1] - delim_list[i] - 1 for i in range(len(delim_list) - 1)),
        default=0,
    )
    tail_span = (
        max(raw_seq_len - 1 - delim_list[-1], 0) if raw_seq_len is not None else 0
    )
    max_item_span = max(inter_span, tail_span, 1)
    max_logical_pos = prefix_len + max_item_span - 1
    return max_logical_pos, max_item_span


def validate_swa_mis_delimiters(
    *,
    prefix_len: int,
    delimiters: Sequence[int] | Any,
    window_left: Optional[int],
    raw_seq_len: Optional[int] = None,
) -> None:
    """Validate serialized MIS delimiter spans against the active SWA window."""
    if window_left is None:
        return
    max_logical_pos, max_item_span = compute_max_logical_pos_from_delimiters(
        prefix_len=prefix_len,
        delimiters=delimiters,
        raw_seq_len=raw_seq_len,
    )
    if max_logical_pos > window_left:
        raise ValueError(
            format_swa_mis_overflow_error(
                max_logical_pos=max_logical_pos,
                window_left=window_left,
                prefix_len=prefix_len,
                max_item_span=max_item_span,
            )
        )


def validate_swa_mis_batch(
    *,
    precomputed_indices: Sequence[Any],
    extend_seq_lens: Sequence[int],
    raw_extend_seq_lens: Optional[Sequence[int]],
    prefix_cache_lens: Optional[Sequence[int]],
    window_left: Optional[int],
) -> None:
    """Validate every MIS sequence in a batch before mutating metadata or positions."""
    if window_left is None:
        return
    for i, seq_len in enumerate(extend_seq_lens):
        delimiter_indices = precomputed_indices[i]
        if len(delimiter_indices) == 0:
            continue
        first_delim_raw = delimiter_indices[0]
        first_delim = (
            int(first_delim_raw.item())
            if hasattr(first_delim_raw, "item")
            else int(first_delim_raw)
        )
        prefix_len = first_delim + (
            int(prefix_cache_lens[i]) if prefix_cache_lens is not None else 0
        )
        raw_seq_len = (
            int(raw_extend_seq_lens[i])
            if raw_extend_seq_lens is not None and i < len(raw_extend_seq_lens)
            else None
        )
        validate_swa_mis_delimiters(
            prefix_len=prefix_len,
            delimiters=delimiter_indices,
            window_left=window_left,
            raw_seq_len=raw_seq_len,
        )


def aggregate_min_positive_swa_window(
    *, windows: Iterable[Optional[int]]
) -> Optional[int]:
    """Return the strictest non-negative SWA window across layers or ranks."""
    active = [
        int(w)
        for w in windows
        if isinstance(w, int) and not isinstance(w, bool) and w >= 0
    ]
    return min(active) if active else None


def merge_scheduler_swa_windows(*, infos: Sequence[dict[str, Any]]) -> Optional[int]:
    """Normalize min_positive_swa_window across all scheduler init info dicts."""
    merged = aggregate_min_positive_swa_window(
        windows=(info.get("min_positive_swa_window") for info in infos)
    )
    for info in infos:
        info["min_positive_swa_window"] = merged
    return merged
