"""Visible-window attention helpers for DeepSeek-V4 vision prefill.

During prefill, tokens inside an image sentinel span attend over a "visible
window" instead of the plain causal sliding window: the left edge is shifted
further left so the whole span stays visible, and the right edge extends past
the diagonal to the span end (bidirectional within the span):

    window(i) = [max(0, i - (W-1) - max(0, left_i - (W-1))), span_end)

where ``left_i`` is the distance of token ``i`` from the span start and ``W``
is the SWA window (128). Tokens outside any span keep the causal window
``[max(0, i - (W-1)), i]``.

Image blocks are atomic across chunk boundaries and cache matches. This module
is torch-free so scheduling can use the same boundaries as attention metadata.
"""

from typing import List, Optional, Sequence, Tuple


def iter_image_spans(mm_input):
    """Yield half-open DSV4 image blocks, including compression padding."""
    if mm_input is None:
        return
    for item in mm_input.mm_items:
        data = item.model_specific_data
        if (
            item.is_image()
            and item.offsets
            and data is not None
            and "perm" in data
            and "types" in data
        ):
            for start, end in item.offsets:
                yield int(start), int(end) + 1


def _iter_visible_window_spans(
    mm_inputs,
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
):
    for req_idx, mm_input in enumerate(mm_inputs or []):
        prefix = int(prefix_lens[req_idx])
        extend_end = prefix + int(extend_lens[req_idx])
        for start, end in iter_image_spans(mm_input):
            if end <= prefix or start >= extend_end:
                continue
            if start < prefix or end > extend_end:
                raise ValueError(
                    f"DeepSeek-V4 image block [{start}, {end}) crosses "
                    f"the prefill range [{prefix}, {extend_end})."
                )
            # Compression padding is causal; IMAGE_START is at position 3 mod 4.
            yield req_idx, start + 3 - start % 4, end


def has_visible_window_span(
    mm_inputs,
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
    swa_window: int,
) -> bool:
    """True iff any request's extend chunk needs per-token visible-window
    overrides (and is therefore ineligible for fixed-shape prefill cuda
    graphs)."""
    return any(
        True for _ in _iter_visible_window_spans(mm_inputs, prefix_lens, extend_lens)
    )


def compute_visible_window_overrides(
    *,
    mm_inputs,
    extend_prefix_lens: Sequence[int],
    extend_seq_lens: Sequence[int],
    swa_window: int,
    padded_num_tokens: int,
) -> Optional[Tuple[List[int], List[int]]]:
    """Per-token SWA window (start, length) for every extend token.

    Defaults to the causal window ``[max(0, pos-(W-1)), pos]``; tokens inside
    an image span get the visible window instead. Returns None on text chunks.
    Partial image blocks are rejected instead of silently changing attention.

    The returned lists have length ``padded_num_tokens``; padding rows carry
    the (start=0, len=1) values that match the padded causal metadata
    (``seq_lens_casual == 1``).
    """
    spans = list(
        _iter_visible_window_spans(mm_inputs, extend_prefix_lens, extend_seq_lens)
    )
    if not spans:
        return None

    win_starts: List[int] = []
    win_lens: List[int] = []
    req_flat_base: List[int] = []
    for prefix, extend_len in zip(extend_prefix_lens, extend_seq_lens):
        prefix, extend_len = int(prefix), int(extend_len)
        req_flat_base.append(len(win_starts) - prefix)
        for pos in range(prefix, prefix + extend_len):
            win_starts.append(max(pos - (swa_window - 1), 0))
            win_lens.append(min(pos + 1, swa_window))

    for req_idx, span_start, span_end in spans:
        flat_base = req_flat_base[req_idx]
        for pos in range(span_start, span_end):
            left = pos - span_start
            win_start = max(0, pos - (swa_window - 1) - max(0, left - (swa_window - 1)))
            i = flat_base + pos
            win_starts[i] = win_start
            win_lens[i] = span_end - win_start

    if padded_num_tokens > len(win_starts):
        pad = padded_num_tokens - len(win_starts)
        win_starts.extend([0] * pad)
        win_lens.extend([1] * pad)
    return win_starts, win_lens


def image_span_aligned_extend_end(mm_input, extend_end: int) -> int:
    """Shrink a chunk boundary to the start of any image block it cuts.

    The result never exceeds the token/KV budget. Callers must defer the
    request if shrinking leaves no tokens to prefill.
    """
    for start, end in iter_image_spans(mm_input):
        if start < extend_end < end:
            extend_end = start
    return extend_end


def image_span_cut_point(mm_input, position: int) -> Optional[int]:
    """Cap a device/host prefix match at an image block's start."""
    aligned = image_span_aligned_extend_end(mm_input, position)
    return aligned if aligned < position else None
