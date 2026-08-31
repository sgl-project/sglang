"""Visible-window attention helpers for DeepSeek-V4 vision prefill.

During prefill, tokens inside an image sentinel span attend over a "visible
window" instead of the plain causal sliding window: the left edge is shifted
further left so the whole span stays visible, and the right edge extends past
the diagonal to the span end (bidirectional within the span):

    window(i) = [max(0, i - (W-1) - max(0, left_i - (W-1))), span_end)

where ``left_i`` is the distance of token ``i`` from the span start and ``W``
is the SWA window (128). Tokens outside any span keep the causal window
``[max(0, i - (W-1)), i]``.

A span that is not fully contained in the current extend chunk (chunked
prefill split or radix-cache hit in the middle of the span) degrades to the
causal window, with a one-time warning (the supported deployment disables
both, so this should never fire there).

This module is deliberately torch-free so scheduler-side code (dp_attn vote,
prefill cuda graph runner) can import it without pulling in the attention
backend.
"""

import logging
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_degraded_span_warned = False


def _warn_degraded_span_once() -> None:
    global _degraded_span_warned
    if _degraded_span_warned:
        return
    logger.warning(
        "DSV4 visible-window: an image span is not fully contained in this "
        "extend chunk (chunked prefill or radix-cache hit); the span falls "
        "back to the plain causal window."
    )
    _degraded_span_warned = True


def _iter_fully_contained_image_spans(
    mm_inputs,
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
):
    """Yield (req_idx, span_start, span_end_exclusive) for each image sentinel
    span fully contained in that request's extend range."""
    for req_idx, mm_input in enumerate(mm_inputs or []):
        if mm_input is None:
            continue
        prefix = int(prefix_lens[req_idx])
        extend_end = prefix + int(extend_lens[req_idx])
        for item in mm_input.mm_items:
            if not item.is_image() or not item.offsets:
                continue
            for span_start, span_end_incl in item.offsets:
                if span_start >= prefix and span_end_incl < extend_end:
                    yield req_idx, int(span_start), int(span_end_incl) + 1
                elif span_start < extend_end and span_end_incl >= prefix:
                    # Genuine cut: the span overlaps this chunk (or the prefix)
                    # but is not fully inside it — chunked prefill without
                    # span alignment, or a radix hit mid-span. Spans that lie
                    # entirely before the prefix or entirely after this chunk
                    # are irrelevant here and must not warn.
                    _warn_degraded_span_once()


def has_fully_contained_image_span(
    mm_inputs,
    prefix_lens: Sequence[int],
    extend_lens: Sequence[int],
) -> bool:
    """True iff any request's extend chunk fully contains an image span, i.e.
    the batch needs per-token visible-window overrides and is ineligible for
    fixed-shape prefill cuda graphs."""
    return any(
        True
        for _ in _iter_fully_contained_image_spans(mm_inputs, prefix_lens, extend_lens)
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
    a fully contained image span get the visible window instead. Returns None
    when no span is fully contained in this chunk (pure causal behavior).

    The returned lists have length ``padded_num_tokens``; padding rows carry
    the (start=0, len=1) values that match the padded causal metadata
    (``seq_lens_casual == 1``).
    """
    spans = list(
        _iter_fully_contained_image_spans(
            mm_inputs, extend_prefix_lens, extend_seq_lens
        )
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
    """Move a chunked-prefill truncation point past any image span it cuts.

    The visible-window attention above requires every image sentinel span to
    be prefilled in a single extend. If ``extend_end`` (absolute position in
    the request's padded input ids) falls strictly inside a span, it is moved
    to the span end, overshooting the chunk budget by at most one span (a few
    hundred tokens — the budget is a scheduling heuristic, not a hard memory
    limit). Extending is preferred over shrinking to the span start because
    shrinking can make zero progress when the span starts at the current
    position.
    """
    if mm_input is None:
        return extend_end
    for item in mm_input.mm_items:
        if not item.is_image() or not item.offsets:
            continue
        for span_start, span_end_incl in item.offsets:
            span_end = span_end_incl + 1
            if span_start < extend_end < span_end:
                extend_end = span_end
    return extend_end
