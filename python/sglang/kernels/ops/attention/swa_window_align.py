"""Align a SWA paged-KV sequence so FA4 n-blocks start at the window.

A 511-token window at ctx=60000 starts at token 59488. tile_n=128 then covers
logical keys [59392, 60000) — five n-blocks — even though only 512 keys are
visible. Shifting the logical origin to the window start (leftpad=59488,
seqused=512) yields four n-blocks without changing which keys are attended.

PagedKVManager already maps ``row + leftpad`` through the page table; this
helper only computes the host/device integers.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def swa_window_align_supported(query_len: int) -> bool:
    """True when FA4 leftpad is safe for this query width.

    T=1 decode/draft is the priced SM103 shape. ``query_len > 1`` is MTP
    verify / draft-extend; pack_gqa plus leftpad changes n-block coverage
    and is not bit-equivalent to the unaligned sliding window.
    """
    return int(query_len) == 1


def align_swa_cache_seqlens(
    cache_seqlens: torch.Tensor,
    *,
    window_left: int,
    query_len: int,
    out_seqlens: Optional[torch.Tensor] = None,
    out_leftpad: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(aligned_seqlens, leftpad)`` for the visible SWA span.

    ``leftpad = relu(cache_seqlens - (window_left + query_len))`` so the
    leftmost query still sees a full window. ``aligned_seqlens`` is the
    remaining K length in that shifted space. Keep the original sliding-window
    mask: rows>1 still need it for the extra keys on the left of later queries.

    Serving only applies this shift for ``query_len == 1``. MTP target-verify
    (T=4, pack_gqa) was never the priced shape; using the T=1 leftpad there
    moves later verify queries off the draft window and pins accept length
    at ~2. See ``swa_window_align_supported``.
    """
    if cache_seqlens.dtype != torch.int32:
        raise ValueError("cache_seqlens must be int32")
    if window_left < 0:
        raise ValueError("window_left must be >= 0")
    if query_len <= 0:
        raise ValueError("query_len must be positive")
    span = int(window_left) + int(query_len)
    if out_leftpad is None:
        leftpad = torch.clamp(cache_seqlens - span, min=0)
    else:
        torch.clamp(cache_seqlens - span, min=0, out=out_leftpad)
        leftpad = out_leftpad
    if out_seqlens is None:
        seqlens = cache_seqlens - leftpad
    else:
        torch.sub(cache_seqlens, leftpad, out=out_seqlens)
        seqlens = out_seqlens
    return seqlens, leftpad
