from __future__ import annotations

from typing import Any, Callable, List, Tuple

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup

from .wrap_method import wrap_method

BufInfoTriple = Tuple[List[int], List[int], List[int]]


def patch_buf_info_method(
    pool: object,
    *,
    method_name: str,
    group: CanaryBufferGroup,
    has_v_half: bool,
    page_size: int,
) -> None:
    """Wrap ``pool.<method_name>()`` so its (ptrs, lens, item_lens) triple is spliced with K/V
    head and tail entries from ``group``.

    Per Rule: PD layout is ``k0 k1 ... kN v0 v1 ... vN`` — head/tail sit at index 0 / N+1 within
    EACH half, not at the absolute ends of the combined list.
    """

    def _with_splice(original: Callable, *args: Any, **kwargs: Any) -> BufInfoTriple:
        ptrs, lens, item_lens = original(*args, **kwargs)
        return _splice_kv_buf_info(
            ptrs=ptrs,
            lens=lens,
            item_lens=item_lens,
            group=group,
            has_v_half=has_v_half,
            page_size=page_size,
        )

    wrap_method(pool, method_name, wrapper=_with_splice)


def splice_segmented_buf_info(
    *,
    ptrs: List[int],
    lens: List[int],
    item_lens: List[int],
    segment_offsets: List[int],
    group: CanaryBufferGroup,
    page_size: int,
) -> BufInfoTriple:
    """For multi-segment packed pools (DSV4): prepend/append head + tail canary entries around EACH
    segment, using only ``group.k_head`` / ``group.k_tail`` (these pools have no V half).
    """
    entries = list(zip(ptrs, lens, item_lens))
    head = _entry_triple(group.k_head, page_size=page_size)
    tail = _entry_triple(group.k_tail, page_size=page_size)

    out: List[Tuple[int, int, int]] = []
    for seg_idx in range(len(segment_offsets)):
        start = segment_offsets[seg_idx]
        stop = (
            segment_offsets[seg_idx + 1]
            if seg_idx + 1 < len(segment_offsets)
            else len(entries)
        )
        out.append(head)
        out.extend(entries[start:stop])
        out.append(tail)

    return _untranspose_entries(out)


def _splice_kv_buf_info(
    *,
    ptrs: List[int],
    lens: List[int],
    item_lens: List[int],
    group: CanaryBufferGroup,
    has_v_half: bool,
    page_size: int,
) -> BufInfoTriple:
    entries = list(zip(ptrs, lens, item_lens))
    k_head = _entry_triple(group.k_head, page_size=page_size)
    k_tail = _entry_triple(group.k_tail, page_size=page_size)

    if not has_v_half:
        out = [k_head, *entries, k_tail]
    else:
        assert group.v_head is not None and group.v_tail is not None
        v_head = _entry_triple(group.v_head, page_size=page_size)
        v_tail = _entry_triple(group.v_tail, page_size=page_size)
        if len(entries) % 2 != 0:
            raise RuntimeError(
                f"kv-canary: K/V split adapter expects even-length buf_info list, got {len(entries)}"
            )
        mid = len(entries) // 2
        out = [k_head, *entries[:mid], k_tail, v_head, *entries[mid:], v_tail]

    return _untranspose_entries(out)


def _entry_triple(buf: torch.Tensor, *, page_size: int) -> Tuple[int, int, int]:
    return (
        buf.data_ptr(),
        buf.nbytes,
        buf[0].nbytes * page_size,
    )


def _untranspose_entries(entries: List[Tuple[int, int, int]]) -> BufInfoTriple:
    out_ptrs, out_lens, out_item_lens = (list(col) for col in zip(*entries))
    return out_ptrs, out_lens, out_item_lens
