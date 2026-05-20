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
    head_entry = _entry_triple(group.k_head, page_size=page_size)
    tail_entry = _entry_triple(group.k_tail, page_size=page_size)

    out_ptrs: List[int] = []
    out_lens: List[int] = []
    out_item_lens: List[int] = []

    for seg_idx in range(len(segment_offsets)):
        start = segment_offsets[seg_idx]
        stop = (
            segment_offsets[seg_idx + 1]
            if seg_idx + 1 < len(segment_offsets)
            else len(ptrs)
        )
        out_ptrs.append(head_entry[0])
        out_lens.append(head_entry[1])
        out_item_lens.append(head_entry[2])
        out_ptrs.extend(ptrs[start:stop])
        out_lens.extend(lens[start:stop])
        out_item_lens.extend(item_lens[start:stop])
        out_ptrs.append(tail_entry[0])
        out_lens.append(tail_entry[1])
        out_item_lens.append(tail_entry[2])

    return out_ptrs, out_lens, out_item_lens


def _splice_kv_buf_info(
    *,
    ptrs: List[int],
    lens: List[int],
    item_lens: List[int],
    group: CanaryBufferGroup,
    has_v_half: bool,
    page_size: int,
) -> BufInfoTriple:
    k_head_entry = _entry_triple(group.k_head, page_size=page_size)
    k_tail_entry = _entry_triple(group.k_tail, page_size=page_size)

    if not has_v_half:
        return (
            [k_head_entry[0]] + list(ptrs) + [k_tail_entry[0]],
            [k_head_entry[1]] + list(lens) + [k_tail_entry[1]],
            [k_head_entry[2]] + list(item_lens) + [k_tail_entry[2]],
        )

    assert group.v_head is not None and group.v_tail is not None
    v_head_entry = _entry_triple(group.v_head, page_size=page_size)
    v_tail_entry = _entry_triple(group.v_tail, page_size=page_size)

    if len(ptrs) % 2 != 0:
        raise RuntimeError(
            f"kv-canary: K/V split adapter expects even-length buf_info list, got {len(ptrs)}"
        )
    mid = len(ptrs) // 2
    return (
        [k_head_entry[0]]
        + list(ptrs[:mid])
        + [k_tail_entry[0], v_head_entry[0]]
        + list(ptrs[mid:])
        + [v_tail_entry[0]],
        [k_head_entry[1]]
        + list(lens[:mid])
        + [k_tail_entry[1], v_head_entry[1]]
        + list(lens[mid:])
        + [v_tail_entry[1]],
        [k_head_entry[2]]
        + list(item_lens[:mid])
        + [k_tail_entry[2], v_head_entry[2]]
        + list(item_lens[mid:])
        + [v_tail_entry[2]],
    )


def _entry_triple(buf: torch.Tensor, *, page_size: int) -> Tuple[int, int, int]:
    return (
        buf.data_ptr(),
        buf.nbytes,
        buf[0].nbytes * page_size,
    )
