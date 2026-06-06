from __future__ import annotations

from typing import Any, Callable, List, Tuple

import torch

from sglang.srt.kv_canary.buffer_group import CanaryBufferGroup
from sglang.srt.kv_canary.pool_patcher.utils import wrap_method

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
    head and tail entries from ``group``."""

    def _with_splice(original: Callable, *args: Any, **kwargs: Any) -> BufInfoTriple:
        ptrs, lens, item_lens = original(*args, **kwargs)
        return splice_kv_buf_info(
            ptrs=ptrs,
            lens=lens,
            item_lens=item_lens,
            group=group,
            has_v_half=has_v_half,
            page_size=page_size,
        )

    wrap_method(pool, method_name, wrapper=_with_splice)


def splice_kv_buf_info(
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
