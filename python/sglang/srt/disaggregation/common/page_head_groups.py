"""Whole-page KV transfer into a decode with fewer attention-TP ranks than its prefill.

Prefill rank g of the G = prefill / decode attention-TP ranks writes each of its pages
into the g-th 1/G of the decode page, so a landed page is [group, token, share]. The
decode reorders it into its own [token, group, share] layout before the request is
scheduled.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.srt.disaggregation.base.conn import StateType

# Cap on the scratch of one reorder launch pair.
_SCRATCH_BYTES = 256 << 20


def page_group_plan(
    src_pages: np.ndarray,
    dst_pages: np.ndarray,
    src_item_len: int,
    groups: int,
    group: int,
) -> np.ndarray:
    """[local offset, remote offset, size] rows: each source page whole, into the
    group-th 1/groups of its destination page."""
    n = min(len(src_pages), len(dst_pages))
    plan = np.empty((n, 3), dtype=np.uint64)
    plan[:, 0] = np.asarray(src_pages[:n], dtype=np.int64) * src_item_len
    plan[:, 1] = (
        np.asarray(dst_pages[:n], dtype=np.int64) * (src_item_len * groups)
        + group * src_item_len
    )
    plan[:, 2] = src_item_len
    return plan


def _units(
    ptrs: Sequence[int], item_lens: Sequence[int], pages: Optional[np.ndarray], out
) -> None:
    if pages is None or len(pages) == 0:
        return
    pages = np.asarray(pages, dtype=np.int64).ravel()
    for ptr, item_len in zip(ptrs, item_lens):
        if ptr and item_len:
            out.setdefault(int(item_len), []).append((int(ptr), pages))


def landed_units(
    kv_args, kv_pages: np.ndarray, state_pages: Optional[List[Optional[np.ndarray]]]
) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    """{page bytes: [(buffer pointer, pages)]} of the KV and window-state entries."""
    units: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    _units(kv_args.kv_data_ptrs, kv_args.kv_item_lens, kv_pages, units)
    state_types = getattr(kv_args, "state_types", None) or []
    for i, state_type in enumerate(state_types):
        if state_type != StateType.SWA or not state_pages or i >= len(state_pages):
            continue
        _units(
            kv_args.state_data_ptrs[i],
            kv_args.state_item_lens[i],
            state_pages[i],
            units,
        )
    return units


@triton.jit
def _gather_rows_kernel(
    ptr_table,
    unit_buf,
    unit_page,
    scratch,
    page_words,
    ROWS: tl.constexpr,
    GROUPS: tl.constexpr,
    CHUNK_WORDS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    # A landed page is [group, row, chunk]; write it to scratch as [row, group, chunk].
    unit = tl.program_id(0)
    row0 = tl.program_id(1) * BLOCK_ROWS
    base = tl.load(ptr_table + tl.load(unit_buf + unit)).to(scratch.dtype)
    page = tl.load(unit_page + unit).to(tl.int64)
    src = base + page * page_words
    dst = scratch + unit.to(tl.int64) * page_words
    r = row0 + tl.arange(0, BLOCK_ROWS)[:, None, None]
    g = tl.arange(0, GROUPS)[None, :, None]
    c = tl.arange(0, CHUNK_WORDS)[None, None, :]
    x = tl.load(src + g * (ROWS * CHUNK_WORDS) + r * CHUNK_WORDS + c)
    tl.store(dst + r * (GROUPS * CHUNK_WORDS) + g * CHUNK_WORDS + c, x)


@triton.jit
def _copy_back_kernel(
    ptr_table,
    unit_buf,
    unit_page,
    scratch,
    page_words,
    BLOCK: tl.constexpr,
):
    unit = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    base = tl.load(ptr_table + tl.load(unit_buf + unit)).to(scratch.dtype)
    page = tl.load(unit_page + unit).to(tl.int64)
    x = tl.load(scratch + unit.to(tl.int64) * page_words + offs)
    tl.store(base + page * page_words + offs, x)


def _reorder_on_device(
    entries: List[Tuple[int, np.ndarray]],
    item_len: int,
    page_size: int,
    groups: int,
    device: torch.device,
) -> None:
    words = item_len // 4
    chunk_words = words // (page_size * groups)
    block_rows = max(1, min(page_size, 4096 // (groups * chunk_words)))
    block = min(words, 4096)
    if (
        words * 4 != item_len
        or chunk_words * page_size * groups != words
        or chunk_words & (chunk_words - 1)
        or groups & (groups - 1)
        or page_size % block_rows
        or words % block
    ):
        raise RuntimeError(
            f"Cannot reorder {item_len}-byte pages of {page_size} rows into {groups} "
            "head groups on the device"
        )
    ptr_table = torch.tensor([p for p, _ in entries], dtype=torch.int64, device=device)
    bufs = np.concatenate(
        [np.full(len(pages), i, dtype=np.int64) for i, (_, pages) in enumerate(entries)]
    )
    pages = np.concatenate([pages for _, pages in entries])
    step = max(1, _SCRATCH_BYTES // item_len)
    scratch = torch.empty(
        min(step, len(pages)) * words, dtype=torch.int32, device=device
    )
    for start in range(0, len(pages), step):
        unit_buf = torch.from_numpy(bufs[start : start + step]).to(
            device, non_blocking=True
        )
        unit_page = torch.from_numpy(pages[start : start + step]).to(
            device, non_blocking=True
        )
        n = unit_buf.numel()
        _gather_rows_kernel[(n, page_size // block_rows)](
            ptr_table,
            unit_buf,
            unit_page,
            scratch,
            words,
            ROWS=page_size,
            GROUPS=groups,
            CHUNK_WORDS=chunk_words,
            BLOCK_ROWS=block_rows,
        )
        _copy_back_kernel[(n, words // block)](
            ptr_table, unit_buf, unit_page, scratch, words, BLOCK=block
        )


def reorder_landed_pages(
    kv_args,
    groups: int,
    kv_pages: np.ndarray,
    state_pages: Optional[List[Optional[np.ndarray]]],
    device: torch.device,
) -> None:
    """Reorder the pages a whole-page transfer landed into the decode's head layout."""
    page_size = kv_args.page_size
    for item_len, entries in landed_units(kv_args, kv_pages, state_pages).items():
        if item_len % (page_size * groups):
            raise RuntimeError(
                f"{item_len}-byte pages of {page_size} rows do not split into {groups} head groups"
            )
        _reorder_on_device(entries, item_len, page_size, groups, device)
