"""The deferred context-KV commit's staging in one launch: the verify rows' write
locations, positions and (with a ring) state slots, and the gated commit lengths
with zeros past the staged requests."""

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _stage_commit_kernel(
    loc_src_ptr,
    loc_dst_ptr,
    pos_src_ptr,
    pos_dst_ptr,
    slot_src_ptr,
    slot_dst_ptr,
    lens_src_ptr,
    lens_dst_ptr,
    rows,
    bs,
    max_bs,
    HAS_SLOT: tl.constexpr,
    ROWS_BLOCK: tl.constexpr,
    LENS_BLOCK: tl.constexpr,
):
    r = tl.arange(0, ROWS_BLOCK)
    m = r < rows
    tl.store(loc_dst_ptr + r, tl.load(loc_src_ptr + r, mask=m), mask=m)
    tl.store(pos_dst_ptr + r, tl.load(pos_src_ptr + r, mask=m), mask=m)
    if HAS_SLOT:
        tl.store(slot_dst_ptr + r, tl.load(slot_src_ptr + r, mask=m), mask=m)
    q = tl.arange(0, LENS_BLOCK)
    lens = tl.load(lens_src_ptr + q, mask=q < bs, other=0)
    tl.store(lens_dst_ptr + q, lens, mask=q < max_bs)


def stage_commit(
    *,
    cache_loc_dst: torch.Tensor,
    positions_dst: torch.Tensor,
    commit_lens_dst: torch.Tensor,
    state_slot_dst: Optional[torch.Tensor],
    cache_loc: torch.Tensor,
    positions: torch.Tensor,
    commit_lens: torch.Tensor,
    state_slot: Optional[torch.Tensor],
    rows: int,
) -> None:
    """``cache_loc_dst[:rows] = cache_loc[:rows]``, likewise positions / state slots,
    ``commit_lens_dst[:bs] = commit_lens`` and ``commit_lens_dst[bs:] = 0``."""
    assert cache_loc.is_contiguous() and positions.is_contiguous()
    assert commit_lens.is_contiguous()
    assert (state_slot is None) == (state_slot_dst is None)
    bs, max_bs = commit_lens.shape[0], commit_lens_dst.shape[0]
    _stage_commit_kernel[(1,)](
        cache_loc,
        cache_loc_dst,
        positions,
        positions_dst,
        state_slot if state_slot is not None else cache_loc,
        state_slot_dst if state_slot_dst is not None else cache_loc_dst,
        commit_lens,
        commit_lens_dst,
        rows,
        bs,
        max_bs,
        HAS_SLOT=state_slot is not None,
        ROWS_BLOCK=triton.next_power_of_2(max(rows, 1)),
        LENS_BLOCK=triton.next_power_of_2(max_bs),
    )
