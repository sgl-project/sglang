"""Translate committed cache locations to the draft SWA pool."""

import torch
import triton
import triton.language as tl


@triton.jit
def _committed_swa_locations(
    LOC,
    MAP,
    LENS,
    OUT,
    N: tl.constexpr,
    WIDTH: tl.constexpr,
    MAP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(LENS + i // WIDTH, i < N, other=0)
    committed = (i < N) & (i % WIDTH < length)
    loc = tl.load(LOC + i, committed, other=0).to(tl.int64)
    # Preserve torch indexing for an unused negative location.
    loc = tl.where(loc < 0, loc + MAP_SIZE, loc)
    swa = tl.load(MAP + loc, committed, other=-1).to(tl.int32)
    tl.store(OUT + i, swa, i < N)


def committed_swa_locations(cache_loc, full_to_swa_mapping, commit_lens, width):
    assert cache_loc.numel() == commit_lens.numel() * width
    out = torch.empty_like(cache_loc, dtype=torch.int32)
    if cache_loc.numel():
        _committed_swa_locations[(triton.cdiv(cache_loc.numel(), 256),)](
            cache_loc,
            full_to_swa_mapping,
            commit_lens,
            out,
            cache_loc.numel(),
            width,
            full_to_swa_mapping.numel(),
            256,
        )
    return out
