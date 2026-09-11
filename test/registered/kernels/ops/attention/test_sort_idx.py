"""Correctness tests for the DeepSeek-V4.1 JIT candidate block table sort.

``sort_candidate_blocks`` turns a row's selected block ids (any order, ``-1``
padded) in place into the ascending, ``INT32_MAX``-padded table DeepGEMM's
sparse indexer schedule reads, and writes the same blocks as pool slots / 8
through the row's index page table. Rows with at most ``k`` blocks get the
identity table. The kernel is a bitmap counting sort whose dense words are
drained by a block-wide queue, so the id distributions below cover sparse,
clustered and completely full words.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import (
    sort_candidate_blocks,
    transform_candidate_blocks,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")

BLOCK = 8
PAD = torch.iinfo(torch.int32).max
MAX_BLOCKS = 1 << 17


def _select(nblocks: int, k: int, dist: str, gen: torch.Generator) -> torch.Tensor:
    """k distinct block ids below nblocks (all of them if nblocks <= k), shuffled."""
    n = min(k, nblocks)
    if dist == "uniform":
        ids = torch.randperm(nblocks, device="cuda", generator=gen)[:n]
    elif dist == "clustered":  # runs of 64 consecutive blocks mixed with scattered ones
        runs = max(n // 96, 1)
        starts = torch.randint(
            0, max(nblocks - 64, 1), (runs,), device="cuda", generator=gen
        )
        runs_ids = (
            starts[:, None] + torch.arange(64, device="cuda")[None, :]
        ).flatten()
        rest = torch.randperm(nblocks, device="cuda", generator=gen)[:n]
        ids = torch.unique(torch.cat([runs_ids, rest]))[:n]
    elif dist == "newest":  # the tail of the row: every word full
        ids = torch.arange(nblocks - n, nblocks, device="cuda")
    else:
        raise ValueError(dist)
    ids = ids.to(torch.int32)
    ids = ids[torch.randperm(ids.numel(), device="cuda", generator=gen)]
    return torch.nn.functional.pad(ids, (0, k - n), value=-1)


def _case(lens, k, dist, page_size, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    lens_t = torch.tensor(lens, dtype=torch.int32, device="cuda")
    max_pages = (max(lens) + page_size - 1) // page_size
    page_table = torch.stack(
        [torch.randperm(max_pages, device="cuda", generator=gen) for _ in lens]
    ).to(torch.int32)
    blocks = torch.stack(
        [_select((l + BLOCK - 1) // BLOCK, k, dist, gen) for l in lens]
    )
    return lens_t, page_table, blocks


def _check(blocks, pages, selected, lens, page_table, page_size, k):
    bpp = page_size // BLOCK
    for b, seq in enumerate(lens):
        nblocks = (seq + BLOCK - 1) // BLOCK
        n = min(k, nblocks)
        if nblocks <= k:
            ref = torch.arange(n, device="cuda", dtype=torch.int32)
        else:
            ref = selected[b][selected[b] >= 0].sort().values
        assert torch.equal(blocks[b, :n], ref), f"row {b} ids"
        assert torch.all(blocks[b, n:] == PAD), f"row {b} padding"
        ref_pages = page_table[b][ref // bpp] * bpp + ref % bpp
        assert torch.equal(pages[b, :n], ref_pages), f"row {b} slots"
        assert torch.all(pages[b, n:] == PAD), f"row {b} slot padding"


@pytest.mark.parametrize("dist", ["uniform", "clustered", "newest"])
@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize(
    "lens,k",
    [
        # rows that fit (identity), rows just above k, long rows up to 1M tokens
        ([1, 8, 16384, 16385, 16392], 2048),
        ([40000, 131072, 65537, 1048576], 2048),
        ([1048576, 1048571, 300000], 2048),
        ([2049, 9000, 20000], 256),
    ],
)
def test_matches_sorted_selection(lens, k, dist, page_size):
    lens_t, page_table, blocks = _case(lens, k, dist, page_size)
    selected = blocks.clone()
    pages = sort_candidate_blocks(blocks, lens_t, page_table, page_size)
    _check(blocks, pages, selected, lens, page_table, page_size, k)


def test_dense_words_only():
    """Every selected id inside k / 32 full words, in random order: the queue path only."""
    lens = [MAX_BLOCKS * BLOCK, 100000]
    lens_t, page_table, _ = _case(lens, 2048, "uniform", 128)
    gen = torch.Generator(device="cuda").manual_seed(3)
    rows = []
    for seq in lens:
        nblocks = (seq + BLOCK - 1) // BLOCK
        start = (nblocks - 2048) // 2 // 32 * 32
        ids = torch.arange(start, start + 2048, device="cuda", dtype=torch.int32)
        rows.append(ids[torch.randperm(2048, device="cuda", generator=gen)])
    blocks = torch.stack(rows)
    selected = blocks.clone()
    pages = torch.empty_like(blocks)
    assert (
        sort_candidate_blocks(blocks, lens_t, page_table, 128, out_pages=pages) is pages
    )
    _check(blocks, pages, selected, lens, page_table, 128, 2048)


@pytest.mark.parametrize("page_size", [64, 128])
def test_page_transform_of_sorted_ids(page_size):
    """Ascending ids in, INT32_MAX past the row's count (a DeepSelect-style
    input): the same pages as the sort, blocks untouched."""
    lens = [1, 16384, 16385, 131072, 1048576]
    k = 2048
    lens_t, page_table, blocks = _case(lens, k, "clustered", page_size, seed=11)
    ref_pages = sort_candidate_blocks(blocks.clone(), lens_t, page_table, page_size)
    ascending = torch.where(blocks < 0, PAD, blocks).sort(dim=1).values
    kept = ascending.clone()
    pages = transform_candidate_blocks(ascending, lens_t, page_table, page_size)
    assert torch.equal(ascending, kept)
    assert torch.equal(pages, ref_pages)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
