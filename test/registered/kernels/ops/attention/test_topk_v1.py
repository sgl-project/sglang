"""Correctness tests for the DeepSeek-V4 JIT top-k transform v1.

``topk_transform_paged`` dispatches to ``deepseek_v4/topk_v1.cuh`` on CUDA. Its
radix select first bins scores by the high byte of their ordered FP16 value and
stashes the threshold bin in shared memory (8192 entries). These tests put the
threshold bin on both sides of that capacity, with strictly higher values
emitted before it, ties at the cutoff, and clusters that only separate at a
deeper FP32 key byte. They cover runtime ``topk`` values, the ``seq_len <= topk``
path, strided scores, and both the page-table mapped and raw outputs.

Selections are compared with ``torch.topk`` as value multisets: any subset of
values tied at the cutoff is a valid selection, but indices must be distinct
and in range, and the mapped output must be the page-table transform of the
raw output.
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import topk_transform_paged
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or is_hip(),
    reason="topk_transform_paged uses the JIT v1 kernel on CUDA only",
)

# kSMEM / (2 * sizeof(int32_t)) in topk_v1.cuh: entries per stash round.
STASH_ENTRIES = 8192
PAGE_SIZE = 64
TOPKS = (512, 1024, 300)


def _generator(seed: int = 0) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _coarse_keys(values: torch.Tensor) -> torch.Tensor:
    """Mirror the kernel's high-byte ordered-FP16 key on the CPU."""
    bits = values.to(torch.float16).view(torch.int16).to(torch.int32) & 0xFFFF
    ordered = torch.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits | 0x8000)
    return ordered >> 8


def _tight_cluster(length: int, base: float = 1.0) -> torch.Tensor:
    # Increasing values in one coarse bin, best candidates last, so an
    # arrival-order stash prefix is wrong.
    return base + torch.arange(length, dtype=torch.float32) * (1e-3 / length)


def _byte_depth_clusters(depth: int, per_side: int, gen) -> torch.Tensor:
    # Two clusters sharing the coarse bin whose keys first differ at bit
    # ``depth``; the cluster with that bit set holds the true top-k.
    keys = torch.full((2 * per_side,), 0x3F000000, dtype=torch.int64)
    keys += torch.randint(0, 1 << depth, (2 * per_side,), generator=gen)
    keys[:per_side] += 1 << depth
    keys = keys[torch.randperm(2 * per_side, generator=gen)]
    return keys.to(torch.int32).view(torch.float32)


def _run_and_check(rows: list[torch.Tensor], topk: int, stride_pad: int = 0):
    gen = _generator(topk)
    batch = len(rows)
    stride = max(row.numel() for row in rows) + stride_pad
    # Pad with +inf so any read past ``seq_lens`` would corrupt the selection.
    scores = torch.full((batch, stride), float("inf"), dtype=torch.float32)
    for i, row in enumerate(rows):
        scores[i, : row.numel()] = row
    seq_lens = torch.tensor([row.numel() for row in rows], dtype=torch.int32)
    num_pages = (stride + PAGE_SIZE - 1) // PAGE_SIZE
    page_table = torch.stack(
        [torch.randperm(4 * num_pages, generator=gen)[:num_pages] for _ in range(batch)]
    ).to(torch.int32)

    out_page = torch.full((batch, topk), -7, dtype=torch.int32, device="cuda")
    out_raw = torch.full((batch, topk), -7, dtype=torch.int32, device="cuda")
    topk_transform_paged(
        scores.cuda(),
        seq_lens.cuda(),
        page_table.cuda(),
        out_page,
        PAGE_SIZE,
        out_raw,
    )
    out_page = out_page.cpu().to(torch.int64)
    out_raw = out_raw.cpu().to(torch.int64)

    for i, row in enumerate(rows):
        length = row.numel()
        raw = out_raw[i]
        if length <= topk:
            expected = torch.full((topk,), -1, dtype=torch.int64)
            expected[:length] = torch.arange(length)
            assert torch.equal(raw, expected), f"row {i}: trivial raw indices"
        else:
            assert bool((raw >= 0).all()) and bool((raw < length).all()), f"row {i}"
            assert torch.unique(raw).numel() == topk, f"row {i}: duplicate index"
            selected = row[raw].sort().values
            reference = torch.topk(row, topk).values.sort().values
            assert torch.equal(selected, reference), f"row {i}: wrong selection"
        valid = raw >= 0
        mapped = torch.full_like(raw, -1)
        mapped[valid] = (
            page_table[i, raw[valid] // PAGE_SIZE].to(torch.int64) * PAGE_SIZE
            + raw[valid] % PAGE_SIZE
        )
        assert torch.equal(out_page[i], mapped), f"row {i}: page mapping"


@pytest.mark.parametrize("topk", TOPKS)
def test_threshold_bin_around_stash_capacity(topk):
    gen = _generator(1)
    below = _tight_cluster(STASH_ENTRIES - 1)
    exact = _tight_cluster(STASH_ENTRIES)
    above = _tight_cluster(STASH_ENTRIES + 1)
    for row in (below, exact, above):
        assert torch.unique(_coarse_keys(row)).numel() == 1
    rows = [
        below,
        exact,
        exact.flip(0),
        exact[torch.randperm(exact.numel(), generator=gen)],
        _tight_cluster(STASH_ENTRIES, base=-0.9),
        above,
        _tight_cluster(3 * STASH_ENTRIES + 17),
    ]
    _run_and_check(rows, topk)


@pytest.mark.parametrize("topk", TOPKS)
def test_oversized_bin_after_strictly_higher_values(topk):
    gen = _generator(2)
    high = 2.0 + torch.arange(37, dtype=torch.float32) * 1e-3
    low = torch.full((257,), -2.0)
    row = torch.cat([high, _tight_cluster(STASH_ENTRIES + 5311), low])
    exact_bin_row = torch.cat([high, _tight_cluster(STASH_ENTRIES), low])
    rows = [row, row[torch.randperm(row.numel(), generator=gen)], exact_bin_row]
    _run_and_check(rows, topk)


@pytest.mark.parametrize("topk", TOPKS)
def test_cutoff_ties(topk):
    gen = _generator(3)
    top = torch.linspace(2.0, 3.0, topk)
    tie_row = torch.cat([top, torch.full((65536,), 1.0)])
    rows = [
        torch.full((STASH_ENTRIES + 1,), 0.75),
        torch.full((STASH_ENTRIES + 1,), 0.0),
        torch.full((STASH_ENTRIES + 1,), -0.0),
        torch.full((topk + 1,), 0.5),
        tie_row[torch.randperm(tie_row.numel(), generator=gen)],
        torch.cat([top[: topk // 2], torch.full((2 * STASH_ENTRIES,), 1.0)]),
    ]
    _run_and_check(rows, topk)


@pytest.mark.parametrize("topk", TOPKS)
def test_clusters_separating_at_each_key_byte(topk):
    gen = _generator(4)
    rows = [_byte_depth_clusters(depth, STASH_ENTRIES + 1, gen) for depth in (16, 8, 0)]
    _run_and_check(rows, topk)


@pytest.mark.parametrize("topk", TOPKS)
def test_short_and_random_rows_with_strided_scores(topk):
    gen = _generator(5)
    rows = [
        torch.rand(1, generator=gen),
        torch.rand(topk - 1, generator=gen),
        torch.rand(topk, generator=gen),
        torch.rand(topk + 1, generator=gen),
        torch.rand(3000, generator=gen),
        torch.randn(40000, generator=gen),
    ]
    _run_and_check(rows, topk, stride_pad=PAGE_SIZE + 3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
