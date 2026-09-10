"""Correctness tests for the DeepSeek-V4.1 JIT bf16 top-k transform.

``topk_transform_bf16_small`` selects the per-row top-k of bf16 ``scores`` within each
row's ``seq_lens`` (rows of at most 16384) and writes the page-table transform of
the selected indices, ``-1`` past ``min(k, seq_len)``, in no particular order.
It is the consumer selection of the two-level sparse indexer: there the "page
table" is the per-row physical block table at page size 8.

Selection uses a 13-bit fp16-derived key, exact for bf16 in fp16's normal
range, so it is validated against ``torch.topk`` on the multiset of selected
values (elements of equal value may swap).
"""

from __future__ import annotations

import sys

import pytest
import torch

from sglang.kernels.ops.attention.dsv4.topk import topk_transform_bf16_small
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

MAX_SEQ = 16384
CONFIGS = [
    # (batch, seq): short rows select everything, full rows the 16384 maximum
    (1, 1),
    (4, 300),
    (8, 512),
    (8, 513),
    (16, 4097),
    (32, 16000),
    (32, MAX_SEQ),
    (148, MAX_SEQ),
    (300, MAX_SEQ),
]


def _rows(batch: int, seq: int, k: int, ties: bool, ragged: bool):
    torch.manual_seed(batch * 131 + seq)
    lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
    if ragged:
        lens = torch.randint(1, seq + 1, (batch,), dtype=torch.int32, device="cuda")
        lens[0] = seq
    scores = torch.randn(batch, MAX_SEQ, device="cuda") * 2
    if ties:
        scores = (scores * 2).round() / 2  # a handful of distinct values per row
    # everything past a row is garbage that must never be selected
    scores.masked_fill_(
        torch.arange(MAX_SEQ, device="cuda")[None, :] >= lens[:, None], 1e4
    )
    return scores.to(torch.bfloat16), lens


def _check(scores, lens, table, page_size, out):
    """Every row: min(k, len) valid slots, the rest -1; slots invert to unique
    in-row indices whose values are torch's top-k values."""
    k = out.shape[1]
    nblocks = MAX_SEQ // page_size
    inv = torch.empty_like(table)
    inv.scatter_(
        1,
        table.long(),
        torch.arange(nblocks, device="cuda", dtype=torch.int32).expand_as(table),
    )
    for b in range(scores.shape[0]):
        n = min(k, int(lens[b]))
        chosen = out[b] >= 0
        assert int(chosen.sum()) == n, (
            f"row {b}: {int(chosen.sum())} selected, want {n}"
        )
        slots = out[b][chosen].long()
        idx = inv[b][slots // page_size].long() * page_size + slots % page_size
        assert bool((idx < int(lens[b])).all()), f"row {b}: index past the row"
        assert idx.unique().numel() == n, f"row {b}: duplicate index"
        got = scores[b, idx].float().sort(descending=True).values
        ref = scores[b, : int(lens[b])].float().topk(n).values
        assert torch.equal(got, ref), f"row {b}: selected values differ from torch.topk"


@pytest.mark.parametrize("page_mode", ["identity", "perm"])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("batch,seq", CONFIGS)
def test_topk_bf16(
    batch: int, seq: int, k: int, ties: bool, page_mode: str
) -> None:
    page_size = 8
    scores, lens = _rows(batch, seq, k, ties, ragged=False)
    nblocks = MAX_SEQ // page_size
    if page_mode == "identity":
        table = torch.arange(nblocks, device="cuda", dtype=torch.int32).repeat(batch, 1)
    else:
        table = torch.stack(
            [torch.randperm(nblocks, device="cuda") for _ in range(batch)]
        ).to(torch.int32)
    out = torch.full((batch, k), -7, dtype=torch.int32, device="cuda")
    topk_transform_bf16_small(scores, lens, table, out, page_size)
    torch.cuda.synchronize()
    _check(scores, lens, table, page_size, out)


@pytest.mark.parametrize("page_size", [8, 64])
def test_topk_bf16_ragged_lengths(page_size: int) -> None:
    batch, k = 64, 512
    scores, lens = _rows(batch, MAX_SEQ, k, ties=False, ragged=True)
    nblocks = MAX_SEQ // page_size
    table = torch.stack(
        [torch.randperm(nblocks, device="cuda") for _ in range(batch)]
    ).to(torch.int32)
    out = torch.full((batch, k), -7, dtype=torch.int32, device="cuda")
    topk_transform_bf16_small(scores, lens, table, out, page_size)
    torch.cuda.synchronize()
    _check(scores, lens, table, page_size, out)


def test_topk_bf16_padded_output() -> None:
    """The output may be a view whose last dimension was padded."""
    batch, k, page_size = 3, 512, 8
    scores, lens = _rows(batch, MAX_SEQ, k, ties=False, ragged=False)
    table = torch.arange(MAX_SEQ // page_size, device="cuda", dtype=torch.int32).repeat(
        batch, 1
    )
    buf = torch.full((batch, k + 64), -7, dtype=torch.int32, device="cuda")
    out = buf[:, :k]
    topk_transform_bf16_small(scores, lens, table, out, page_size)
    torch.cuda.synchronize()
    assert bool((buf[:, k:] == -7).all()), "wrote past the view"
    _check(scores, lens, table, page_size, out)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
