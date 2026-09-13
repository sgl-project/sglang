"""Correctness tests for the DeepSeek-V4.1 JIT bf16 top-k transform.

``topk_transform_bf16_small`` selects the per-row top-k of bf16 ``scores`` within each
row's ``seq_lens`` (rows of at most 16384, ``k`` at most 2048) and writes the
page-table transform of the selected indices, ``-1`` past ``min(k, seq_len)``, in
no particular order. It is the consumer selection of the two-level sparse
indexer: there the "page table" is the per-row physical block table at page
size 8.

Selection is exact, so it is validated against ``torch.topk`` on the multiset of
selected values (elements of equal value may swap). NaN scores are not
supported and never generated here.
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


def _identity_table(batch: int, page_size: int = 8) -> torch.Tensor:
    return torch.arange(MAX_SEQ // page_size, device="cuda", dtype=torch.int32).repeat(
        batch, 1
    )


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


def _run_full_rows(scores: torch.Tensor, k: int) -> None:
    """Whole rows of `scores` (any width up to MAX_SEQ) through the identity table."""
    batch, seq = scores.shape
    lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
    table = _identity_table(batch)
    out = torch.full((batch, k), -7, dtype=torch.int32, device="cuda")
    topk_transform_bf16_small(scores, lens, table, out, 8)
    torch.cuda.synchronize()
    _check(scores, lens, table, 8, out)


def _randn_aligned(batch: int, seq: int) -> torch.Tensor:
    # rows must start vector-aligned, so odd widths come from a wider padded tensor
    padded = (seq + 7) // 8 * 8
    return torch.randn(batch, padded, device="cuda", dtype=torch.bfloat16)[:, :seq]


@pytest.mark.parametrize("page_mode", ["identity", "perm"])
@pytest.mark.parametrize("ties", [False, True])
@pytest.mark.parametrize("k", [512, 2048])
@pytest.mark.parametrize("batch,seq", CONFIGS)
def test_topk_bf16(batch: int, seq: int, k: int, ties: bool, page_mode: str) -> None:
    page_size = 8
    scores, lens = _rows(batch, seq, k, ties, ragged=False)
    nblocks = MAX_SEQ // page_size
    if page_mode == "identity":
        table = _identity_table(batch, page_size)
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
    table = _identity_table(batch, page_size)
    buf = torch.full((batch, k + 64), -7, dtype=torch.int32, device="cuda")
    out = buf[:, :k]
    topk_transform_bf16_small(scores, lens, table, out, page_size)
    torch.cuda.synchronize()
    assert bool((buf[:, k:] == -7).all()), "wrote past the view"
    _check(scores, lens, table, page_size, out)


@pytest.mark.parametrize("k", [1, 512, 2048])
@pytest.mark.parametrize("seq", [1000, 3000, 8191, 12345])
def test_topk_bf16_odd_widths(seq: int, k: int) -> None:
    """Rows whose width is not a multiple of the vector: the last vector is partial."""
    torch.manual_seed(seq * 7 + k)
    _run_full_rows(_randn_aligned(33, seq), k)


@pytest.mark.parametrize(
    "seq,k", [(4096, 1024), (16384, 2048), (16384, 512), (1000, 512)]
)
def test_topk_bf16_heavy_ties(seq: int, k: int) -> None:
    """Heavy ties, all-equal rows and all -inf rows exercise the equal-quota path."""
    torch.manual_seed(seq + k)
    _run_full_rows(torch.randint(0, 5, (32, seq), device="cuda").bfloat16(), k)
    _run_full_rows(torch.randint(-3, 3, (32, seq), device="cuda").bfloat16(), k)
    _run_full_rows(torch.full((32, seq), 1.5, device="cuda", dtype=torch.bfloat16), k)
    _run_full_rows(
        torch.full((32, seq), float("-inf"), device="cuda", dtype=torch.bfloat16), k
    )


@pytest.mark.parametrize("seq,k", [(4096, 1024), (16384, 2048), (16384, 1024)])
def test_topk_bf16_signed_zero(seq: int, k: int) -> None:
    """+0 and -0 compare equal as floats but sit in different histogram bins: every
    mix of pivot sign / neighbours must still fill exactly k slots."""
    torch.manual_seed(seq + k)
    x = torch.randn(32, seq, device="cuda", dtype=torch.bfloat16)
    zero = torch.zeros_like(x)
    neg_zero = -zero
    c = torch.rand(32, seq, device="cuda")
    _run_full_rows(
        torch.where(c < 0.45, zero, torch.where(c < 0.9, neg_zero, x.abs())), k
    )
    _run_full_rows(torch.where(c < 0.98, neg_zero, x.abs()), k)  # pivot is -0
    _run_full_rows(torch.where(c < 0.98, zero, x.abs()), k)  # pivot is +0
    _run_full_rows(
        torch.where(c < 0.5, neg_zero, torch.where(c < 0.98, zero, x.abs())), k
    )
    _run_full_rows(
        torch.where(c < 0.9, -x.abs(), torch.where(c < 0.95, neg_zero, zero)), k
    )


@pytest.mark.parametrize("seq,k", [(4096, 1024), (16384, 2048)])
def test_topk_bf16_bit_patterns(seq: int, k: int) -> None:
    """Denormals, and the full bf16 range minus NaN."""
    torch.manual_seed(seq + k)
    _run_full_rows(
        torch.randint(0, 64, (32, seq), dtype=torch.int16, device="cuda").view(
            torch.bfloat16
        ),
        k,
    )
    x = torch.randint(
        -(2**15), 2**15, (32, seq), dtype=torch.int16, device="cuda"
    ).view(torch.bfloat16)
    _run_full_rows(torch.where(x.isnan(), torch.zeros_like(x), x), k)
    _run_full_rows(
        torch.randint(0, 0x7F80, (32, seq), dtype=torch.int16, device="cuda").view(
            torch.bfloat16
        ),
        k,
    )


@pytest.mark.parametrize(
    "nan_bits,n_nan", [(0x7FC0, 5), (0x7FC0, 100), (0x7FC0, 600), (0xFFC0, 600)]
)
def test_topk_bf16_nan_scores(nan_bits: int, n_nan: int) -> None:
    """NaN scores are never selected: the slots they would have taken are -1, the
    rest are the top of the real scores, and nothing reads stale shared memory."""
    torch.manual_seed(nan_bits + n_nan)
    batch, seq, k = 8, MAX_SEQ, 512
    scores = (torch.randn(batch, seq, device="cuda") * 2).to(torch.bfloat16)
    bits = scores.view(
        torch.int16
    )  # write the NaN by bit pattern: .item() would lose its sign
    for b in range(batch):
        bits[b, torch.randperm(seq, device="cuda")[:n_nan]] = nan_bits - (
            0x10000 if nan_bits >= 0x8000 else 0
        )
    lens = torch.full((batch,), seq, dtype=torch.int32, device="cuda")
    table = _identity_table(batch)
    out = torch.full((batch, k), -7, dtype=torch.int32, device="cuda")
    topk_transform_bf16_small(scores, lens, table, out, 8)
    torch.cuda.synchronize()
    for b in range(batch):
        real = scores[b][~scores[b].isnan()].float()
        n_real = (
            max(0, k - n_nan) if nan_bits < 0x8000 else k
        )  # positive NaNs eat slots
        chosen = out[b] >= 0
        assert int(chosen.sum()) == n_real, (
            f"row {b}: {int(chosen.sum())} selected, want {n_real}"
        )
        assert bool((out[b][~chosen] == -1).all()), (
            f"row {b}: unselected slots are not -1"
        )
        idx = out[b][chosen].long()
        assert bool((idx < seq).all()) and idx.unique().numel() == n_real, (
            f"row {b}: bad index"
        )
        got = scores[b, idx].float().sort(descending=True).values
        assert torch.equal(got, real.topk(n_real).values), (
            f"row {b}: not the top real scores"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
