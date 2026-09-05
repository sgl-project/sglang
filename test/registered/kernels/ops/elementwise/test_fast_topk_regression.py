"""Regression tests for sgl-project/sglang#36807 (JIT fast_topk overflow).

A concentrated row (e.g. 10.0 + 0.1*N(0,1)) puts ~16-34K candidates into the
coarse fp16 threshold bin, far past the 4096-entry smem staging buffer. The
buggy kernel silently drops the overflow (plus its fused histogram counts) and
still returns exactly-k in-range, non-duplicate indices — wrong and unstable.

Place in test/registered/kernels/ops/elementwise/ alongside test_fast_topk.py
(same _check_topk_values semantics), or run standalone with:
    PYTHONPATH=<sglang-src>/python pytest test_fast_topk_regression.py

Before the fix these tests FAIL (wrong rows / unstable selections); after the
exact-fallback fix they PASS.
"""
import pytest
import torch

from sglang.kernels.ops.elementwise.fast_topk import fast_topk
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _check_topk_values(score, lengths, indices, topk, row_starts):
    """Selected indices must pick exactly the top-k value multiset.

    Order is unspecified and tie-breaking may differ from torch.topk, so we
    compare sorted score values, not index sets. Mirrors test_fast_topk.py.
    """
    for b in range(score.shape[0]):
        start = int(row_starts[b]) if row_starts is not None else 0
        length = int(lengths[b])
        section = score[b, start : start + length]
        row = indices[b]
        if length <= topk:
            assert torch.equal(
                row[:length].cpu(), torch.arange(length, dtype=torch.int32)
            )
            assert (row[length:] == -1).all()
            continue
        assert (row >= 0).all(), "long rows must fill every slot"
        picked = section[row.long()]
        expected = torch.topk(section, topk).values
        assert torch.equal(
            picked.sort(descending=True).values, expected.sort(descending=True).values
        ), f"row {b}: top-{topk} value multiset mismatch"


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("length", [30938, 65504])
def test_fast_topk_threshold_bin_overflow(topk, length):
    # Regression for sgl-project/sglang#36807 (mochgolf): concentrated scores
    # overflow the 4096-entry staging buffer; selection must stay exact.
    torch.manual_seed(0)
    batch = 4
    score = 10.0 + 0.1 * torch.randn(batch, length, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_non_overflow_control(topk):
    # N(0,1) rows at the same length never overflow: selection stays exact.
    torch.manual_seed(0)
    batch, length = 4, 65504
    score = torch.randn(batch, length, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize("length", [30938, 65504])
def test_fast_topk_overflow_stable_across_runs(topk, length):
    # The overflow used to be unstable (which candidates survive depends on
    # atomic arrival order). The fixed kernel must pick the same set every run.
    torch.manual_seed(0)
    batch = 4
    score = 10.0 + 0.1 * torch.randn(batch, length, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    first = (
        score.gather(1, fast_topk(score, lengths, topk).long())
        .sort(dim=1, descending=True)
        .values
    )
    for _ in range(5):
        got = (
            score.gather(1, fast_topk(score, lengths, topk).long())
            .sort(dim=1, descending=True)
            .values
        )
        assert torch.equal(got, first), "top-k selection unstable across runs"
