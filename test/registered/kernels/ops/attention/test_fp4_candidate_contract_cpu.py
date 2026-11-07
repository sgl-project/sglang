"""Portable negative controls for the declared candidate acceptance contract."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels.fp4_candidate import (
    SUPPORTED_TOPK,
    assert_topk_intervals,
    assert_topk_ties,
    check,
    make_inputs,
    oracle,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def providers(reference, actual):
    return {
        "triton": lambda *_: reference,
        "cuda": lambda *_: actual,
    }


def test_real_oracle_rejects_outside_interval(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    x = make_inputs(
        requests=1, queries=1, candidates=8, max_len=8, analytic=True, device="cpu"
    )
    expected, _, _, _ = oracle(x)
    reference = expected.float()
    wrong = reference.clone()
    wrong[0, 0] = 1.0e6
    with pytest.raises(AssertionError, match="independent FP64 interval failure"):
        check(x, providers(reference, wrong))


@pytest.mark.parametrize("k", SUPPORTED_TOPK)
def test_topk_rejects_below_cutoff(k):
    expected = torch.arange(2048, dtype=torch.float32)[None, :]
    wrong = expected.clone()
    wrong[0, 0] = 4096
    with pytest.raises(AssertionError, match="selected a below-cutoff score"):
        assert_topk_ties(wrong, expected, k=k)


@pytest.mark.parametrize("k", SUPPORTED_TOPK)
def test_topk_rejects_omitted_strictly_above_cutoff(k):
    expected = torch.ones((1, k + 2))
    expected[0, 0] = 2
    wrong = expected.clone()
    wrong[0, 0] = 0
    with pytest.raises(AssertionError, match="omitted an above-cutoff score"):
        assert_topk_ties(wrong, expected, k=k)


@pytest.mark.parametrize("k", SUPPORTED_TOPK)
def test_topk_allows_equal_cutoff_alternative(k):
    expected = torch.ones((1, k + 2))
    expected[0, 0] = 2
    original = expected.topk(k).indices[0]
    excluded = (~torch.isin(torch.arange(k + 2), original)).nonzero()[0, 0]
    alternative = expected.clone()
    alternative[0, excluded] = 1.25
    assert not torch.equal(
        original.sort().values, alternative.topk(k).indices[0].sort().values
    )
    assert_topk_ties(alternative, expected, k=k)


@pytest.mark.parametrize("k", SUPPORTED_TOPK)
def test_oracle_selection_rejects_provably_worse_choice(k):
    lower = torch.zeros((1, k + 1), dtype=torch.float64)
    lower[0, 0] = 2
    upper = lower.clone()
    wrong = torch.ones((1, k + 1))
    wrong[0, 0] = 0
    with pytest.raises(AssertionError, match="FP64 selection intervals"):
        assert_topk_intervals(wrong, lower, upper, k)


def test_exactness_report_and_optional_signed_zero_gate(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    x = make_inputs(
        requests=1, queries=1, candidates=8, max_len=8, analytic=True, device="cpu"
    )
    x.q.zero_()
    expected, _, _, _ = oracle(x)
    reference = expected.float()
    negative_zero = reference.clone()
    negative_zero[0, 0] = -0.0
    report = check(x, providers(reference, negative_zero))
    assert report["cuda_triton_exact"] is False
    assert report["cuda_triton_different_elements"] == 1
    assert report["cuda_triton_signed_zero_differences"] == 1
    assert report["oracle_columns"] == "all"
    assert report["topk"] == [512, 1024]
    with pytest.raises(AssertionError, match="CUDA/Triton bitwise mismatch"):
        check(x, providers(reference, negative_zero), require_bitwise=True)
    assert check(x, providers(reference, reference))["cuda_triton_exact"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
