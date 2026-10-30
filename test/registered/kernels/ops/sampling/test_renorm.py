# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/main/tests/test_sampling.py
# and /sgl-workspace/sglang/python/sglang/kernels/aot/tests/test_sampling.py

import sys

import pytest
import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

if is_hip():
    from sglang.kernels.ops.sampling.renorm_triton import (
        top_k_renorm_probs_triton as top_k_renorm_prob,
    )
    from sglang.kernels.ops.sampling.renorm_triton import (
        top_p_renorm_probs_triton as top_p_renorm_prob,
    )
else:
    from sgl_kernel import top_k_renorm_prob, top_p_renorm_prob


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("k", [10, 100, 500])
def test_top_k_renorm_probs(batch_size, vocab_size, k):
    """Test top_k_renorm_probs kernel for correctness.

    This test validates that the kernel correctly:
    1. Identifies the top-k probabilities
    2. Masks out non-top-k values
    3. Renormalizes the remaining probabilities to sum to 1
    """
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, _ = torch.sort(normalized_prob, descending=True)
    pivot = sorted_prob[:, k - 1]
    mask = (normalized_prob >= pivot.unsqueeze(-1)).int()
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        dim=-1, keepdim=True
    )

    renorm_prob = top_k_renorm_prob(normalized_prob, k)
    for i in range(batch_size):
        torch.testing.assert_close(
            renorm_prob_ground_truth[i],
            renorm_prob[i],
            rtol=1e-3,
            atol=1e-3,
        )


@pytest.mark.parametrize("batch_size", [1, 99, 989])
@pytest.mark.parametrize("vocab_size", [111, 32000, 128256])
@pytest.mark.parametrize("p", [0.1, 0.5, 0.9])
def test_top_p_renorm_probs(batch_size, vocab_size, p):
    """Test top_p_renorm_probs kernel for correctness.

    This test validates that the kernel correctly:
    1. Computes the cumulative probability distribution
    2. Identifies tokens in the top-p threshold
    3. Masks out tokens outside the threshold
    4. Renormalizes the remaining probabilities to sum to 1
    """
    torch.manual_seed(42)
    pre_norm_prob = torch.rand(batch_size, vocab_size, device="cuda:0")
    normalized_prob = pre_norm_prob / pre_norm_prob.sum(dim=-1, keepdim=True)
    sorted_prob, indices = torch.sort(normalized_prob, descending=False)
    cdf = torch.cumsum(sorted_prob, dim=-1)
    mask = torch.zeros(batch_size, vocab_size, dtype=torch.int32, device="cuda:0")
    mask.scatter_add_(1, indices, (cdf >= (1 - p)).int())
    renorm_prob_ground_truth = normalized_prob.clone()
    renorm_prob_ground_truth[mask == 0] = 0
    renorm_prob_ground_truth = renorm_prob_ground_truth / renorm_prob_ground_truth.sum(
        dim=-1, keepdim=True
    )

    renorm_prob = top_p_renorm_prob(normalized_prob, p)
    torch.testing.assert_close(
        renorm_prob_ground_truth,
        renorm_prob,
        rtol=1e-3,
        atol=1e-3,
    )


def _top_p_float64(probs: torch.Tensor, top_ps: torch.Tensor) -> torch.Tensor:
    """FlashInfer's top-p (keep every probability at or above the pivot, ties included),
    the pivot found on a float64 cumulative mass."""
    sorted_probs = torch.sort(probs.double(), dim=-1).values
    cdf = torch.cumsum(sorted_probs, dim=-1)
    cutoff = torch.searchsorted(cdf, (1.0 - top_ps.double()).unsqueeze(1))
    pivots = sorted_probs.gather(1, cutoff.clamp(max=probs.shape[-1] - 1))
    kept = torch.where(probs.double() >= pivots, probs.double(), 0.0)
    return (kept / kept.sum(-1, keepdim=True)).float()


@pytest.mark.skipif(not is_hip(), reason="the fused kernel serves the ROCm path")
@pytest.mark.parametrize("rows_per_request", [5, 6])
# one request fits the CUs: one launch, whose work words must be zero again after it
@pytest.mark.parametrize("requests", [1, 4])
def test_softmax_top_p_matches_softmax_then_top_p(rows_per_request, requests):
    """A row split over many blocks keeps the same tokens as softmax then top-p, and
    without top-p it is the temperature softmax, launch after launch."""
    from sglang.kernels.ops.sampling.softmax_top_p import softmax_top_p_probs

    torch.manual_seed(0)
    vocab = 129280
    rows = requests * rows_per_request
    # strided rows, as a captured logits buffer hands them over
    logits = (torch.randn(rows, vocab + 64, device="cuda:0") * 6)[:, :vocab]
    for temps, top_ps in (
        ([0.3, 0.7, 1.0, 2.5], [0.95, 0.5, 1.0, 0.9]),
        ([2.5, 1.0, 0.7, 0.3], [0.9, 0.95, 0.5, 1.0]),
    ):
        temps = torch.tensor(temps[:requests], device="cuda:0")
        top_ps = torch.tensor(top_ps[:requests], device="cuda:0")
        probs = torch.softmax(
            logits / temps.repeat_interleave(rows_per_request)[:, None], -1
        )
        soft = softmax_top_p_probs(logits, temps, None, rows_per_request)
        torch.testing.assert_close(soft, probs, rtol=1e-4, atol=1e-7)
        expected = _top_p_float64(probs, top_ps.repeat_interleave(rows_per_request))
        got = softmax_top_p_probs(logits, temps, top_ps, rows_per_request)
        assert torch.equal(got > 0, expected > 0)
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-7)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
