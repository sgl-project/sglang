# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/main/tests/test_sampling.py
# and /sgl-workspace/sglang/sgl-kernel/tests/test_sampling.py

import sys

import pytest
import sgl_kernel
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, suite="stage-b-kernel-unit-1-gpu-large")
register_cuda_ci(est_time=120, suite="nightly-kernel-1-gpu", nightly=True)


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

    renorm_prob = sgl_kernel.top_k_renorm_prob(normalized_prob, k)
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

    renorm_prob = sgl_kernel.top_p_renorm_prob(normalized_prob, p)
    torch.testing.assert_close(
        renorm_prob_ground_truth,
        renorm_prob,
        rtol=1e-3,
        atol=1e-3,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
