# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/main/tests/test_sampling.py
# and /sgl-workspace/sglang/python/sglang/kernels/aot/tests/test_sampling.py

import sys

import pytest
import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")
register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")

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


@pytest.mark.skipif(not is_hip(), reason="checks the HIP Triton pivot search")
@pytest.mark.parametrize("batch_size", [1, 48, 384])
@pytest.mark.parametrize("vocab_size", [33, 4096, 154880])
@pytest.mark.parametrize("scale", [2.0, 8.0])
@pytest.mark.parametrize("ties", [False, True])
def test_top_p_renorm_probs_peaked(batch_size, vocab_size, scale, ties):
    """Per-row top_p on softmax rows: the pivot keeps every tie at its value."""
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda:0") * scale
    if ties:
        logits = logits.round()
    probs = torch.softmax(logits, dim=-1)
    top_ps = torch.rand(batch_size, device="cuda:0") * 0.99 + 0.01
    top_ps[0] = 1.0
    sorted_prob = torch.sort(probs, dim=-1).values
    cdf = torch.cumsum(sorted_prob, dim=-1)
    cutoff = torch.searchsorted(cdf, (1 - top_ps).unsqueeze(1)).clamp_(
        max=vocab_size - 1
    )
    pivot = sorted_prob.gather(1, cutoff)
    renorm_prob_ground_truth = torch.where(probs >= pivot, probs, 0.0)
    renorm_prob_ground_truth /= renorm_prob_ground_truth.sum(dim=-1, keepdim=True)

    renorm_prob = top_p_renorm_prob(probs, top_ps)
    torch.testing.assert_close(
        renorm_prob_ground_truth,
        renorm_prob,
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.skipif(not is_hip(), reason="checks the HIP Triton pivot search")
@pytest.mark.parametrize("batch_size", [1, 48])
@pytest.mark.parametrize("vocab_size", [4096, 154880])
@pytest.mark.parametrize("scale", [0.0, 0.01, 1.0, 4.0])
@pytest.mark.parametrize("top_p", [1e-7, 1e-5, 1e-4, 1e-2, 0.5, 0.95])
def test_top_p_renorm_probs_support_and_sum(batch_size, vocab_size, scale, top_p):
    """Rows sum to 1 and keep exactly the reference tokens.

    A small top_p keeps a few tokens out of a cumulative mass near 1, so the
    denominator must come from the kept tokens, not from a difference of two
    sums near 1.
    """
    torch.manual_seed(0)
    logits = torch.randn(batch_size, vocab_size, device="cuda:0") * scale
    probs = torch.softmax(logits, dim=-1)
    top_ps = torch.full((batch_size,), top_p, device="cuda:0")

    renorm_prob = top_p_renorm_prob(probs, top_ps)

    torch.testing.assert_close(
        renorm_prob.double().sum(dim=-1),
        torch.ones(batch_size, dtype=torch.float64, device="cuda:0"),
        rtol=0,
        atol=1e-5,
    )

    sorted_prob = torch.sort(probs.double(), dim=-1).values
    cdf = torch.cumsum(sorted_prob, dim=-1)
    target = 1.0 - top_ps.double().unsqueeze(1)
    cutoff = torch.searchsorted(cdf, target).clamp_(max=vocab_size - 1)
    pivot = sorted_prob.gather(1, cutoff)
    # Within fp32 rounding of a cdf step the kept set may differ by one token.
    margin = torch.minimum(
        (cdf.gather(1, cutoff) - target).abs(),
        (cdf.gather(1, (cutoff - 1).clamp(min=0)) - target).abs(),
    ).squeeze(1)
    clear = margin > 1e-5
    expected = probs.double() >= pivot
    assert torch.equal((renorm_prob > 0)[clear], expected[clear])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
