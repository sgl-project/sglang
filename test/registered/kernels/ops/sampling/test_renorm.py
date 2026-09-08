# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/main/tests/test_sampling.py
# and /sgl-workspace/sglang/python/sglang/kernels/aot/tests/test_sampling.py

import os
import sys

import pytest
import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd-mi35x")

if is_hip():
    from sglang.kernels.ops.sampling.renorm_triton import (
        top_k_renorm_probs_triton as top_k_renorm_prob,
    )
    from sglang.kernels.ops.sampling.renorm_triton import (
        top_p_renorm_probs_triton as top_p_renorm_prob,
    )
else:
    from sglang.srt.layers.sampling_renorm import (
        top_k_renorm_prob,
        top_p_renorm_prob,
    )


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


def _default_is_deterministic() -> bool:
    return os.environ.get("SGLANG_RENORM_DETERMINISTIC", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _flat_distribution(batch_size: int, vocab_size: int) -> torch.Tensor:
    """A heavy-tailed, high-entropy distribution: thousands of tokens survive a
    top-p 0.95 cutoff, so the renorm kernels' histogram sums have many terms.
    This is the regime in which the flashinfer float-atomic kernels return
    different bytes call to call (see SGLANG_RENORM_DETERMINISTIC)."""
    gen = torch.Generator(device="cuda:0").manual_seed(7)
    ranks = torch.arange(1, vocab_size + 1, device="cuda:0", dtype=torch.float32)
    zipf = -1.1 * torch.log(ranks)
    logits = torch.empty(batch_size, vocab_size, device="cuda:0")
    for r in range(batch_size):
        perm = torch.randperm(vocab_size, device="cuda:0", generator=gen)
        logits[r] = zipf[perm] + 0.6 * torch.randn(
            vocab_size, device="cuda:0", generator=gen
        )
    return torch.softmax(logits, dim=-1)


@pytest.mark.skipif(is_hip(), reason="CUDA-only: exercises the flashinfer dispatch")
@pytest.mark.parametrize("batch_size,vocab_size", [(256, 128256)])
def test_top_p_renorm_probs_is_deterministic(batch_size, vocab_size):
    """Repeated calls on the same input must return bit-identical output.

    Every TP rank runs this kernel independently on the same logits, and its
    output feeds sampled tokens and speculative accept decisions that are
    committed to per-rank KV/radix state. A last-bit difference between ranks
    desynchronizes them and eventually deadlocks a collective (#33549, #33289).
    """
    probs = _flat_distribution(batch_size, vocab_size)
    top_p = torch.full((batch_size,), 0.95, device="cuda:0")
    ref = top_p_renorm_prob(probs, top_p, deterministic=True)
    for _ in range(30):
        out = top_p_renorm_prob(probs, top_p, deterministic=True)
        assert torch.equal(out, ref), "top_p_renorm_prob output changed between calls"
    # the default path is the deterministic one unless the env opts out
    if _default_is_deterministic():
        assert torch.equal(top_p_renorm_prob(probs, top_p), ref)


@pytest.mark.skipif(is_hip(), reason="CUDA-only: exercises the flashinfer dispatch")
@pytest.mark.parametrize("batch_size,vocab_size,k", [(256, 128256, 40)])
def test_top_k_renorm_probs_is_deterministic(batch_size, vocab_size, k):
    """Same invariant for top-k renorm; flashinfer's radix multi-CTA kernel
    accumulates the kept mass with float atomics and is not reproducible."""
    probs = _flat_distribution(batch_size, vocab_size)
    top_k = torch.full((batch_size,), k, device="cuda:0", dtype=torch.int32)
    ref = top_k_renorm_prob(probs, top_k, deterministic=True)
    for _ in range(30):
        out = top_k_renorm_prob(probs, top_k, deterministic=True)
        assert torch.equal(out, ref), "top_k_renorm_prob output changed between calls"
    if _default_is_deterministic():
        assert torch.equal(top_k_renorm_prob(probs, top_k), ref)
    # and the deterministic kernel must agree with the fast one up to rounding
    fast = top_k_renorm_prob(probs, top_k, deterministic=False)
    torch.testing.assert_close(fast, ref, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
