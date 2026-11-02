# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/main/tests/test_sampling.py
# and /sgl-workspace/sglang/python/sglang/kernels/aot/tests/test_sampling.py

import sys

import pytest
import torch

from sglang.srt.runtime_context import get_context, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")
register_amd_ci(est_time=20, stage="jit-kernel-unit", runner_config="amd")


@pytest.fixture(autouse=True)
def runtime_context():
    """The renorm policy reads the published exec / parallel config (tp_size=1
    here: single rank, so the default path picks the fast kernels unless a test
    overrides tp_size)."""
    publish(ServerArgs(model_path="dummy"), role="test")
    try:
        yield
    finally:
        reset_context()


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


def _flat_distribution(batch_size: int, vocab_size: int) -> torch.Tensor:
    """High-entropy rows: the regime where the float-atomic kernels return different bytes."""
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
    """Repeated calls on the same input return bit-identical output."""
    probs = _flat_distribution(batch_size, vocab_size)
    top_p = torch.full((batch_size,), 0.95, device="cuda:0")
    ref = top_p_renorm_prob(probs, top_p, deterministic=True)
    for _ in range(30):
        out = top_p_renorm_prob(probs, top_p, deterministic=True)
        assert torch.equal(out, ref), "top_p_renorm_prob output changed between calls"
    # the default path on a multi-rank config must reach the same kernel
    with get_context().override_server_args(tp_size=2):
        assert torch.equal(top_p_renorm_prob(probs, top_p), ref)


@pytest.mark.skipif(is_hip(), reason="CUDA-only: exercises the flashinfer dispatch")
@pytest.mark.parametrize("batch_size,vocab_size,k", [(256, 128256, 40)])
def test_top_k_renorm_probs_is_deterministic(batch_size, vocab_size, k):
    """Same invariant for top-k; the radix multi-CTA kernel sums with float atomics."""
    probs = _flat_distribution(batch_size, vocab_size)
    top_k = torch.full((batch_size,), k, device="cuda:0", dtype=torch.int32)
    ref = top_k_renorm_prob(probs, top_k, deterministic=True)
    for _ in range(30):
        out = top_k_renorm_prob(probs, top_k, deterministic=True)
        assert torch.equal(out, ref), "top_k_renorm_prob output changed between calls"
    with get_context().override_server_args(tp_size=2):
        assert torch.equal(top_k_renorm_prob(probs, top_k), ref)
    # default path agrees with the deterministic one up to rounding
    fast = top_k_renorm_prob(probs, top_k, deterministic=False)
    torch.testing.assert_close(fast, ref, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
