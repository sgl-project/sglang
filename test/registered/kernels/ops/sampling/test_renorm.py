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


def _make_probs(kind, batch_size, vocab_size):
    g = torch.Generator(device="cuda:0").manual_seed(42)
    if kind == "uniform_rand":
        x = torch.rand(batch_size, vocab_size, device="cuda:0", generator=g)
        return x / x.sum(dim=-1, keepdim=True)
    if kind.startswith("softmax"):
        scale = float(kind.split("_")[1])
        logits = torch.randn(batch_size, vocab_size, device="cuda:0", generator=g)
        return torch.softmax(logits * scale, dim=-1)
    if kind == "one_hot":
        x = torch.zeros(batch_size, vocab_size, device="cuda:0")
        x[:, 0] = 1.0
        return x
    if kind == "all_equal":
        return torch.full((batch_size, vocab_size), 1.0 / vocab_size, device="cuda:0")
    if kind == "sparse_ties":
        x = torch.randint(0, 4, (batch_size, vocab_size), device="cuda:0", generator=g)
        x = x.float()
        x[:, 0] += 1.0
        return x / x.sum(dim=-1, keepdim=True)
    if kind == "tiny":
        x = torch.rand(batch_size, vocab_size, device="cuda:0", generator=g) * 1e-30
        return x
    raise ValueError(kind)


def _check_pivot_definition(probs, top_ps, pivots, eps=1e-6):
    """pivot = min{v in row : sum(x <= v) >= 1 - p}, checked in fp64 on the row values."""
    x = probs.double()
    target = (1.0 - top_ps.double()).unsqueeze(1)
    v = pivots.double().unsqueeze(1)
    # With 1 - p <= 0 every entry is kept; the pivot may then be 0 rather than a row value.
    keep_all = (target <= 0).squeeze(1)
    assert bool((x[keep_all] >= v[keep_all]).all()), "top_p >= 1 must keep every entry"
    x, target, v = x[~keep_all], target[~keep_all], v[~keep_all]
    if x.shape[0] == 0:
        return
    assert bool((x == v).any(dim=1).all()), "pivot must be a value of its row"
    mass_le = torch.where(x <= v, x, 0.0).sum(dim=1, keepdim=True)
    below = (
        torch.where(x < v, x, torch.full_like(x, -1.0)).max(dim=1, keepdim=True).values
    )
    mass_le_below = torch.where(x <= below, x, 0.0).sum(dim=1, keepdim=True)
    has_below = below >= 0
    total = x.sum(dim=1, keepdim=True)
    reaches = (mass_le >= target - eps) | (v == x.max(dim=1, keepdim=True).values)
    minimal = (
        ~has_below | (mass_le_below < target + eps) | (target <= 0) | (total < target)
    )
    assert bool(reaches.all()), "kept mass below the top_p budget"
    assert bool(minimal.all()), "a smaller pivot would already reach the budget"


@pytest.mark.skipif(not is_hip(), reason="radix-select pivot is the ROCm Triton path")
@pytest.mark.parametrize("batch_size", [1, 6, 99])
@pytest.mark.parametrize("vocab_size", [111, 32000, 154880])
@pytest.mark.parametrize("p", [1e-6, 0.1, 0.5, 0.9, 0.95, 1.0])
@pytest.mark.parametrize(
    "kind",
    [
        "uniform_rand",
        "softmax_1",
        "softmax_8",
        "one_hot",
        "all_equal",
        "sparse_ties",
        "tiny",
    ],
)
def test_top_p_pivots_radix(batch_size, vocab_size, p, kind):
    """The radix-select pivot satisfies the top-p pivot definition exactly (fp64 check)."""
    from sglang.kernels.ops.sampling.topp_radix_triton import top_p_pivots_radix

    probs = _make_probs(kind, batch_size, vocab_size).float().contiguous()
    top_ps = torch.full((batch_size,), p, device="cuda:0")
    pivots = top_p_pivots_radix(probs, top_ps)
    _check_pivot_definition(probs, top_ps, pivots)


@pytest.mark.skipif(not is_hip(), reason="radix-select pivot is the ROCm Triton path")
@pytest.mark.parametrize("batch_size", [1, 6, 99])
@pytest.mark.parametrize("vocab_size", [111, 154880])
@pytest.mark.parametrize("p", [0.5, 0.95])
def test_top_p_renorm_probs_radix_env(monkeypatch, batch_size, vocab_size, p):
    """SGLANG_TOPP_RADIX=1 routes top_p_renorm_probs_triton through the radix pivot."""
    from sglang.kernels.ops.sampling.renorm_triton import top_p_renorm_probs_triton

    probs = _make_probs("softmax_8", batch_size, vocab_size).float().contiguous()
    top_ps = torch.full((batch_size,), p, device="cuda:0")
    monkeypatch.setenv("SGLANG_TOPP_RADIX", "0")
    ref = top_p_renorm_probs_triton(probs, top_ps)
    monkeypatch.setenv("SGLANG_TOPP_RADIX", "1")
    out = top_p_renorm_probs_triton(probs, top_ps)

    kept = out > 0
    pivots = torch.where(kept, probs, torch.full_like(probs, 2.0)).min(dim=1).values
    _check_pivot_definition(probs, top_ps, pivots)
    torch.testing.assert_close(
        out.sum(dim=1), torch.ones_like(out[:, 0]), rtol=0, atol=1e-5
    )
    assert (kept != (ref > 0)).sum(dim=1).max().item() <= 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
