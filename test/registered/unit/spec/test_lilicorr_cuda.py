"""GPU parity tests for the LiLiCorr Triton kernels.

The CPU suite in ``test_lilicorr.py`` cannot reach these: both kernel entry
points return a value-identical torch implementation for non-CUDA input, so on a
CPU runner every assertion there exercises the reference path and the Triton code
is never compiled. These tests pin the two against each other on device, which is
the only place the claim "value-identical" is actually checked.

Both kernels are exact rather than approximate, so the assertions are equality,
not tolerance, wherever the input dtype makes equality meaningful.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.speculative.lilicorr import (
    _greedy_path_torch,
    _topk_lse_torch,
    lilicorr_greedy_path,
    lilicorr_topk_lse,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="LiLiCorr Triton kernels require CUDA"
)


# --- the tiled top-k + log-partition ---------------------------------------


# 151936 is Qwen3's vocabulary. The rest straddle the kernel's 1024-wide tile
# boundary in both directions, because the tile pre-selection argument is where
# an off-by-one would hide: a vocabulary that is an exact multiple of the tile
# width, one element past it, and one element short of it.
@pytest.mark.parametrize("vocab", [1024, 1025, 2047, 8192, 8193, 151936])
def test_tiled_topk_lse_is_exact_against_the_reference(vocab):
    torch.manual_seed(0)
    logits = torch.randn(6, vocab, device="cuda", dtype=torch.float32)

    vals, ids, lse = lilicorr_topk_lse(logits, 8)
    ref_vals, ref_ids, ref_lse = _topk_lse_torch(logits.cpu(), 8)

    torch.testing.assert_close(vals.cpu(), ref_vals)
    torch.testing.assert_close(ids.cpu(), ref_ids)
    torch.testing.assert_close(lse.cpu(), ref_lse)


def test_tiled_topk_lse_is_the_full_vocabulary_log_softmax():
    """The head consumes ``val - lse``, which must be log_softmax over the whole
    vocabulary. Raw top-k logits would score a different function, and the tiling
    is exactly where the partition could silently become partial."""
    torch.manual_seed(1)
    logits = torch.randn(4, 151936, device="cuda", dtype=torch.float32)

    vals, ids, lse = lilicorr_topk_lse(logits, 8)
    expected = torch.log_softmax(logits.double(), dim=-1)

    torch.testing.assert_close(
        (vals - lse.unsqueeze(-1)).double(),
        torch.gather(expected, 1, ids),
        rtol=1e-5,
        atol=1e-5,
    )


def test_tiled_topk_lse_survives_bf16_logits():
    """Production logits are bf16, because they come off the target lm_head.

    Asserted on the selected *values*, not the ids: bf16 spacing near the maximum
    of a 151936-wide row is coarse enough that several entries can round to the
    same value, and which of an exactly-tied set gets returned is unspecified in
    both implementations. The log-probs the head consumes are what must agree.
    """
    torch.manual_seed(2)
    logits = torch.randn(4, 151936, device="cuda", dtype=torch.bfloat16)

    vals, ids, lse = lilicorr_topk_lse(logits, 8)
    ref_vals, ref_ids, ref_lse = _topk_lse_torch(logits.cpu(), 8)

    torch.testing.assert_close(
        (vals - lse.unsqueeze(-1)).cpu(),
        ref_vals - ref_lse.unsqueeze(-1),
        rtol=2e-3,
        atol=2e-3,
    )
    # The partition is the whole point of fusing the two, so pin it on its own --
    # but at a tolerance that reflects summation order rather than exactness. The
    # kernel reduces per-program online-softmax partials; the reference sums
    # 151936 fp32 terms sequentially. Measured divergence between the two is
    # ~1.7e-5 relative, so the fp32 default (1.3e-6) is unphysical here. The
    # fp32-input case above *is* checked against the default and passes.
    torch.testing.assert_close(lse.cpu(), ref_lse, rtol=1e-4, atol=1e-3)


def test_a_vocabulary_narrower_than_k_tiles_takes_the_exact_reference_path():
    """When the vocabulary spans fewer than k tiles the pre-selection is vacuous
    -- every tile is chosen -- and the lane group cannot be expressed as a power
    of two. That case must fall back rather than fail to compile."""
    torch.manual_seed(3)
    logits = torch.randn(3, 3072, device="cuda", dtype=torch.float32)

    vals, ids, lse = lilicorr_topk_lse(logits, 8)
    ref_vals, ref_ids, ref_lse = _topk_lse_torch(logits.cpu(), 8)

    torch.testing.assert_close(vals.cpu(), ref_vals)
    torch.testing.assert_close(ids.cpu(), ref_ids)
    torch.testing.assert_close(lse.cpu(), ref_lse)


# --- the fused greedy commit ------------------------------------------------


@pytest.mark.parametrize("topk", [1, 2, 4, 8, 16])
def test_fused_greedy_path_matches_the_reference(topk):
    torch.manual_seed(4)
    bs, slots = 5, 15
    log_start = torch.randn(bs, topk, device="cuda")
    log_pair = torch.randn(bs, slots - 1, topk, topk, device="cuda")
    ids = torch.randint(0, 151936, (bs, slots, topk), device="cuda")

    actual = lilicorr_greedy_path(log_start, log_pair, ids)
    expected = _greedy_path_torch(log_start.cpu(), log_pair.cpu(), ids.cpu())

    torch.testing.assert_close(actual.cpu(), expected)


def test_fused_greedy_path_breaks_ties_toward_the_lower_candidate_on_device():
    """tl.argmax and Tensor.argmax must agree on ties, or the two paths commit
    different tokens on exactly the inputs where the head is least certain."""
    log_start = torch.zeros(2, 8, device="cuda")
    log_pair = torch.zeros(2, 14, 8, 8, device="cuda")
    ids = torch.arange(2 * 15 * 8, device="cuda").view(2, 15, 8)

    actual = lilicorr_greedy_path(log_start, log_pair, ids)
    expected = _greedy_path_torch(log_start.cpu(), log_pair.cpu(), ids.cpu())

    torch.testing.assert_close(actual.cpu(), expected)
    # Every slot should have taken candidate 0, i.e. the lowest index.
    assert actual[:, 0].cpu().equal(ids[:, 0, 0].cpu())


def test_fused_greedy_path_accepts_a_non_unit_stride_last_dim():
    """The kernel reads the candidate (last) dim contiguously and takes every
    other dim through a passed stride, so a factor tensor whose last dim is not
    unit-stride has to be copied first. This pins that copy rather than trusting
    it: a transposed view is passed, and compared against the same transposed
    values laid out normally.
    """
    torch.manual_seed(5)
    bs, slots, topk = 3, 15, 8
    log_start = torch.randn(bs, topk, device="cuda")
    base = torch.randn(bs, slots - 1, topk, topk, device="cuda")

    ids = torch.randint(0, 1000, (bs, slots, topk), device="cuda")

    transposed = base.transpose(-1, -2)
    assert transposed.stride(-1) != 1, "the view under test must be non-contiguous"

    actual = lilicorr_greedy_path(log_start, transposed, ids)
    expected = _greedy_path_torch(
        log_start.cpu(), transposed.cpu().contiguous(), ids.cpu()
    )
    torch.testing.assert_close(actual.cpu(), expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
