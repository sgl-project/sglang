"""GPU parity tests for the LiLiCorr Triton kernels.

Both kernel entry points return a value-identical torch implementation for
non-CUDA input, so on a CPU runner ``test_lilicorr.py`` only ever exercises the
reference path and the Triton code is never compiled. These tests pin the two
against each other on device, which is the only place "value-identical" is
actually checked.
"""

import sys

import pytest
import torch

from sglang.kernels.ops.speculative.lilicorr import (
    _greedy_path_torch,
    _sample_path_torch,
    _topk_lse_torch,
    lilicorr_greedy_path,
    lilicorr_sample_path,
    lilicorr_topk_lse,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="LiLiCorr Triton kernels require CUDA"
)


# --- the tiled top-k + log-partition ---------------------------------------


# 151936 is Qwen3's vocabulary; the rest straddle the kernel's 1024-wide tile
# boundary in both directions, where an off-by-one in the tile pre-selection
# would hide.
@pytest.mark.parametrize("vocab", [1024, 1025, 2047, 151936])
def test_tiled_topk_lse_is_exact_against_the_reference(vocab):
    torch.manual_seed(0)
    logits = torch.randn(6, vocab, device="cuda", dtype=torch.float32)

    vals, tokens, lse = lilicorr_topk_lse(logits, 8)
    ref_vals, ref_tokens, ref_lse = _topk_lse_torch(logits.cpu(), 8)

    torch.testing.assert_close(vals.cpu(), ref_vals)
    torch.testing.assert_close(tokens.cpu(), ref_tokens)
    torch.testing.assert_close(lse.cpu(), ref_lse)


def test_tiled_topk_lse_is_the_full_vocabulary_log_softmax():
    """The head consumes ``val - lse``, which must be log_softmax over the whole
    vocabulary. The tiling is exactly where the partition could silently become
    partial."""
    torch.manual_seed(1)
    logits = torch.randn(4, 151936, device="cuda", dtype=torch.float32)

    vals, tokens, lse = lilicorr_topk_lse(logits, 8)
    expected = torch.log_softmax(logits.double(), dim=-1)

    torch.testing.assert_close(
        (vals - lse.unsqueeze(-1)).double(),
        torch.gather(expected, 1, tokens),
        rtol=1e-5,
        atol=1e-5,
    )


def test_tiled_topk_lse_survives_bf16_logits():
    """Production logits are bf16, because they come off the target lm_head.

    Asserted on the selected values, not the tokens: bf16 spacing near the
    maximum of a 151936-wide row is coarse enough that several entries can round
    to the same value, and which of an exactly-tied set is returned is
    unspecified in both implementations.
    """
    torch.manual_seed(2)
    logits = torch.randn(4, 151936, device="cuda", dtype=torch.bfloat16)

    vals, tokens, lse = lilicorr_topk_lse(logits, 8)
    ref_vals, ref_tokens, ref_lse = _topk_lse_torch(logits.cpu(), 8)

    torch.testing.assert_close(
        (vals - lse.unsqueeze(-1)).cpu(),
        ref_vals - ref_lse.unsqueeze(-1),
        rtol=2e-3,
        atol=2e-3,
    )
    # Tolerance reflects summation order, not exactness: the kernel reduces
    # per-program online-softmax partials while the reference sums 151936 fp32
    # terms sequentially, measured ~1.7e-5 relative apart. The fp32-input case
    # above is checked against the default tolerance and passes.
    torch.testing.assert_close(lse.cpu(), ref_lse, rtol=1e-4, atol=1e-3)


def test_tied_bf16_logits_return_a_valid_selection():
    """Which of an exactly-tied set is returned is unspecified here and in CUDA
    ``torch.topk`` alike, so the two can name different tokens for the same row.
    The invariant that does hold, and that the head depends on, is that every
    returned id really carries the value reported for it and that the values
    match the reference. Built with deliberate ties rather than hoping for them.
    """
    torch.manual_seed(6)
    # 64 distinct values over a wide row: every value recurs many times, so the
    # top-8 is almost entirely ties.
    logits = torch.randint(0, 64, (4, 8192), device="cuda").to(torch.bfloat16)

    vals, tokens, _ = lilicorr_topk_lse(logits, 8)
    ref_vals, _, _ = _topk_lse_torch(logits.cpu(), 8)

    torch.testing.assert_close(vals.cpu(), ref_vals)
    # Each returned id must actually hold the value returned beside it.
    gathered = torch.gather(logits, 1, tokens).float()
    torch.testing.assert_close(gathered.cpu(), vals.cpu())


def test_a_vocabulary_narrower_than_k_tiles_takes_the_exact_reference_path():
    """When the vocabulary spans fewer than k tiles the lane group cannot be
    expressed as a power of two, so that case must fall back rather than fail to
    compile."""
    torch.manual_seed(3)
    logits = torch.randn(3, 3072, device="cuda", dtype=torch.float32)

    vals, tokens, lse = lilicorr_topk_lse(logits, 8)
    ref_vals, ref_tokens, ref_lse = _topk_lse_torch(logits.cpu(), 8)

    torch.testing.assert_close(vals.cpu(), ref_vals)
    torch.testing.assert_close(tokens.cpu(), ref_tokens)
    torch.testing.assert_close(lse.cpu(), ref_lse)


# --- the fused greedy commit ------------------------------------------------


# 16 is the widest pool the fused commit holds in one lane group.
@pytest.mark.parametrize("topk", [1, 8, 16])
def test_fused_greedy_path_matches_the_reference(topk):
    torch.manual_seed(4)
    bs, slots = 5, 15
    log_start = torch.randn(bs, topk, device="cuda")
    log_pair = torch.randn(bs, slots - 1, topk, topk, device="cuda")
    tokens = torch.randint(0, 151936, (bs, slots, topk), device="cuda")

    actual = lilicorr_greedy_path(log_start, log_pair, tokens)
    expected = _greedy_path_torch(log_start.cpu(), log_pair.cpu(), tokens.cpu())

    torch.testing.assert_close(actual.cpu(), expected)


def test_fused_greedy_path_breaks_ties_toward_the_lower_candidate_on_device():
    """tl.argmax and Tensor.argmax must agree on ties, or the two paths commit
    different tokens on exactly the inputs where the head is least certain."""
    log_start = torch.zeros(2, 8, device="cuda")
    log_pair = torch.zeros(2, 14, 8, 8, device="cuda")
    tokens = torch.arange(2 * 15 * 8, device="cuda").view(2, 15, 8)

    actual = lilicorr_greedy_path(log_start, log_pair, tokens)
    expected = _greedy_path_torch(log_start.cpu(), log_pair.cpu(), tokens.cpu())

    torch.testing.assert_close(actual.cpu(), expected)
    assert actual[:, 0].cpu().equal(tokens[:, 0, 0].cpu())


def test_fused_greedy_path_accepts_a_non_unit_stride_last_dim():
    """The kernel reads the candidate dim contiguously and takes every other dim
    through a passed stride, so a factor tensor whose last dim is not unit-stride
    has to be copied first. This pins that copy rather than trusting it."""
    torch.manual_seed(5)
    bs, slots, topk = 3, 15, 8
    log_start = torch.randn(bs, topk, device="cuda")
    base = torch.randn(bs, slots - 1, topk, topk, device="cuda")
    tokens = torch.randint(0, 1000, (bs, slots, topk), device="cuda")

    transposed = base.transpose(-1, -2)
    assert transposed.stride(-1) != 1, "the view under test must be non-contiguous"

    actual = lilicorr_greedy_path(log_start, transposed, tokens)
    expected = _greedy_path_torch(
        log_start.cpu(), transposed.cpu().contiguous(), tokens.cpu()
    )
    torch.testing.assert_close(actual.cpu(), expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def _sampled_inputs(bs, slots, k, *, seed=0):
    torch.manual_seed(seed)
    return dict(
        log_start=torch.randn(bs, k, device="cuda"),
        log_pair=torch.randn(bs, slots - 1, k, k, device="cuda"),
        candidate_tokens=torch.arange(bs * slots * k, device="cuda").view(bs, slots, k),
        uniforms=torch.rand(bs, slots, device="cuda"),
        temperatures=torch.rand(bs, device="cuda") + 0.5,
        greedy_mask=torch.zeros(bs, dtype=torch.bool, device="cuda"),
    )


def test_sampled_path_commits_the_same_ids_as_the_reference():
    """The committed ids are integers, so there is no tolerance to hide in: the
    Triton draw and the torch draw must consume the uniforms identically or the two
    walk different paths from the same random numbers."""
    a = _sampled_inputs(6, 5, 8, seed=0)
    tokens, q = lilicorr_sample_path(**a)
    ref_tokens, ref_q = _sample_path_torch(**{k: v.cpu() for k, v in a.items()})

    assert torch.equal(tokens.cpu(), ref_tokens)
    torch.testing.assert_close(q.cpu(), ref_q, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("k", [1, 2, 4, 16])
def test_sampled_path_matches_the_reference_at_the_lane_group_edges(k):
    """``k`` is the Triton lane group, so 1 and 16 are the degenerate and full cases
    where an off-by-one in the cumulative-sum draw would show up."""
    a = _sampled_inputs(4, 4, k, seed=k)
    tokens, q = lilicorr_sample_path(**a)
    ref_tokens, ref_q = _sample_path_torch(**{key: v.cpu() for key, v in a.items()})

    assert torch.equal(tokens.cpu(), ref_tokens)
    torch.testing.assert_close(q.cpu(), ref_q, atol=1e-6, rtol=1e-6)
