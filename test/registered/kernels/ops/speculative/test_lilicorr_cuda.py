import sys

import pytest
import torch

from sglang.kernels.ops.speculative.lilicorr import (
    _topk_lse_torch,
    lilicorr_sample_path,
    lilicorr_topk_lse,
)
from sglang.srt.sampling.draft_sampling import DraftSamplingParams
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="LiLiCorr Triton kernels require CUDA"
)


# 1025 and 2047 straddle the tile boundary; 3072 has fewer tiles than k.
@pytest.mark.parametrize("vocab", [1025, 2047, 3072, 151936])
def test_tiled_topk_lse_is_exact_against_the_reference(vocab):
    torch.manual_seed(0)
    logits = torch.randn(6, vocab, device="cuda", dtype=torch.float32)

    vals, tokens, lse = lilicorr_topk_lse(logits, 8)
    ref_vals, ref_tokens, ref_lse = _topk_lse_torch(logits.cpu(), 8)

    torch.testing.assert_close(vals.cpu(), ref_vals)
    torch.testing.assert_close(tokens.cpu(), ref_tokens)
    torch.testing.assert_close(lse.cpu(), ref_lse)


def test_tied_bf16_logits_return_a_valid_selection():
    torch.manual_seed(6)
    logits = torch.randint(0, 64, (4, 8192), device="cuda").to(torch.bfloat16)

    vals, tokens, _ = lilicorr_topk_lse(logits, 8)
    ref_vals, _, _ = _topk_lse_torch(logits.cpu(), 8)

    torch.testing.assert_close(vals.cpu(), ref_vals)
    gathered = torch.gather(logits, 1, tokens).float()
    torch.testing.assert_close(gathered.cpu(), vals.cpu())


def _walk_reference(log_start, log_pair, candidate_tokens, uniforms, params):
    if params is not None:
        params = DraftSamplingParams(
            params.temperatures.cpu(), params.top_ks.cpu(), params.top_ps.cpu()
        )
    return lilicorr_sample_path(
        log_start.cpu(),
        log_pair.cpu(),
        candidate_tokens.cpu(),
        uniforms=uniforms.cpu(),
        params=params,
    )


def _greedy_state(bs, slots, device):
    return dict(uniforms=torch.zeros(bs, slots, device=device), params=None)


def _greedy(log_start, log_pair, candidate_tokens):
    bs, slots, _ = candidate_tokens.shape
    tokens, _ = lilicorr_sample_path(
        log_start,
        log_pair,
        candidate_tokens,
        **_greedy_state(bs, slots, candidate_tokens.device),
    )
    return tokens


def _greedy_reference(log_start, log_pair, candidate_tokens):
    bs, slots, _ = candidate_tokens.shape
    tokens, _ = _walk_reference(
        log_start,
        log_pair,
        candidate_tokens,
        **_greedy_state(bs, slots, torch.device("cpu")),
    )
    return tokens


@pytest.mark.parametrize("topk", [1, 8, 16])
def test_greedy_path_matches_the_reference(topk):
    torch.manual_seed(4)
    bs, slots = 5, 15
    log_start = torch.randn(bs, topk, device="cuda")
    log_pair = torch.randn(bs, slots - 1, topk, topk, device="cuda")
    tokens = torch.randint(0, 151936, (bs, slots, topk), device="cuda")

    actual = _greedy(log_start, log_pair, tokens)
    expected = _greedy_reference(log_start, log_pair, tokens)

    torch.testing.assert_close(actual.cpu(), expected)


def test_greedy_path_breaks_ties_toward_the_lower_candidate_on_device():
    log_start = torch.zeros(2, 8, device="cuda")
    log_pair = torch.zeros(2, 14, 8, 8, device="cuda")
    tokens = torch.arange(2 * 15 * 8, device="cuda").view(2, 15, 8)

    actual = _greedy(log_start, log_pair, tokens)
    expected = _greedy_reference(log_start, log_pair, tokens)

    torch.testing.assert_close(actual.cpu(), expected)
    assert actual[:, 0].cpu().equal(tokens[:, 0, 0].cpu())


@pytest.mark.parametrize("nan_rows", ["all", "mixed"])
def test_greedy_path_matches_the_reference_on_nan_scores(nan_rows):
    torch.manual_seed(5)
    bs, slots, topk = 3, 6, 8
    log_start = torch.randn(bs, topk, device="cuda")
    log_pair = torch.randn(bs, slots - 1, topk, topk, device="cuda")
    if nan_rows == "all":
        log_start[0] = float("nan")
        log_pair[0] = float("nan")
    else:
        log_pair[torch.rand_like(log_pair) < 0.3] = float("nan")
    candidate_tokens = torch.arange(bs * slots * topk, device="cuda").view(
        bs, slots, topk
    )
    state = _greedy_state(bs, slots, candidate_tokens.device)

    tokens, q = lilicorr_sample_path(log_start, log_pair, candidate_tokens, **state)
    ref_tokens, ref_q = _walk_reference(log_start, log_pair, candidate_tokens, **state)

    assert torch.equal(tokens.cpu(), ref_tokens)
    assert torch.equal(q.cpu(), ref_q)


def _sampled_inputs(bs, slots, k, *, seed=0):
    torch.manual_seed(seed)
    return dict(
        log_start=torch.randn(bs, k, device="cuda"),
        log_pair=torch.randn(bs, slots - 1, k, k, device="cuda"),
        candidate_tokens=torch.arange(bs * slots * k, device="cuda").view(bs, slots, k),
        uniforms=torch.rand(bs, slots, device="cuda"),
        params=DraftSamplingParams(
            torch.rand(bs, device="cuda") + 0.5,
            torch.full((bs,), TOP_K_ALL, dtype=torch.int32, device="cuda"),
            torch.ones(bs, device="cuda"),
        ),
    )


@pytest.mark.parametrize("k", [1, 2, 8, 16])
def test_sampled_path_matches_the_reference(k):
    a = _sampled_inputs(4, 5, k, seed=k)
    tokens, q = lilicorr_sample_path(**a)
    ref_tokens, ref_q = _walk_reference(**a)

    assert torch.equal(tokens.cpu(), ref_tokens)
    torch.testing.assert_close(q.cpu(), ref_q, atol=1e-6, rtol=1e-6)


def test_truncated_walk_roundoff_fallback_stays_in_support():
    from sglang.kernels.ops.speculative.dflash import selector_walk_triton

    probs = torch.tensor([0.25, 0.0, 0.7499998, 0.0], device="cuda")
    tokens, q = selector_walk_triton(
        candidate_ids=torch.arange(8, device="cuda").reshape(1, 2, 4),
        probs=probs.expand(1, 2, 4, 4),
        uniforms=torch.full((1, 2), 0.9999999, device="cuda"),
    )
    assert tokens.tolist() == [[2, 6]]
    assert torch.all(q.gather(-1, (tokens % 4).unsqueeze(-1)) > 0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
