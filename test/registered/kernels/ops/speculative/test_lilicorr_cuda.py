import sys

import pytest
import torch

from sglang.kernels.ops.speculative.lilicorr import (
    _lattice_scores,
    _selector_walk_torch,
    lilicorr_sample_path,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="extra-a", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="LiLiCorr Triton kernels require CUDA"
)


def _walk_reference(
    log_start, log_pair, candidate_tokens, uniforms, temperatures, greedy_mask
):
    return _selector_walk_torch(
        candidate_ids=candidate_tokens.cpu(),
        scores=_lattice_scores(log_start.cpu(), log_pair.cpu()),
        uniforms=uniforms.cpu(),
        temperatures=temperatures.cpu(),
        greedy_mask=greedy_mask.cpu(),
    )


def _greedy_state(bs, slots, device):
    return dict(
        uniforms=torch.zeros(bs, slots, device=device),
        temperatures=torch.ones(bs, device=device),
        greedy_mask=torch.ones(bs, dtype=torch.bool, device=device),
    )


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


@pytest.mark.parametrize("k", [1, 2, 8, 16])
def test_sampled_path_matches_the_reference(k):
    a = _sampled_inputs(4, 5, k, seed=k)
    tokens, q = lilicorr_sample_path(**a)
    ref_tokens, ref_q = _walk_reference(**a)

    assert torch.equal(tokens.cpu(), ref_tokens)
    torch.testing.assert_close(q.cpu(), ref_q, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
