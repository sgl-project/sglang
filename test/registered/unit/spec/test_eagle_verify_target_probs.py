"""EAGLE verify's target distribution must match the pytorch sampler's filter."""

from types import SimpleNamespace

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")

# The renorm kernels the fix replaces only run on GPU.
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _sampler_target_probs(logits, top_ks, top_ps):
    from sglang.srt.layers.sampler import top_k_top_p_filter_probs_torch

    n = logits.shape[0]
    sorted_probs, sorted_idx = top_k_top_p_filter_probs_torch(
        torch.softmax(logits.to(_DEVICE), -1),
        top_ks.repeat(n).to(_DEVICE),
        top_ps.repeat(n).to(_DEVICE),
    )
    expected = torch.zeros_like(sorted_probs).scatter_(-1, sorted_idx, sorted_probs)
    return (expected / expected.sum(-1, keepdim=True)).cpu()


def _verify_target_probs(logits, top_ks, top_ps):
    from sglang.srt.runtime_context import get_context
    from sglang.srt.speculative.eagle_utils import _build_verify_target_probs

    sampling_info = SimpleNamespace(
        temperatures=torch.ones(1, 1, device=_DEVICE),
        top_ks=top_ks.to(_DEVICE),
        top_ps=top_ps.to(_DEVICE),
        need_top_k_sampling=bool((top_ks < logits.shape[-1]).any()),
        need_top_p_sampling=bool((top_ps < 1.0).any()),
        max_top_k=None,
        sampling_seed=None,
    )
    with get_context().override_server_args(sampling_backend="pytorch"):
        return _build_verify_target_probs(
            logits.to(_DEVICE), sampling_info, logits.shape[0]
        ).cpu()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="softmax kernel needs GPU")
def test_target_probs_match_the_pytorch_sampler_at_a_top_p_tie():
    probs = torch.tensor([[0.6, 0.2, 0.2, 0.0]]).clamp_min(1e-9)
    logits = probs.log().repeat(2, 1)
    top_ks = torch.tensor([1 << 30], dtype=torch.int32)
    top_ps = torch.tensor([0.7])

    target = _verify_target_probs(logits, top_ks, top_ps)

    torch.testing.assert_close(
        target, _sampler_target_probs(logits, top_ks, top_ps), rtol=0, atol=1e-6
    )
    assert (target > 0).sum(-1).tolist() == [2, 2]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="softmax kernel needs GPU")
def test_target_probs_match_the_pytorch_sampler_under_top_k():
    torch.manual_seed(0)
    logits = torch.randn(3, 64) * 3
    top_ks = torch.tensor([5], dtype=torch.int32)
    top_ps = torch.tensor([0.9])

    target = _verify_target_probs(logits, top_ks, top_ps)

    torch.testing.assert_close(
        target, _sampler_target_probs(logits, top_ks, top_ps), rtol=0, atol=1e-6
    )
