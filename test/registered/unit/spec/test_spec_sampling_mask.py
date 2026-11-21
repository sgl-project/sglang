"""Speculative replay capture must describe the complete verification policy."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers import sampling_renorm
from sglang.srt.layers.logits_processor import SamplingMaskStatus
from sglang.srt.speculative import dflash_utils
from sglang.srt.speculative.spec_sampling_mask import SpeculativeSamplingMaskCapture
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=12, stage="base-b", runner_config="1-gpu-small")


def _top_p(probs, top_ps):
    values, indices = probs.sort(dim=-1, descending=True)
    values = values.masked_fill(values.cumsum(-1) - values >= top_ps[:, None], 0)
    values = values / values.sum(-1, keepdim=True)
    return torch.zeros_like(probs).scatter_(1, indices, values)


def _inclusive_top_k(probs, top_ks):
    cutoff = probs.sort(dim=-1, descending=True).values.gather(
        1, (top_ks.clamp(max=probs.shape[-1]) - 1).long()[:, None]
    )
    kept = probs.masked_fill(probs < cutoff, 0)
    return kept / kept.sum(-1, keepdim=True)


def _sampling_info(ks, top_p, known_bound):
    return SimpleNamespace(
        top_ks=torch.tensor(ks, dtype=torch.int32),
        top_ps=torch.full((len(ks),), top_p),
        temperatures=torch.full((len(ks), 1), 0.7),
        need_top_k_sampling=True,
        need_top_p_sampling=top_p < 1,
        is_all_greedy=False,
        sampling_mask_batch_indices=torch.tensor([0]),
        sampling_mask_top_ks=[ks[0]] + [1] * (len(ks) - 1),
        sampling_support_logprobs_capture_indices=torch.tensor([0]),
        max_bounded_top_k=max(k for k in ks if k < 128) if known_bound else None,
    )


@pytest.mark.parametrize("ties", [64, 65, 68, 72])
@pytest.mark.parametrize("top_p", [1.0, 0.8])
@pytest.mark.parametrize("known_bound", [True, False])
@pytest.mark.parametrize("replay_row", [0, 1])
def test_scoring_neighbor_does_not_expand_replay_support(
    monkeypatch, ties, top_p, known_bound, replay_row
):
    monkeypatch.setattr(
        dflash_utils, "radix_topk", lambda logits, k: torch.topk(logits, k, dim=-1)
    )
    monkeypatch.setattr(dflash_utils, "_dflash_top_k_renorm_prob", _inclusive_top_k)
    monkeypatch.setattr(dflash_utils, "_dflash_top_p_renorm_prob", _top_p)
    logits = torch.full((6, 128), -20.0)
    logits[:, :ties] = 0.0
    alone = dflash_utils.build_speculative_verify_target_probs(
        next_token_logits=logits[:3],
        sampling_info=_sampling_info([64], top_p, known_bound),
        draft_token_num=3,
        bs=1,
        max_top_k=64,
        uniform_top_k_value=64,
    )
    ks = [64, 1 << 30] if replay_row == 0 else [1 << 30, 64]
    info = _sampling_info(ks, top_p, known_bound)
    info.sampling_mask_batch_indices = torch.tensor([replay_row])
    info.sampling_mask_top_ks = [64 if row == replay_row else 1 for row in range(2)]
    draft = SimpleNamespace(max_top_k=1 << 30, uniform_top_k_value=None)
    scope = patch.object(torch.Tensor, "item", side_effect=AssertionError("host sync"))
    if known_bound:
        scope.start()
    try:
        capture = SpeculativeSamplingMaskCapture.from_logits(
            info,
            next_token_logits=logits,
            draft_input=draft,
            draft_token_num=3,
            bs=2,
        )
    finally:
        if known_bound:
            scope.stop()
    torch.testing.assert_close(capture.target_probs[replay_row], alone[0])
    expected_scoring = (logits[3:] / 0.7).softmax(-1)
    if top_p < 1:
        expected_scoring = _top_p(expected_scoring, torch.full((3,), top_p))
    torch.testing.assert_close(capture.target_probs[1 - replay_row], expected_scoring)
    chosen = capture.target_probs.argmax(-1)
    output = capture.build_output(out_tokens=chosen, commit_lens=torch.tensor([3, 1]))
    assert (output.statuses == SamplingMaskStatus.OK).all()
    assert (output.lengths <= 64).all()
    torch.testing.assert_close(output.support_logprobs.exp().sum(-1), torch.ones(1, 3))


@pytest.mark.parametrize("ties", [64, 65, 72])
@pytest.mark.parametrize("greedy", [False, True])
@pytest.mark.parametrize("support", [False, True])
def test_capture_detects_truncation_even_when_selected_token_is_finite(
    ties, greedy, support
):
    probs = torch.zeros(1, 2, 128)
    probs[..., :ties] = 1.0 / ties
    capture = SpeculativeSamplingMaskCapture(
        target_probs=probs,
        batch_indices=torch.tensor([0]),
        max_top_k=64,
        greedy_mask=torch.tensor([greedy]),
        support_capture_indices=torch.tensor([0]) if support else None,
    )
    output = capture.build_output(
        out_tokens=torch.tensor([[0, 1]]), commit_lens=torch.tensor([2])
    )
    expected = (
        SamplingMaskStatus.OVERFLOW
        if ties > 64 and not greedy
        else SamplingMaskStatus.OK
    )
    assert (output.statuses == expected).all()
    if greedy:
        assert output.lengths.tolist() == [[1, 1]]
        assert output.token_ids[0, :, 0].tolist() == [0, 1]


def test_unbounded_and_greedy_capture_still_work():
    probs = torch.ones(1, 2, 4) / 4
    for target in (probs, None):
        capture = SpeculativeSamplingMaskCapture(
            target_probs=target,
            batch_indices=torch.tensor([0]),
            max_top_k=4,
            support_capture_indices=torch.tensor([0]),
        )
        output = capture.build_output(
            out_tokens=torch.tensor([[0, 1]]), commit_lens=torch.tensor([2])
        )
        assert (output.statuses == SamplingMaskStatus.OK).all()
        torch.testing.assert_close(
            output.support_logprobs.exp().sum(-1), torch.ones(1, 2)
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA sampling kernels"
)
@pytest.mark.parametrize("ties", [65, 72])
@pytest.mark.parametrize("top_p", [1.0, 0.8])
@pytest.mark.parametrize("deterministic", [False, True])
def test_cuda_mixed_batch_captures_complete_policy(
    monkeypatch, ties, top_p, deterministic
):
    monkeypatch.setattr(sampling_renorm, "renorm_deterministic", lambda: deterministic)
    info = _sampling_info([64, 1 << 30], top_p, True)
    for name, value in vars(info).items():
        if isinstance(value, torch.Tensor):
            setattr(info, name, value.cuda())
    logits = torch.full((6, 128), -20.0, device="cuda")
    logits[:, :ties] = 0.0
    capture = SpeculativeSamplingMaskCapture.from_logits(
        info,
        next_token_logits=logits,
        draft_input=SimpleNamespace(max_top_k=1 << 30, uniform_top_k_value=None),
        draft_token_num=3,
        bs=2,
    )
    assert ((capture.target_probs[0] > 0).sum(-1) <= 64).all()
    expected = (logits[3:] / 0.7).softmax(-1)
    if top_p < 1:
        expected = dflash_utils._dflash_top_p_renorm_prob(
            expected, info.top_ps[1:].expand(3)
        )
    torch.testing.assert_close(capture.target_probs[1], expected)
    output = capture.build_output(
        out_tokens=capture.target_probs.argmax(-1),
        commit_lens=torch.tensor([3, 1], device="cuda"),
    )
    assert (output.statuses == SamplingMaskStatus.OK).all()
    torch.testing.assert_close(
        output.support_logprobs.exp().sum(-1), torch.ones(1, 3, device="cuda")
    )
    capture.target_probs[0].zero_()
    capture.target_probs[0, :, :ties] = 1.0 / ties
    overflow = capture.build_output(
        out_tokens=capture.target_probs.argmax(-1),
        commit_lens=torch.tensor([3, 1], device="cuda"),
    )
    assert (overflow.statuses == SamplingMaskStatus.OVERFLOW).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-x"]))
