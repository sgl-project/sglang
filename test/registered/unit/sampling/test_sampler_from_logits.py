"""Unit tests for the sampler's fused sample-from-logits fast path."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.sampler import Sampler
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")

VOCAB = 32000


def _make_sampler(enable_deterministic=False, disable_fast_path=False) -> Sampler:
    sampler = Sampler.__new__(Sampler)
    torch.nn.Module.__init__(sampler)
    sampler.enable_deterministic = enable_deterministic
    sampler.disable_sampling_from_logits = disable_fast_path
    return sampler


def _sampling_info(
    sampling_seed=None,
    need_min_p_sampling=False,
    top_k=50,
    top_p=0.95,
    batch_size=4,
):
    return SimpleNamespace(
        sampling_seed=sampling_seed,
        need_min_p_sampling=need_min_p_sampling,
        top_ks=torch.full((batch_size,), top_k, device="cuda", dtype=torch.int32),
        top_ps=torch.full((batch_size,), top_p, device="cuda"),
    )


def _exec_with_backend(backend: str):
    return SimpleNamespace(kernel=SimpleNamespace(sampling_backend=backend))


class TestSampleFromLogitsGate(CustomTestCase):
    """The fast path must only engage when nothing downstream needs probs."""

    def _gate(self, sampler, sampling_info, backend="flashinfer", **kwargs):
        with patch(
            "sglang.srt.layers.sampler.get_exec",
            return_value=_exec_with_backend(backend),
        ):
            return sampler._should_sample_from_scaled_logits(
                sampling_info,
                kwargs.get("return_logprob", False),
                kwargs.get("return_sampling_mask", False),
            )

    def test_engages_on_clean_flashinfer_batch(self):
        self.assertTrue(self._gate(_make_sampler(), _sampling_info()))

    def test_disqualifiers(self):
        cases = {
            "deterministic": (
                _make_sampler(enable_deterministic=True),
                _sampling_info(),
                {},
            ),
            "env_kill_switch": (
                _make_sampler(disable_fast_path=True),
                _sampling_info(),
                {},
            ),
            "sampling_seed": (
                _make_sampler(),
                _sampling_info(sampling_seed=torch.ones(4, device="cuda")),
                {},
            ),
            "min_p": (
                _make_sampler(),
                _sampling_info(need_min_p_sampling=True),
                {},
            ),
            "return_logprob": (
                _make_sampler(),
                _sampling_info(),
                {"return_logprob": True},
            ),
            "return_sampling_mask": (
                _make_sampler(),
                _sampling_info(),
                {"return_sampling_mask": True},
            ),
        }
        for name, (sampler, info, kwargs) in cases.items():
            with self.subTest(disqualifier=name):
                self.assertFalse(self._gate(sampler, info, **kwargs))

    def test_pytorch_backend_not_eligible(self):
        self.assertFalse(
            self._gate(_make_sampler(), _sampling_info(), backend="pytorch")
        )


class TestSampleFromLogitsParity(CustomTestCase):
    """The fused kernels must match the softmax + from-probs pipeline."""

    def test_top_k_top_p_bitwise_matches_probs_path(self):
        # Mirrors the AOT alignment test: with the same RNG state, sampling
        # from logits and from softmax(logits) picks identical tokens.
        from flashinfer.sampling import top_k_top_p_sampling_from_probs

        sampler = _make_sampler()
        info = _sampling_info(batch_size=64)
        logits = torch.randn(64, VOCAB, device="cuda") * 4

        torch.manual_seed(42)
        fast = sampler._sample_from_scaled_logits(
            logits.clone(), info, simple_sampling_case=False
        )
        torch.manual_seed(42)
        ref = top_k_top_p_sampling_from_probs(
            torch.softmax(logits, dim=-1).contiguous(),
            info.top_ks,
            info.top_ps,
            filter_apply_order="joint",
        )
        self.assertTrue(torch.equal(fast, ref))

    def test_simple_case_distribution_matches_multinomial(self):
        small_vocab = 64
        num_samples = 100_000
        torch.manual_seed(0)
        logits = torch.randn(1, small_vocab, device="cuda") * 2
        expanded = logits.expand(num_samples, small_vocab).contiguous()

        sampler = _make_sampler()
        info = _sampling_info(batch_size=num_samples)
        fast = sampler._sample_from_scaled_logits(
            expanded, info, simple_sampling_case=True
        )
        ref = torch.multinomial(torch.softmax(expanded, dim=-1), 1).view(-1)

        freq_fast = torch.bincount(fast.long(), minlength=small_vocab).float()
        freq_ref = torch.bincount(ref, minlength=small_vocab).float()
        l1 = (freq_fast - freq_ref).abs().sum().item() / num_samples
        self.assertLess(l1, 0.03)
        self.assertTrue(bool((fast >= 0).all()))
        self.assertTrue(bool((fast < small_vocab).all()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
