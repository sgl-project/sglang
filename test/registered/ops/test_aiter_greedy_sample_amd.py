"""Unit tests for aiter greedy_sample kernel and Sampler integration.

Validates that:
1. aiter.greedy_sample produces identical results to torch.argmax (kernel level)
2. The global and greedy-only environment flags select the AITER kernel
3. Sampler.forward() correctly dispatches to aiter when _use_aiter=True
4. The fallback to torch.argmax works when _use_aiter=False
5. return_logprob path works with the aiter greedy branch

The kernel is designed for production LLM inference (large vocab, bf16) and is
used when SGLANG_USE_AITER=1 or SGLANG_USE_AITER_GREEDY_SAMPLE=1 on ROCm.
"""

import os
import unittest
from unittest import mock

import torch

from sglang.srt.utils.common import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd")


class TestAiterGreedySelection(unittest.TestCase):
    def test_greedy_only_flag_enables_aiter_sampler(self):
        from sglang.srt.layers import sampler as sampler_mod

        with (
            mock.patch.object(sampler_mod, "is_hip", return_value=True),
            mock.patch.dict(
                os.environ,
                {
                    "SGLANG_USE_AITER": "0",
                    "SGLANG_USE_AITER_GREEDY_SAMPLE": "1",
                },
                clear=True,
            ),
        ):
            self.assertTrue(sampler_mod._should_use_aiter_greedy_sample())
            self.assertEqual(os.environ["USE_ROCM_AITER_ROPE_BACKEND"], "0")

    def test_both_flags_disabled_use_argmax(self):
        from sglang.srt.layers import sampler as sampler_mod

        with (
            mock.patch.object(sampler_mod, "is_hip", return_value=True),
            mock.patch.dict(
                os.environ,
                {
                    "SGLANG_USE_AITER": "0",
                    "SGLANG_USE_AITER_GREEDY_SAMPLE": "0",
                },
                clear=True,
            ),
        ):
            self.assertFalse(sampler_mod._should_use_aiter_greedy_sample())


def _mock_global_server_args(backend="pytorch"):
    from sglang.srt.layers import sampler as sampler_mod
    from sglang.srt.server_args import (
        ServerArgs,
        set_global_server_args_for_scheduler,
    )

    # Publish for real: the sampler reads the context slot through
    # get_server_args(), which a module-attribute rebinding cannot intercept.
    set_global_server_args_for_scheduler(
        ServerArgs(model_path="dummy", sampling_backend=backend)
    )

    class _DummyTPGroup:
        device_group = None

    sampler_mod.get_tp_group = lambda: _DummyTPGroup()
    from sglang.srt.runtime_context import get_flags

    get_flags().dp.enabled = False


def _make_sampling_info(batch_size, vocab_size, device="cuda"):
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

    return SamplingBatchInfo(
        temperatures=torch.ones(batch_size, 1, device=device, dtype=torch.float),
        top_ps=torch.ones(batch_size, device=device),
        top_ks=torch.zeros(batch_size, device=device, dtype=torch.int32),
        min_ps=torch.zeros(batch_size, device=device),
        is_all_greedy=True,
        is_any_greedy=True,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=vocab_size,
        device=device,
    )


@unittest.skipUnless(is_hip(), "aiter greedy_sample requires ROCm")
class TestAiterGreedySample(unittest.TestCase):
    """Kernel-level correctness: aiter.greedy_sample vs torch.argmax."""

    @classmethod
    def setUpClass(cls):
        try:
            from aiter import greedy_sample

            cls.greedy_sample = staticmethod(greedy_sample)
        except ImportError:
            raise unittest.SkipTest("aiter not installed")
        cls.device = "cuda"

    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def _run_and_compare(self, batch_size, vocab_size):
        logits = torch.randn(
            batch_size, vocab_size, device=self.device, dtype=torch.bfloat16
        )

        expected = torch.argmax(logits, dim=-1)

        actual = torch.empty(logits.shape[0], device=logits.device, dtype=torch.int32)
        self.greedy_sample(actual, logits)

        self.assertTrue(
            torch.equal(actual.to(expected.dtype), expected),
            f"Mismatch for shape ({batch_size}, {vocab_size}): "
            f"expected={expected[:8].tolist()}, got={actual[:8].tolist()}",
        )

    def test_single_request(self):
        self._run_and_compare(1, 32000)

    def test_small_batch(self):
        self._run_and_compare(4, 32000)

    def test_medium_batch(self):
        self._run_and_compare(32, 32000)

    def test_large_batch(self):
        self._run_and_compare(128, 32000)

    def test_realistic_vocab_deepseek(self):
        self._run_and_compare(64, 129280)

    def test_realistic_vocab_llama3(self):
        self._run_and_compare(64, 128256)

    def test_realistic_vocab_qwen(self):
        self._run_and_compare(64, 151936)

    def test_various_batch_sizes(self):
        configs = [
            (1, 128256),
            (2, 128256),
            (8, 128256),
            (16, 128256),
            (32, 129280),
            (64, 129280),
            (128, 129280),
            (256, 129280),
        ]
        for batch_size, vocab_size in configs:
            with self.subTest(batch_size=batch_size, vocab_size=vocab_size):
                self._run_and_compare(batch_size, vocab_size)

    def test_tied_values(self):
        vocab_size = 32000
        logits = torch.zeros(8, vocab_size, device=self.device, dtype=torch.bfloat16)
        logits[:, 0] = 1.0

        expected = torch.argmax(logits, dim=-1)
        actual = torch.empty(8, device=self.device, dtype=torch.int32)
        self.greedy_sample(actual, logits)

        self.assertTrue(torch.equal(actual.to(expected.dtype), expected))

    def test_negative_logits(self):
        vocab_size = 32000
        logits = (
            torch.randn(16, vocab_size, device=self.device, dtype=torch.bfloat16) - 5.0
        )

        expected = torch.argmax(logits, dim=-1)
        actual = torch.empty(16, device=self.device, dtype=torch.int32)
        self.greedy_sample(actual, logits)

        self.assertTrue(torch.equal(actual.to(expected.dtype), expected))

    def test_extreme_values(self):
        vocab_size = 32000
        logits = torch.randn(16, vocab_size, device=self.device, dtype=torch.bfloat16)
        logits[0, 42] = 1e4
        logits[1, 100] = -1e4

        expected = torch.argmax(logits, dim=-1)
        actual = torch.empty(16, device=self.device, dtype=torch.int32)
        self.greedy_sample(actual, logits)

        self.assertTrue(torch.equal(actual.to(expected.dtype), expected))


@unittest.skipUnless(is_hip(), "aiter greedy_sample requires ROCm")
class TestAiterGreedyIntegration(unittest.TestCase):
    """Integration: Sampler.forward() with _use_aiter on/off."""

    @classmethod
    def setUpClass(cls):
        try:
            from aiter import greedy_sample

            cls._greedy_sample_fn = staticmethod(greedy_sample)
        except ImportError:
            raise unittest.SkipTest("aiter not installed")
        cls.device = "cuda"

    def setUp(self):
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def _run_sampler(self, use_aiter, logits, sampling_info, return_logprob=False):
        from sglang.srt.layers import sampler as sampler_mod
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput

        _mock_global_server_args()

        patches = {"_use_aiter": use_aiter}
        if use_aiter:
            patches["_aiter_greedy_sample"] = self._greedy_sample_fn

        with mock.patch.multiple(sampler_mod, **patches):
            sampler = sampler_mod.Sampler()
            batch_size = logits.shape[0]
            positions = torch.arange(batch_size, device=self.device, dtype=torch.int32)

            return sampler.forward(
                logits_output=LogitsProcessorOutput(next_token_logits=logits.clone()),
                sampling_info=sampling_info,
                return_logprob=return_logprob,
                top_logprobs_nums=[0] * batch_size,
                token_ids_logprobs=[None] * batch_size,
                positions=positions,
            )

    def test_aiter_matches_argmax_through_sampler(self):
        batch_size, vocab_size = 64, 129280
        logits = torch.randn(
            batch_size, vocab_size, device=self.device, dtype=torch.bfloat16
        )
        sampling_info = _make_sampling_info(batch_size, vocab_size, self.device)

        out_aiter = self._run_sampler(True, logits, sampling_info)
        out_argmax = self._run_sampler(False, logits, sampling_info)

        self.assertTrue(
            torch.equal(out_aiter.cpu(), out_argmax.cpu()),
            f"Sampler mismatch: aiter={out_aiter[:8].tolist()}, "
            f"argmax={out_argmax[:8].tolist()}",
        )

    def test_fallback_to_argmax_when_disabled(self):
        batch_size, vocab_size = 32, 32000
        logits = torch.randn(
            batch_size, vocab_size, device=self.device, dtype=torch.bfloat16
        )
        sampling_info = _make_sampling_info(batch_size, vocab_size, self.device)

        out = self._run_sampler(False, logits, sampling_info)
        expected = torch.argmax(logits, dim=-1)

        self.assertTrue(
            torch.equal(out.cpu(), expected.cpu()),
            "Fallback path should produce torch.argmax results",
        )

    def test_aiter_greedy_with_return_logprob(self):
        batch_size, vocab_size = 16, 32000
        logits = torch.randn(
            batch_size, vocab_size, device=self.device, dtype=torch.bfloat16
        )
        sampling_info = _make_sampling_info(batch_size, vocab_size, self.device)

        out_aiter = self._run_sampler(True, logits, sampling_info, return_logprob=True)
        out_argmax = self._run_sampler(
            False, logits, sampling_info, return_logprob=True
        )

        self.assertTrue(
            torch.equal(out_aiter.cpu(), out_argmax.cpu()),
            "Token IDs should match with return_logprob=True",
        )

    def test_aiter_output_dtype(self):
        """Document that the aiter path returns int32 (vs int64 from argmax)."""
        batch_size, vocab_size = 16, 32000
        logits = torch.randn(
            batch_size, vocab_size, device=self.device, dtype=torch.bfloat16
        )
        sampling_info = _make_sampling_info(batch_size, vocab_size, self.device)

        out_aiter = self._run_sampler(True, logits, sampling_info)
        out_argmax = self._run_sampler(False, logits, sampling_info)

        self.assertEqual(out_aiter.dtype, torch.int32)
        self.assertEqual(out_argmax.dtype, torch.int64)

    def test_various_batch_sizes_through_sampler(self):
        vocab_size = 129280
        for batch_size in [1, 4, 16, 64, 128]:
            with self.subTest(batch_size=batch_size):
                logits = torch.randn(
                    batch_size,
                    vocab_size,
                    device=self.device,
                    dtype=torch.bfloat16,
                )
                sampling_info = _make_sampling_info(batch_size, vocab_size, self.device)

                out_aiter = self._run_sampler(True, logits, sampling_info)
                out_argmax = self._run_sampler(False, logits, sampling_info)

                self.assertTrue(
                    torch.equal(out_aiter.cpu(), out_argmax.cpu()),
                    f"Mismatch at batch_size={batch_size}",
                )


if __name__ == "__main__":
    unittest.main()
