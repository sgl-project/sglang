"""MPS greedy sampling is opt-in and leaves eager Torch dispatch unchanged."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.test_utils import CustomTestCase

register_mps_ci(est_time=10, suite="stage-a-unit-test-mps")


@unittest.skipUnless(torch.backends.mps.is_available(), "Requires Torch MPS")
class TestMpsArgmaxDispatch(CustomTestCase):
    def test_public_argmax_opt_in_preserves_lowest_index_ties(self):
        from sglang.kernels.ops.sampling import argmax

        logits = torch.tensor([[2.0, 2.0, 1.0], [-3.0, -2.0, -2.0]], device="mps")
        for opt_in in (False, True):
            with self.subTest(opt_in=opt_in):
                actual = argmax(logits, use_mps_kernel=opt_in)
                torch.testing.assert_close(actual, torch.tensor([0, 1], device="mps"))

    def test_eager_sampler_does_not_use_compiled_mps_kernel(self):
        from sglang.srt.layers.logits_processor import LogitsProcessorOutput
        from sglang.srt.layers.sampler import Sampler
        from sglang.srt.runtime_context import get_context

        sampler = Sampler.__new__(Sampler)
        torch.nn.Module.__init__(sampler)
        info = SimpleNamespace(
            is_all_greedy=True,
            has_custom_logit_processor=False,
            sampling_mask_batch_indices=None,
            sampling_support_logprobs_capture_indices=None,
            grammars=None,
        )
        logits = torch.tensor([[1.0, 3.0, 3.0]], device="mps")
        with (
            get_context().override_server_args(
                device="mps", mps_execution_backend="eager"
            ),
            patch(
                "sglang.kernels.ops.sampling.mps.greedy_argmax",
                side_effect=AssertionError(
                    "compiled-only kernel used by eager sampler"
                ),
            ),
        ):
            actual = sampler(
                LogitsProcessorOutput(next_token_logits=logits),
                info,
                False,
                [],
                [],
                torch.tensor([0], device="mps"),
            )
        torch.testing.assert_close(actual, torch.tensor([1], device="mps"))


if __name__ == "__main__":
    unittest.main()
