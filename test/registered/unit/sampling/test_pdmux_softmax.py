"""Sampling on SM partitions must avoid whole-device cluster launches."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers import sampler
from sglang.srt.multiplex import pdmux_context
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestPDMuxSoftmax(unittest.TestCase):
    def test_green_stream_uses_torch_without_mutating_logits(self):
        logits = torch.tensor([[1.0, -2.0, 4.0], [-5.0, 3.0, 0.0]])
        original = logits.clone()
        with (
            patch.object(pdmux_context, "_GREEN_CONTEXT_STREAMS", {17}),
            patch.object(
                pdmux_context.torch.cuda,
                "current_stream",
                return_value=SimpleNamespace(cuda_stream=17),
            ),
            patch.object(sampler, "sampler_softmax") as fast_softmax,
        ):
            probs = sampler.Sampler._softmax(
                SimpleNamespace(enable_deterministic=False), logits
            )
        fast_softmax.assert_not_called()
        torch.testing.assert_close(probs, original.softmax(dim=-1))
        torch.testing.assert_close(logits, original)

    def test_full_device_decode_keeps_fast_path(self):
        logits = torch.tensor([[1.0, 2.0]])
        expected = object()
        with (
            patch.object(pdmux_context, "_GREEN_CONTEXT_STREAMS", {17}),
            patch.object(
                pdmux_context.torch.cuda,
                "current_stream",
                return_value=SimpleNamespace(cuda_stream=18),
            ),
            patch.object(sampler, "sampler_softmax", return_value=expected) as fast,
        ):
            result = sampler.Sampler._softmax(
                SimpleNamespace(enable_deterministic=False), logits
            )
        fast.assert_called_once_with(logits)
        self.assertIs(result, expected)

    def test_deterministic_path_does_not_inspect_cuda_stream(self):
        logits = torch.tensor([[1.0, 2.0]])
        with patch.object(sampler, "is_pdmux_green_context_stream") as green:
            result = sampler.Sampler._softmax(
                SimpleNamespace(enable_deterministic=True), logits
            )
        green.assert_not_called()
        torch.testing.assert_close(result, logits.softmax(dim=-1))


if __name__ == "__main__":
    unittest.main()
