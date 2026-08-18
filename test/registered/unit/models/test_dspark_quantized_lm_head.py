import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _QuantMethod:
    def __init__(self, output: torch.Tensor):
        self.output = output
        self.calls = []

    def apply(self, layer, hidden, bias=None):
        self.calls.append((layer, hidden, bias))
        return self.output


def _compute_base_logits(lm_head, hidden):
    model = types.SimpleNamespace(lm_head=lm_head, logits_mup_width_multiplier=None)
    return DSparkDraftMixin.compute_base_logits(model, hidden)


class TestDSparkQuantizedLmHead(CustomTestCase):
    @patch(
        "sglang.srt.models.dspark.tensor_model_parallel_all_gather",
        side_effect=lambda value, dim: value,
    )
    def test_quantized_lm_head_uses_quant_method(self, _gather):
        hidden = torch.randn(3, 8)
        expected = torch.randn(3, 11)
        lm_head = nn.Module()
        # Simulate packed storage: its physical input dimension is half of the
        # logical hidden size and therefore cannot be used by torch.matmul.
        lm_head.weight = nn.Parameter(
            torch.empty(11, 4, dtype=torch.uint8), requires_grad=False
        )
        lm_head.org_vocab_size = 11
        lm_head.quant_method = _QuantMethod(expected)

        logits, confidence_tap = _compute_base_logits(lm_head, hidden)

        self.assertIsNone(confidence_tap)
        self.assertTrue(torch.equal(logits, expected))
        self.assertEqual(len(lm_head.quant_method.calls), 1)
        layer, actual_hidden, bias = lm_head.quant_method.calls[0]
        self.assertIs(layer, lm_head)
        self.assertIs(actual_hidden, hidden)
        self.assertIsNone(bias)

    @patch(
        "sglang.srt.models.dspark.tensor_model_parallel_all_gather",
        side_effect=lambda value, dim: value,
    )
    def test_dense_lm_head_keeps_matmul_path(self, _gather):
        hidden = torch.randn(3, 8, dtype=torch.float32)
        lm_head = nn.Module()
        lm_head.weight = nn.Parameter(torch.randn(13, 8, dtype=torch.float16))
        lm_head.org_vocab_size = 11

        logits, confidence_tap = _compute_base_logits(lm_head, hidden)

        expected = torch.matmul(hidden.to(torch.float16), lm_head.weight.T)[..., :11]
        self.assertIsNone(confidence_tap)
        self.assertTrue(torch.equal(logits, expected))


if __name__ == "__main__":
    unittest.main()
