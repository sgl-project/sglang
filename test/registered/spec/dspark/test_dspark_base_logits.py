"""DSpark base-logit projection tests."""

import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=30, suite="base-a-test-cpu")


class _FakeQuantMethod:
    def __init__(self):
        self.called = False

    def apply(self, layer, hidden, bias):
        del bias
        self.called = True
        return hidden.new_zeros((hidden.shape[0], layer.org_vocab_size))


class TestDSparkBaseLogits(CustomTestCase):
    def test_quantized_shared_lm_head_uses_quant_method(self):
        quant_method = _FakeQuantMethod()
        # Packed weights intentionally have a physical K dimension that does
        # not match the draft hidden size, like an NVFP4 LM head.
        lm_head = types.SimpleNamespace(
            weight=torch.empty((16, 4), dtype=torch.uint8),
            org_vocab_size=16,
            quant_method=quant_method,
        )
        model = types.SimpleNamespace(
            lm_head=lm_head,
            logits_mup_width_multiplier=None,
        )
        compute_base_logits = DSparkDraftMixin.compute_base_logits.__get__(
            model, type(model)
        )

        with patch(
            "sglang.srt.models.dspark.tensor_model_parallel_all_gather",
            side_effect=lambda logits, dim: logits,
        ):
            logits, confidence = compute_base_logits(torch.randn(2, 8))

        self.assertTrue(quant_method.called)
        self.assertEqual((2, 16), tuple(logits.shape))
        self.assertIsNone(confidence)


if __name__ == "__main__":
    unittest.main()
