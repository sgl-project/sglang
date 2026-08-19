import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _make_fake_lm_head(vocab_size: int, in_features: int) -> nn.Module:
    head = nn.Module()
    head.weight = nn.Parameter(torch.randn(vocab_size, in_features))
    head.bias = None
    head.org_vocab_size = vocab_size
    head.quant_method = None
    return head


def _make_mixin(lm_head: nn.Module) -> DSparkDraftMixin:
    mixin = DSparkDraftMixin.__new__(DSparkDraftMixin)
    mixin.logits_mup_width_multiplier = None
    mixin.lm_head = lm_head
    return mixin


class TestDsparkQuantizedLmHead(CustomTestCase):
    """DSpark shares the target lm_head; a quantized target stores it packed.

    When the target model is quantized (e.g. NVFP4, weight layout
    [vocab, hidden*4/8] with per-block scales), a dense matmul against the
    packed layout is invalid. compute_base_logits must route through the
    quant method (the LogitsProcessor path) instead.
    """

    def setUp(self):
        self.patch_gather = patch(
            "sglang.srt.models.dspark.gather_and_crop_vocab",
            side_effect=lambda logits, lm_head: logits,
        )
        self.patch_gather.start()
        self.addCleanup(self.patch_gather.stop)

    def test_packed_weight_routes_to_quant_method(self):
        hidden = torch.randn(7, 5120)
        head = _make_fake_lm_head(vocab_size=320, in_features=2560)
        expected = torch.randn(7, 320)
        head.quant_method = MagicMock()
        head.quant_method.apply = MagicMock(return_value=expected)
        mixin = _make_mixin(head)

        logits, extra = mixin.compute_base_logits(hidden)

        head.quant_method.apply.assert_called_once_with(head, hidden, None)
        self.assertTrue(torch.equal(logits, expected))
        self.assertIsNone(extra)

    def test_unquantized_weight_uses_matmul(self):
        hidden = torch.randn(7, 5120)
        head = _make_fake_lm_head(vocab_size=320, in_features=5120)
        mixin = _make_mixin(head)

        logits, extra = mixin.compute_base_logits(hidden)

        expected = torch.matmul(hidden, head.weight.T)
        self.assertTrue(torch.allclose(logits, expected))
        self.assertIsNone(extra)

    def test_packed_weight_without_quant_method_raises(self):
        hidden = torch.randn(7, 5120)
        head = _make_fake_lm_head(vocab_size=320, in_features=2560)
        mixin = _make_mixin(head)

        with self.assertRaisesRegex(ValueError, "packed weight layout"):
            mixin.compute_base_logits(hidden)


if __name__ == "__main__":
    unittest.main()
