import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from sglang.srt.models.dspark import DSparkDraftMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


class _FakePackedHead(nn.Module):
    """A minimal NVFP4-style packed lm_head: 2 fp4 values per uint8 byte.

    quant_method.apply dequantizes the packed weight and runs the dense
    matmul, standing in for the real ModelOpt/Marlin apply path.
    """

    def __init__(self, vocab_size: int, in_features: int):
        super().__init__()
        assert in_features % 2 == 0
        # Quantized weights are plain tensors, not nn.Parameter (uint8 cannot
        # carry requires_grad).
        self.weight = torch.randint(
            0, 256, (vocab_size, in_features // 2), dtype=torch.uint8
        )
        self.bias = None
        self.org_vocab_size = vocab_size
        self.quant_method = self

    @staticmethod
    def _dequant(weight: torch.Tensor) -> torch.Tensor:
        low = (weight & 0x0F).float() / 15.0
        high = (weight >> 4).float() / 15.0
        return torch.stack([low, high], dim=-1).reshape(
            weight.shape[0], weight.shape[1] * 2
        )

    def apply(self, layer: nn.Module, x: torch.Tensor, bias=None) -> torch.Tensor:
        return torch.matmul(x, self._dequant(layer.weight).T)


def _make_unquantized_head(vocab_size: int, in_features: int) -> nn.Module:
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
    """DSpark shares the target lm_head; a quantized target needs the quant path.

    When the target model is quantized (NVFP4, FP8, W8A8, GPTQ, ...), a dense
    matmul against the raw weight is invalid or incorrect. compute_base_logits
    must mirror LogitsProcessor._compute_lm_head and route through the quant
    method instead.
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
        head = _make_unquantized_head(vocab_size=320, in_features=2560)
        expected = torch.randn(7, 320)
        head.quant_method = MagicMock()
        head.quant_method.apply = MagicMock(return_value=expected)
        mixin = _make_mixin(head)

        logits, extra = mixin.compute_base_logits(hidden)

        head.quant_method.apply.assert_called_once_with(head, hidden, None)
        self.assertTrue(torch.equal(logits, expected))
        self.assertIsNone(extra)

    def test_packed_weight_numerically_matches_dequantized_matmul(self):
        # Real dequantize path (not a mock): the quant-method output must equal
        # a dense matmul against the dequantized weight.
        hidden = torch.randn(7, 5120)
        head = _FakePackedHead(vocab_size=320, in_features=5120)
        mixin = _make_mixin(head)

        logits, extra = mixin.compute_base_logits(hidden)

        expected = torch.matmul(hidden, head._dequant(head.weight).T)
        self.assertTrue(torch.allclose(logits, expected, atol=1e-6))
        self.assertIsNone(extra)

    def test_unquantized_weight_uses_matmul(self):
        hidden = torch.randn(7, 5120)
        head = _make_unquantized_head(vocab_size=320, in_features=5120)
        mixin = _make_mixin(head)

        logits, extra = mixin.compute_base_logits(hidden)

        expected = torch.matmul(hidden, head.weight.T)
        self.assertTrue(torch.allclose(logits, expected))
        self.assertIsNone(extra)


if __name__ == "__main__":
    unittest.main()
