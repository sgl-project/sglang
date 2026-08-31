import types
import unittest

import torch

from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo
from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")
register_cpu_ci(est_time=2, suite="base-b-test-cpu")


class TestMxfp4GetTritonQuantInfo(CustomTestCase):
    """Mxfp4MoEMethod.get_triton_quant_info wires the (bf16-upcast) w13/w2 and
    their biases into a TritonMoeQuantInfo for the MoE LoRA triton runner.

    The method only reads ``layer`` attributes, so it is exercised here as an
    unbound call with a lightweight stub layer -- no GPU or weight loading is
    required, keeping this a CPU unit test.
    """

    def _make_layer(self, with_bias: bool = True) -> types.SimpleNamespace:
        layer = types.SimpleNamespace()
        layer.w13_weight = torch.randn(4, 16, 8, dtype=torch.bfloat16)
        layer.w2_weight = torch.randn(4, 8, 8, dtype=torch.bfloat16)
        if with_bias:
            layer.w13_weight_bias = torch.randn(4, 16, dtype=torch.float32)
            layer.w2_weight_bias = torch.randn(4, 8, dtype=torch.float32)
        return layer

    def test_returns_triton_quant_info_passing_through_weights_and_biases(self):
        layer = self._make_layer(with_bias=True)
        info = Mxfp4MoEMethod.get_triton_quant_info(object(), layer)

        self.assertIsInstance(info, TritonMoeQuantInfo)
        # The same tensors are passed through (identity, no copy).
        self.assertIs(info.w13_weight, layer.w13_weight)
        self.assertIs(info.w2_weight, layer.w2_weight)
        self.assertIs(info.b13, layer.w13_weight_bias)
        self.assertIs(info.b2, layer.w2_weight_bias)

    def test_missing_biases_default_to_none(self):
        layer = self._make_layer(with_bias=False)
        info = Mxfp4MoEMethod.get_triton_quant_info(object(), layer)

        self.assertIsInstance(info, TritonMoeQuantInfo)
        self.assertIs(info.w13_weight, layer.w13_weight)
        self.assertIs(info.w2_weight, layer.w2_weight)
        self.assertIsNone(info.b13)
        self.assertIsNone(info.b2)


if __name__ == "__main__":
    unittest.main()
