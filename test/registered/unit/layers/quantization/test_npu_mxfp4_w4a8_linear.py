"""CPU regression tests for Ascend MXFP4 W4A8 Linear helpers."""

import unittest

import torch

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    _prepare_mxfp4_w4a8_bias,
)
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.npu_mxfp4 import Mxfp4W4A8Config
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNPUMXFP4W4A8Linear(CustomTestCase):
    def test_preserves_no_bias_path(self):
        self.assertIsNone(_prepare_mxfp4_w4a8_bias(None))

    def test_formats_vector_bias_for_a8w4_matmul(self):
        bias = torch.arange(8, dtype=torch.float32)

        quant_bias = _prepare_mxfp4_w4a8_bias(bias)

        self.assertEqual(quant_bias.dtype, torch.bfloat16)
        self.assertEqual(quant_bias.shape, (1, 8))
        self.assertTrue(torch.equal(quant_bias, bias.to(torch.bfloat16).unsqueeze(0)))

    def test_preserves_matrix_shape(self):
        bias = torch.arange(8, dtype=torch.bfloat16).reshape(1, 8)

        quant_bias = _prepare_mxfp4_w4a8_bias(bias)

        self.assertEqual(quant_bias.dtype, torch.bfloat16)
        self.assertEqual(quant_bias.shape, (1, 8))
        self.assertIs(quant_bias, bias)


class TestMxfp4W4A8ConfigLinearDispatch(CustomTestCase):
    """Online W4A8 is limited to MoE experts during accuracy isolation."""

    @staticmethod
    def _make_linear_layer(input_size: int) -> LinearBase:
        # Build a bare LinearBase without the full (distributed) __init__; only
        # input_size is read by the alignment gate. Mirrors test_modelslim_config.
        layer = LinearBase.__new__(LinearBase)
        torch.nn.Module.__init__(layer)
        layer.input_size = input_size
        return layer

    def test_all_linear_layers_fall_back_to_bf16(self):
        config = Mxfp4W4A8Config()
        cases = (
            (4096, "model.layers.0.self_attn.o_proj"),
            (4304, "model.visual.blocks.0.mlp.linear_fc2"),
            (3072, "model.layers.0.linear_attn.out_proj"),
        )

        for input_size, prefix in cases:
            with self.subTest(prefix=prefix):
                method = config.get_quant_method(
                    self._make_linear_layer(input_size), prefix
                )
                self.assertIsInstance(method, UnquantizedLinearMethod)


if __name__ == "__main__":
    unittest.main()
