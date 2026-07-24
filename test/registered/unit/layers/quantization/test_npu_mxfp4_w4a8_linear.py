"""CPU regression tests for Ascend MXFP4 W4A8 Linear helpers."""

import unittest

import torch

from sglang.srt.hardware_backend.npu.quantization.linear_method_npu import (
    _prepare_mxfp4_w4a8_bias,
)
from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.npu_mxfp4 import (
    MXFP4_W4A8_GROUP_SIZE,
    Mxfp4W4A8Config,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import is_npu
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


class TestMxfp4W4A8ConfigGroupAlignment(CustomTestCase):
    """Guards the K (input dim) 32-alignment gate in ``get_quant_method``.

    The FP4 (A8W4) ``npu_quant_matmul`` requires the reduction dim K to be a
    multiple of the MXFP4 group size (32); it has no partial-last-block support.
    Qwen3.5's vision MLP ``linear_fc2`` has K=4304 (not 32-aligned) and used to
    crash the kernel with ``the k dim must to be aligned to 32, which is 4304``.
    Group-unaligned Linear layers must now fall back to unquantized BF16.
    """

    @staticmethod
    def _make_linear_layer(input_size: int) -> LinearBase:
        # Build a bare LinearBase without the full (distributed) __init__; only
        # input_size is read by the alignment gate. Mirrors test_modelslim_config.
        layer = LinearBase.__new__(LinearBase)
        torch.nn.Module.__init__(layer)
        layer.input_size = input_size
        return layer

    def test_skips_group_unaligned_linear(self):
        # K=4304 is not a multiple of 32 -> must fall back to BF16, not the FP4
        # W4A8 method (which would crash the kernel).
        self.assertNotEqual(4304 % MXFP4_W4A8_GROUP_SIZE, 0)
        layer = self._make_linear_layer(4304)

        method = Mxfp4W4A8Config().get_quant_method(
            layer, "model.visual.blocks.0.mlp.linear_fc2"
        )

        self.assertIsInstance(method, UnquantizedLinearMethod)

    def test_aligned_linear_not_skipped(self):
        # A 32-aligned K must NOT hit the fallback: on CPU (is_npu() False) the
        # method resolution reaches the NPU-only branch and raises, proving the
        # alignment gate did not short-circuit it.
        self.assertEqual(4096 % MXFP4_W4A8_GROUP_SIZE, 0)
        layer = self._make_linear_layer(4096)
        config = Mxfp4W4A8Config()

        if is_npu():
            self.skipTest("aligned-path fallback assertion is CPU-only")
        with self.assertRaises(NotImplementedError):
            config.get_quant_method(layer, "model.layers.0.self_attn.o_proj")


if __name__ == "__main__":
    unittest.main()
