"""CPU regression tests for the Ascend MXFP4 W4A4 online config (Linear gate)."""

import unittest

import torch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.npu_mxfp4_w4a4 import (
    MXFP4_W4A4_GROUP_SIZE,
    Mxfp4W4A4Config,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMxfp4W4A4ConfigGroupAlignment(CustomTestCase):
    """Guards the K (input dim) 32-alignment gate in ``get_quant_method``.

    MXFP4 block scales use group_size=32, so a reduction dim K that is not a
    multiple of 32 does not tile evenly. Qwen3.5's vision MLP ``linear_fc2`` has
    K=4304 (4304/32=134.5); serving Qwen3.5-VL online with ``--quantization
    mxfp4`` must fall back to BF16 for it rather than route it to the dual-level
    FP4 method, mirroring how the offline msmodelslim yaml excludes linear_fc2.
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
        # K=4304 is not a multiple of 32 -> must fall back to BF16, not the
        # dual-level FP4 method (whose block scales cannot tile K=4304).
        self.assertNotEqual(4304 % MXFP4_W4A4_GROUP_SIZE, 0)
        layer = self._make_linear_layer(4304)

        method = Mxfp4W4A4Config().get_quant_method(
            layer, "model.visual.blocks.0.mlp.linear_fc2"
        )

        self.assertIsInstance(method, UnquantizedLinearMethod)

    def test_aligned_linear_not_skipped(self):
        # A 32-aligned K must NOT hit the fallback: on CPU (is_npu() False) the
        # method resolution reaches the NPU-only branch and raises, proving the
        # alignment gate did not short-circuit it.
        self.assertEqual(4096 % MXFP4_W4A4_GROUP_SIZE, 0)
        layer = self._make_linear_layer(4096)
        config = Mxfp4W4A4Config()

        if is_npu():
            self.skipTest("aligned-path fallback assertion is CPU-only")
        with self.assertRaises(NotImplementedError):
            config.get_quant_method(layer, "model.layers.0.self_attn.o_proj")


if __name__ == "__main__":
    unittest.main()
