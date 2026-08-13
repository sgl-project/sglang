import unittest
from unittest.mock import patch

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1, suite="stage-a-unit-test-npu")

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod


class TestNPUFP4MethodGate(unittest.TestCase):
    def test_pre_a5_keeps_fp8_moe_method(self):
        config = Fp8Config(is_fp4_experts=True)
        layer = FusedMoE.__new__(FusedMoE)

        with (
            patch("sglang.srt.layers.quantization.fp8.is_npu", return_value=True),
            patch(
                "sglang.srt.layers.quantization.fp8.has_npu_a5_support",
                return_value=False,
            ),
        ):
            method = config.get_quant_method(layer, "model.layers.0.experts")

        self.assertIsInstance(method, Fp8MoEMethod)


if __name__ == "__main__":
    unittest.main()
