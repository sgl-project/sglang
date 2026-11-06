"""W4AFP8 expert method selection for optional MXFP4 Mega conversion."""

import unittest
from unittest.mock import Mock, patch

from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization import w4afp8
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestW4AFP8Config(CustomTestCase):
    def test_expert_quantization_is_preserved(self):
        from sglang.test.test_utils import published_topology

        config = w4afp8.W4AFp8Config.from_config(
            {"quant_method": "w4afp8", "modules_to_not_convert": ["mlp.gate"]}
        )
        with published_topology():
            self.assertIsInstance(
                config.get_quant_method(
                    Mock(spec=FusedMoE), "model.layers.0.mlp.experts"
                ),
                w4afp8.W4AFp8MoEMethod,
            )

    def test_mxfp4_selects_conversion_from_unpacked_checkpoint(self):
        config = w4afp8.W4AFp8Config.from_config({"quant_method": "w4afp8"})
        # Check method selection without requiring a CUDA device on CPU CI.
        with patch("sglang.srt.layers.moe.sm90_mxfp4_mega.enabled", return_value=True):
            method = config.get_quant_method(
                Mock(spec=FusedMoE), "model.layers.0.mlp.experts"
            )
        self.assertIsInstance(method, w4afp8.W4AFp8Mxfp4MegaMoEMethod)
        self.assertIs(method.quant_config, config)


if __name__ == "__main__":
    unittest.main()
