import unittest

from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_MOE_PREFIX = "mtp.layers.0.mlp.experts"


def _moe_quant_description(gate: str, up: str, down: str):
    return {
        f"{_MOE_PREFIX}.0.gate_proj.weight": gate,
        f"{_MOE_PREFIX}.0.up_proj.weight": up,
        f"{_MOE_PREFIX}.0.down_proj.weight": down,
    }


def _w123_quant_description(w1: str, w3: str, w2: str):
    return {
        f"{_MOE_PREFIX}.0.w1.weight": w1,
        f"{_MOE_PREFIX}.0.w3.weight": w3,
        f"{_MOE_PREFIX}.0.w2.weight": w2,
    }


class TestModelSlimMoESkip(CustomTestCase):
    def test_all_float_experts_are_skipped(self):
        config = ModelSlimConfig(_moe_quant_description("FLOAT", "FLOAT", "FLOAT"))

        self.assertTrue(config._is_moe_layer_skipped(_MOE_PREFIX))

    def test_all_float_w123_experts_are_skipped(self):
        config = ModelSlimConfig(_w123_quant_description("FLOAT", "FLOAT", "FLOAT"))

        self.assertTrue(config._is_moe_layer_skipped(_MOE_PREFIX))

    def test_quantized_experts_are_not_skipped(self):
        config = ModelSlimConfig(
            _moe_quant_description(
                "W8A8_DYNAMIC", "W8A8_DYNAMIC", "W8A8_DYNAMIC"
            )
        )

        self.assertFalse(config._is_moe_layer_skipped(_MOE_PREFIX))

    def test_mixed_precision_experts_raise(self):
        config = ModelSlimConfig(
            _moe_quant_description("FLOAT", "FLOAT", "W8A8_DYNAMIC")
        )

        with self.assertRaisesRegex(ValueError, "some but not all shards"):
            config._is_moe_layer_skipped(_MOE_PREFIX)

    def test_unknown_scheme_is_not_treated_as_unquantized(self):
        config = ModelSlimConfig(
            _moe_quant_description("UNKNOWN", "UNKNOWN", "UNKNOWN")
        )

        self.assertFalse(config._is_moe_layer_skipped(_MOE_PREFIX))


if __name__ == "__main__":
    unittest.main()
