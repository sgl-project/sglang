import unittest
from types import SimpleNamespace

import torch

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
    def test_910b_w4a8_intermediate_padding_preserves_gate_up_pairs(self):
        from sglang.srt.layers.quantization.modelslim.schemes.modelslim_w4a8_int8_moe import (
            ModelSlimW4A8Int8MoE,
        )

        width = 288
        fields = {
            "w13_weight": torch.arange(width * 4).to(torch.int8).view(1, width, 4),
            "w13_weight_scale": torch.arange(width * 2).float().view(1, -1, 1),
            "w13_weight_offset": torch.ones(1, width * 2, 1),
            "w13_scale_bias": torch.arange(width * 2).float().view(1, -1, 1),
            "w2_weight": torch.arange(width * 2).to(torch.int8).view(1, 2, width),
        }
        layer = SimpleNamespace(
            **{
                name: torch.nn.Parameter(value.clone(), requires_grad=False)
                for name, value in fields.items()
            }
        )
        ModelSlimW4A8Int8MoE._align_910b_intermediate(layer)
        self.assertEqual(layer.intermediate_size_per_partition, 320)
        for name, before in fields.items():
            after = getattr(layer, name)
            if name == "w2_weight":
                torch.testing.assert_close(after[..., :width], before)
                self.assertEqual(after[..., width:].count_nonzero().item(), 0)
                continue
            for got, expected in zip(after.chunk(2, dim=1), before.chunk(2, dim=1)):
                torch.testing.assert_close(got[:, : expected.shape[1]], expected)
                fill = int(name == "w13_weight_scale")
                self.assertTrue((got[:, expected.shape[1] :] == fill).all())
        weight_pointer = layer.w13_weight.data_ptr()
        ModelSlimW4A8Int8MoE._align_910b_intermediate(layer)
        self.assertEqual(layer.w13_weight.data_ptr(), weight_pointer)

    def test_all_float_experts_are_skipped(self):
        config = ModelSlimConfig(_moe_quant_description("FLOAT", "FLOAT", "FLOAT"))

        self.assertTrue(config._is_moe_layer_skipped(_MOE_PREFIX))

    def test_all_float_w123_experts_are_skipped(self):
        config = ModelSlimConfig(_w123_quant_description("FLOAT", "FLOAT", "FLOAT"))

        self.assertTrue(config._is_moe_layer_skipped(_MOE_PREFIX))

    def test_quantized_experts_are_not_skipped(self):
        config = ModelSlimConfig(
            _moe_quant_description("W8A8_DYNAMIC", "W8A8_DYNAMIC", "W8A8_DYNAMIC")
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
