"""CPU regression tests for ModelSlim MXFP4 W4A8 MoE support."""

import unittest

import torch

from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
    NPUMXFP4W4A8MoEMethod,
)
from sglang.srt.hardware_backend.npu.quantization.online_moe_methods import (
    NPUMXFP4W4A8FusedMoEMethod,
)
from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.srt.layers.quantization.modelslim.schemes import (
    ModelSlimMXFP4W4A8MoEScheme,
)
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestModelSlimMXFP4W4A8MoE(CustomTestCase):
    def test_online_method_reuses_generic_fused_moe_plumbing(self):
        self.assertTrue(
            issubclass(
                NPUMXFP4W4A8FusedMoEMethod,
                UnquantizedFusedMoEMethod,
            )
        )

    def test_config_resolves_standard_qwen_moe_projections(self):
        prefix = "model.layers.0.mlp.experts"
        quant_config = ModelSlimConfig(
            {
                f"{prefix}.0.gate_proj.weight": "W4A8_MXFP",
                f"{prefix}.0.up_proj.weight": "W4A8_MXFP",
                f"{prefix}.0.down_proj.weight": "W4A8_MXFP",
            }
        )

        w13_scheme, w2_scheme = quant_config.get_moe_scheme(torch.nn.Module(), prefix)

        self.assertIsInstance(w13_scheme, ModelSlimMXFP4W4A8MoEScheme)
        self.assertIsInstance(w2_scheme, ModelSlimMXFP4W4A8MoEScheme)
        self.assertEqual(w13_scheme.weight_prefix, "w13")
        self.assertEqual(w2_scheme.weight_prefix, "w2")

    def test_creates_packed_weights_scales_and_empty_offsets(self):
        layer = torch.nn.Module()
        w13_scheme = ModelSlimMXFP4W4A8MoEScheme({}, "w13")
        w2_scheme = ModelSlimMXFP4W4A8MoEScheme({}, "w2")

        for scheme in (w13_scheme, w2_scheme):
            scheme.create_weights(
                layer=layer,
                num_experts=2,
                hidden_size=64,
                intermediate_size_per_partition=96,
            )

        self.assertEqual(layer.w13_weight.shape, (2, 192, 32))
        self.assertEqual(layer.w13_weight_scale.shape, (2, 192, 2))
        self.assertIsNone(layer.w13_weight_offset)
        self.assertEqual(layer.w2_weight.shape, (2, 64, 48))
        self.assertEqual(layer.w2_weight_scale.shape, (2, 64, 3))
        self.assertIsNone(layer.w2_weight_offset)

    def test_normalizes_flat_and_pair_split_scales_to_same_layout(self):
        flat_scale = torch.arange(24, dtype=torch.uint8).reshape(2, 3, 4)
        pair_split_scale = flat_scale.reshape(2, 3, 2, 2)
        expected = pair_split_scale.transpose(-3, -2)

        self.assertTrue(
            torch.equal(NPUMXFP4W4A8MoEMethod._process_scale_fp4(flat_scale), expected)
        )
        self.assertTrue(
            torch.equal(
                NPUMXFP4W4A8MoEMethod._process_scale_fp4(pair_split_scale),
                expected,
            )
        )


if __name__ == "__main__":
    unittest.main()
