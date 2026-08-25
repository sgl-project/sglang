"""CPU regression tests for W4A8 MXFP4 MoE support on Ascend NPU."""

import unittest

import torch

from sglang.srt.hardware_backend.npu.quantization.online_moe_methods import (
    NPUW4A8MXFP4OnlineMoEMethod,
)
from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimW4A8MXFP4MoE
from sglang.srt.layers.quantization.unquant import UnquantizedFusedMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestNPUW4A8MXFP4MoE(CustomTestCase):
    def test_online_method_stays_on_the_unquant_fused_moe_base(self):
        """FusedMoE picks its w1/w3 shard order from this base class.

        Dropping it makes the weight loader swap gate and up on every expert, so
        gmm1 computes silu(up) * gate with no error -- only degenerate output.
        """
        self.assertTrue(
            issubclass(NPUW4A8MXFP4OnlineMoEMethod, UnquantizedFusedMoEMethod)
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

        self.assertIsInstance(w13_scheme, ModelSlimW4A8MXFP4MoE)
        self.assertIsInstance(w2_scheme, ModelSlimW4A8MXFP4MoE)
        self.assertEqual(w13_scheme.weight_prefix, "w13")
        self.assertEqual(w2_scheme.weight_prefix, "w2")

    def test_creates_packed_weights_and_ceil_divided_scales(self):
        """K=80 is not a multiple of the 32-element block.

        A truncating scale count would allocate 2 scales instead of 3 and drop
        the last partial block, so the checkpoint fails to load.
        """
        layer = torch.nn.Module()
        for weight_prefix in ("w13", "w2"):
            ModelSlimW4A8MXFP4MoE({}, weight_prefix).create_weights(
                layer=layer,
                num_experts=2,
                hidden_size=64,
                intermediate_size_per_partition=80,
            )

        self.assertEqual(layer.w13_weight.shape, (2, 160, 32))
        self.assertEqual(layer.w13_weight_scale.shape, (2, 160, 2))
        self.assertEqual(layer.w2_weight.shape, (2, 64, 40))
        self.assertEqual(layer.w2_weight_scale.shape, (2, 64, 3))


if __name__ == "__main__":
    unittest.main()
