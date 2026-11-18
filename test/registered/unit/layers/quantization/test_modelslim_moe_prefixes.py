"""Unit tests for ModelSlim MoE checkpoint projection-name resolution."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.srt.layers.quantization.modelslim.schemes import (
    ModelSlimW8A8Int8MoE,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestModelSlimMoEPrefixes(CustomTestCase):
    # ModelSlimW8A8Int8MoE.__init__ builds NPUW8A8Int8MoEMethod, which resolves
    # torch.ops.npu.npu_dynamic_quant - absent on the CPU runner. Only that
    # construction is mocked; prefix resolution and scheme selection stay real.
    @patch(
        "sglang.srt.layers.quantization.modelslim.schemes."
        "modelslim_w8a8_int8_moe.NPUW8A8Int8MoEMethod"
    )
    def test_projection_name_families_resolve_to_fused_weight_groups(
        self, _mock_npu_moe_method
    ):
        """Bug regression: ModelSlim checkpoints may describe experts with
        gate_proj/up_proj/down_proj or MiniMax-style w1/w3/w2 names. Both must
        select the W13 and W2 schemes instead of reporting missing metadata.
        """
        prefix = "model.layers.0.mlp.experts"
        projection_name_families = [
            ("gate_proj", "up_proj", "down_proj"),
            ("w1", "w3", "w2"),
        ]

        for gate_name, up_name, down_name in projection_name_families:
            with self.subTest(
                gate_name=gate_name, up_name=up_name, down_name=down_name
            ):
                quant_description = {
                    f"{prefix}.0.{gate_name}.weight": "W8A8_DYNAMIC",
                    f"{prefix}.0.{up_name}.weight": "W8A8_DYNAMIC",
                    f"{prefix}.0.{down_name}.weight": "W8A8_DYNAMIC",
                }
                config = ModelSlimConfig(quant_description)

                w13_scheme, w2_scheme = config.get_moe_scheme(torch.nn.Module(), prefix)

                self.assertIsInstance(w13_scheme, ModelSlimW8A8Int8MoE)
                self.assertIsInstance(w2_scheme, ModelSlimW8A8Int8MoE)
                self.assertEqual(w13_scheme.weight_prefix, "w13")
                self.assertEqual(w2_scheme.weight_prefix, "w2")


if __name__ == "__main__":
    unittest.main()
