"""CPU-only regression tests for ModelSlim quantization config resolution."""

import unittest

import torch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.modelslim.modelslim import (
    ModelSlimConfig,
    ModelSlimLinearMethod,
)
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimMXFP8Scheme
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestModelSlimPackedModuleResolution(CustomTestCase):
    def setUp(self):
        super().setUp()
        prefix = "model.language_model.layers.0.linear_attn"
        self.quant_config = ModelSlimConfig(
            {
                f"{prefix}.in_proj_qkv.weight": "W8A8_MXFP8",
                f"{prefix}.in_proj_z.weight": "W8A8_MXFP8",
                f"{prefix}.in_proj_b.weight": "W8A8_MXFP8",
                f"{prefix}.in_proj_a.weight": "W8A8_MXFP8",
                "packed_modules_mapping": {
                    # Model-specific mappings exposed by Qwen3.5.
                    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
                    "in_proj_ba": ["in_proj_b", "in_proj_a"],
                    # Generic NPU mappings are grouped by model scope.
                    "model": {
                        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
                        "gate_up_proj": ["gate_proj", "up_proj"],
                    },
                },
            }
        )
        self.prefix = prefix

    @staticmethod
    def _make_linear_layer() -> LinearBase:
        layer = LinearBase.__new__(LinearBase)
        torch.nn.Module.__init__(layer)
        return layer

    def test_resolves_qwen35_gdn_packed_projections(self):
        for projection in ("in_proj_qkvz", "in_proj_ba"):
            with self.subTest(projection=projection):
                layer = self._make_linear_layer()
                method = self.quant_config.get_quant_method(
                    layer, f"{self.prefix}.{projection}"
                )

                self.assertIsInstance(method, ModelSlimLinearMethod)
                self.assertIsInstance(layer.scheme, ModelSlimMXFP8Scheme)

    def test_resolves_qwen35_visual_projections(self):
        visual_prefix = "model.visual.blocks.0"
        quant_config = ModelSlimConfig(
            {
                f"{visual_prefix}.attn.qkv.weight": "W8A8_MXFP8",
                f"{visual_prefix}.attn.proj.weight": "W8A8_MXFP8",
                f"{visual_prefix}.mlp.linear_fc1.weight": "W8A8_MXFP8",
                f"{visual_prefix}.mlp.linear_fc2.weight": "W8A8_MXFP8",
                "packed_modules_mapping": {
                    "visual": {
                        "qkv_proj": ["qkv"],
                    }
                },
            }
        )

        for projection in (
            "attn.qkv_proj",
            "attn.proj",
            "mlp.linear_fc1",
            "mlp.linear_fc2",
        ):
            with self.subTest(projection=projection):
                layer = self._make_linear_layer()
                method = quant_config.get_quant_method(
                    layer, f"{visual_prefix}.{projection}"
                )

                self.assertIsInstance(method, ModelSlimLinearMethod)
                self.assertIsInstance(layer.scheme, ModelSlimMXFP8Scheme)


if __name__ == "__main__":
    unittest.main()
