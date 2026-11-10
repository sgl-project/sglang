"""Regression coverage for Laguna's regex-only INT4 expert scheme."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16MoE,
    CompressedTensorsWNA16TritonMoE,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
    _get_marlin_moe_scale_transform,
)
from sglang.srt.layers.quantization.marlin_utils import (
    marlin_moe_permute_scales,
    sm70_marlin_moe_logical_scales,
)


class TestLagunaCompressedTensors(unittest.TestCase):
    def test_regex_only_group_resolves_w4a16_moe(self):
        quant_config = CompressedTensorsConfig.from_config(
            {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {
                    "group_0": {
                        "targets": [
                            r"re:.*layers\.\d+\..*"
                            r"(w[1-3]|gate_proj|up_proj|down_proj)$"
                        ],
                        "weights": {
                            "dynamic": False,
                            "group_size": 32,
                            "num_bits": 4,
                            "strategy": "group",
                            "symmetric": True,
                            "type": "int",
                        },
                        "input_activations": None,
                    }
                },
                "ignore": [
                    "lm_head",
                    r"re:.*\.self_attn\..*$",
                    r"re:.*\.mlp\.shared_expert\..*$",
                ],
            }
        )

        self.assertNotIn("Linear", quant_config.target_scheme_map)
        scheme = quant_config.get_moe_scheme(
            torch.nn.Module(),
            layer_name="model.layers.1.mlp.experts",
        )

        self.assertIsInstance(
            scheme,
            (CompressedTensorsWNA16MoE, CompressedTensorsWNA16TritonMoE),
        )
        self.assertEqual(scheme.num_bits, 4)
        self.assertEqual(scheme.group_size, 32)

    def test_triton_w2_scales_apply_wide_output_compensation(self):
        scheme = CompressedTensorsWNA16TritonMoE.__new__(
            CompressedTensorsWNA16TritonMoE
        )
        scheme.moe_runner_config = SimpleNamespace(wide_output_scale=4.0)

        layer = torch.nn.Module()
        layer.w13_weight_packed = torch.nn.Parameter(
            torch.zeros((1, 2, 4), dtype=torch.int32), requires_grad=False
        )
        layer.w2_weight_packed = torch.nn.Parameter(
            torch.zeros((1, 2, 4), dtype=torch.int32), requires_grad=False
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.full((1, 2, 4), 8.0), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.full((1, 2, 4), 8.0), requires_grad=False
        )

        scheme.process_weights_after_loading(layer)

        torch.testing.assert_close(
            layer.w13_weight_scale,
            torch.full_like(layer.w13_weight_scale, 8.0),
        )
        torch.testing.assert_close(
            layer.w2_weight_scale,
            torch.full_like(layer.w2_weight_scale, 2.0),
        )

    def test_marlin_scale_layout_selects_sm70_logical_order(self):
        with unittest.mock.patch(
            "torch.cuda.get_device_capability", return_value=(7, 0)
        ):
            self.assertIs(
                _get_marlin_moe_scale_transform(torch.device("cuda")),
                sm70_marlin_moe_logical_scales,
            )

        self.assertIs(
            _get_marlin_moe_scale_transform(torch.device("cpu")),
            marlin_moe_permute_scales,
        )


if __name__ == "__main__":
    unittest.main()
