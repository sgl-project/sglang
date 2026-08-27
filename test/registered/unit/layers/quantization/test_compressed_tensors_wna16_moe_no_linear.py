import unittest
from unittest import mock

import torch

from sglang.srt.layers.moe import MoeRunnerBackend
from sglang.srt.layers.quantization.compressed_tensors import compressed_tensors
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16MoE,
    CompressedTensorsWNA16TritonMoE,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_WNA16_MOE_SCHEMES = (CompressedTensorsWNA16MoE, CompressedTensorsWNA16TritonMoE)
EXPERTS_LAYER = "model.layers.0.mlp.experts"
PER_LAYER_EXPERT_TARGETS = [
    f"{EXPERTS_LAYER}.0.gate_proj",
    f"{EXPERTS_LAYER}.0.up_proj",
    f"{EXPERTS_LAYER}.0.down_proj",
]


def _make_wna16_moe_config(targets, num_bits, **weight_overrides):
    weights = {
        "num_bits": num_bits,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": 128,
    }
    weights.update(weight_overrides)
    return {
        "quant_method": "compressed-tensors",
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": targets,
                "weights": weights,
                "input_activations": None,
            }
        },
        "ignore": ["lm_head", "re:.*self_attn.*", "re:.*mlp.gate$"],
    }


class TestWNA16MoENoLinearGroup(CustomTestCase):
    def _assert_wna16_moe(self, config_dict, expected_bits):
        quant_config = CompressedTensorsConfig.from_config(config_dict)
        self.assertNotIn("Linear", quant_config.target_scheme_map)

        layer = torch.nn.Module()
        scheme = quant_config.get_moe_scheme(layer, layer_name=EXPERTS_LAYER)

        self.assertIsInstance(scheme, _WNA16_MOE_SCHEMES)
        self.assertEqual(scheme.num_bits, expected_bits)
        self.assertEqual(scheme.group_size, 128)

    def test_regex_expert_targets_int4(self):
        config = _make_wna16_moe_config(["re:.*mlp.experts.*"], num_bits=4)
        self._assert_wna16_moe(config, expected_bits=4)

    def test_regex_expert_targets_int8(self):
        config = _make_wna16_moe_config(["re:.*mlp.experts.*"], num_bits=8)
        self._assert_wna16_moe(config, expected_bits=8)

    def test_per_layer_fqn_expert_targets_int4(self):
        config = _make_wna16_moe_config(PER_LAYER_EXPERT_TARGETS, num_bits=4)
        self._assert_wna16_moe(config, expected_bits=4)

    def test_blackwell_int4_auto_uses_triton(self):
        for group_size in (32, 128):
            with self.subTest(group_size=group_size):
                quant_config = CompressedTensorsConfig.from_config(
                    _make_wna16_moe_config(
                        ["re:.*mlp.experts.*"],
                        num_bits=4,
                        group_size=group_size,
                    )
                )

                with (
                    mock.patch.object(
                        compressed_tensors,
                        "get_moe_runner_backend",
                        return_value=MoeRunnerBackend.AUTO,
                    ),
                    mock.patch.object(
                        compressed_tensors, "is_sm100_supported", return_value=True
                    ),
                ):
                    scheme = quant_config.get_moe_scheme(
                        torch.nn.Module(), layer_name=EXPERTS_LAYER
                    )

                self.assertIsInstance(scheme, CompressedTensorsWNA16TritonMoE)

    def test_blackwell_auto_rejects_unvalidated_triton_layouts(self):
        cases = {
            "asymmetric": {"symmetric": False},
            "channel": {"strategy": "channel", "group_size": None},
            "group64": {"group_size": 64},
            "actorder": {"actorder": "group"},
        }
        for name, overrides in cases.items():
            with self.subTest(name=name):
                quant_config = CompressedTensorsConfig.from_config(
                    _make_wna16_moe_config(
                        ["re:.*mlp.experts.*"], num_bits=4, **overrides
                    )
                )
                with (
                    mock.patch.object(
                        compressed_tensors,
                        "get_moe_runner_backend",
                        return_value=MoeRunnerBackend.AUTO,
                    ),
                    mock.patch.object(
                        compressed_tensors, "is_sm100_supported", return_value=True
                    ),
                ):
                    scheme = quant_config.get_moe_scheme(
                        torch.nn.Module(), layer_name=EXPERTS_LAYER
                    )

                self.assertIsInstance(scheme, CompressedTensorsWNA16MoE)
                self.assertNotIsInstance(scheme, CompressedTensorsWNA16TritonMoE)

    def test_explicit_triton_rejects_unvalidated_layout(self):
        quant_config = CompressedTensorsConfig.from_config(
            _make_wna16_moe_config(["re:.*mlp.experts.*"], num_bits=4, symmetric=False)
        )

        with (
            mock.patch.object(
                compressed_tensors,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.TRITON,
            ),
            self.assertRaisesRegex(ValueError, "only supports symmetric INT4"),
        ):
            quant_config.get_moe_scheme(torch.nn.Module(), layer_name=EXPERTS_LAYER)

    def test_blackwell_explicit_marlin_is_preserved(self):
        quant_config = CompressedTensorsConfig.from_config(
            _make_wna16_moe_config(["re:.*mlp.experts.*"], num_bits=4)
        )

        with (
            mock.patch.object(
                compressed_tensors,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.MARLIN,
            ),
            mock.patch.object(
                compressed_tensors, "is_sm100_supported", return_value=True
            ),
        ):
            scheme = quant_config.get_moe_scheme(
                torch.nn.Module(), layer_name=EXPERTS_LAYER
            )

        self.assertIsInstance(scheme, CompressedTensorsWNA16MoE)

    def test_blackwell_int8_auto_keeps_marlin(self):
        quant_config = CompressedTensorsConfig.from_config(
            _make_wna16_moe_config(["re:.*mlp.experts.*"], num_bits=8)
        )

        with (
            mock.patch.object(
                compressed_tensors,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.AUTO,
            ),
            mock.patch.object(
                compressed_tensors, "is_sm100_supported", return_value=True
            ),
        ):
            scheme = quant_config.get_moe_scheme(
                torch.nn.Module(), layer_name=EXPERTS_LAYER
            )

        self.assertIsInstance(scheme, CompressedTensorsWNA16MoE)
        self.assertNotIsInstance(scheme, CompressedTensorsWNA16TritonMoE)


if __name__ == "__main__":
    unittest.main()
