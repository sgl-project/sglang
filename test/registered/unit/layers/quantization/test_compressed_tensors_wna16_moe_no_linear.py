"""CPU regression test for WNA16 compressed-tensors MoE with no "Linear" group.

CompressedTensorsWNA16MoE used to read ``target_scheme_map["Linear"]`` in its
constructor. That raised ``KeyError: 'Linear'`` for compressed-tensors MoE
checkpoints whose ``config_groups`` only target the expert projections through a
regex or per-layer FQN target and therefore have no group literally named
"Linear" (e.g. mixed-precision INT4/INT8 MoE quant configs). ``get_moe_scheme``
already resolves the per-layer weight scheme by matching the layer against the
config_groups targets, so it now threads that ``weight_quant`` into the scheme
constructor instead of assuming a "Linear" group.

These tests pin that contract: building a MoE compressed-tensors config with no
"Linear" group and calling ``get_moe_scheme`` must return the correct WNA16 MoE
scheme rather than raising ``KeyError``. This is pure config-parsing logic (no
weights are created and no kernels run), so it runs on CPU.

The configs mirror real Laguna-style MoE quant configs: WNA16 int4/int8, group
strategy, group_size 128, symmetric, expert projections targeted by regex or by
per-layer FQN, with attention / router layers ignored.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16MoE,
    CompressedTensorsWNA16TritonMoE,
)
from sglang.test.test_utils import CustomTestCase

# WNA16 MoE Marlin (default) and Triton backends are both valid resolutions for
# this config; only the "no KeyError, correct WNA16 int-N scheme" contract matters.
_WNA16_MOE_SCHEMES = (CompressedTensorsWNA16MoE, CompressedTensorsWNA16TritonMoE)

# Layer whose experts we resolve a scheme for. get_moe_scheme() expands this into
# ".0.gate_proj" / ".0.up_proj" / ".0.down_proj" and matches each against targets.
EXPERTS_LAYER = "model.layers.0.mlp.experts"

# Per-layer FQN targets: the three expert projections of layer 0, named
# explicitly rather than via regex. Still no "Linear" group.
PER_LAYER_EXPERT_TARGETS = [
    f"{EXPERTS_LAYER}.0.gate_proj",
    f"{EXPERTS_LAYER}.0.up_proj",
    f"{EXPERTS_LAYER}.0.down_proj",
]


def _make_wna16_moe_config(targets, num_bits):
    """A WNA16 compressed-tensors MoE quant config with NO "Linear" group.

    Only the expert projections are quantized, targeted via ``targets`` (regex or
    per-layer FQN). Attention / router / lm_head are ignored, exactly as a real
    mixed-precision MoE checkpoint would express it.
    """
    return {
        "quant_method": "compressed-tensors",
        # pack-quantized => WNA16 (weight-only, int, no input activations).
        "format": "pack-quantized",
        "config_groups": {
            "group_0": {
                "targets": targets,
                "weights": {
                    "num_bits": num_bits,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": 128,
                },
                # Weight-only: no activation quantization.
                "input_activations": None,
            }
        },
        "ignore": ["lm_head", "re:.*self_attn.*", "re:.*mlp.gate$"],
    }


class TestWNA16MoENoLinearGroup(CustomTestCase):
    """Regression: get_moe_scheme() must not assume a "Linear" config group."""

    def _assert_wna16_moe(self, config_dict, expected_bits):
        quant_config = CompressedTensorsConfig.from_config(config_dict)

        # Precondition that reproduces the original bug: the parsed scheme map
        # has no "Linear" group, so the old target_scheme_map["Linear"] lookup
        # would KeyError.
        self.assertNotIn("Linear", quant_config.target_scheme_map)

        layer = torch.nn.Module()
        # Would raise KeyError: 'Linear' before the fix.
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


if __name__ == "__main__":
    unittest.main()
