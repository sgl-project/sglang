"""CPU regression tests for multi-group ("mixed-precision") compressed-tensors configs.

Such a checkpoint used to load completely unquantized. ``ignore`` was matched by
substring, so a parent module entry swallowed its quantized children, and the
activation-quantization gate read the top-level format -- which compressed-tensors
sets to ``mixed-precision`` when groups disagree -- dropping ``input_activations``
for every group. Both are config-parsing paths, so these tests run on CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    check_equal_or_regex_match,
    should_ignore_layer,
)
from sglang.test.test_utils import CustomTestCase

EXPERTS_LAYER = "model.language_model.layers.0.mlp.experts"
GATE_PROJ = f"{EXPERTS_LAYER}.0.gate_proj"
MLP_LAYER = "model.language_model.layers.0.mlp.gate_proj"

FP8_TARGET = "re:.*self_attn\\.(q|k|v|o)_proj$"
NVFP4_TARGET = "re:.*mlp\\.experts\\.\\d+\\.(gate|up|down)_proj$"
WNA16_TARGET = "re:.*mlp\\.(gate|up|down)_proj$"

# FP8 W8A8 attention projections.
FP8_GROUP = {
    "format": "float-quantized",
    "targets": [FP8_TARGET],
    "weights": {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": "channel",
        "dynamic": False,
    },
    "input_activations": {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": "token",
        "dynamic": True,
    },
}

# NVFP4 W4A4 expert projections.
NVFP4_GROUP = {
    "format": "nvfp4-pack-quantized",
    "targets": [NVFP4_TARGET],
    "weights": {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "strategy": "tensor_group",
        "group_size": 16,
        "dynamic": False,
    },
    "input_activations": {
        "num_bits": 4,
        "type": "float",
        "symmetric": True,
        "strategy": "tensor_group",
        "group_size": 16,
        "dynamic": "local",
    },
}

# Weight-only INT4: a group whose format is not an activation format.
WNA16_GROUP = {
    "format": "pack-quantized",
    "targets": [WNA16_TARGET],
    "weights": {
        "num_bits": 4,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": 128,
        "dynamic": False,
    },
    "input_activations": None,
}


def _mixed_precision_config(*groups, ignore=()):
    """Groups disagree, so compressed-tensors writes format="mixed-precision"."""
    return {
        "quant_method": "compressed-tensors",
        "format": "mixed-precision",
        "config_groups": {f"group_{i}": g for i, g in enumerate(groups)},
        "ignore": list(ignore),
    }


class TestIgnoreListPrefixMatching(CustomTestCase):
    """A parent module in `ignore` must not de-quantize its children."""

    def test_parent_module_does_not_swallow_children(self):
        ignore = [f"{EXPERTS_LAYER}.0", "model.language_model.layers.0.linear_attn"]

        # The parent itself is still ignored: semantics are unchanged.
        self.assertTrue(should_ignore_layer(f"{EXPERTS_LAYER}.0", ignore=ignore))
        self.assertTrue(
            should_ignore_layer(
                "model.language_model.layers.0.linear_attn", ignore=ignore
            )
        )

        # Its quantized children are not.
        self.assertFalse(should_ignore_layer(GATE_PROJ, ignore=ignore))
        self.assertFalse(
            should_ignore_layer(
                "model.language_model.layers.0.linear_attn.out_proj", ignore=ignore
            )
        )

    def test_module_suffix_target_still_matches(self):
        # Targets written as a module suffix keep working (dotted-path boundary).
        self.assertTrue(
            check_equal_or_regex_match(
                "model.layers.0.self_attn.kv_b_proj", ["self_attn.kv_b_proj"]
            )
        )
        self.assertTrue(check_equal_or_regex_match("model.lm_head", ["lm_head"]))
        # ... but only on a boundary, never mid-token.
        self.assertFalse(
            check_equal_or_regex_match("model.layers.0.gate_proj", ["ate"])
        )
        self.assertFalse(
            check_equal_or_regex_match(
                "model.layers.0.mlp.shared_expert_gate", ["gate"]
            )
        )

    def test_exact_and_regex_targets_unchanged(self):
        self.assertTrue(check_equal_or_regex_match(GATE_PROJ, [GATE_PROJ]))
        self.assertTrue(check_equal_or_regex_match(GATE_PROJ, ["re:.*gate_proj$"]))
        self.assertFalse(check_equal_or_regex_match(GATE_PROJ, ["re:.*down_proj$"]))


class TestMixedPrecisionFormat(CustomTestCase):
    """The per-group `format` must win over a top-level "mixed-precision"."""

    def test_input_activations_survive_mixed_precision(self):
        config = _mixed_precision_config(FP8_GROUP, NVFP4_GROUP)
        quant_config = CompressedTensorsConfig.from_config(config)

        for target, scheme in quant_config.target_scheme_map.items():
            self.assertIsNotNone(
                scheme["input_activations"],
                f"input_activations dropped for target {target}",
            )

        expert_scheme = quant_config.target_scheme_map[NVFP4_TARGET]
        self.assertEqual(expert_scheme["format"], "nvfp4-pack-quantized")
        self.assertEqual(expert_scheme["weights"].num_bits, 4)
        self.assertEqual(expert_scheme["input_activations"].num_bits, 4)

    def test_linear_scheme_uses_per_group_format(self):
        # WNA16 is selected only when the format is "pack-quantized". Reading the
        # top-level format instead of the matched group's would see
        # "mixed-precision" and resolve no scheme at all.
        config = _mixed_precision_config(WNA16_GROUP, NVFP4_GROUP)
        quant_config = CompressedTensorsConfig.from_config(config)

        with mock.patch.object(
            CompressedTensorsConfig, "_check_scheme_supported", return_value=True
        ):
            scheme = quant_config.get_linear_scheme(
                torch.nn.Module(), layer_name=MLP_LAYER
            )

        self.assertIsInstance(scheme, CompressedTensorsWNA16)
        self.assertEqual(scheme.pack_factor, 32 // 4)
        self.assertEqual(scheme.strategy, "group")
        self.assertEqual(scheme.group_size, 128)


if __name__ == "__main__":
    unittest.main()
