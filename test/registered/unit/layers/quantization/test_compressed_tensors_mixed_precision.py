"""CPU regression tests for multi-group ("mixed-precision") compressed-tensors configs.

Such a checkpoint used to load completely unquantized. ``ignore`` was matched by
substring, so a parent module entry swallowed its quantized children, and the
activation-quantization gate read the top-level format -- which compressed-tensors
sets to ``mixed-precision`` when groups disagree -- dropping ``input_activations``
for every group. Both are config-parsing paths, so these tests run on CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4AFP8MoE,
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


# Block-FP8 attention / shared-expert / dense-MLP projections, as produced by
# requantizing a DeepSeek-style FP8 release.
FP8_BLOCK_TARGET = "re:.*\\.mlp\\.shared_experts\\.(gate|up|down)_proj$"
FP8_BLOCK_GROUP = {
    "format": "float-quantized",
    "targets": [FP8_BLOCK_TARGET],
    "weights": {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": "block",
        "block_structure": [128, 128],
        "dynamic": False,
    },
    "input_activations": {
        "num_bits": 8,
        "type": "float",
        "symmetric": True,
        "strategy": "group",
        "group_size": 128,
        "dynamic": True,
    },
}

# W4A8: group-128 INT4 routed experts with dynamic per-token FP8 activations.
W4A8_TARGET = "re:.*\\.mlp\\.experts\\.\\d+\\.(gate|up|down)_proj$"
W4A8_GROUP = {
    "format": "pack-quantized",
    "targets": [W4A8_TARGET],
    "weights": {
        "num_bits": 4,
        "type": "int",
        "symmetric": True,
        "strategy": "group",
        "group_size": 128,
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


class TestW4AFP8MixedPrecision(CustomTestCase):
    """INT4 routed experts beside block-FP8 shared experts.

    Every lookup below used to key off ``target_scheme_map["Linear"]`` or the
    top-level format, neither of which a multi-group checkpoint has to provide.
    """

    def _config(self):
        return CompressedTensorsConfig.from_config(
            _mixed_precision_config(W4A8_GROUP, FP8_BLOCK_GROUP)
        )

    def test_wint4afp8_detected_from_group_format(self):
        # The top-level format is "mixed-precision"; the INT4 group declares
        # "pack-quantized". Reading only the former detects nothing.
        quant_config = self._config()
        scheme = quant_config.target_scheme_map[W4A8_TARGET]
        self.assertTrue(
            quant_config._is_wint4afp8(
                scheme["weights"], scheme["input_activations"], scheme["format"]
            )
        )
        self.assertFalse(
            quant_config._is_wint4afp8(
                scheme["weights"], scheme["input_activations"], "mixed-precision"
            )
        )

    def test_weight_block_size_falls_back_to_any_block_group(self):
        # No "Linear" target at all: the block size still has to be found, or
        # block-FP8 layers cannot shard their scales.
        quant_config = self._config()
        self.assertNotIn("Linear", quant_config.target_scheme_map)
        self.assertEqual(quant_config.weight_block_size, [128, 128])

    def test_weight_block_size_ignores_non_block_linear_target(self):
        # A "Linear" target that describes the group-wise INT4 experts has no
        # block structure; the block-FP8 group's must still win.
        groups = (dict(W4A8_GROUP, targets=["Linear"]), FP8_BLOCK_GROUP)
        quant_config = CompressedTensorsConfig.from_config(
            _mixed_precision_config(*groups)
        )
        self.assertIn("Linear", quant_config.target_scheme_map)
        self.assertEqual(quant_config.weight_block_size, [128, 128])

    def test_moe_scheme_reads_matched_group_not_linear(self):
        # Top-level format stays "mixed-precision"; the scheme must still
        # take num_bits / group_size from the matched INT4 group.
        quant_config = self._config()
        scheme_dict = quant_config.target_scheme_map[W4A8_TARGET]
        self.assertEqual(quant_config.quant_format, "mixed-precision")

        scheme = CompressedTensorsW4AFP8MoE(
            quant_config,
            weight_quant=scheme_dict["weights"],
            input_quant=scheme_dict["input_activations"],
        )
        self.assertEqual(scheme.num_bits, 4)
        self.assertEqual(scheme.group_size, 128)
        self.assertEqual(scheme.packed_factor, 32 // 4)

    def test_shared_experts_fusion_refused_on_precision_mismatch(self):
        # Folding block-FP8 shared experts into the INT4 routed-expert kernel
        # would load them through the wrong scheme.
        self.assertFalse(self._config().can_fuse_shared_expert())

    def test_shared_experts_fusion_allowed_when_schemes_agree(self):
        uniform = dict(W4A8_GROUP, targets=["Linear"])
        quant_config = CompressedTensorsConfig.from_config(
            _mixed_precision_config(uniform)
        )
        self.assertTrue(quant_config.can_fuse_shared_expert())

    def test_shared_experts_fusion_refused_when_shared_is_unquantized(self):
        # Shared experts listed in `ignore` stay in the model dtype.
        quant_config = CompressedTensorsConfig.from_config(
            _mixed_precision_config(
                W4A8_GROUP,
                FP8_BLOCK_GROUP,
                ignore=["re:.*\\.mlp\\.shared_experts\\..*"],
            )
        )
        self.assertFalse(quant_config.can_fuse_shared_expert())


if __name__ == "__main__":
    unittest.main()
