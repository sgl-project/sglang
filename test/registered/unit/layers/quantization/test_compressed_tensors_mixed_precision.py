"""CPU regression tests for compressed-tensors target and ignore matching.

A multi-group ("mixed-precision") checkpoint used to load completely unquantized.
``ignore`` was matched by substring, so a parent module entry swallowed its
quantized children, and the activation-quantization gate read the top-level format
-- which compressed-tensors sets to ``mixed-precision`` when groups disagree --
dropping ``input_activations`` for every group. Both are config-parsing paths, so
these tests run on CPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import logging
import unittest
from unittest import mock

import torch

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.compressed_tensors import compressed_tensors
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsWNA16,
)
from sglang.srt.layers.quantization.compressed_tensors.utils import (
    check_equal_or_regex_match,
    find_matched_target,
    should_ignore_layer,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.test.test_utils import CustomTestCase

EXPERTS_LAYER = "model.language_model.layers.0.mlp.experts"
GATE_PROJ = f"{EXPERTS_LAYER}.0.gate_proj"
MLP_LAYER = "model.language_model.layers.0.mlp.gate_proj"
IN_PROJ_QKVZ = "model.language_model.layers.0.linear_attn.in_proj_qkvz"

# Qwen3.5's packed_modules_mapping, which Qwen4-Exp reuses.
PACKED_MODULES_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}

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


def _single_group_config(group):
    """Nothing in `ignore`, as llm-compressor writes experts-only checkpoints."""
    config = CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": group["format"],
            "config_groups": {"group_0": group},
            "ignore": [],
        }
    )
    config.update_packed_modules_mapping(PACKED_MODULES_MAPPING)

    return config


def _linear():
    # __new__ is enough: get_quant_method only isinstance-checks the layer.
    return LinearBase.__new__(LinearBase)


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


class TestUntargetedLayers(CustomTestCase):
    """A layer that no group targets and `ignore` does not list is stored unquantized;
    scheme lookup must not raise "Unable to find matching target" for it."""

    def test_untargeted_linear_loads_unquantized(self):
        config = _single_group_config(NVFP4_GROUP)
        method = config.get_quant_method(_linear(), IN_PROJ_QKVZ)

        self.assertIsInstance(method, UnquantizedLinearMethod)

    def test_untargeted_experts_load_unquantized(self):
        config = _single_group_config(FP8_GROUP)
        scheme = config.get_moe_scheme(torch.nn.Module(), layer_name=EXPERTS_LAYER)

        self.assertIsNone(scheme)

    def test_untargeted_layers_are_logged(self):
        """warning_once is lru_cache-wrapped; clear it to stay order-independent."""
        logging.Logger.warning_once.cache_clear()
        self.addCleanup(logging.Logger.warning_once.cache_clear)

        config = _single_group_config(NVFP4_GROUP)
        with self.assertLogs(compressed_tensors.logger, level="WARNING") as captured:
            config.get_quant_method(_linear(), IN_PROJ_QKVZ)

        self.assertIn("match no compressed-tensors target", "\n".join(captured.output))

    def test_fused_layer_matches_when_every_shard_matches(self):
        matched = find_matched_target(
            layer_name=IN_PROJ_QKVZ,
            module=torch.nn.Module(),
            targets=["re:.*in_proj_qkv$", "re:.*in_proj_z$"],
            fused_mapping=PACKED_MODULES_MAPPING,
        )

        self.assertEqual(matched, "re:.*in_proj_qkv$")

    def test_partly_targeted_fused_layer_raises(self):
        # Its shards would need different methods inside one fused layer.
        with self.assertRaisesRegex(ValueError, "different quantization schemes"):
            find_matched_target(
                layer_name=IN_PROJ_QKVZ,
                module=torch.nn.Module(),
                targets=["re:.*in_proj_qkv$"],
                fused_mapping=PACKED_MODULES_MAPPING,
            )


if __name__ == "__main__":
    unittest.main()
