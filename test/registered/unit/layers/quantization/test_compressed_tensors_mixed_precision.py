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

# A vision tower whose ignore list is spelled against the CompressedLinear
# wrapper. The names below are the prefixes the matcher actually receives:
# ClippableQKVParallelLinear and ClippableGateUpParallelLinear hand their own
# prefix to the inner fused linear, while ClippableRowParallelLinear nests its
# inner linear at ".linear".
VISION_PREFIX = "model.vision_tower.encoder.layers"
VISION_QKV_LAYER = f"{VISION_PREFIX}.0.self_attn.qkv_proj"
VISION_GATE_UP_LAYER = f"{VISION_PREFIX}.0.mlp.gate_up_proj"
VISION_O_PROJ_LAYER = f"{VISION_PREFIX}.0.self_attn.o_proj.linear"
VISION_DOWN_PROJ_LAYER = f"{VISION_PREFIX}.0.mlp.down_proj.linear"
VISION_FUSED_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}
VISION_NUM_LAYERS = 27
TEXT_PREFIX = "model.language_model.layers.0"


def _wrapped_tower_ignore_list():
    """27 layers x 7 projections, every entry spelled ``<module>.linear``.

    135 name fused shards (q/k/v, gate/up) and 54 name unfused layers
    (o_proj, down_proj).
    """
    entries = []
    for idx in range(VISION_NUM_LAYERS):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            entries.append(f"{VISION_PREFIX}.{idx}.self_attn.{proj}.linear")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            entries.append(f"{VISION_PREFIX}.{idx}.mlp.{proj}.linear")
    return entries


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


class TestFusedIgnoreLinearWrapper(CustomTestCase):
    """Fused shard names must honor ignores written for the CT wrapper."""

    def test_all_wrapped_shards_are_ignored(self):
        ignore = [
            f"{VISION_PREFIX}.0.self_attn.{proj}.linear"
            for proj in ("q_proj", "k_proj", "v_proj")
        ]

        self.assertTrue(
            should_ignore_layer(
                VISION_QKV_LAYER,
                ignore=ignore,
                fused_mapping=VISION_FUSED_MAPPING,
            )
        )

    def test_partial_wrapped_shards_extend_inconsistent_scheme_error(self):
        """Wrapped shards join the consistency check, so a partial list now raises.

        Before this change the wrapped entry matched nothing, every shard agreed
        on "not ignored", and the layer loaded quantized. The error boundary is
        extended to the wrapper spelling, not preserved unchanged.
        """
        ignore = [f"{VISION_PREFIX}.0.mlp.gate_proj.linear"]

        with self.assertRaisesRegex(ValueError, "different quantization schemes"):
            should_ignore_layer(
                VISION_GATE_UP_LAYER,
                ignore=ignore,
                fused_mapping=VISION_FUSED_MAPPING,
            )

    def test_mixed_spelling_shards_resolve_to_ignored(self):
        """A list mixing plain and wrapped shard spellings is now consistent.

        Before this change the plain entry matched and the wrapped ones did not,
        which raised the inconsistent-scheme error. Both spellings now match, so
        the layer is ignored instead of aborting the load.
        """
        ignore = [f"{VISION_PREFIX}.0.self_attn.q_proj"] + [
            f"{VISION_PREFIX}.0.self_attn.{proj}.linear"
            for proj in ("k_proj", "v_proj")
        ]

        self.assertTrue(
            should_ignore_layer(
                VISION_QKV_LAYER,
                ignore=ignore,
                fused_mapping=VISION_FUSED_MAPPING,
            )
        )

    def test_wrapped_tower_list_is_honored_at_runtime_names(self):
        """The whole wrapped list, queried at the names the tower really has.

        The two fused names need the retry; the two unfused ones are already
        nested at ".linear" by ClippableRowParallelLinear and match directly.
        """
        ignore = _wrapped_tower_ignore_list()
        self.assertEqual(len(ignore), VISION_NUM_LAYERS * 7)

        for idx in range(VISION_NUM_LAYERS):
            for layer_name in (
                f"{VISION_PREFIX}.{idx}.self_attn.qkv_proj",
                f"{VISION_PREFIX}.{idx}.mlp.gate_up_proj",
                f"{VISION_PREFIX}.{idx}.self_attn.o_proj.linear",
                f"{VISION_PREFIX}.{idx}.mlp.down_proj.linear",
            ):
                self.assertTrue(
                    should_ignore_layer(
                        layer_name,
                        ignore=ignore,
                        fused_mapping=VISION_FUSED_MAPPING,
                    ),
                    layer_name,
                )

    def test_wrapped_tower_list_leaves_the_text_model_quantized(self):
        """A tower-only ignore list must not reach the language model."""
        ignore = _wrapped_tower_ignore_list()

        for layer_name in (
            f"{TEXT_PREFIX}.self_attn.qkv_proj",
            f"{TEXT_PREFIX}.mlp.gate_up_proj",
            f"{TEXT_PREFIX}.self_attn.o_proj.linear",
            f"{TEXT_PREFIX}.mlp.down_proj.linear",
            "lm_head",
        ):
            self.assertFalse(
                should_ignore_layer(
                    layer_name,
                    ignore=ignore,
                    fused_mapping=VISION_FUSED_MAPPING,
                ),
                layer_name,
            )

    def test_intentionally_quantized_tower_keeps_its_scheme(self):
        config = CompressedTensorsConfig.from_config(
            {
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "config_groups": {"group_0": WNA16_GROUP},
                "ignore": [],
            }
        )
        config.update_packed_modules_mapping(VISION_FUSED_MAPPING)

        scheme = config.get_scheme_dict(torch.nn.Linear(1, 1), VISION_GATE_UP_LAYER)

        self.assertIsNotNone(
            scheme, "an empty ignore list must not force the tower to bf16"
        )

    def test_parent_ignore_does_not_swallow_fused_children(self):
        """The no-prefix invariant, in both spellings of the parent entry."""
        for entry in (f"{VISION_PREFIX}.0.mlp", f"{VISION_PREFIX}.0.mlp.linear"):
            self.assertFalse(
                should_ignore_layer(
                    VISION_GATE_UP_LAYER,
                    ignore=[entry],
                    fused_mapping=VISION_FUSED_MAPPING,
                ),
                entry,
            )

    def test_retry_stays_keyed_on_the_ignore_list(self):
        """Only the wrapper suffix, only one level, only listed modules."""
        for entry in (
            "some.other.module.linear",
            f"{VISION_PREFIX}.0.self_attn.q_proj.linear.linear",
            f"{VISION_PREFIX}.0.self_attn.q_proj.weight",
        ):
            self.assertFalse(
                should_ignore_layer(
                    VISION_QKV_LAYER,
                    ignore=[entry],
                    fused_mapping=VISION_FUSED_MAPPING,
                ),
                entry,
            )

    def test_class_name_target_spelling_stays_inert(self):
        """Configs name the class as "Linear", and matching is case-sensitive."""
        for layer_name in (
            VISION_QKV_LAYER,
            VISION_O_PROJ_LAYER,
            VISION_DOWN_PROJ_LAYER,
        ):
            self.assertFalse(
                should_ignore_layer(
                    layer_name,
                    ignore=["Linear"],
                    fused_mapping=VISION_FUSED_MAPPING,
                ),
                layer_name,
            )

    def test_bare_linear_token_reaches_fused_names_only(self):
        """A bare "linear" entry is a suffix target, so it now reaches fused names.

        It cannot reach further than that: unfused names that are not already
        nested at ".linear" are matched exactly as before.
        """
        self.assertTrue(
            should_ignore_layer(
                VISION_QKV_LAYER,
                ignore=["linear"],
                fused_mapping=VISION_FUSED_MAPPING,
            )
        )
        for layer_name in (f"{VISION_PREFIX}.0.self_attn.o_proj", "lm_head"):
            self.assertFalse(
                should_ignore_layer(
                    layer_name,
                    ignore=["linear"],
                    fused_mapping=VISION_FUSED_MAPPING,
                ),
                layer_name,
            )


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
