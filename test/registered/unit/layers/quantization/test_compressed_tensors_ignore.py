"""Unit tests for compressed-tensors ``should_ignore_layer`` — no server, no model loading.

Covers the case where the ``ignore`` list in a checkpoint's
``quantization_config`` contains submodule paths (e.g., ``...q_proj.linear``)
while the runtime layer name is the parent layer (e.g., ``...q_proj``), which
typically arises when a model's ``packed_modules_mapping`` expands a fused
layer (e.g., ``qkv_proj``) into shard names. This pattern appears in NVFP4 VLM
checkpoints where the vision tower is kept in BF16 via the ``ignore`` list.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")

import unittest

from sglang.srt.layers.quantization.compressed_tensors.utils import (
    _is_equal_or_regex_match,
    should_ignore_layer,
)
from sglang.test.test_utils import CustomTestCase


class TestIsEqualOrRegexMatch(CustomTestCase):
    """Direct tests for the substring/path-prefix matcher."""

    def test_exact_match(self):
        self.assertTrue(
            _is_equal_or_regex_match(
                "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.q_proj"
            )
        )

    def test_exact_mismatch(self):
        self.assertFalse(_is_equal_or_regex_match("foo", "bar"))

    def test_regex_match(self):
        self.assertTrue(
            _is_equal_or_regex_match(
                "model.layers.5.self_attn.q_proj", "re:.*layers\\.\\d+.*q_proj"
            )
        )

    def test_regex_no_match(self):
        self.assertFalse(
            _is_equal_or_regex_match("model.layers.5.self_attn.q_proj", "re:.*o_proj")
        )

    def test_check_contains_substring_match(self):
        # Existing behavior: target is a substring of value.
        self.assertTrue(
            _is_equal_or_regex_match(
                "model.layers.0.self_attn.q_proj",
                "self_attn.q_proj",
                check_contains=True,
            )
        )

    def test_check_contains_path_extension_match(self):
        # New behavior: target is a path-extension of value (i.e., target
        # starts with value + "."). This covers compressed-tensors checkpoints
        # whose ignore list uses ".linear" submodule suffix while the runtime
        # layer name is the parent layer.
        self.assertTrue(
            _is_equal_or_regex_match(
                "model.vision_tower.encoder.layers.0.self_attn.q_proj",
                "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear",
                check_contains=True,
            )
        )

    def test_check_contains_no_over_match_on_partial_token(self):
        # Path-extension match must respect the "." path boundary; a partial
        # token like "q" should NOT match "q_proj.linear".
        self.assertFalse(
            _is_equal_or_regex_match(
                "model.layers.0.self_attn.q",
                "model.layers.0.self_attn.q_proj.linear",
                check_contains=True,
            )
        )

    def test_check_contains_disjoint_names(self):
        self.assertFalse(
            _is_equal_or_regex_match(
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.mlp.gate_proj.linear",
                check_contains=True,
            )
        )


class TestShouldIgnoreLayerLinearSuffix(CustomTestCase):
    """End-to-end ``should_ignore_layer`` cases for the ``.linear`` suffix pattern.

    These cases mirror what happens when a VLM checkpoint
    (e.g., ``RedHatAI/gemma-4-31B-it-NVFP4``,
    ``cyankiwi/gemma-4-31B-it-AWQ-4bit``) lists every vision tower projection
    with a ``.linear`` suffix in ``quantization_config.ignore``, and a model
    such as Gemma 4 fuses Q/K/V into ``qkv_proj`` via
    ``packed_modules_mapping``.
    """

    FUSED_MAPPING = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    VISION_IGNORE = [
        "model.vision_tower.encoder.layers.0.self_attn.q_proj.linear",
        "model.vision_tower.encoder.layers.0.self_attn.k_proj.linear",
        "model.vision_tower.encoder.layers.0.self_attn.v_proj.linear",
        "model.vision_tower.encoder.layers.0.self_attn.o_proj.linear",
        "model.vision_tower.encoder.layers.0.mlp.gate_proj.linear",
        "model.vision_tower.encoder.layers.0.mlp.up_proj.linear",
        "model.vision_tower.encoder.layers.0.mlp.down_proj.linear",
    ]

    def test_fused_qkv_proj_is_ignored_when_all_shards_have_linear_suffix(self):
        """Fused qkv_proj layer should be ignored when each shard's path
        appears in the ignore list with a ``.linear`` submodule suffix."""
        self.assertTrue(
            should_ignore_layer(
                "model.vision_tower.encoder.layers.0.self_attn.qkv_proj",
                ignore=self.VISION_IGNORE,
                fused_mapping=self.FUSED_MAPPING,
            )
        )

    def test_fused_gate_up_proj_is_ignored_when_all_shards_have_linear_suffix(self):
        """Fused gate_up_proj layer should be ignored when both shards
        appear in the ignore list with a ``.linear`` submodule suffix."""
        self.assertTrue(
            should_ignore_layer(
                "model.vision_tower.encoder.layers.0.mlp.gate_up_proj",
                ignore=self.VISION_IGNORE,
                fused_mapping=self.FUSED_MAPPING,
            )
        )

    def test_unfused_o_proj_is_ignored_with_linear_suffix(self):
        """Non-fused linear (e.g., o_proj) should also be ignored when its
        ``.linear`` submodule path is in the ignore list."""
        self.assertTrue(
            should_ignore_layer(
                "model.vision_tower.encoder.layers.0.self_attn.o_proj",
                ignore=self.VISION_IGNORE,
                fused_mapping=self.FUSED_MAPPING,
            )
        )

    def test_language_model_layer_not_ignored(self):
        """Language-model layers (not in ignore list) should remain quantized."""
        self.assertFalse(
            should_ignore_layer(
                "model.language_model.layers.0.self_attn.qkv_proj",
                ignore=self.VISION_IGNORE,
                fused_mapping=self.FUSED_MAPPING,
            )
        )

    def test_other_vision_layer_not_in_ignore_is_not_ignored(self):
        """Vision layers from a layer index not present in the ignore list
        should not be ignored (sanity check that match is path-specific)."""
        self.assertFalse(
            should_ignore_layer(
                "model.vision_tower.encoder.layers.5.self_attn.qkv_proj",
                ignore=self.VISION_IGNORE,
                fused_mapping=self.FUSED_MAPPING,
            )
        )


if __name__ == "__main__":
    unittest.main()
