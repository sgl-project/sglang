"""Unit tests for is_layer_skipped_awq (fused-layer shard matching, #52208)."""

import unittest

from sglang.srt.layers.quantization.awq.awq import is_layer_skipped_awq
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# Mirrors the modules_to_not_convert shipped with Qwen3.6-27B-AWQ: the dense
# parts are written with the checkpoint's *unfused* names, while SGLang builds
# the fused qkv_proj / gate_up_proj modules.
QWEN36_MODULES_TO_NOT_CONVERT = [
    "visual",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "mtp",
]


class TestIsLayerSkippedAwq(CustomTestCase):
    def test_fused_qkv_is_skipped_when_all_shards_listed(self):
        # Regression for #52208: the dense q/k/v are unfused in the AWQ
        # checkpoint, so only the shard expansion can match the fused module.
        prefix = "model.language_model.layers.11.self_attn.qkv_proj"
        self.assertTrue(is_layer_skipped_awq(prefix, QWEN36_MODULES_TO_NOT_CONVERT))

    def test_fused_gate_up_is_skipped_when_all_shards_listed(self):
        prefix = "model.layers.11.mlp.gate_up_proj"
        self.assertTrue(
            is_layer_skipped_awq(
                prefix, ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
            )
        )

    def test_direct_fused_name_entry_still_matches(self):
        # Workaround configs list the fused name itself; keep it working.
        prefix = "model.layers.11.self_attn.qkv_proj"
        self.assertTrue(
            is_layer_skipped_awq(prefix, ["self_attn.qkv_proj", "mtp"])
        )

    def test_unfused_layer_uses_substring_match(self):
        # Unified GDN tensors are listed directly; behavior must not change.
        self.assertTrue(
            is_layer_skipped_awq(
                "model.layers.1.linear_attn.in_proj_ba", QWEN36_MODULES_TO_NOT_CONVERT
            )
        )
        self.assertTrue(
            is_layer_skipped_awq("model.visual.blocks.0.attn.qkv", QWEN36_MODULES_TO_NOT_CONVERT)
        )
        self.assertTrue(
            is_layer_skipped_awq(
                "model.language_model.mtp.layers.0.self_attn.qkv_proj",
                QWEN36_MODULES_TO_NOT_CONVERT,
            )
        )

    def test_quantized_fused_layer_is_not_skipped(self):
        prefix = "model.layers.0.mlp.gate_up_proj"
        self.assertFalse(is_layer_skipped_awq(prefix, ["visual", "mtp"]))
        self.assertFalse(
            is_layer_skipped_awq("model.layers.0.self_attn.qkv_proj", [])
        )
        self.assertFalse(is_layer_skipped_awq("model.layers.0.self_attn.qkv_proj", None))

    def test_partial_shard_skip_is_rejected(self):
        # Only some shards of the fused layer are dense: the checkpoint is
        # unloadable into a homogeneous fused module either way, so fail
        # loudly instead of building an incompatible quantized layer.
        with self.assertRaises(ValueError):
            is_layer_skipped_awq(
                "model.layers.1.mlp.gate_up_proj", ["mlp.gate_proj"]
            )


if __name__ == "__main__":
    unittest.main()
