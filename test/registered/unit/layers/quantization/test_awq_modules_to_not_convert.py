import unittest

from sglang.srt.layers.quantization.awq.awq import (
    AWQConfig,
    AWQMarlinConfig,
    is_layer_skipped_awq,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

# modules_to_not_convert from QuantTrio/Qwen3.6-27B-AWQ (checkpoint-side names).
QWEN36_AWQ_SKIP = [
    "visual",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "model.layers.0.",
    "mtp",
]
QWEN35_PACKED = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}


class TestAwqModulesToNotConvert(unittest.TestCase):
    def test_fused_layer_is_skipped_when_all_shards_are_listed(self):
        prefix = "model.language_model.layers.11.self_attn.qkv_proj"
        self.assertTrue(is_layer_skipped_awq(prefix, QWEN36_AWQ_SKIP, QWEN35_PACKED))
        # Without packed_modules_mapping the built-in qkv/gate_up table applies.
        self.assertTrue(is_layer_skipped_awq(prefix, QWEN36_AWQ_SKIP))

    def test_fused_layer_stays_quantized_unless_every_shard_is_listed(self):
        prefix = "model.layers.3.self_attn.qkv_proj"
        partial = ["self_attn.q_proj", "self_attn.k_proj"]
        self.assertFalse(is_layer_skipped_awq(prefix, partial, QWEN35_PACKED))
        for prefix in (
            "model.language_model.layers.12.linear_attn.in_proj_qkvz",
            "model.language_model.layers.12.mlp.gate_up_proj",
            "model.language_model.layers.12.self_attn.o_proj",
        ):
            with self.subTest(prefix=prefix):
                self.assertFalse(
                    is_layer_skipped_awq(prefix, QWEN36_AWQ_SKIP, QWEN35_PACKED)
                )

    def test_existing_substring_matches_are_unchanged(self):
        for prefix in (
            "model.language_model.layers.5.linear_attn.in_proj_ba",
            "model.layers.0.mlp.gate_up_proj",
            "model.visual.blocks.0.attn.qkv",
            "mtp.layers.0.self_attn.qkv_proj",
        ):
            with self.subTest(prefix=prefix):
                self.assertTrue(
                    is_layer_skipped_awq(prefix, QWEN36_AWQ_SKIP, QWEN35_PACKED)
                )
        self.assertFalse(is_layer_skipped_awq("model.layers.1.mlp.down_proj", []))

    def test_from_config_reads_packed_modules_mapping(self):
        cfg = {
            "bits": 4,
            "group_size": 128,
            "zero_point": True,
            "modules_to_not_convert": QWEN36_AWQ_SKIP,
            "packed_modules_mapping": QWEN35_PACKED,
        }
        self.assertEqual(
            AWQConfig.from_config(cfg).packed_modules_mapping, QWEN35_PACKED
        )
        self.assertEqual(
            AWQMarlinConfig.from_config(cfg).packed_modules_mapping, QWEN35_PACKED
        )
        cfg.pop("packed_modules_mapping")
        self.assertEqual(AWQConfig.from_config(cfg).packed_modules_mapping, {})


if __name__ == "__main__":
    unittest.main()
