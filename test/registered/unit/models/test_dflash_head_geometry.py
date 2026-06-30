import unittest
from types import SimpleNamespace

from sglang.srt.configs.laguna import LagunaConfig
from sglang.srt.models.dflash import _get_dflash_total_num_heads
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDFlashHeadGeometry(unittest.TestCase):
    def test_per_layer_heads_override_scalar_default(self):
        config = SimpleNamespace(
            num_attention_heads=42,
            num_attention_heads_per_layer=[32, 64, 32, 64, 32],
        )

        self.assertEqual(
            [_get_dflash_total_num_heads(config, i) for i in range(5)],
            [32, 64, 32, 64, 32],
        )

    def test_scalar_heads_used_when_per_layer_heads_are_absent(self):
        config = SimpleNamespace(num_attention_heads=42)

        self.assertEqual(_get_dflash_total_num_heads(config, 3), 42)

    def test_per_layer_heads_require_layer_entry(self):
        config = SimpleNamespace(
            num_hidden_layers=5,
            num_attention_heads=42,
            num_attention_heads_per_layer=[32],
        )

        with self.assertRaisesRegex(ValueError, "expected num_hidden_layers=5, got 1"):
            _get_dflash_total_num_heads(config, 1)

    def test_laguna_config_accepts_non_uniform_all_swa_heads(self):
        config = LagunaConfig(
            num_hidden_layers=5,
            num_attention_heads=42,
            num_attention_heads_per_layer=[32, 64, 32, 64, 32],
            layer_types=["sliding_attention"] * 5,
        )

        self.assertEqual(
            config.num_attention_heads_per_layer,
            [32, 64, 32, 64, 32],
        )
        self.assertEqual(config.num_attention_heads, 32)

    def test_laguna_config_does_not_apply_swa_rope_to_full_layers(self):
        config = LagunaConfig(
            num_hidden_layers=2,
            num_attention_heads=32,
            layer_types=["full_attention", "sliding_attention"],
            rope_theta=10000.0,
            rope_parameters={
                "full_attention": {},
                "sliding_attention": {"rope_theta": 500000.0},
            },
        )

        self.assertEqual(config.rope_theta, 10000.0)
        self.assertEqual(config.swa_rope_theta, 500000.0)

    def test_laguna_config_all_swa_uses_swa_rope_for_default_fields(self):
        config = LagunaConfig(
            num_hidden_layers=2,
            num_attention_heads=32,
            layer_types=["sliding_attention", "sliding_attention"],
            rope_parameters={
                "full_attention": {},
                "sliding_attention": {"rope_theta": 500000.0},
            },
        )

        self.assertEqual(config.rope_theta, 500000.0)
        self.assertEqual(config.swa_rope_theta, 500000.0)


if __name__ == "__main__":
    unittest.main()
