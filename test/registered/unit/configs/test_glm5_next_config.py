"""Unit tests for GLM-5 Next configuration compatibility."""

import unittest

from sglang.srt.configs.glm5_next import Glm5NextTextConfig
from sglang.srt.configs.mamba_utils import KimiLinearStateShape
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestGlm5NextTextConfig(CustomTestCase):
    def test_kimi_linear_state_shape_preserves_channel_slice_axis(self):
        shape = KimiLinearStateShape.create(
            tp_world_size=8,
            num_heads=32,
            head_dim=96,
            conv_kernel_size=3,
        )

        self.assertEqual(shape.conv, [(2, 1152)])
        self.assertEqual(shape.temporal, (4, 96, 96))
        self.assertEqual(shape.conv_slice_axis, 1)
        self.assertEqual(shape.conv_shard_groups, [3072, 3072, 3072])

    def test_transformers_format_builds_legacy_linear_attention_config(self):
        config = Glm5NextTextConfig(
            num_hidden_layers=4,
            layer_types=[
                "linear_attention",
                "linear_attention",
                "deepseek_sparse_attention",
                "linear_attention",
            ],
            linear_head_dim=96,
            linear_num_heads=32,
            linear_conv_kernel_dim=3,
            gate_lower_bound=-5.0,
        )

        self.assertEqual(
            config.linear_attn_config,
            {
                "full_attn_layers": [2],
                "head_dim": 96,
                "kda_layers": [0, 1, 3],
                "num_heads": 32,
                "short_conv_kernel_size": 3,
                "gate_lower_bound": -5.0,
            },
        )
        self.assertEqual(config.linear_layer_ids, [0, 1, 3])
        self.assertEqual(config.full_attention_layer_ids, [2])

    def test_transformers_format_without_lower_bound(self):
        config = Glm5NextTextConfig(
            num_hidden_layers=2,
            layer_types=["linear_attention", "deepseek_sparse_attention"],
        )

        self.assertIsNone(config.linear_attn_config["gate_lower_bound"])

    def test_legacy_linear_lower_bound_is_normalized(self):
        config = Glm5NextTextConfig(
            num_hidden_layers=2,
            layer_types=["linear_attention", "deepseek_sparse_attention"],
            linear_lower_bound=-4.0,
        )

        self.assertEqual(config.linear_attn_config["gate_lower_bound"], -4.0)

    def test_gate_lower_bound_takes_precedence(self):
        config = Glm5NextTextConfig(
            num_hidden_layers=2,
            layer_types=["linear_attention", "deepseek_sparse_attention"],
            linear_lower_bound=-4.0,
            gate_lower_bound=-5.0,
        )

        self.assertEqual(config.linear_attn_config["gate_lower_bound"], -5.0)

    def test_legacy_linear_attention_config_takes_precedence(self):
        linear_attn_config = {
            "full_attn_layers": [1],
            "head_dim": 64,
            "kda_layers": [0],
            "num_heads": 16,
            "short_conv_kernel_size": 2,
            "gate_lower_bound": -4.0,
        }

        config = Glm5NextTextConfig(
            num_hidden_layers=2,
            linear_attn_config=linear_attn_config,
            layer_types=["deepseek_sparse_attention", "linear_attention"],
            linear_head_dim=96,
            linear_num_heads=32,
            linear_conv_kernel_dim=3,
            gate_lower_bound=-5.0,
        )

        self.assertIs(config.linear_attn_config, linear_attn_config)


if __name__ == "__main__":
    unittest.main()
