"""Unit tests for GLM-5 Next configuration compatibility."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from sglang.srt.configs.glm5_next import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)
from sglang.srt.configs.mamba_utils import KimiLinearStateShape
from sglang.srt.models.glm5_next import (
    Glm5NextForConditionalGeneration,
    swiglu_clamped,
)
from sglang.srt.runtime_context import get_context, get_parallel
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


class TestGlm5NextVisionConfig(CustomTestCase):
    def test_vision_config_defaults_swiglu_limit_to_none(self):
        config = Glm5NextVisionConfig()
        self.assertIsNone(config.swiglu_limit)

    def test_vision_config_preserves_explicit_swiglu_limit(self):
        config = Glm5NextVisionConfig(swiglu_limit=10.0)
        self.assertEqual(config.swiglu_limit, 10.0)

    def test_glm5_next_config_instantiates_vision_config_without_swiglu_limit(self):
        config = Glm5NextConfig(vision_config={})
        self.assertIsNotNone(config.vision_config)
        self.assertIsNone(config.vision_config.swiglu_limit)

    def test_glm5_next_config_preserves_vision_swiglu_limit(self):
        config = Glm5NextConfig(vision_config={"swiglu_limit": 10.0})
        self.assertIsNotNone(config.vision_config)
        self.assertEqual(config.vision_config.swiglu_limit, 10.0)

    def test_swiglu_clamped_with_none_matches_unclamped_swiglu(self):
        x = torch.randn(2, 8, dtype=torch.float32)
        out_none = swiglu_clamped(x, None)
        gate, up = torch.chunk(x, 2, dim=-1)
        expected = F.silu(gate) * up
        torch.testing.assert_close(out_none, expected)

    def test_swiglu_clamped_with_limit_clamps_activations(self):
        x = torch.tensor([[100.0, -100.0]], dtype=torch.float32)
        out_clamped = swiglu_clamped(x, 5.0)
        gate, up = torch.chunk(x, 2, dim=-1)
        gate = torch.clamp(gate, max=5.0)
        up = torch.clamp(up, min=-5.0, max=5.0)
        expected = F.silu(gate) * up
        torch.testing.assert_close(out_clamped, expected)

    def test_glm5_next_config_with_none_vision_config(self):
        config = Glm5NextConfig(vision_config=None)
        self.assertIsNone(config.vision_config)

    def test_glm5_next_for_conditional_generation_init_with_none_vision_config(self):
        config = Glm5NextConfig(vision_config=None, encoder_only=True)
        with (
            patch(
                "sglang.srt.models.glm5_next.get_pp_group",
                return_value=SimpleNamespace(is_last_rank=True, world_size=1),
            ),
            get_parallel().override(tp_size=1, attn_tp_size=1),
            get_context().override_server_args(mm_enable_dp_encoder=False),
        ):
            model = Glm5NextForConditionalGeneration(config)
            self.assertIsNone(model.visual)

    def test_glm5_next_for_conditional_generation_init_with_language_only(self):
        config = Glm5NextConfig(vision_config={}, language_only=True, encoder_only=True)
        with (
            patch(
                "sglang.srt.models.glm5_next.get_pp_group",
                return_value=SimpleNamespace(is_last_rank=True, world_size=1),
            ),
            get_parallel().override(tp_size=1, attn_tp_size=1),
            get_context().override_server_args(mm_enable_dp_encoder=False),
        ):
            model = Glm5NextForConditionalGeneration(config)
            self.assertIsNone(model.visual)

    def test_glm5_next_for_conditional_generation_load_weights_with_none_vision_config(
        self,
    ):
        config = Glm5NextConfig(vision_config=None, encoder_only=True)
        with (
            patch(
                "sglang.srt.models.glm5_next.get_pp_group",
                return_value=SimpleNamespace(is_last_rank=True, world_size=1),
            ),
            get_parallel().override(tp_size=1, attn_tp_size=1),
            get_context().override_server_args(mm_enable_dp_encoder=False),
        ):
            model = Glm5NextForConditionalGeneration(config)
            visual_weight = torch.randn(4, 4)
            weights = [("visual.attn.qkv.weight", visual_weight)]
            # Should execute cleanly without dereferencing missing vision_config
            model.load_weights(weights)
            self.assertIsNone(model.visual)


if __name__ == "__main__":
    unittest.main()
