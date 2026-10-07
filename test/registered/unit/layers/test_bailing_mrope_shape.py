"""Regression tests for Bailing multimodal rotary positions and config bounds."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.configs.bailing_hybrid import BailingMoeV3VLConfig
from sglang.srt.layers.rotary_embedding.bailing_mrope import BailingMRotaryEmbedding
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _position_config(max_position_embeddings=131072):
    return SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2),
        text_config=SimpleNamespace(
            image_patch_token=11,
            video_patch_token=12,
            image_start_token=10,
            video_start_token=13,
            use_interleaved_frame_timestamp=False,
            max_position_embeddings=max_position_embeddings,
        ),
    )


class TestBailingMRotaryEmbedding(CustomTestCase):
    def test_text_and_single_multimodal_position_shapes(self):
        """A singleton sequence must retain the [3, batch, seq] contract."""
        config = _position_config()
        text_positions, text_delta = (
            BailingMRotaryEmbedding.bailing_3drope_get_input_positions_tensor(
                torch.tensor([7]), config, None, None
            )
        )
        image_positions, image_delta = (
            BailingMRotaryEmbedding.bailing_3drope_get_input_positions_tensor(
                torch.tensor([10, 11]),
                config,
                image_grid_thw=torch.tensor([[1, 2, 2]]),
                video_grid_thw=None,
            )
        )

        self.assertEqual(text_positions.shape, (3, 1, 1))
        self.assertEqual(text_delta.shape, (1, 1))
        self.assertEqual(image_positions.shape, (3, 1, 2))
        self.assertEqual(image_delta.shape, (1, 1))

    def test_centered_height_positions_can_be_negative(self):
        """Tall images require negative H coordinates instead of clamping to zero."""
        config = _position_config()
        input_ids = torch.tensor([10] + [11] * 7 + [99])

        positions, _ = (
            BailingMRotaryEmbedding.bailing_3drope_get_input_positions_tensor(
                input_ids,
                config,
                image_grid_thw=torch.tensor([[1, 14, 2]]),
                video_grid_thw=None,
            )
        )

        self.assertEqual(positions.shape, (3, 1, 9))
        self.assertLess(int(positions[1].min()), 0)

    def test_checkpoint_position_bound_is_enforced(self):
        """Media positions at or beyond the checkpoint context must fail clearly."""
        config = _position_config(max_position_embeddings=4)
        input_ids = torch.tensor([10] + [11] * 7)

        with self.assertRaisesRegex(ValueError, "checkpoint bounds"):
            BailingMRotaryEmbedding.bailing_3drope_get_input_positions_tensor(
                input_ids,
                config,
                image_grid_thw=torch.tensor([[1, 14, 2]]),
                video_grid_thw=None,
            )

    def test_negative_start_cache_growth_preserves_phase(self):
        """Growing a negative-origin cache must append the next logical phase."""
        with mock.patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True):
            rotary = BailingMRotaryEmbedding(
                head_size=8,
                rotary_dim=8,
                max_position_embeddings=16,
                base=10000,
                is_neox_style=True,
                dtype=torch.float32,
                mrope_section=[2, 1, 1],
                video_rope=True,
            )
        self.assertEqual(rotary.position_start, -16)
        self.assertEqual(rotary.cos_sin_cache.shape[0], 32)

        rotary._ensure_cos_sin_cache_length(32)
        inv_freq = rotary._compute_inv_freq(rotary.base)
        expected = torch.cat(((16 * inv_freq).cos(), (16 * inv_freq).sin()))
        torch.testing.assert_close(rotary.cos_sin_cache[32], expected)

    def test_yarn_scaling_extends_positive_cache_only(self):
        """YaRN factor stretches the positive side; the negative side is fixed."""
        with mock.patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True):
            rotary = BailingMRotaryEmbedding(
                head_size=8,
                rotary_dim=8,
                max_position_embeddings=16,
                base=10000,
                is_neox_style=True,
                dtype=torch.float32,
                mrope_section=[2, 1, 1],
                video_rope=True,
                scaling_factor=2.0,
                original_max_position_embeddings=16,
            )
        self.assertEqual(rotary.position_start, -16)
        self.assertEqual(rotary.cos_sin_cache.shape[0], 16 + 32)
        self.assertGreater(rotary.mscale, 1.0)

        inv_freq = rotary._compute_inv_freq(rotary.base)
        position = 20  # beyond the native bound of 16, row = 20 - (-16)
        expected = torch.cat(
            (
                (position * inv_freq).cos() * rotary.mscale,
                (position * inv_freq).sin() * rotary.mscale,
            )
        )
        torch.testing.assert_close(rotary.cos_sin_cache[36], expected)

        # Growth past the initial cache keeps the same scaled phase.
        rotary._ensure_cos_sin_cache_length(48)
        torch.testing.assert_close(rotary.cos_sin_cache[36], expected)

    def test_yarn_scaling_relaxes_positive_position_bound(self):
        """A yarn rope_parameters entry must raise the allowed positive bound."""
        config = _position_config(max_position_embeddings=4)
        config.text_config.rope_parameters = {
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 4,
        }
        input_ids = torch.tensor([10] + [11] * 7)

        # max position is 7: fails against the native bound of 4, passes once
        # the yarn factor doubles the positive bound to 8.
        positions, _ = (
            BailingMRotaryEmbedding.bailing_3drope_get_input_positions_tensor(
                input_ids,
                config,
                image_grid_thw=torch.tensor([[1, 14, 2]]),
                video_grid_thw=None,
            )
        )
        self.assertEqual(positions.shape, (3, 1, 8))

    def test_rope_scaling_override_merges_into_text_config(self):
        """A top-level rope_scaling override must not drop the mrope markers."""
        config = BailingMoeV3VLConfig(
            mrope_section=[8, 12, 12],
            text_config={"max_position_embeddings": 131072},
            vision_config={},
        )
        config.rope_scaling = {
            "rope_type": "yarn",
            "factor": 2.0,
            "original_max_position_embeddings": 131072,
        }

        rope_parameters = config.text_config.rope_parameters
        self.assertEqual(rope_parameters["rope_type"], "yarn")
        self.assertEqual(rope_parameters["factor"], 2.0)
        self.assertEqual(rope_parameters["mrope_section"], [8, 12, 12])
        self.assertTrue(rope_parameters["video_rope"])

    def test_public_checkpoint_config_contract(self):
        """External Ling-3.0-flash-VL config literals must survive local parsing."""
        config = BailingMoeV3VLConfig(
            image_token_id=157157,
            video_token_id=156909,
            mrope_section=[8, 12, 12],
            text_config={
                "num_hidden_layers": 42,
                "vocab_size": 157184,
                "max_position_embeddings": 131072,
                "moe_router_enable_expert_bias": True,
                "num_experts": 512,
                "num_experts_per_tok": 8,
                "n_group": 8,
                "topk_group": 4,
                "score_function": "sigmoid",
                "routed_scaling_factor": 2.5,
                "short_conv_kernel_size": 4,
            },
            vision_config={"disable_merger_proj": True},
        )

        self.assertEqual(config.text_config.num_hidden_layers, 42)
        self.assertEqual(config.text_config.max_position_embeddings, 131072)
        self.assertTrue(config.text_config.moe_router_enable_expert_bias)
        self.assertEqual(
            config.text_config.rope_parameters["mrope_section"], [8, 12, 12]
        )
        self.assertTrue(config.text_config.rope_parameters["video_rope"])
        self.assertTrue(config.vision_config.disable_merger_proj)


if __name__ == "__main__":
    unittest.main()
