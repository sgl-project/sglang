"""Unit tests for JetBrains Mellum config helpers."""

import unittest

from sglang.srt.configs.mellum import MellumConfig
from sglang.srt.configs.model_config import get_hybrid_layer_ids, is_hybrid_swa_model
from sglang.srt.models.mellum import (
    _get_layer_rope_config,
    _get_layer_type,
    get_attention_sliding_window_size,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


YARN_ATTENTION_FACTOR = 1.2773247398245525


def _make_mellum2_like_config() -> MellumConfig:
    return MellumConfig(
        num_hidden_layers=4,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        mlp_layer_types=["sparse", "sparse", "sparse", "sparse"],
        sliding_window=1024,
        rope_parameters={
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": 16.0,
                "original_max_position_embeddings": 8192,
                "attention_factor": YARN_ATTENTION_FACTOR,
                "beta_fast": 32,
                "beta_slow": 1,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 500000.0,
            },
        },
    )


class TestMellumConfig(unittest.TestCase):
    def test_config_defaults_to_mellum_architecture(self):
        config = MellumConfig()

        self.assertEqual(config.model_type, "mellum")
        self.assertEqual(config.architectures, ["MellumForCausalLM"])
        self.assertEqual(config.mlp_layer_types, ["sparse"] * config.num_hidden_layers)

    def test_mellum2_layer_types_are_hybrid_swa(self):
        config = _make_mellum2_like_config()

        self.assertTrue(config.is_hybrid_swa)
        self.assertTrue(is_hybrid_swa_model(["MellumForCausalLM"], config))
        self.assertEqual(
            get_hybrid_layer_ids(["MellumForCausalLM"], config),
            ([0, 1, 2], [3]),
        )
        self.assertEqual(_get_layer_type(config, 0), "sliding_attention")
        self.assertEqual(_get_layer_type(config, 3), "full_attention")

    def test_sliding_window_matches_exclusive_radix_attention_window(self):
        config = _make_mellum2_like_config()

        self.assertEqual(get_attention_sliding_window_size(config), 1023)

        config.sliding_window = None
        self.assertIsNone(get_attention_sliding_window_size(config))

    def test_per_layer_rope_parameters_are_flattened_for_sglang(self):
        config = _make_mellum2_like_config()

        full_theta, full_scaling, full_config, partial_rotary_factor = (
            _get_layer_rope_config(config, "full_attention")
        )
        self.assertEqual(full_theta, 500000.0)
        self.assertEqual(partial_rotary_factor, 1.0)
        self.assertEqual(
            full_scaling,
            {
                "rope_type": "yarn",
                "factor": 16.0,
                "original_max_position_embeddings": 8192,
                "beta_fast": 32,
                "beta_slow": 1,
                "attn_factor": YARN_ATTENTION_FACTOR,
            },
        )
        self.assertEqual(
            full_config.rope_parameters["attention_factor"],
            YARN_ATTENTION_FACTOR,
        )

        sliding_theta, sliding_scaling, sliding_config, partial_rotary_factor = (
            _get_layer_rope_config(config, "sliding_attention")
        )
        self.assertEqual(sliding_theta, 500000.0)
        self.assertIsNone(sliding_scaling)
        self.assertEqual(partial_rotary_factor, 1.0)
        self.assertEqual(sliding_config.rope_parameters["rope_type"], "default")


if __name__ == "__main__":
    unittest.main()
