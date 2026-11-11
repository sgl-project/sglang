"""Nemotron Shortcut's hybrid layout, derived from the checkpoint's own config."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from transformers import PretrainedConfig

from sglang.srt.configs import all_configs  # noqa: F401  registers the layout
from sglang.srt.configs.linear_attn_model_registry import (
    get_linear_attn_config,
    get_linear_attn_spec,
)
from sglang.srt.configs.nemotron_shortcut import (
    NemotronShortcutLayout,
    gdp_conv_dim,
    sdsa_index_head_dim,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

# The checkpoint's normalized block vocabulary, one of each plus a repeat.
LAYER_TYPES = [
    "linear_attention",
    "mlp",
    "shortcut_moe_attn",
    "shortcut_moe_gdp",
    "full_attention",
    "linear_attention",
]


class _TextConfig(PretrainedConfig):
    """Stand-in for the checkpoint's remote-code text config."""

    model_type = "nemotron4_text"


class _Config(PretrainedConfig):
    model_type = "nemotron4"


def _text_config(**overrides) -> _TextConfig:
    fields = dict(
        layer_types=list(LAYER_TYPES),
        gdp_num_heads=8,
        gdp_num_groups=8,
        gdp_head_dim=64,
        gdp_key_head_dim=128,
        gdp_num_householder=3,
        gdp_conv_kernel=4,
        index_topk=2048,
        index_head_dim=128,
    )
    fields.update(overrides)
    return _TextConfig(**fields)


def _config(text_config=None, arch="Nemotron4ForConditionalGeneration") -> _Config:
    config = _Config(architectures=[arch])
    config.text_config = text_config or _text_config()
    return config


class TestLayout(CustomTestCase):
    def test_layer_ids_follow_the_mixer_of_each_block(self):
        layout = NemotronShortcutLayout(_text_config())
        # Shortcut-MoE blocks carry state of their mixer; MLP blocks carry none.
        self.assertEqual(layout.linear_layer_ids, [0, 3, 5])
        self.assertEqual(layout.full_attention_layer_ids, [2, 4])

    def test_registry_wraps_only_this_model_type(self):
        spec, view = get_linear_attn_config(_config())
        self.assertIsInstance(view, NemotronShortcutLayout)
        self.assertIs(get_linear_attn_spec(_config()), spec)

        other = _Config(architectures=["SomeOtherModel"])
        other.text_config = PretrainedConfig(layer_types=list(LAYER_TYPES))
        result = get_linear_attn_config(other)
        self.assertFalse(
            result is not None and isinstance(result[1], NemotronShortcutLayout)
        )

    def test_state_shape_shards_every_conv_block_by_head(self):
        config = _text_config()
        for tp in (1, 2, 4):
            parallel = SimpleNamespace(attn_tp_size=tp)
            with patch(
                "sglang.srt.runtime_context.get_parallel", return_value=parallel
            ):
                params = NemotronShortcutLayout(config).mamba2_cache_params
            # Conv window over [V0..V2, K0..K2, Q]: 3 * 512 + 4 * 1024 channels.
            self.assertEqual(gdp_conv_dim(config), 3 * 512 + 4 * 1024)
            self.assertEqual(params.shape.conv[0][0], gdp_conv_dim(config) // tp)
            self.assertEqual(params.shape.conv[0][1], config.gdp_conv_kernel - 1)
            # One FP32 [H, V, K] recurrent state per rank's heads.
            self.assertEqual(params.shape.temporal, (8 // tp, 64, 128))
            self.assertEqual(params.layers, [0, 3, 5])

    def test_unequal_gdp_groups_are_rejected(self):
        with self.assertRaises(ValueError):
            NemotronShortcutLayout(_text_config(gdp_num_groups=4))


class TestSdsaGating(CustomTestCase):
    def test_index_cache_only_with_index_topk(self):
        self.assertEqual(sdsa_index_head_dim(_config()), 128)
        self.assertIsNone(sdsa_index_head_dim(_config(_text_config(index_topk=None))))


if __name__ == "__main__":
    unittest.main()
