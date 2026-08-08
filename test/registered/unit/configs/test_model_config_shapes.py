"""Unit tests for ModelConfig shape normalization."""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import ModelConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_text_config(**overrides):
    defaults = dict(
        architectures=["MixtralForCausalLM"],
        model_type="mixtral",
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=2,
        vocab_size=32000,
        num_key_value_heads=8,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestModelConfigShapes(CustomTestCase):
    def _derive_shapes(self, text_config):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.hf_config = text_config
        model_config.hf_text_config = text_config
        model_config._derive_model_shapes()
        return model_config

    def test_optional_head_dims_default_when_none(self):
        text_config = _make_text_config(
            head_dim=None,
            v_head_dim=None,
            swa_head_dim=None,
            swa_v_head_dim=None,
        )

        model_config = self._derive_shapes(text_config)

        self.assertEqual(model_config.head_dim, 128)
        self.assertEqual(model_config.v_head_dim, 128)
        self.assertEqual(model_config.swa_head_dim, 128)
        self.assertEqual(model_config.swa_v_head_dim, 128)
        self.assertEqual(text_config.head_dim, 128)
        self.assertEqual(text_config.v_head_dim, 128)
        self.assertEqual(text_config.swa_head_dim, 128)
        self.assertEqual(text_config.swa_v_head_dim, 128)

    def test_explicit_head_dims_are_preserved(self):
        text_config = _make_text_config(
            head_dim=128,
            v_head_dim=96,
            swa_head_dim=64,
            swa_v_head_dim=48,
        )

        model_config = self._derive_shapes(text_config)

        self.assertEqual(model_config.head_dim, 128)
        self.assertEqual(model_config.v_head_dim, 96)
        self.assertEqual(model_config.swa_head_dim, 64)
        self.assertEqual(model_config.swa_v_head_dim, 48)

    def test_nextn_predict_layers_from_outer_config_for_wrapper_models(self):
        """Draft rewrites set num_nextn_predict_layers on the outer hf_config;
        for wrapper configs (hf_text_config is a nested, different object) the
        value must still be visible, or EAGLE/MTP KV sizing treats the draft
        as a full-depth model (#29857)."""
        text_config = _make_text_config(
            architectures=["Qwen3_5ForCausalLMMTP"], head_dim=128
        )
        outer_config = SimpleNamespace(
            architectures=["Qwen3_5ForCausalLMMTP"],
            model_type="qwen3_5",
            num_nextn_predict_layers=1,
        )

        model_config = ModelConfig.__new__(ModelConfig)
        model_config.hf_config = outer_config
        model_config.hf_text_config = text_config
        model_config._derive_model_shapes()

        self.assertEqual(model_config.num_nextn_predict_layers, 1)

    def test_nextn_predict_layers_nested_value_wins(self):
        """When the nested text config carries its own value, it takes
        precedence over the outer config."""
        text_config = _make_text_config(
            head_dim=128,
            num_nextn_predict_layers=2,
        )
        outer_config = SimpleNamespace(
            architectures=["MixtralForCausalLM"],
            model_type="mixtral",
            num_nextn_predict_layers=1,
        )

        model_config = ModelConfig.__new__(ModelConfig)
        model_config.hf_config = outer_config
        model_config.hf_text_config = text_config
        model_config._derive_model_shapes()

        self.assertEqual(model_config.num_nextn_predict_layers, 2)

    def test_nextn_predict_layers_absent_everywhere_is_none(self):
        text_config = _make_text_config(head_dim=128)

        model_config = self._derive_shapes(text_config)

        self.assertIsNone(model_config.num_nextn_predict_layers)


if __name__ == "__main__":
    unittest.main()
