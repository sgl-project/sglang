"""Unit tests for ModelConfig shape normalization."""

import math
import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import (
    AttentionArch,
    ModelConfig,
    compute_mla_mscale_scaling,
)
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

    def test_mla_branches_derive_scaling(self):
        base_scaling = 1 / math.sqrt(128 + 64)
        rope_scaling = {"rope_type": "yarn", "factor": 2.0, "mscale_all_dim": 1.0}
        cases = [
            ("KimiVLForConditionalGeneration", {}, base_scaling),
            ("MiniCPM3ForCausalLM", {}, base_scaling),
            ("DeepseekVL2ForCausalLM", {"use_mla": True}, base_scaling),
            (
                "KimiVLForConditionalGeneration",
                {"rope_scaling": rope_scaling},
                compute_mla_mscale_scaling(rope_scaling, base_scaling),
            ),
        ]

        for arch, extra, expected_scaling in cases:
            with self.subTest(arch=arch, rope_scaling=extra.get("rope_scaling")):
                config_kwargs = dict(
                    architectures=[arch],
                    kv_lora_rank=512,
                    qk_nope_head_dim=128,
                    qk_rope_head_dim=64,
                    v_head_dim=128,
                    rope_scaling=None,
                )
                config_kwargs.update(extra)
                text_config = _make_text_config(**config_kwargs)
                model_config = self._derive_shapes(text_config)

                self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
                self.assertAlmostEqual(model_config.scaling, expected_scaling)


if __name__ == "__main__":
    unittest.main()
