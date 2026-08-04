import math
import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import ModelConfig, compute_mla_mscale_scaling
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_mla_config(architectures, **overrides):
    defaults = dict(
        architectures=architectures,
        model_type="mla",
        hidden_size=4096,
        num_attention_heads=32,
        num_hidden_layers=2,
        vocab_size=32000,
        num_key_value_heads=32,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        use_mla=True,
        rope_scaling=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestMlaMscaleScaling(CustomTestCase):
    def test_respects_disabled_yarn_scaling(self):
        base_scaling = 1 / math.sqrt(128)
        rope_scaling = {
            "rope_type": "deepseek_yarn",
            "factor": 128,
            "mscale_all_dim": 1,
            "apply_yarn_scaling": False,
        }

        self.assertEqual(
            compute_mla_mscale_scaling(rope_scaling, base_scaling), base_scaling
        )

    def test_applies_yarn_scaling_by_default(self):
        base_scaling = 1 / math.sqrt(128)
        rope_scaling = {
            "rope_type": "deepseek_yarn",
            "factor": 128,
            "mscale_all_dim": 1,
        }

        self.assertGreater(
            compute_mla_mscale_scaling(rope_scaling, base_scaling), base_scaling
        )

    def test_respects_disabled_native_apply_scale(self):
        base_scaling = 1 / math.sqrt(128)
        rope_scaling = {
            "rope_type": "mistral",
            "factor": 128,
            "mscale_all_dim": 1,
            "apply_scale": False,
        }

        self.assertEqual(
            compute_mla_mscale_scaling(rope_scaling, base_scaling), base_scaling
        )


class TestModelConfigMlaScaling(CustomTestCase):
    """MLA branches must derive model_config.scaling; FlashInferMLA reads it
    unconditionally, so a missing value crashes those models at startup."""

    def _derive(self, text_config):
        model_config = ModelConfig.__new__(ModelConfig)
        model_config.hf_config = text_config
        model_config.hf_text_config = text_config
        model_config._derive_model_shapes()
        return model_config

    def test_minicpm3_sets_plain_mla_scaling(self):
        # MiniCPM3 has no yarn mscale: plain 1/sqrt(qk_nope + qk_rope).
        model_config = self._derive(
            _make_mla_config(
                ["MiniCPM3ForCausalLM"], qk_nope_head_dim=96, qk_rope_head_dim=32
            )
        )
        self.assertAlmostEqual(model_config.scaling, 1 / math.sqrt(96 + 32))

    def test_deepseek_vl2_mla_sets_scaling(self):
        model_config = self._derive(
            _make_mla_config(["DeepseekVL2ForCausalLM"], use_mla=True)
        )
        self.assertAlmostEqual(model_config.scaling, 1 / math.sqrt(128 + 64))

    def test_kimi_vl_sets_scaling(self):
        model_config = self._derive(
            _make_mla_config(["KimiVLForConditionalGeneration"])
        )
        self.assertAlmostEqual(model_config.scaling, 1 / math.sqrt(128 + 64))

    def test_kimi_vl_applies_yarn_mscale(self):
        # VL2/Kimi-VL run a DeepseekV2 language model, so yarn mscale applies
        # and pushes the scale above the plain base value.
        rope_scaling = {
            "rope_type": "deepseek_yarn",
            "factor": 128,
            "mscale_all_dim": 1,
        }
        model_config = self._derive(
            _make_mla_config(
                ["KimiVLForConditionalGeneration"], rope_scaling=rope_scaling
            )
        )
        self.assertGreater(model_config.scaling, 1 / math.sqrt(128 + 64))


if __name__ == "__main__":
    unittest.main()
