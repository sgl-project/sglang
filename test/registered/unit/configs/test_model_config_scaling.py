import math
import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import ModelConfig, compute_mla_mscale_scaling
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _mla_scaling(rope_scaling) -> tuple[float, float]:
    """Run ModelConfig._init_mla_scaling on fixed head dims -> (base, result)."""
    config = ModelConfig.__new__(ModelConfig)
    config.qk_nope_head_dim = 128
    config.qk_rope_head_dim = 64
    config._init_mla_scaling(rope_scaling)
    return 1 / math.sqrt(192), config.scaling


class TestMlaMscaleScaling(CustomTestCase):
    def test_ignores_transformers_v5_default_rope_parameters(self):
        base_scaling = 1 / math.sqrt(72)
        rope_scaling = {"rope_theta": 10000.0, "rope_type": "default"}

        with self.assertNoLogs("sglang.srt.configs.model_config", level="WARNING"):
            scaling = compute_mla_mscale_scaling(rope_scaling, base_scaling)

        self.assertEqual(scaling, base_scaling)

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

    def test_applies_legacy_scaling_without_rope_type(self):
        base_scaling = 1 / math.sqrt(128)
        rope_scaling = {"factor": 128, "mscale_all_dim": 1}

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


class TestInitMlaScaling(CustomTestCase):
    """ModelConfig._init_mla_scaling must not re-add a "default" fallback of its
    own: DeepseekV2AttentionMLA stamps rope_type="deepseek_yarn" on any non-empty
    rope_scaling, so a dict without rope_type/type still gets a yarn rope and its
    mscale belongs in the scale FlashInferMLA reads as sm_scale."""

    def test_applies_mscale_without_rope_type(self):
        base, scaling = _mla_scaling({"factor": 40, "mscale_all_dim": 1})
        self.assertGreater(scaling, base)

    def test_ignores_default_rope_type(self):
        base, scaling = _mla_scaling({"rope_type": "default", "factor": 40})
        self.assertEqual(scaling, base)

    def test_no_rope_scaling_keeps_base(self):
        base, scaling = _mla_scaling(None)
        self.assertEqual(scaling, base)

    def test_hyv4_shape_derivation_uses_rope_parameters(self):
        hf_config = SimpleNamespace(
            architectures=["HYV4ForCausalLM"],
            model_type="hy_v4",
            hidden_size=2816,
            num_hidden_layers=34,
            num_attention_heads=32,
            vocab_size=120832,
            head_dim=64,
            v_head_dim=256,
            kv_lora_rank=512,
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            index_topk=2048,
            index_head_dim=128,
            rope_parameters={"rope_theta": 10_000_000, "rope_type": "default"},
        )
        config = ModelConfig.__new__(ModelConfig)
        config.hf_config = hf_config
        config.hf_text_config = hf_config

        config._derive_model_shapes()

        self.assertEqual(config.scaling, 1 / math.sqrt(256))


if __name__ == "__main__":
    unittest.main()
