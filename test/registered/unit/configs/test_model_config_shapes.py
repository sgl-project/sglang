import math
import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import (
    AttentionArch,
    ModelConfig,
    _quant_config_to_dict,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


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

    def test_v_head_dim_zero_falls_back_to_head_dim(self):
        # deepseek-vl2-tiny's language_config sets use_mla=False + v_head_dim=0
        # as an MLA-disabled sentinel. The non-MLA KV pool sizes its V buffer
        # from v_head_dim directly, so a literal 0 collapses V-cache to a
        # zero-width tensor and `v_cache[indices] = v` fails with a shape
        # mismatch inside `_set_kv_buffer_impl`. v_head_dim=0 must be treated
        # identically to `None` and fall back to head_dim.
        text_config = _make_text_config(head_dim=128, v_head_dim=0)

        model_config = self._derive_shapes(text_config)

        self.assertEqual(model_config.v_head_dim, 128)
        self.assertEqual(text_config.v_head_dim, 128)

    def test_ling_mla_nope_shapes(self):
        text_config = _make_text_config(
            architectures=["BailingMoeV3ForCausalLM"],
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            use_mla_nope=True,
            v_head_dim=128,
        )

        model_config = self._derive_shapes(text_config)

        self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
        self.assertEqual(model_config.head_dim, 128)
        self.assertEqual(model_config.qk_rope_head_dim, 0)
        self.assertEqual(model_config.scaling, 1 / math.sqrt(128))

    def test_ling_mla_rope_shapes(self):
        text_config = _make_text_config(
            architectures=["BailingMoeV3ForCausalLM"],
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            use_mla_nope=False,
            v_head_dim=128,
        )

        model_config = self._derive_shapes(text_config)

        self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
        self.assertEqual(model_config.head_dim, 128)
        self.assertEqual(model_config.qk_rope_head_dim, 64)
        self.assertEqual(model_config.scaling, 1 / math.sqrt(192))

    def test_sarvam_mla_shapes(self):
        text_config = _make_text_config(
            architectures=["SarvamMLAForCausalLM"],
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            rope_scaling=None,
            v_head_dim=128,
        )

        model_config = self._derive_shapes(text_config)

        self.assertEqual(model_config.attention_arch, AttentionArch.MLA)
        self.assertEqual(model_config.head_dim, 192)
        self.assertEqual(model_config.qk_rope_head_dim, 64)
        self.assertEqual(model_config.scaling, 1 / math.sqrt(192))

    def test_quant_config_objects_are_normalized(self):
        quant_config = SimpleNamespace(to_dict=lambda: {"quant_method": "test"})

        self.assertEqual(_quant_config_to_dict(quant_config), {"quant_method": "test"})


if __name__ == "__main__":
    unittest.main()
