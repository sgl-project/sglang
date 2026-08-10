"""Unit tests for hybrid attention model configuration."""

import math
import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import (
    get_deepseek_v4_compress_ratios,
    get_hybrid_layer_ids,
    get_num_indexer_layers,
    is_embedding_gemma,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LEGACY = dict(
    architectures=["DeepseekV4ForCausalLM"],
    compress_ratios=[128, 4, 128, 4],
    rope_scaling={"rope_type": "yarn"},
)

MODERN = dict(
    architectures=["DeepseekV4ForCausalLM"],
    layer_types=[
        "heavily_compressed_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
        "compressed_sparse_attention",
    ],
    compress_rates={
        "heavily_compressed_attention": 128,
        "compressed_sparse_attention": 4,
    },
    rope_parameters={"rope_type": "yarn"},
)


class TestDeepseekV4ConfigCompat(CustomTestCase):
    def test_deepseek_v4_compress_ratios_legacy_and_modern(self):
        legacy = get_deepseek_v4_compress_ratios(SimpleNamespace(**LEGACY))
        modern = get_deepseek_v4_compress_ratios(SimpleNamespace(**MODERN))
        self.assertEqual(legacy, [128, 4, 128, 4])
        self.assertEqual(legacy, modern)

    def test_deepseek_v4_num_indexer_layers_legacy_and_modern(self):
        legacy = SimpleNamespace(**LEGACY)
        modern = SimpleNamespace(**MODERN)
        # two layers have compress_ratio == 4
        self.assertEqual(get_num_indexer_layers(legacy), 2)
        self.assertEqual(get_num_indexer_layers(modern), 2)

    def test_deepseek_v4_empty_compress_ratios_falls_back_to_modern(self):
        # SGLang's own DeepSeekV4Config defaults compress_ratios to [], so an
        # empty legacy placeholder must not shadow a populated modern
        # representation. This is hardening rather than the #34092 repro
        # itself: HF's V4 config drops the legacy kwarg instead of emptying it.
        config = SimpleNamespace(compress_ratios=[], **MODERN)
        self.assertEqual(get_deepseek_v4_compress_ratios(config), [128, 4, 128, 4])
        self.assertEqual(get_num_indexer_layers(config), 2)

    def test_deepseek_v4_layer_type_absent_from_compress_rates_is_uncompressed(self):
        # sliding_attention carries no compress_rates entry and must resolve to
        # ratio 0, matching the legacy mapping. Ratios also stay one-per-layer:
        # consumers index by layer id and slice the list per PP rank, so an
        # unmapped type must yield 0 rather than drop an entry and shift every
        # later layer's ratio.
        config = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"],
            layer_types=["sliding_attention", "compressed_sparse_attention"],
            compress_rates={"compressed_sparse_attention": 4},
        )
        self.assertEqual(get_deepseek_v4_compress_ratios(config), [0, 4])
        self.assertEqual(get_num_indexer_layers(config), 1)

    def test_deepseek_v4_model_config_accepts_modern_transformers_config(self):
        # transformers >= 4.57 config (compress_rates/layer_types/rope_parameters)
        # must derive shapes without error and match the legacy form (#34092).
        from sglang.srt.configs.model_config import AttentionArch, ModelConfig

        rope = {"rope_type": "deepseek_yarn", "factor": 16, "mscale_all_dim": 1}
        # _derive_model_shapes reads these unconditionally after the arch branch
        # (num_attention_heads/model_type/hidden_size/num_hidden_layers/vocab_size),
        # plus the V4 branch's head/window/index dims.
        common = dict(
            model_type="deepseek_v4",
            num_attention_heads=64,
            num_hidden_layers=4,
            hidden_size=4096,
            vocab_size=129280,
            head_dim=576,
            qk_rope_head_dim=64,
            sliding_window=128,
            index_head_dim=128,
        )
        legacy_cfg = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"],
            compress_ratios=[128, 4, 128, 4],
            rope_scaling=rope,
            **common,
        )
        modern_cfg = SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"],
            layer_types=MODERN["layer_types"],
            compress_rates=MODERN["compress_rates"],
            rope_parameters=rope,
            **common,
        )

        def derive(cfg):
            mc = ModelConfig.__new__(ModelConfig)  # bypass the heavy __init__
            mc.hf_config = cfg
            mc.hf_text_config = cfg
            mc._derive_model_shapes()
            return mc

        legacy = derive(legacy_cfg)
        modern = derive(modern_cfg)
        self.assertEqual(modern.compress_ratios, [128, 4, 128, 4])
        self.assertEqual(legacy.compress_ratios, modern.compress_ratios)
        self.assertEqual(modern.attention_arch, AttentionArch.MHA)
        # scaling is equal across both forms and yarn actually applied (not base)
        self.assertEqual(legacy.scaling, modern.scaling)
        self.assertNotEqual(modern.scaling, 1 / math.sqrt(576))


class TestHybridLayerIds(CustomTestCase):
    def test_layer_type_architectures(self):
        config = SimpleNamespace(
            num_hidden_layers=4,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
        )

        for architecture in (
            "Gemma4ForCausalLM",
            "Gemma4ForConditionalGeneration",
            "LagunaForCausalLM",
            "MellumForCausalLM",
        ):
            with self.subTest(architecture=architecture):
                self.assertEqual(
                    get_hybrid_layer_ids([architecture], config),
                    ([0, 2], [1, 3]),
                )


class TestEmbeddingGemmaConfig(CustomTestCase):
    def test_detects_bidirectional_gemma3_text_config(self):
        config = SimpleNamespace(
            model_type="gemma3_text", use_bidirectional_attention=True
        )
        self.assertTrue(is_embedding_gemma(config))

    def test_does_not_misclassify_causal_gemma3(self):
        config = SimpleNamespace(
            model_type="gemma3_text", use_bidirectional_attention=False
        )
        self.assertFalse(is_embedding_gemma(config))


if __name__ == "__main__":
    unittest.main()
