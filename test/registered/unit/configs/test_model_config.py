"""Unit tests for hybrid attention model configuration."""

import math
import unittest
from types import SimpleNamespace

from sglang.srt.configs.model_config import (
    AttentionArch,
    ModelConfig,
    get_hybrid_layer_ids,
    get_num_indexer_layers,
    is_embedding_gemma,
)
from sglang.srt.utils.hf_transformers_patches import normalize_deepseek_v4_compat
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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


class TestDraftModelConfig(CustomTestCase):
    def test_qwen35_mtp_depth_is_synced_to_text_config(self):
        config = object.__new__(ModelConfig)
        config.is_draft_model = True
        config.speculative_algorithm = "EAGLE"
        config.hf_config = SimpleNamespace(
            architectures=["Qwen3_5MoeForConditionalGeneration"]
        )
        config.hf_text_config = SimpleNamespace()

        config._config_draft_model()

        self.assertEqual(config.hf_config.architectures, ["Qwen3_5ForCausalLMMTP"])
        self.assertEqual(config.hf_config.num_nextn_predict_layers, 1)
        self.assertEqual(config.hf_text_config.num_nextn_predict_layers, 1)


# ---------------------------------------------------------------------------
# End-to-end equivalence: legacy vs transformers >= 4.57 DeepSeek V4 configs
# ---------------------------------------------------------------------------


def _make_v4_alias():
    """Return the same DeepSeek V3 subclass sglang uses for ``deepseek_v4``.

    This mirrors ``_DeepseekV4ConfigAlias`` in
    ``python/sglang/srt/utils/hf_transformers/common.py`` — subclassing
    upstream ``DeepseekV3Config`` and overriding ``model_type`` — so the
    tests exercise the same load path a real checkpoint takes. Using a
    real ``PreTrainedConfig`` subclass (rather than ``SimpleNamespace`` or
    a bare ``PretrainedConfig()``) is what makes the equivalence assertion
    meaningful: the upstream class carries the ``rope_parameters`` <->
    ``rope_scaling`` bidirectional alias that sglang relies on to *not*
    duplicate rope-rename handling in this repo.
    """
    from transformers import DeepseekV3Config

    class _DeepseekV4ConfigAlias(DeepseekV3Config):
        model_type = "deepseek_v4"

    return _DeepseekV4ConfigAlias


_LT_CSA = "compressed_sparse_attention"
_LT_HCA = "heavily_compressed_attention"
_LT_SWA = "sliding_attention"

_YARN_ROPE = {
    "rope_type": "deepseek_yarn",
    "factor": 16,
    "mscale_all_dim": 1,
    "beta_fast": 32,
    "beta_slow": 1,
    "original_max_position_embeddings": 4096,
}


class TestDeepseekV4ConfigCompat(CustomTestCase):
    """Guard #34092 end-to-end: a DeepSeek V4 config loaded with
    ``transformers >= 4.57`` (``compress_rates`` dict + ``layer_types``
    list, plus ``rope_parameters``) must derive the same ``ModelConfig``
    shape as the legacy ``compress_ratios`` list + ``rope_scaling`` dict
    representation. This is the assertion that closes the issue — the
    field-rename fix in ``normalize_deepseek_v4_compat`` alone is
    necessary but not sufficient; only the equivalence assertion catches
    a rebuild that silently drops rope scaling, corrupts per-layer
    ratios, or breaks the indexer-layer count."""

    _COMMON = dict(
        num_attention_heads=64,
        num_hidden_layers=4,
        hidden_size=4096,
        vocab_size=129280,
        head_dim=576,
        qk_rope_head_dim=64,
        sliding_window=128,
        index_head_dim=128,
    )

    def _legacy_config(self):
        cfg = _make_v4_alias()(**self._COMMON)
        cfg.architectures = ["DeepseekV4ForCausalLM"]
        cfg.compress_ratios = [128, 4, 128, 4]
        cfg.rope_scaling = dict(_YARN_ROPE)
        return cfg

    def _modern_config(self):
        cfg = _make_v4_alias()(**self._COMMON)
        cfg.architectures = ["DeepseekV4ForCausalLM"]
        cfg.compress_rates = {_LT_CSA: 4, _LT_HCA: 128}
        cfg.layer_types = [_LT_HCA, _LT_CSA, _LT_HCA, _LT_CSA]
        cfg.rope_parameters = dict(_YARN_ROPE)
        return cfg

    def _derive(self, cfg):
        # Match how the loader actually calls things: normalize once at
        # the config boundary, then run the DSv4 branch of the shape
        # derivation. ``ModelConfig.__new__`` bypasses the heavy
        # ``__init__`` (which downloads weights).
        normalize_deepseek_v4_compat(cfg)
        mc = ModelConfig.__new__(ModelConfig)
        mc.hf_config = cfg
        mc.hf_text_config = cfg
        mc._derive_model_shapes()
        return mc

    def test_compress_ratios_match_legacy(self):
        # The heart of #34092: rebuilding via layer_types + compress_rates
        # must land on exactly the same per-layer list a legacy config
        # would have shipped.
        modern = self._derive(self._modern_config())
        legacy = self._derive(self._legacy_config())
        self.assertEqual(modern.compress_ratios, [128, 4, 128, 4])
        self.assertEqual(modern.compress_ratios, legacy.compress_ratios)

    def test_indexer_layer_count_matches_legacy(self):
        # ``get_num_indexer_layers`` filters by ``ratio == 4``. If the
        # rebuild left ratios in the wrong slots, indexer allocation
        # would silently drift.
        modern = self._modern_config()
        normalize_deepseek_v4_compat(modern)
        legacy = self._legacy_config()
        self.assertEqual(get_num_indexer_layers(modern), 2)
        self.assertEqual(get_num_indexer_layers(legacy), 2)

    def test_rope_scaling_survives_both_representations(self):
        # ``_derive_model_shapes`` in the DSv4 branch reads
        # ``self.hf_config.rope_scaling`` directly. Upstream
        # ``PreTrainedConfig`` bidirectionally aliases ``rope_parameters``
        # onto ``rope_scaling``; this test pins that dependency down so a
        # future upstream change (or a switch to a subclass that breaks
        # the alias) is caught here rather than at model-load time.
        modern = self._derive(self._modern_config())
        legacy = self._derive(self._legacy_config())
        self.assertEqual(modern.scaling, legacy.scaling)
        # And yarn actually applied — not the base 1/sqrt(head_dim).
        self.assertNotEqual(modern.scaling, 1 / math.sqrt(576))
        self.assertEqual(modern.attention_arch, AttentionArch.MHA)

    def test_sliding_attention_layers_use_legacy_zero(self):
        # Real V4 checkpoints interleave sliding-window layers. Those
        # slots are absent from ``compress_rates`` (SWA has no
        # compression rate). Upstream's legacy encoding uses ``0``; the
        # rebuild must preserve that.
        cfg = _make_v4_alias()(**self._COMMON)
        cfg.architectures = ["DeepseekV4ForCausalLM"]
        cfg.compress_rates = {_LT_CSA: 4, _LT_HCA: 128}
        cfg.layer_types = [_LT_SWA, _LT_HCA, _LT_SWA, _LT_CSA]
        cfg.rope_parameters = dict(_YARN_ROPE)
        derived = self._derive(cfg)
        self.assertEqual(derived.compress_ratios, [0, 128, 0, 4])
        # The two SWA layers must not count as C4 indexer layers.
        self.assertEqual(get_num_indexer_layers(cfg), 1)


if __name__ == "__main__":
    unittest.main()
