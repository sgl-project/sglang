"""Unit tests for ``sglang.srt.configs.zaya.ZayaConfig``."""

import unittest

from transformers import AutoConfig

from sglang.srt.configs.zaya import ZayaConfig, register_zaya_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


# Field set copied verbatim (layer list compressed) from Zyphra/ZAYA1-8B
# config.json at pinned revision 67d34da515b30409ee8daa67c3605c4402d03f6f —
# the transformers-native (>= v5.13) ZAYA schema.
_NATIVE_8B_CONFIG = {
    "architectures": ["ZayaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 2,
    "cca_time0": 2,
    "cca_time1": 2,
    "dtype": "bfloat16",
    "eos_token_id": 106,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "layer_types": ["hybrid"] * 40,
    "lm_head_bias": False,
    "max_position_embeddings": 131072,
    "model_type": "zaya",
    "moe_intermediate_size": 2048,
    "num_attention_heads": 8,
    "num_experts": 16,
    "num_experts_per_tok": 1,
    "num_hidden_layers": 40,
    "num_key_value_heads": 2,
    "output_router_logits": False,
    "pad_token_id": 0,
    "partial_rotary_factor": 0.5,
    "rms_norm_eps": 1e-05,
    "rope_parameters": {
        "hybrid": {
            "partial_rotary_factor": 0.5,
            "rope_theta": 5000000,
            "rope_type": "default",
        },
        "hybrid_sliding": {
            "partial_rotary_factor": 0.5,
            "rope_theta": 10000.0,
            "rope_type": "default",
        },
        "rope_type": "default",
    },
    "router_hidden_size": 256,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "use_cache": True,
    "vocab_size": 262272,
}

# Field set copied verbatim from Zyphra/ZAYA1-8B-legacy config.json at pinned
# revision 1bef1fb587fc4bcb78f318f49ff2ef2b2510f073 — the legacy
# Megatron-style schema SGLang already understands.
_LEGACY_8B_CONFIG = {
    "activation_func": "swiglu",
    "activation_func_fp8_input_store": False,
    "add_bias_linear": False,
    "architectures": ["ZayaForCausalLM"],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bias_activation_fusion": True,
    "bos_token_id": 2,
    "cca": True,
    "dtype": "bfloat16",
    "eos_token_id": 106,
    "ffn_hidden_size": 4096,
    "gated_linear_unit": True,
    "hidden_size": 2048,
    "head_dim": 128,
    "kv_channels": 128,
    "lm_head_bias": False,
    "mamba_cache_dtype": "float32",
    "max_position_embeddings": 131072,
    "model_type": "zaya",
    "moe_router_topk": 1,
    "norm_epsilon": 1e-05,
    "normalization": "RMSNorm",
    "num_attention_heads": 8,
    "num_experts": 16,
    "num_hidden_layers": 80,
    "num_key_value_heads": 2,
    "num_query_groups": 2,
    "pad_token_id": 0,
    "partial_rotary_factor": 0.5,
    "residual_in_fp32": True,
    "rope_scaling": False,
    "rope_theta": 5000000,
    "scale_residual_merge": True,
    "sliding_window": None,
    "use_cache": True,
    "vocab_size": 262272,
    "zaya_mlp_expansion": 256,
    "zaya_use_eda": True,
    "zaya_use_mod": True,
}


class TestZayaConfig(CustomTestCase):
    def test_default_fields_match_zaya1_base(self):
        """Defaults reflect Zyphra/ZAYA1-base reference checkpoint."""
        cfg = ZayaConfig()
        self.assertEqual(cfg.model_type, "zaya")
        self.assertEqual(cfg.hidden_size, 2048)
        self.assertEqual(cfg.head_dim, 128)
        self.assertEqual(cfg.num_attention_heads, 8)
        self.assertEqual(cfg.num_query_groups, 2)
        self.assertEqual(cfg.num_key_value_heads, 2)
        self.assertEqual(cfg.num_experts, 16)
        self.assertEqual(cfg.moe_router_topk, 1)
        self.assertEqual(cfg.ffn_hidden_size, 4096)
        self.assertEqual(cfg.zaya_mlp_expansion, 256)
        self.assertEqual(cfg.cca_time0, 2)
        self.assertEqual(cfg.cca_time1, 2)
        self.assertTrue(cfg.tie_word_embeddings)
        self.assertTrue(cfg.zaya_use_eda)
        self.assertTrue(cfg.zaya_use_mod)
        self.assertTrue(cfg.scale_residual_merge)
        self.assertEqual(cfg.partial_rotary_factor, 0.5)
        self.assertEqual(cfg.rope_theta, 1_000_000.0)

    def test_rope_parameters_auto_derived(self):
        """When neither ``rope_scaling`` nor ``rope_parameters`` is supplied,
        both ``rope_theta`` and ``partial_rotary_factor`` should still appear
        inside ``rope_parameters`` together with a default ``rope_type``.
        """
        cfg = ZayaConfig()
        rp = cfg.rope_parameters
        self.assertEqual(rp["rope_type"], "default")
        self.assertEqual(rp["rope_theta"], 1_000_000.0)
        self.assertEqual(rp["partial_rotary_factor"], 0.5)

    def test_rope_parameters_explicit_takes_priority(self):
        cfg = ZayaConfig(rope_parameters={"type": "linear", "factor": 4.0})
        rp = cfg.rope_parameters
        # ``type`` is normalized to ``rope_type``.
        self.assertEqual(rp["rope_type"], "linear")
        self.assertEqual(rp["factor"], 4.0)
        # Defaults are still merged in.
        self.assertEqual(rp["rope_theta"], 1_000_000.0)

    def test_head_dim_required(self):
        with self.assertRaises(AssertionError):
            ZayaConfig(head_dim=None)

    def test_num_query_groups_must_equal_kv_heads(self):
        with self.assertRaises(AssertionError):
            ZayaConfig(num_query_groups=4, num_key_value_heads=2)

    def test_hybrid_model_properties(self):
        """Verify properties required for HybridReqToTokenPool integration."""
        cfg = ZayaConfig()
        # Default 80 layers: even layers are attention, odd are MoE
        self.assertEqual(cfg.full_attention_layer_ids, list(range(0, 80, 2)))
        self.assertEqual(cfg.linear_layer_ids, cfg.full_attention_layer_ids)
        self.assertEqual(cfg.mamba_chunk_size, 1)

        params = cfg.mamba2_cache_params
        self.assertIsNotNone(params)
        # conv[0] = conv_state: (in_out_ch, total_padding)
        in_out_ch = (cfg.num_attention_heads + cfg.num_key_value_heads) * cfg.head_dim
        total_padding = (cfg.cca_time0 - 1) + (cfg.cca_time1 - 1)
        self.assertEqual(params.shape.conv[0], (in_out_ch, total_padding))
        # conv[1] = prev_hs: (hidden_size, 1)
        self.assertEqual(params.shape.conv[1], (cfg.hidden_size, 1))
        self.assertEqual(params.layers, cfg.linear_layer_ids)

    def test_hybrid_model_properties_with_zaya_layers(self):
        """When zaya_layers is provided, layer IDs derive from the list."""
        cfg = ZayaConfig(zaya_layers=["a", 16, "a", 16])
        self.assertEqual(cfg.num_hidden_layers, 4)
        self.assertEqual(cfg.full_attention_layer_ids, [0, 2])
        self.assertEqual(cfg.linear_layer_ids, [0, 2])

    def test_auto_config_registration_is_idempotent(self):
        # Calling the helper twice must not raise even though importing the
        # module already registered the model type.
        register_zaya_config()
        register_zaya_config()
        # ``AutoConfig.for_model`` now resolves to ``ZayaConfig``.
        cfg = AutoConfig.for_model("zaya")
        self.assertIsInstance(cfg, ZayaConfig)


class TestZayaNativeConfigTranslation(CustomTestCase):
    """Translation of the transformers-native (>= v5.13) ZAYA config schema.

    Zyphra's native checkpoints (ZAYA1-8B, ZAYA1-74B-preview) describe L
    hybrid layers (attention + MoE folded per layer) with native field
    names, while SGLang's internal zaya model is legacy-shaped: 2L
    interleaved layers with Megatron-style field names. Feeding a native
    config into ``ZayaConfig`` today silently keeps the legacy defaults
    (e.g. ``num_hidden_layers`` stays at the native 40 while the default
    4096 ``ffn_hidden_size`` no longer corresponds to it), so the model is
    built with the wrong geometry and cannot load the checkpoint.
    """

    def test_native_8b_translates_to_internal_geometry(self):
        cfg = ZayaConfig(**_NATIVE_8B_CONFIG)
        self.assertEqual(cfg.checkpoint_format, "native")
        # L native hybrid layers -> 2L interleaved internal layers.
        self.assertEqual(cfg.num_hidden_layers, 80)
        self.assertEqual(cfg.full_attention_layer_ids, list(range(0, 80, 2)))
        # moe_intermediate_size I -> gated ffn_hidden_size 2I.
        self.assertEqual(cfg.ffn_hidden_size, 4096)
        self.assertEqual(cfg.activation_func, "swiglu")
        self.assertTrue(cfg.gated_linear_unit)
        # Nested per-layer-type rope -> flat full-attention rope.
        self.assertEqual(cfg.rope_theta, 5_000_000.0)
        self.assertEqual(cfg.partial_rotary_factor, 0.5)
        self.assertEqual(cfg.rope_parameters["rope_theta"], 5_000_000.0)
        # Field renames.
        self.assertEqual(cfg.num_query_groups, 2)
        self.assertEqual(cfg.norm_epsilon, 1e-5)
        self.assertEqual(cfg.zaya_mlp_expansion, 256)
        self.assertEqual(cfg.moe_router_topk, 1)
        # Architecture features that are always on in the native schema.
        self.assertTrue(cfg.zaya_use_eda)
        self.assertTrue(cfg.zaya_use_mod)
        self.assertTrue(cfg.scale_residual_merge)
        # Unchanged pass-through.
        self.assertEqual(cfg.vocab_size, 262272)
        self.assertEqual(cfg.hidden_size, 2048)
        self.assertEqual(cfg.head_dim, 128)
        self.assertEqual(cfg.num_experts, 16)

    def test_legacy_8b_stays_legacy(self):
        cfg = ZayaConfig(**_LEGACY_8B_CONFIG)
        self.assertEqual(cfg.checkpoint_format, "legacy")
        self.assertEqual(cfg.num_hidden_layers, 80)
        self.assertEqual(cfg.ffn_hidden_size, 4096)
        self.assertEqual(cfg.rope_theta, 5_000_000.0)
        self.assertEqual(cfg.zaya_mlp_expansion, 256)

    def test_mixed_markers_resolve_as_legacy(self):
        """A config carrying legacy markers must not be re-translated even if
        native-only keys are also present (e.g. a hand-converted config that
        kept both field sets): translating it would double the layer count."""
        cfg = ZayaConfig(**{**_LEGACY_8B_CONFIG, "moe_intermediate_size": 2048})
        self.assertEqual(cfg.checkpoint_format, "legacy")
        self.assertEqual(cfg.num_hidden_layers, 80)
        self.assertEqual(cfg.ffn_hidden_size, 4096)


if __name__ == "__main__":
    unittest.main()
