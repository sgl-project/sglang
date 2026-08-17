"""Unit tests for ``sglang.srt.configs.zaya.ZayaConfig``."""

import unittest

from transformers import AutoConfig

from sglang.srt.configs.zaya import ZayaConfig, register_zaya_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


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


if __name__ == "__main__":
    unittest.main()
