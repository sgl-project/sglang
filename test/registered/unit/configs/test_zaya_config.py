"""Unit tests for ``sglang.srt.configs.zaya.ZayaConfig``."""

import unittest

from sglang.srt.configs.zaya import ZayaConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestZayaConfig(CustomTestCase):
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


if __name__ == "__main__":
    unittest.main()
