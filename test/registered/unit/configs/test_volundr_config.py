"""Unit tests for srt/configs/volundr.py: layer map, KDA state layout, backend routing."""

import unittest
from types import SimpleNamespace

from sglang.srt.configs.linear_attn_model_registry import (
    get_linear_attn_config,
    get_linear_attn_spec_by_arch,
)
from sglang.srt.configs.model_config import uses_kda_attention
from sglang.srt.configs.volundr import VolundrConfig, VolundrTextConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

MIXERS = ["kda", "kda", "bcsa", "kda", "kda", "attn"]


def _text_config(**kw):
    return VolundrTextConfig(
        mixer_types=MIXERS,
        num_hidden_layers=len(MIXERS),
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        linear_num_key_heads=2,
        linear_num_value_heads=6,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        **kw,
    )


class TestVolundrConfig(CustomTestCase):
    def test_layer_map_from_mixer_types(self):
        c = _text_config()
        # full_attention_interval=4 would put attention at layer 3 only.
        self.assertEqual(c.linear_layer_ids, [0, 1, 3, 4])
        self.assertEqual(c.full_attention_layer_ids, [2, 5])
        self.assertEqual(
            c.layer_types,
            ["linear_attention"] * 2
            + ["full_attention"]
            + ["linear_attention"] * 2
            + ["full_attention"],
        )

    def test_mixer_types_length_checked(self):
        with self.assertRaises(ValueError):
            VolundrTextConfig(mixer_types=["kda"], num_hidden_layers=2)

    def test_defaults(self):
        c = _text_config()
        self.assertEqual((c.bcsa_window, c.bcsa_compress, c.bcsa_topk), (4096, 4, 512))
        self.assertEqual((c.bcsa_index_heads, c.bcsa_index_head_dim), (4, 128))
        self.assertEqual(c.engram_orders, [2, 3])

    def test_routed_to_kda_backend(self):
        spec = get_linear_attn_spec_by_arch("VolundrForCausalLM")
        self.assertIsNotNone(spec)
        self.assertTrue(spec.backend_class_name.endswith(".KDAAttnBackend"))
        full = VolundrConfig(text_config=_text_config().to_dict())
        full.architectures = ["VolundrForConditionalGeneration"]
        matched = get_linear_attn_config(full)
        self.assertIsNotNone(matched)
        self.assertIsInstance(matched[1], VolundrTextConfig)
        self.assertTrue(uses_kda_attention(full))

    def test_hybrid_gdn_config_defers_to_registry(self):
        from sglang.srt.configs.hybrid_arch import hybrid_gdn_config

        c = _text_config()
        model_config = SimpleNamespace(
            hf_config=c, linear_attn_registry_result=get_linear_attn_config(c)
        )
        self.assertIsNone(hybrid_gdn_config(model_config))


if __name__ == "__main__":
    unittest.main()
