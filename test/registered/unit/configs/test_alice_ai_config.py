"""AliceAI layout, serialization and recurrent-cache geometry."""

import tempfile
import unittest

from sglang.srt.configs.alice_ai import AliceAIConfig
from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_config
from sglang.srt.runtime_context import get_context
from sglang.srt.utils.hf_transformers_utils import get_config, get_rope_config
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestAliceAIConfig(CustomTestCase):
    def test_published_layout(self):
        config = AliceAIConfig()
        self.assertEqual(config.full_attention_layer_ids, list(range(3, 48, 4)))
        self.assertEqual(len(config.linear_layer_ids), 36)
        self.assertEqual(config.block_attn_res_block_size, 4)
        spec, resolved_config = get_linear_attn_config(config)
        self.assertIs(resolved_config, config)
        self.assertTrue(spec.backend_class_name.endswith("KDAAttnBackend"))

    def test_checkpoint_layer_types_take_precedence_over_periodic_layout(self):
        config = AliceAIConfig(
            num_hidden_layers=3,
            layer_types=["full_attention", "linear_attention", "full_attention"],
        )
        self.assertEqual(config.linear_layer_ids, [1])
        self.assertEqual(config.full_attention_layer_ids, [0, 2])

    def test_kda_cache_geometry_is_sharded_per_head(self):
        config = AliceAIConfig()
        with get_context().override_server_args(tp_size=4):
            cache = config.mamba2_cache_params
        self.assertEqual(cache.shape.temporal, (8, 128, 128))
        self.assertEqual(cache.shape.conv, [(3, 3072)])
        self.assertEqual(cache.layers, config.linear_layer_ids)

    def test_invalid_checkpoint_geometry_is_rejected(self):
        cases = [
            {"block_attn_res_block_size": 0},
            {"block_attn_res_block_size": True},
            {"linear_num_value_heads": 64},
            {"linear_value_head_dim": 64},
            {"linear_conv_kernel_dim": 1},
            {"router_score_function": "softmax"},
            {"kda_allow_negative_eigenvalues": True},
            {"num_experts_per_tok": 513},
            {"layer_types": ["full_attention"]},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                AliceAIConfig(**overrides)

    def test_local_checkpoint_loads_through_sglang_config_registry(self):
        config = AliceAIConfig(
            mtp_num_hidden_layers=1,
            auto_map={"AutoConfig": "configuration_alice_ai.AliceAIConfig"},
            architectures=["AliceAIForCausalLM"],
        )
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            loaded = get_config(directory, trust_remote_code=False)
        self.assertIsInstance(loaded, AliceAIConfig)
        self.assertEqual(loaded.layer_types, config.layer_types)
        self.assertEqual(loaded.mtp_num_hidden_layers, 1)
        self.assertEqual(loaded.num_nextn_predict_layers, 1)
        self.assertEqual(get_rope_config(loaded)[0], 1_000_000.0)
        self.assertEqual(loaded.partial_rotary_factor, 0.25)

    def test_rope_parameters_survive_checkpoint_roundtrip(self):
        """Saving a legacy scaling config must not erase it on a v5 reload."""
        cases = [
            (
                {
                    "rope_theta": 2_000_000.0,
                    "rope_scaling": {"type": "linear", "factor": 2.0},
                    "partial_rotary_factor": 0.5,
                },
                {
                    "rope_type": "linear",
                    "factor": 2.0,
                    "rope_theta": 2_000_000.0,
                    "partial_rotary_factor": 0.5,
                },
            ),
            (
                {
                    "rope_scaling": {"rope_type": "linear", "factor": 2.0},
                    "rope_parameters": {
                        "rope_type": "linear",
                        "factor": 3.0,
                        "rope_theta": 3_000_000.0,
                        "partial_rotary_factor": 0.75,
                    },
                },
                {
                    "rope_type": "linear",
                    "factor": 3.0,
                    "rope_theta": 3_000_000.0,
                    "partial_rotary_factor": 0.75,
                },
            ),
            (
                {"rope_parameters": {"rope_theta": 123456.0}},
                {
                    "rope_type": "default",
                    "rope_theta": 123456.0,
                    "partial_rotary_factor": 0.25,
                },
            ),
        ]
        for overrides, expected in cases:
            with self.subTest(overrides=overrides):
                config = AliceAIConfig(**overrides)
                with tempfile.TemporaryDirectory() as directory:
                    config.save_pretrained(directory)
                    loaded = get_config(directory, trust_remote_code=False)
                theta, parameters = get_rope_config(loaded)
                self.assertIsNotNone(parameters)
                self.assertEqual(theta, expected["rope_theta"])
                self.assertEqual(
                    loaded.partial_rotary_factor, expected["partial_rotary_factor"]
                )
                for name, value in expected.items():
                    self.assertEqual(parameters.get(name), value)


if __name__ == "__main__":
    unittest.main()
