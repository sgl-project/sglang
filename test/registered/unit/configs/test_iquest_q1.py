import tempfile
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.configs.iquest_q1 import IQuestQ1Config, IQuestQ1MTPConfig
from sglang.srt.configs.model_config import ModelConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestIQuestQ1Config(CustomTestCase):
    def test_checkpoint_names_resolve_without_remote_code(self):
        with tempfile.TemporaryDirectory() as directory:
            IQuestQ1Config().save_pretrained(directory)
            config = ModelConfig(directory)
        self.assertIsInstance(config.hf_config, IQuestQ1Config)
        self.assertEqual(config.hf_config.model_type, "iquest_q1")
        self.assertEqual(config.hf_config.architectures, ["IQuestQ1ForCausalLM"])
        self.assertEqual(len(config.hf_config.layer_types), 88)
        self.assertEqual(config.hf_config.hybrid_layer_pattern.count(1), 63)

    def test_mtp_config_keeps_independent_sliding_attention(self):
        from sglang.srt.model_executor.model_runner_components.layer_setup import (
            _compute_model_num_layers,
        )

        target = IQuestQ1Config(hidden_size=64, vocab_size=128).to_dict()
        original = dict(target)
        config = IQuestQ1MTPConfig.from_dict(
            {
                "model_type": "iquest_q1_mtp",
                "architectures": ["IQuestQ1MTP"],
                "target_config": target,
                "num_target_layers": 88,
                "sliding_window": 512,
                "swa_rope_theta": 10000.0,
                "fp32_residual_connection": True,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            draft = ModelConfig(directory, is_draft_model=True)
        self.assertIsInstance(draft.hf_config, IQuestQ1MTPConfig)
        self.assertEqual(draft.hf_config.model_type, "iquest_q1_mtp")
        self.assertEqual(draft.hf_config.architectures, ["IQuestQ1MTP"])
        self.assertEqual(draft.hf_config.hidden_size, 64)
        self.assertEqual(draft.hf_config.vocab_size, 128)
        self.assertEqual(draft.num_hidden_layers, 1)
        self.assertIsNone(draft.num_nextn_predict_layers)
        self.assertNotIn("num_nextn_predict_layers", config.to_dict())
        self.assertEqual(
            _compute_model_num_layers(
                model=SimpleNamespace(), model_config=draft, is_draft_worker=True
            ),
            1,
        )
        self.assertEqual(draft.swa_attention_layer_ids, [0])
        self.assertEqual(draft.full_attention_layer_ids, [])
        self.assertEqual(draft.sliding_window_size, 512)
        self.assertEqual(draft.hf_config.swa_rope_theta, 10000.0)
        self.assertTrue(draft.hf_config.fp32_residual_connection)
        self.assertEqual(draft.hf_config.num_draft_slots, 7)
        self.assertEqual(draft.hf_config.num_target_layers, 88)
        self.assertTrue(draft.hf_config.enable_lm_head_fp32)
        self.assertEqual(target, original)
        self.assertEqual(target["num_hidden_layers"], 88)
        self.assertEqual(target["sliding_window"], 4096)
        full = IQuestQ1MTPConfig(target_config=target)
        self.assertEqual(full.layer_types, ["full_attention"])
        self.assertIsNone(full.sliding_window)

    def test_mtp_config_rejects_other_draft_graphs(self):
        for kwargs in (
            {"num_hidden_layers": 2},
            {"num_draft_slots": 0},
            {"num_target_layers": 0},
            {"num_target_layers": 87, "target_config": {"num_hidden_layers": 88}},
            {"sliding_window": 512},
            {"sliding_window": -1, "swa_rope_theta": 10000.0},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                IQuestQ1MTPConfig(**kwargs)

    def test_checkpoint_defaults_and_roundtrip(self):
        config = IQuestQ1Config()
        expected = {
            "vocab_size": 160000,
            "hidden_size": 3072,
            "intermediate_size": 1536,
            "dense_intermediate_size": 12288,
            "max_position_embeddings": 524288,
            "num_hidden_layers": 88,
            "num_attention_heads": 48,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_router_dtype": "fp32",
            "num_mtp_layers": 0,
            "eos_token_id": 0,
            "pad_token_id": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            restored = IQuestQ1Config.from_pretrained(directory)
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(getattr(config, key), value)
                self.assertEqual(getattr(restored, key), value)
        self.assertEqual(restored.layer_types, config.layer_types)
        self.assertEqual(restored.rope_parameters["partial_rotary_factor"], 0.25)
        self.assertNotIn("num_nextn_predict_layers", config.to_dict())
        self.assertNotIn("num_nextn_predict_layers", restored.to_dict())
        self.assertEqual(restored.dtype, torch.bfloat16)

    def test_full_attention_configuration(self):
        for kwargs in (
            {"use_hybrid_layers": False, "num_hidden_layers": 3},
            {"use_sliding_window": False},
        ):
            config = IQuestQ1Config(**kwargs)
            self.assertEqual(
                config.layer_types, ["full_attention"] * config.num_hidden_layers
            )
            self.assertFalse(config.is_hybrid_swa)

    def test_rope_scaling_survives_checkpoint_roundtrip(self):
        for key, scaling in (
            ("rope_parameters", {"rope_type": "linear", "factor": 2.0}),
            ("rope_scaling", {"type": "linear", "factor": 2.0}),
        ):
            with self.subTest(key=key):
                config = IQuestQ1Config(**{key: scaling})
                restored = IQuestQ1Config.from_dict(config.to_dict())
                self.assertEqual(restored.rope_parameters["rope_type"], "linear")
                self.assertEqual(restored.rope_parameters["factor"], 2.0)
                self.assertEqual(
                    restored.rope_parameters["partial_rotary_factor"], 0.25
                )

    def test_invalid_layer_types_and_rotary_dimensions(self):
        with self.assertRaisesRegex(ValueError, "full_attention"):
            IQuestQ1Config(first_layers_types=["invalid"])
        for rotary_dim in (0, 3, 130):
            with self.subTest(rotary_dim=rotary_dim):
                with self.assertRaisesRegex(ValueError, "rotary_dim"):
                    IQuestQ1Config(rotary_dim=rotary_dim)

    def test_layer_types_expand_to_the_hybrid_schedule(self):
        config = IQuestQ1Config(dtype="bfloat16")
        self.assertEqual(len(config.layer_types), config.num_hidden_layers)
        self.assertEqual(config.layer_types[0], "full_attention")
        self.assertEqual(config.layer_types[-3:], ["full_attention"] * 3)
        self.assertEqual(sum(config.hybrid_layer_pattern), 63)
        self.assertTrue(config.is_hybrid_swa)

    def test_layer_pattern_length_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "layer type pattern"):
            IQuestQ1Config(dtype="bfloat16", num_hybrid_layers_block=20)

    def test_mtp_config_loads_flat_release_configuration(self):
        config = IQuestQ1MTPConfig.from_dict(
            {
                "model_type": "iquest_q1_mtp",
                "architectures": ["IQuestQ1MTP"],
                "hidden_size": 64,
                "vocab_size": 128,
                "sliding_window": 512,
                "swa_rope_theta": 10000.0,
            }
        )
        self.assertIsInstance(config, IQuestQ1MTPConfig)
        self.assertEqual(config.hidden_size, 64)
        self.assertEqual(config.layer_types, ["sliding_attention"])

    def test_router_requires_fp32(self):
        with self.assertRaisesRegex(ValueError, "moe_router_dtype"):
            IQuestQ1Config(moe_router_dtype="bf16")


if __name__ == "__main__":
    unittest.main()
