import tempfile
import unittest
from types import SimpleNamespace

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
        full = IQuestQ1MTPConfig(target_config=target, sliding_window=None)
        self.assertEqual(full.layer_types, ["full_attention"])
        self.assertIsNone(full.sliding_window)


if __name__ == "__main__":
    unittest.main()
