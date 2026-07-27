import unittest

from sglang.srt.configs.embedding_model_spec import (
    BCGPrefillPolicy,
    EmbeddingTask,
    PoolingStrategy,
    resolve_embedding_model_spec,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestEmbeddingModelSpec(unittest.TestCase):
    def test_embedding_gemma_declares_full_encoder_bcg_contract(self):
        spec = resolve_embedding_model_spec(
            ["Gemma3TextModel"],
            is_embedding_requested=False,
            is_embedding_gemma=True,
        )

        self.assertEqual(spec.family, "embeddinggemma")
        self.assertEqual(spec.task, EmbeddingTask.EMBED)
        self.assertEqual(spec.pooling, PoolingStrategy.MEAN)
        self.assertTrue(spec.auto_enable_embedding)
        self.assertTrue(spec.safe_disable_kv_cache)
        self.assertEqual(spec.bcg_prefill_policy, BCGPrefillPolicy.FULL_ENCODER)

    def test_encoder_embedding_models_enable_embedding_mode_automatically(self):
        spec = resolve_embedding_model_spec(
            ["BertModel"],
            is_embedding_requested=False,
            is_embedding_gemma=False,
        )

        self.assertEqual(spec.family, "bert")
        self.assertEqual(spec.task, EmbeddingTask.EMBED)
        self.assertEqual(spec.pooling, PoolingStrategy.CLS)
        self.assertFalse(spec.requires_embedding_flag)
        self.assertTrue(spec.auto_enable_embedding)

    def test_decoder_embedding_intent_does_not_assume_encoder_fast_path(self):
        spec = resolve_embedding_model_spec(
            ["Qwen3ForCausalLM"],
            is_embedding_requested=True,
            is_embedding_gemma=False,
        )

        self.assertEqual(spec.family, "explicit_decoder_embedding")
        self.assertEqual(spec.task, EmbeddingTask.EMBED)
        self.assertFalse(spec.safe_disable_kv_cache)
        self.assertEqual(spec.bcg_prefill_policy, BCGPrefillPolicy.DEFAULT)

    def test_unknown_generation_model_has_no_embedding_contract_without_intent(self):
        spec = resolve_embedding_model_spec(
            ["Qwen3ForCausalLM"],
            is_embedding_requested=False,
            is_embedding_gemma=False,
        )

        self.assertEqual(spec.task, EmbeddingTask.NONE)
        self.assertEqual(spec.family, "none")


if __name__ == "__main__":
    unittest.main()
