import unittest
from types import SimpleNamespace

from sglang.srt.configs.embedding_model_spec import (
    AttentionPattern,
    BCGEligibility,
    BCGPrefillPolicy,
    EmbeddingExecution,
    EmbeddingTask,
    PoolingStrategy,
    embedding_support_matrix,
    resolve_embedding_model_spec,
    resolved_embedding_plan,
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
        self.assertEqual(spec.execution, EmbeddingExecution.ENCODER_ONLY)
        self.assertEqual(spec.attention, AttentionPattern.BIDIRECTIONAL)
        self.assertEqual(spec.bcg_eligibility, BCGEligibility.FULL_ENCODER)

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

    def test_support_matrix_is_derived_from_the_same_registry(self):
        matrix = embedding_support_matrix()
        by_architecture = {row["architecture"]: row for row in matrix}

        self.assertEqual(len(matrix), 7)
        self.assertEqual(by_architecture["BertModel"]["family"], "bert")
        self.assertEqual(by_architecture["BertModel"]["attention"], "bidirectional")
        self.assertTrue(by_architecture["CLIPModel"]["supports_multimodal"])
        self.assertEqual(
            by_architecture["Gemma3TextModel (use_bidirectional_attention=true)"][
                "bcg_eligibility"
            ],
            "full_encoder",
        )

    def test_resolved_plan_reports_effective_runtime_knobs(self):
        spec = resolve_embedding_model_spec(
            ["Gemma3TextModel"],
            is_embedding_requested=False,
            is_embedding_gemma=True,
        )
        plan = resolved_embedding_plan(
            spec,
            config=SimpleNamespace(
                is_embedding=True,
                cuda_graph_config=SimpleNamespace(
                    prefill=SimpleNamespace(
                        backend="breakable", max_bs=16384, bs=[1024, 16384]
                    )
                ),
                prefill_only_disable_kv_cache=True,
                disable_radix_cache=True,
                chunked_prefill_size=-1,
            ),
            model_config=SimpleNamespace(
                is_matryoshka=False, matryoshka_dimensions=None
            ),
        )

        self.assertTrue(plan["enabled"])
        self.assertTrue(plan["bcg"]["enabled"])
        self.assertEqual(plan["bcg"]["capture_token_budget"], 16384)
        self.assertEqual(plan["bcg"]["capture_batch_sizes"], [1024, 16384])
        self.assertTrue(plan["cache"]["kv_cache_disabled"])
        self.assertTrue(plan["cache"]["radix_cache_disabled"])
        self.assertTrue(plan["cache"]["chunked_prefill_disabled"])

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
