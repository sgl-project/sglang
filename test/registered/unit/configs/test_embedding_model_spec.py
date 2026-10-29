import copy
import unittest
from types import SimpleNamespace

from transformers import AutoConfig, PretrainedConfig

from sglang.srt.configs.embedding_model_spec import (
    AttentionPattern,
    BCGEligibility,
    BCGPrefillPolicy,
    EmbeddingExecution,
    EmbeddingTask,
    PoolingStrategy,
    RerankInputFormat,
    embedding_support_matrix,
    resolve_embedding_model_spec,
    resolved_embedding_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _NoSerializationConfig(PretrainedConfig):
    def to_dict(self):
        raise AssertionError("Unrelated configs must not be serialized")


class TestEmbeddingModelSpec(unittest.TestCase):
    def test_mistral3_reranker_defaults_require_the_supported_scoring_layout(self):
        config = {
            "architectures": ["Mistral3ForSequenceClassification"],
            "model_type": "mistral3",
            "is_causal": False,
            "pooling": "avg",
            "id2label": {"0": "LABEL_0"},
            "text_config": {
                "model_type": "ministral3",
                "is_causal": False,
                "pooling": "avg",
            },
            "vision_config": {"model_type": "pixtral"},
        }
        spec = resolve_embedding_model_spec(
            config["architectures"],
            is_embedding_requested=False,
            is_embedding_gemma=False,
            hf_config=AutoConfig.for_model(**config),
        )
        self.assertEqual(spec.task, EmbeddingTask.CLASSIFY)
        self.assertEqual(spec.attention, AttentionPattern.BIDIRECTIONAL)
        self.assertTrue(spec.auto_enable_embedding)
        self.assertTrue(spec.safe_disable_radix_cache)
        self.assertTrue(spec.safe_disable_chunked_prefill)
        self.assertFalse(spec.normalize)
        self.assertEqual(
            spec.rerank_input_format, RerankInputFormat.QUERY_DOCUMENT_CHAT
        )

        for path, value in [
            (("architectures",), ["Mistral3ForConditionalGeneration"]),
            (("model_type",), "llama"),
            (("text_config", "model_type"), "mistral"),
            (("vision_config", "model_type"), "clip_vision_model"),
            (("is_causal",), True),
            (("text_config", "is_causal"), True),
            (("pooling",), "last"),
            (("text_config", "pooling"), "last"),
            (("id2label",), {"0": "LABEL_0", "1": "LABEL_1"}),
        ]:
            with self.subTest(field=".".join(path), value=value):
                other = copy.deepcopy(config)
                target = other
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                spec = resolve_embedding_model_spec(
                    other["architectures"],
                    is_embedding_requested=False,
                    is_embedding_gemma=False,
                    hf_config=AutoConfig.for_model(**other),
                )
                self.assertEqual(spec.task, EmbeddingTask.NONE)
                self.assertFalse(spec.auto_enable_embedding)
                self.assertFalse(spec.safe_disable_radix_cache)
                self.assertFalse(spec.safe_disable_chunked_prefill)
                self.assertEqual(spec.rerank_input_format, RerankInputFormat.DEFAULT)

    def test_unrelated_models_ignore_config_without_serializing_it(self):
        for architecture, requested, embedding_gemma in [
            ("Qwen3ForCausalLM", False, False),
            ("Qwen3ForCausalLM", True, False),
            ("BertModel", False, False),
            ("XLMRobertaForSequenceClassification", False, False),
            ("Gemma3TextModel", False, True),
            ("EmbeddingGemma2Model", False, False),
            ("Mistral3ForConditionalGeneration", False, False),
            (None, False, False),
        ]:
            with self.subTest(architecture=architecture, embedding=requested):
                architectures = [architecture] if architecture else None
                expected = resolve_embedding_model_spec(
                    architectures,
                    is_embedding_requested=requested,
                    is_embedding_gemma=embedding_gemma,
                )
                actual = resolve_embedding_model_spec(
                    architectures,
                    is_embedding_requested=requested,
                    is_embedding_gemma=embedding_gemma,
                    hf_config=_NoSerializationConfig(),
                )
                self.assertEqual(actual, expected)
                self.assertEqual(actual.rerank_input_format, RerankInputFormat.DEFAULT)

    def test_mistral3_architecture_alone_does_not_enable_checkpoint_defaults(self):
        spec = resolve_embedding_model_spec(
            ["Mistral3ForSequenceClassification"],
            is_embedding_requested=False,
            is_embedding_gemma=False,
        )
        self.assertEqual(spec.task, EmbeddingTask.NONE)
        self.assertEqual(spec.rerank_input_format, RerankInputFormat.DEFAULT)
        self.assertFalse(spec.auto_enable_embedding)
        self.assertFalse(spec.safe_disable_radix_cache)
        self.assertFalse(spec.safe_disable_chunked_prefill)

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

        self.assertEqual(len(matrix), 8)
        self.assertIn("EmbeddingGemma2Model", by_architecture)
        self.assertTrue(by_architecture["EmbeddingGemma2Model"]["supports_multimodal"])
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

    def test_embedding_gemma2_declares_multimodal_spec(self):
        spec = resolve_embedding_model_spec(
            ["EmbeddingGemma2Model"],
            is_embedding_requested=False,
            is_embedding_gemma=False,
        )

        self.assertEqual(spec.family, "embedding_gemma2")
        self.assertEqual(spec.task, EmbeddingTask.EMBED)
        self.assertEqual(spec.execution, EmbeddingExecution.MULTIMODAL)
        self.assertEqual(spec.pooling, PoolingStrategy.MEAN)
        self.assertTrue(spec.supports_multimodal)
        self.assertFalse(spec.requires_embedding_flag)
        self.assertTrue(spec.auto_enable_embedding)
        self.assertEqual(spec.attention, AttentionPattern.BIDIRECTIONAL)


if __name__ == "__main__":
    unittest.main()
