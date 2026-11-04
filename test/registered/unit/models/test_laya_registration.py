"""Unit tests for native registration of the ``LayaForDecision`` architecture.

``convaiinnovations/laya`` is a non-autoregressive decision model: a ModernBERT
encoder plus a small transformer decision head. It must resolve to the native
SGLang implementation and be served as an embedding/pooling model, never fall
back to the Transformers backend or be treated as a generative model.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest

from sglang.srt.configs.model_config import is_generation_model
from sglang.test.test_utils import CustomTestCase


class TestLayaRegistration(CustomTestCase):
    def test_entry_class_is_native_laya(self):
        from sglang.srt.models import laya

        entry = laya.EntryClass
        self.assertEqual(entry.__name__, "LayaForDecision")
        self.assertEqual(entry.__module__, "sglang.srt.models.laya")
        self.assertTrue(hasattr(entry, "forward"))
        self.assertTrue(hasattr(entry, "load_weights"))

    def test_registry_resolves_native_not_transformers_fallback(self):
        from sglang.srt.models.registry import ModelRegistry

        model_cls, resolved_arch = ModelRegistry.resolve_model_cls("LayaForDecision")
        self.assertEqual(resolved_arch, "LayaForDecision")
        self.assertEqual(model_cls.__name__, "LayaForDecision")
        self.assertEqual(model_cls.__module__, "sglang.srt.models.laya")
        self.assertNotIn("Transformers", model_cls.__name__)

    def test_laya_is_never_generative(self):
        """A decision model has no LM head, with or without --is-embedding."""
        self.assertFalse(is_generation_model(["LayaForDecision"]))
        self.assertFalse(is_generation_model(["LayaForDecision"], is_embedding=False))
        self.assertFalse(is_generation_model(["LayaForDecision"], is_embedding=True))

    def test_embedding_spec_is_auto_enabled_and_not_normalized(self):
        from sglang.srt.configs.embedding_model_spec import (
            AttentionPattern,
            EmbeddingExecution,
            EmbeddingTask,
            PoolingStrategy,
            resolve_embedding_model_spec,
        )

        spec = resolve_embedding_model_spec(
            ["LayaForDecision"],
            is_embedding_requested=False,
            is_embedding_gemma=False,
        )
        self.assertEqual(spec.task, EmbeddingTask.EMBED)
        self.assertEqual(spec.execution, EmbeddingExecution.ENCODER_ONLY)
        self.assertEqual(spec.attention, AttentionPattern.BIDIRECTIONAL)
        self.assertEqual(spec.pooling, PoolingStrategy.MODEL_DEFINED)
        self.assertFalse(spec.normalize)
        self.assertTrue(spec.auto_enable_embedding)

    def test_other_architectures_unaffected(self):
        self.assertTrue(is_generation_model(["LlamaForCausalLM"]))
        self.assertFalse(is_generation_model(["BertForSequenceClassification"]))
        self.assertFalse(is_generation_model(["BertModel"]))


if __name__ == "__main__":
    unittest.main()
