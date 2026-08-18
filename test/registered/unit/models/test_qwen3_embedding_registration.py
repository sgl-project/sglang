"""Unit tests for native registration of the bare ``Qwen3Model`` embedding arch.

Checkpoints such as ``microsoft/harrier-oss-v1-0.6b`` declare
``architectures=["Qwen3Model"]`` (a bare Qwen3 backbone). These must resolve to
the native SGLang implementation (``sglang.srt.models.qwen3_embedding.Qwen3Model``)
and be served as an embedding model, NOT fall back to the Transformers backend.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

from sglang.srt.configs.model_config import is_generation_model
from sglang.test.test_utils import CustomTestCase


class TestQwen3ModelEmbeddingRegistration(CustomTestCase):
    def test_entry_class_is_native_qwen3model(self):
        """The bare arch string maps to a native EntryClass named 'Qwen3Model'."""
        from sglang.srt.models import qwen3_embedding

        entry = qwen3_embedding.EntryClass
        self.assertEqual(entry.__name__, "Qwen3Model")
        self.assertEqual(entry.__module__, "sglang.srt.models.qwen3_embedding")
        # It carries a LAST-token / normalized pooler, i.e. an embedding head.
        self.assertTrue(hasattr(entry, "forward"))
        self.assertTrue(hasattr(entry, "load_weights"))

    def test_registry_resolves_native_not_transformers_fallback(self):
        """ModelRegistry resolves 'Qwen3Model' to the native class, not the
        TransformersForCausalLM fallback."""
        from sglang.srt.models.registry import ModelRegistry

        model_cls, resolved_arch = ModelRegistry.resolve_model_cls("Qwen3Model")
        self.assertEqual(resolved_arch, "Qwen3Model")
        self.assertEqual(model_cls.__name__, "Qwen3Model")
        self.assertEqual(model_cls.__module__, "sglang.srt.models.qwen3_embedding")
        self.assertNotIn("Transformers", model_cls.__name__)

    def test_bare_qwen3model_classified_as_embedding(self):
        """'Qwen3Model' is non-generative regardless of the --is-embedding flag."""
        self.assertFalse(is_generation_model(["Qwen3Model"]))
        self.assertFalse(is_generation_model(["Qwen3Model"], is_embedding=False))
        self.assertFalse(is_generation_model(["Qwen3Model"], is_embedding=True))

    def test_existing_qwen3_archs_unaffected(self):
        """The generative / classification archs keep their prior behavior."""
        # Qwen3ForCausalLM is generative by default, embedding only with the flag
        # (this is how Qwen3-Embedding-0.6B is served).
        self.assertTrue(is_generation_model(["Qwen3ForCausalLM"]))
        self.assertTrue(is_generation_model(["Qwen3ForCausalLM"], is_embedding=False))
        self.assertFalse(is_generation_model(["Qwen3ForCausalLM"], is_embedding=True))
        # Sequence-classification / reward archs stay non-generative.
        self.assertFalse(is_generation_model(["Qwen3ForSequenceClassification"]))
        self.assertFalse(is_generation_model(["Qwen3ForRewardModel"]))


if __name__ == "__main__":
    unittest.main()
