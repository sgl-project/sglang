"""Unit tests for the ``Qwen3BidirectionalModel`` embedding architecture.

Checkpoints such as ``ai-sage/Giga-Embeddings-instruct-3B-0826`` declare
``architectures=["Qwen3BidirectionalModel"]``: a Qwen3 backbone served as an
encoder-style embedding model (``is_causal=False``). These must resolve to the
native SGLang implementation, be classified as an embedding model, and use MEAN
pooling (not the LAST pooling of the plain ``Qwen3Model`` embedding arch).
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.configs.embedding_model_spec import (
    AttentionPattern,
    PoolingStrategy,
    resolve_embedding_model_spec,
)
from sglang.srt.configs.model_config import is_generation_model
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.test_utils import CustomTestCase


class TestQwen3BidirectionalRegistration(CustomTestCase):
    def test_entry_class_registered(self):
        from sglang.srt.models import qwen3_embedding

        names = {c.__name__ for c in qwen3_embedding.EntryClass}
        # The bidirectional arch is registered alongside the plain Qwen3Model.
        self.assertIn("Qwen3BidirectionalModel", names)
        self.assertIn("Qwen3Model", names)

    def test_registry_resolves_native_class(self):
        from sglang.srt.models.registry import ModelRegistry

        model_cls, resolved_arch = ModelRegistry.resolve_model_cls(
            "Qwen3BidirectionalModel"
        )
        self.assertEqual(resolved_arch, "Qwen3BidirectionalModel")
        self.assertEqual(model_cls.__name__, "Qwen3BidirectionalModel")
        self.assertEqual(
            model_cls.__module__, "sglang.srt.models.qwen3_embedding"
        )
        self.assertNotIn("Transformers", model_cls.__name__)

    def test_classified_as_embedding(self):
        """Non-generative regardless of the --is-embedding flag."""
        self.assertFalse(is_generation_model(["Qwen3BidirectionalModel"]))
        self.assertFalse(
            is_generation_model(["Qwen3BidirectionalModel"], is_embedding=True)
        )
        # The generative arch keeps its prior behavior.
        self.assertTrue(is_generation_model(["Qwen3ForCausalLM"]))

    def test_uses_mean_pooling_not_last(self):
        """Giga-Embeddings requires mean pooling; last-token pooling (used by
        the plain Qwen3Model embedding arch) would produce wrong embeddings.
        """
        import inspect

        from sglang.srt.models.qwen3_embedding import (
            Qwen3BidirectionalModel,
            Qwen3Model,
        )

        # The bidirectional class overrides __init__ to swap the pooler to MEAN;
        # the plain Qwen3Model keeps LAST pooling.
        self.assertIsNot(Qwen3BidirectionalModel.__init__, Qwen3Model.__init__)
        src = inspect.getsource(Qwen3BidirectionalModel.__init__)
        self.assertIn("PoolingType.MEAN", src)

    def test_embedding_spec_is_bidirectional_mean(self):
        spec = resolve_embedding_model_spec(
            ["Qwen3BidirectionalModel"],
            is_embedding_requested=False,
            is_embedding_gemma=False,
        )
        self.assertEqual(spec.attention, AttentionPattern.BIDIRECTIONAL)
        self.assertEqual(spec.pooling, PoolingStrategy.MEAN)
        self.assertTrue(spec.bidirectional_attention)
        self.assertTrue(spec.auto_enable_embedding)

    def test_capability_adjustment_disables_cuda_graph(self):
        """Regression: the captured prefill CUDA graph corrupts the non-causal
        attention (embeddings came out wrong / batch-size dependent), and
        prefix/split-prefill reuse is invalid under bidirectional attention.
        Serving this arch must disable CUDA graph, radix cache, and chunked
        prefill.
        """
        args = ServerArgs(model_path="dummy")
        args.model_config = SimpleNamespace(
            is_embedding_gemma=False,
            is_multimodal=False,
            embedding_model_spec=None,
            hf_config=SimpleNamespace(architectures=["Qwen3BidirectionalModel"]),
        )
        args.cuda_graph_config = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL),
            prefill=PhaseConfig(backend=Backend.BREAKABLE),
        )
        args.disable_cuda_graph = False
        args.disable_radix_cache = False
        args.chunked_prefill_size = 2048

        with patch.object(
            args, "get_model_config", return_value=args.model_config
        ):
            args._handle_model_capability_adjustments()

        self.assertTrue(args.disable_cuda_graph)
        self.assertTrue(args.disable_radix_cache)
        self.assertEqual(args.chunked_prefill_size, -1)
        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.DISABLED)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)


if __name__ == "__main__":
    unittest.main()
