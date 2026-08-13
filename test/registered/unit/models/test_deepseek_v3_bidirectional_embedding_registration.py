"""Unit tests for the ``DeepseekV3BidirectionalModel`` embedding architecture.

Checkpoints such as ``ai-sage/Giga-Embeddings-instruct-10B-A1.8B-0826`` declare
``architectures=["DeepseekV3BidirectionalModel"]``: a DeepSeek-V3 MoE backbone
served as an encoder-style embedding model (``is_causal=False``). These must
resolve to the native SGLang implementation, be classified as an embedding
model, and -- because the absorbed-MLA kernels are causal-only -- force the
runtime onto the non-absorbed MHA prefill path by disabling CUDA-graph capture.
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


class TestDeepseekV3BidirectionalRegistration(CustomTestCase):
    def test_registry_resolves_native_class(self):
        from sglang.srt.models.registry import ModelRegistry

        model_cls, resolved_arch = ModelRegistry.resolve_model_cls(
            "DeepseekV3BidirectionalModel"
        )
        self.assertEqual(resolved_arch, "DeepseekV3BidirectionalModel")
        self.assertEqual(model_cls.__name__, "DeepseekV3BidirectionalModel")
        self.assertEqual(
            model_cls.__module__, "sglang.srt.models.deepseek_v2_embedding"
        )
        self.assertNotIn("Transformers", model_cls.__name__)
        # Reuses the DeepSeek-V3 weight-loading path.
        self.assertTrue(hasattr(model_cls, "load_weights"))

    def test_classified_as_embedding(self):
        """Non-generative regardless of the --is-embedding flag."""
        self.assertFalse(is_generation_model(["DeepseekV3BidirectionalModel"]))
        self.assertFalse(
            is_generation_model(["DeepseekV3BidirectionalModel"], is_embedding=True)
        )
        # The base causal-LM archs stay generative.
        self.assertTrue(is_generation_model(["DeepseekV3ForCausalLM"]))

    def test_embedding_spec_is_bidirectional_mean(self):
        spec = resolve_embedding_model_spec(
            ["DeepseekV3BidirectionalModel"],
            is_embedding_requested=False,
            is_embedding_gemma=False,
        )
        self.assertEqual(spec.attention, AttentionPattern.BIDIRECTIONAL)
        self.assertEqual(spec.pooling, PoolingStrategy.MEAN)
        self.assertTrue(spec.bidirectional_attention)
        self.assertTrue(spec.auto_enable_embedding)

    def test_capability_adjustment_forces_triton_and_disables_cuda_graph(self):
        """Only the Triton backend honors ENCODER_ONLY on the MLA MHA prefill
        path (flashinfer silently runs causal). The absorbed-MLA kernel is
        causal-only and is what graph capture pins, so serving this arch must
        force triton and disable CUDA-graph capture and the prefix/split-prefill
        reuse that is invalid under bidirectional attention. A regression here
        would silently produce causal embeddings.
        """
        args = ServerArgs(model_path="dummy")
        args.attention_backend = "flashinfer"
        args.model_config = SimpleNamespace(
            is_embedding_gemma=False,
            is_multimodal=False,
            embedding_model_spec=None,
            hf_config=SimpleNamespace(architectures=["DeepseekV3BidirectionalModel"]),
        )
        args.cuda_graph_config = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL),
            prefill=PhaseConfig(backend=Backend.TC_PIECEWISE),
        )
        args.disable_radix_cache = False
        args.disable_cuda_graph = False
        args.chunked_prefill_size = 2048

        with patch.object(args, "get_model_config", return_value=args.model_config):
            args._handle_model_capability_adjustments()

        # The backend override is declared through the resolution stash (it
        # materializes onto the field during full server-arg resolution).
        declared = {}
        for _, fields in args._resolved_overrides or []:
            declared.update(fields)
        self.assertEqual(declared.get("attention_backend"), "triton")
        self.assertTrue(args.disable_cuda_graph)
        self.assertTrue(args.disable_radix_cache)
        self.assertEqual(args.chunked_prefill_size, -1)
        self.assertEqual(args.cuda_graph_config.decode.backend, Backend.DISABLED)
        self.assertEqual(args.cuda_graph_config.prefill.backend, Backend.DISABLED)


if __name__ == "__main__":
    unittest.main()
