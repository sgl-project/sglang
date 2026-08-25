"""Unit tests for native Qwen3.5 sequence-classification support."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.configs.embedding_model_spec import (
    EmbeddingExecution,
    EmbeddingTask,
    resolve_embedding_model_spec,
)
from sglang.srt.configs.model_config import is_generation_model
from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.models.qwen3_5 import Qwen3_5ForSequenceClassification
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestQwen3_5ForSequenceClassification(CustomTestCase):
    def test_registry_resolves_native_class(self):
        from sglang.srt.models.registry import ModelRegistry

        model_cls, resolved_arch = ModelRegistry.resolve_model_cls(
            "Qwen3_5ForSequenceClassification"
        )

        self.assertIs(model_cls, Qwen3_5ForSequenceClassification)
        self.assertEqual(resolved_arch, "Qwen3_5ForSequenceClassification")

    def test_architecture_is_multimodal_classification(self):
        architectures = ["Qwen3_5ForSequenceClassification"]

        self.assertFalse(is_generation_model(architectures))
        spec = resolve_embedding_model_spec(
            architectures,
            is_embedding_requested=True,
            is_embedding_gemma=False,
        )
        self.assertEqual(spec.task, EmbeddingTask.CLASSIFY)
        self.assertEqual(spec.execution, EmbeddingExecution.CLASSIFICATION)
        self.assertFalse(spec.normalize)
        self.assertTrue(spec.supports_multimodal)

    def test_score_head_uses_raw_last_token_hidden_state(self):
        model = Qwen3_5ForSequenceClassification.__new__(
            Qwen3_5ForSequenceClassification
        )
        nn.Module.__init__(model)
        model.score = nn.Linear(2, 1, bias=False)
        model.score.weight.data.copy_(torch.tensor([[2.0, -1.0]]))
        model.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)

        hidden_states = torch.tensor(
            [
                [1.0, 1.0],
                [3.0, 4.0],
                [2.0, 2.0],
                [6.0, 8.0],
            ]
        )
        forward_batch = SimpleNamespace(
            extend_seq_lens=torch.tensor([2, 2]),
            extend_seq_lens_cpu=[2, 2],
            multi_item_delimiter_indices=None,
            return_pooled_hidden_states=True,
            is_prefill_only=True,
            dimensions=None,
        )

        output = model._pool_hidden_states(
            torch.arange(4), hidden_states, forward_batch
        )

        raw_last_tokens = hidden_states[torch.tensor([1, 3])]
        torch.testing.assert_close(output.embeddings, model.score(raw_last_tokens))
        torch.testing.assert_close(output.pooled_hidden_states, raw_last_tokens)

        normalized_scores = model.score(
            nn.functional.normalize(raw_last_tokens, p=2, dim=-1)
        )
        self.assertFalse(torch.allclose(output.embeddings, normalized_scores))


if __name__ == "__main__":
    unittest.main()
