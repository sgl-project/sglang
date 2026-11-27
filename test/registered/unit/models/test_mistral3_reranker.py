import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers import Mistral3Config

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.models.mistral import Mistral3ForConditionalGeneration
from sglang.srt.models.mistral3_reranker import (
    Ministral3ForSequenceClassification,
    Mistral3ForSequenceClassification,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FixedBackbone(nn.Module):
    def __init__(self, hidden_states):
        super().__init__()
        self.hidden_states = hidden_states

    def forward(self, *args, **kwargs):
        return self.hidden_states


class TestMistral3Reranker(CustomTestCase):
    def test_packed_mean_scores_do_not_lose_a_short_request_after_a_large_sum(self):
        model = Ministral3ForSequenceClassification.__new__(
            Ministral3ForSequenceClassification
        )
        nn.Module.__init__(model)
        model.is_last_rank = True
        model.pooler = Pooler(pooling_type=PoolingType.MEAN, normalize=False)
        model.score = nn.Linear(2, 1, bias=False, dtype=torch.float32)
        model.score.weight.data.copy_(torch.tensor([[-0.25, 0.5]]))
        short = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16)
        hidden = torch.cat((torch.tensor([[1000.0, -1000.0]]).repeat(64, 1), short))
        model.model = _FixedBackbone(hidden.to(torch.bfloat16))
        batch = SimpleNamespace(
            extend_seq_lens=torch.tensor([64, 2]),
            dimensions=None,
            return_pooled_hidden_states=True,
        )
        result = model(None, None, batch)
        torch.testing.assert_close(result.embeddings, torch.tensor([[-750.0], [1.0]]))
        torch.testing.assert_close(
            result.pooled_hidden_states,
            torch.tensor([[1000, -1000], [2, 3]], dtype=torch.bfloat16),
        )
        model.model = _FixedBackbone(short)
        batch.extend_seq_lens = torch.tensor([2])
        single = model(None, None, batch)
        torch.testing.assert_close(single.embeddings, result.embeddings[1:])

    def test_causal_checkpoint_is_rejected_before_allocating_the_backbone(self):
        config = Mistral3Config(
            architectures=["Mistral3ForSequenceClassification"],
            text_config={
                "model_type": "ministral3",
                "is_causal": True,
                "pooling": "avg",
            },
            vision_config={"model_type": "pixtral"},
            is_causal=False,
            pooling="avg",
            num_labels=1,
        )
        with patch.object(
            Mistral3ForConditionalGeneration,
            "__init__",
            side_effect=AssertionError("Backbone allocation started"),
        ):
            with self.assertRaisesRegex(ValueError, "bidirectional attention"):
                Mistral3ForSequenceClassification(config)
            config.text_config.is_causal = False
            with self.assertRaisesRegex(AssertionError, "Backbone allocation started"):
                Mistral3ForSequenceClassification(config)

    def test_canonical_classifier_weight_is_loaded_without_losing_fp32_scoring(self):
        text = Ministral3ForSequenceClassification.__new__(
            Ministral3ForSequenceClassification
        )
        nn.Module.__init__(text)
        text.config = SimpleNamespace(tie_word_embeddings=False)
        text.model = nn.Module()
        text.score = nn.Linear(2, 1, bias=False, dtype=torch.float32)
        text.score.weight.data.fill_(float("nan"))
        model = Mistral3ForSequenceClassification.__new__(
            Mistral3ForSequenceClassification
        )
        nn.Module.__init__(model)
        model.vision_feature_select_strategy = "full"
        model.language_model = text
        weight = torch.tensor([[0.25, -0.5]], dtype=torch.bfloat16)
        model.load_weights([("score.weight", weight)])
        torch.testing.assert_close(text.score.weight, weight.float())

    def test_scoring_rejects_generation_instead_of_returning_hidden_states(self):
        model = Ministral3ForSequenceClassification.__new__(
            Ministral3ForSequenceClassification
        )
        nn.Module.__init__(model)
        with self.assertRaisesRegex(ValueError, "embedding mode"):
            model(None, None, None, get_embedding=False)


if __name__ == "__main__":
    unittest.main()
