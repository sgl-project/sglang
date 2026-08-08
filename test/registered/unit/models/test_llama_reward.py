"""Unit tests for Llama sequence-classification pooling and MIS scoring."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang.srt.layers.pooler import Pooler, PoolingType
from sglang.srt.models.llama_reward import LlamaForSequenceClassification
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeLlamaModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, input_ids, positions, forward_batch, input_embeds=None):
        return self.embed_tokens(input_ids) if input_embeds is None else input_embeds


class TestLlamaForSequenceClassification(CustomTestCase):
    def setUp(self):
        hidden_size = 8
        self.model = LlamaForSequenceClassification.__new__(
            LlamaForSequenceClassification
        )
        nn.Module.__init__(self.model)
        self.model.model = _FakeLlamaModel(vocab_size=32, hidden_size=hidden_size)
        self.model.score = nn.Linear(hidden_size, 2, bias=False)
        self.model.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=False)
        with torch.no_grad():
            self.model.score.weight.copy_(
                torch.arange(2 * hidden_size).reshape(2, hidden_size) / hidden_size
            )

    @staticmethod
    def _forward_batch(
        seq_lens,
        *,
        delimiter_indices=None,
        return_pooled_hidden_states=True,
    ):
        return SimpleNamespace(
            extend_seq_lens=torch.tensor(seq_lens),
            extend_seq_lens_cpu=seq_lens,
            multi_item_delimiter_indices=delimiter_indices,
            dimensions=None,
            return_pooled_hidden_states=return_pooled_hidden_states,
            is_prefill_only=True,
        )

    def test_get_input_embeddings_delegates_to_backbone(self):
        self.assertIs(
            self.model.get_input_embeddings(),
            self.model.model.get_input_embeddings(),
        )

    def test_non_mis_forward_matches_last_token_pooling(self):
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        forward_batch = self._forward_batch([2, 3], delimiter_indices=None)

        output = self.model(
            input_ids,
            torch.arange(len(input_ids)),
            forward_batch,
        )

        hidden_states = self.model.model.get_input_embeddings()(input_ids)
        expected_pooled_hidden = hidden_states[torch.tensor([1, 4])]
        expected_scores = self.model.score(expected_pooled_hidden)

        torch.testing.assert_close(output.embeddings, expected_scores)
        torch.testing.assert_close(output.pooled_hidden_states, expected_pooled_hidden)

    def test_mis_embedding_overrides_match_non_mis_scores(self):
        placeholder_id = 0
        delimiter_id = 9
        query_ids = torch.tensor([1, 2])
        doc_embeds = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8) / 8

        # Non-MIS reference: one [query, doc-placeholder] sequence per document.
        non_mis_input_ids = torch.cat(
            [torch.cat([query_ids, torch.tensor([placeholder_id])])] * 3
        )
        query_embeds = self.model.get_input_embeddings()(query_ids)
        non_mis_input_embeds = torch.cat(
            [
                torch.cat([query_embeds, doc_embed.unsqueeze(0)])
                for doc_embed in doc_embeds
            ]
        )
        non_mis_output = self.model(
            non_mis_input_ids,
            torch.arange(len(non_mis_input_ids)),
            self._forward_batch([3, 3, 3], delimiter_indices=None),
            input_embeds=non_mis_input_embeds,
        )

        # MIS: query<D>doc1<D>doc2<D>doc3<D>. The first pooled row belongs
        # to the query delimiter and is removed by score result processing.
        mis_input_ids = torch.tensor(
            [1, 2, delimiter_id, 0, delimiter_id, 0, delimiter_id, 0, delimiter_id]
        )
        delimiter_embed = self.model.get_input_embeddings()(
            torch.tensor([delimiter_id])
        )
        mis_input_embeds = torch.cat(
            [
                query_embeds,
                delimiter_embed,
                doc_embeds[0:1],
                delimiter_embed,
                doc_embeds[1:2],
                delimiter_embed,
                doc_embeds[2:3],
                delimiter_embed,
            ]
        )
        mis_output = self.model(
            mis_input_ids,
            torch.arange(len(mis_input_ids)),
            self._forward_batch(
                [len(mis_input_ids)],
                delimiter_indices=[torch.tensor([2, 4, 6, 8])],
            ),
            input_embeds=mis_input_embeds,
        )

        self.assertIsInstance(mis_output.embeddings, list)
        self.assertEqual(mis_output.embeddings[0].shape, (4, 2))
        self.assertEqual(mis_output.pooled_hidden_states[0].shape, (4, 8))
        torch.testing.assert_close(
            mis_output.embeddings[0][1:], non_mis_output.embeddings
        )
        torch.testing.assert_close(
            mis_output.pooled_hidden_states[0][1:],
            non_mis_output.pooled_hidden_states,
        )
        self.assertTrue(torch.isfinite(mis_output.embeddings[0]).all())
        self.assertEqual(torch.unique(mis_output.embeddings[0][1:], dim=0).shape[0], 3)

    def test_mis_forward_omits_pooled_hidden_states_when_not_requested(self):
        input_ids = torch.tensor([1, 2, 9, 3, 4, 9])
        forward_batch = self._forward_batch(
            [len(input_ids)],
            delimiter_indices=[torch.tensor([2, 5])],
            return_pooled_hidden_states=False,
        )
        output = self.model(
            input_ids,
            torch.arange(len(input_ids)),
            forward_batch,
        )

        self.assertIsInstance(output.embeddings, list)
        self.assertEqual(output.embeddings[0].shape, (2, 2))
        self.assertIsNone(output.pooled_hidden_states)


if __name__ == "__main__":
    unittest.main()
