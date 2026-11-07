"""Unit tests for the Qwen3.5-VL sequence classification entry classes.

The heavy backbone (vision tower, hybrid GDN stack) is replaced by a stub; the
classification __init__, the parent's load_weights and the parent's forward all
run for real, so the checkpoint's `score.weight` must load and reach the logits.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from sglang.srt.models.qwen3_5_classification import (
    Qwen3_5ForSequenceClassification,
    Qwen3_5VLForSequenceClassification,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

HIDDEN_SIZE = 8


class _BackboneStub(nn.Module):
    """Returns preset final hidden states in place of the language model."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_states = None

    def forward(self, input_ids, forward_batch, positions, pp_proxy_tensors):
        return self.hidden_states


def _stub_parent_init(self, config, quant_config=None, prefix=""):
    # Only the state the parent forward/load_weights read; the parent rebinds
    # config to text_config, which is where hidden_size lives.
    nn.Module.__init__(self)
    self.config = SimpleNamespace(hidden_size=HIDDEN_SIZE, tie_word_embeddings=False)
    self.model = _BackboneStub()
    self.language_model_only = True
    self.is_mrope_enabled = False
    self.capture_aux_hidden_states = False
    self.pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)


def _forward_batch(extend_seq_lens):
    return SimpleNamespace(
        extend_seq_lens=torch.tensor(extend_seq_lens, dtype=torch.long),
        extend_seq_lens_cpu=extend_seq_lens,
        multi_item_delimiter_indices=None,
        token_indices_to_pool=None,
        dimensions=None,
        return_pooled_hidden_states=False,
        is_prefill_only=True,
    )


class TestQwen3_5SequenceClassification(CustomTestCase):
    def test_checkpoint_score_weight_produces_last_token_logits(self):
        """`score.weight` from the checkpoint loads and scores the last token of
        each request; the head has len(id2label) rows, as llama.cpp converts it."""
        torch.manual_seed(0)
        id2label = {0: "a", 1: "b", 2: "c"}
        score_weight = torch.randn(len(id2label), HIDDEN_SIZE)
        seq_lens = [3, 2]
        hidden_states = torch.randn(sum(seq_lens), HIDDEN_SIZE)
        expected = hidden_states[[2, 4]] @ score_weight.T

        for cls in (
            Qwen3_5VLForSequenceClassification,
            Qwen3_5ForSequenceClassification,
        ):
            with self.subTest(cls=cls.__name__), patch.object(
                Qwen3_5ForConditionalGeneration, "__init__", _stub_parent_init
            ):
                model = cls(SimpleNamespace(id2label=id2label))
                loaded = model.load_weights([("score.weight", score_weight)])
                self.assertIn("score.weight", loaded)

                model.model.hidden_states = hidden_states
                out = model(
                    input_ids=torch.zeros(sum(seq_lens), dtype=torch.long),
                    positions=torch.arange(sum(seq_lens)),
                    forward_batch=_forward_batch(seq_lens),
                    get_embedding=True,
                )
                torch.testing.assert_close(out.embeddings, expected)


if __name__ == "__main__":
    unittest.main()
