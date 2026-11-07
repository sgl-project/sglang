"""Qwen3.5 scoring must preserve checkpoint weights and raw hidden-state scale."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.configs.model_config import (
    is_generation_model,
    is_multimodal_model,
    is_qwen3_5,
    is_score_and_pool_model,
)
from sglang.srt.models.qwen3_5 import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
)
from sglang.srt.models.qwen3_5_classification import Qwen3_5ForSequenceClassification
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _backbone_init(self, config, **kwargs):
    nn.Module.__init__(self)
    self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    self._start_layer = 0
    self._end_layer = 1


def _backbone_forward(self, input_ids, **kwargs):
    return self.embed_tokens(input_ids)


class TestQwen3_5SequenceClassification(CustomTestCase):
    def setUp(self):
        self.boundaries = ExitStack()
        self.addCleanup(self.boundaries.close)
        parallel = SimpleNamespace(
            pp_group=SimpleNamespace(
                is_first_rank=True, is_last_rank=True, world_size=1
            ),
            enable_dp_lm_head=False,
        )
        self.boundaries.enter_context(
            patch("sglang.srt.models.qwen3_vl.get_parallel", return_value=parallel)
        )
        self.boundaries.enter_context(
            patch(
                "sglang.srt.models.qwen3_vl.get_mm",
                return_value=SimpleNamespace(mm_enable_dp_encoder=False),
            )
        )
        self.boundaries.enter_context(
            patch(
                "sglang.srt.models.qwen3_vl.LogitsProcessor", return_value=nn.Identity()
            )
        )
        # Replace only the expensive transformer computation. Construction,
        # checkpoint mapping, forward dispatch, pooling and scoring remain real.
        self.boundaries.enter_context(
            patch.object(Qwen3_5ForCausalLM, "__init__", _backbone_init)
        )
        self.boundaries.enter_context(
            patch.object(Qwen3_5ForCausalLM, "forward", _backbone_forward)
        )
        self.config = SimpleNamespace(
            architectures=["Qwen3_5ForSequenceClassification"],
            language_model_only=True,
            tie_word_embeddings=True,
            num_labels=2,
            text_config=SimpleNamespace(
                hidden_size=2, vocab_size=8, rope_scaling={}, rope_parameters={}
            ),
        )

    @staticmethod
    def _batch(lengths, **overrides):
        return SimpleNamespace(
            **{
                "extend_seq_lens": torch.tensor(lengths),
                "extend_seq_lens_cpu": lengths,
                "multi_item_delimiter_indices": None,
                "token_indices_to_pool": None,
                "return_pooled_hidden_states": True,
                "is_prefill_only": True,
                "dimensions": None,
                **overrides,
            }
        )

    def _model(self, num_labels=2):
        self.config.num_labels = num_labels
        model = Qwen3_5ForSequenceClassification(self.config)
        embeddings = torch.tensor(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [3.0, 4.0],
                [2.0, 2.0],
                [5.0, 1.0],
                [6.0, 8.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        head = torch.tensor([[2.0, -1.0], [-1.0, 3.0]])[:num_labels]
        loaded = model.load_weights(
            [
                ("model.language_model.embed_tokens.weight", embeddings),
                ("score.weight", head),
                ("lm_head.weight", torch.full((8, 2), -99.0)),
            ]
        )
        self.assertEqual(loaded, {"model.embed_tokens.weight", "score.weight"})
        self.assertIsNone(model.lm_head)
        self.assertIsNone(model.score.bias)
        torch.testing.assert_close(model.score.weight, head)
        torch.testing.assert_close(model.model.embed_tokens.weight, embeddings)
        return model

    def test_batched_scores_use_raw_last_token_and_checkpoint_head(self):
        # Unequal request lengths catch selecting a fixed row or the final row
        # of the whole batch; non-unit states catch embedding normalization.
        for num_labels in (1, 2):
            with self.subTest(num_labels=num_labels):
                model = self._model(num_labels)
                output = model(
                    torch.tensor([1, 2, 3, 4, 5]),
                    torch.tensor([0, 1, 0, 1, 2]),
                    self._batch([2, 3]),
                    get_embedding=True,
                )
                torch.testing.assert_close(
                    output.embeddings,
                    torch.tensor([[2.0, 9.0], [4.0, 18.0]])[:, :num_labels],
                )
                torch.testing.assert_close(
                    output.pooled_hidden_states, torch.tensor([[3.0, 4.0], [6.0, 8.0]])
                )

    def test_requested_positions_preserve_per_request_scores(self):
        model = self._model()
        output = model(
            torch.tensor([1, 2, 3, 4, 5]),
            torch.tensor([0, 1, 0, 1, 2]),
            self._batch(
                [2, 3], token_indices_to_pool=[torch.tensor([0, 1]), torch.tensor([1])]
            ),
            get_embedding=True,
        )
        self.assertEqual(len(output.embeddings), 2)
        torch.testing.assert_close(
            output.embeddings[0], torch.tensor([[1.0, 2.0], [2.0, 9.0]])
        )
        torch.testing.assert_close(output.embeddings[1], torch.tensor([[9.0, -2.0]]))

    def test_tied_checkpoint_load_has_no_amx_generation_head(self):
        with (
            patch("sglang.srt.models.qwen3_5._is_cpu", True),
            patch("sglang.srt.models.qwen3_5._is_amx_available", True),
        ):
            self._model()

    def test_generation_class_keeps_embedding_normalization(self):
        # The shared forward hook must preserve the existing embedding output.
        self.config.architectures = ["Qwen3_5ForConditionalGeneration"]
        with (
            patch("sglang.srt.models.qwen3_vl._is_cpu_amx_available", False),
            patch(
                "sglang.srt.models.qwen3_5.get_model",
                return_value=SimpleNamespace(model_path="unused", revision=None),
            ),
            patch(
                "sglang.srt.models.qwen3_5.load_joint_head_config", return_value=None
            ),
        ):
            model = Qwen3_5ForConditionalGeneration(self.config)
        with torch.no_grad():
            model.model.embed_tokens.weight[2].copy_(torch.tensor([3.0, 4.0]))
        output = model(
            torch.tensor([2]), torch.tensor([0]), self._batch([1]), get_embedding=True
        )
        self.assertIs(model.lm_head, model.model.embed_tokens)
        torch.testing.assert_close(output.embeddings, torch.tensor([[0.6, 0.8]]))

    def test_registry_and_serving_capabilities_agree(self):
        from sglang.srt.models.registry import ModelRegistry

        arch = "Qwen3_5ForSequenceClassification"
        cls, resolved = ModelRegistry.resolve_model_cls(arch)
        self.assertIs(cls, Qwen3_5ForSequenceClassification)
        self.assertEqual(resolved, arch)
        self.assertFalse(is_generation_model([arch]))
        self.assertTrue(is_multimodal_model([arch]))
        self.assertTrue(is_qwen3_5(self.config))
        self.assertTrue(is_score_and_pool_model([arch]))


if __name__ == "__main__":
    unittest.main()
