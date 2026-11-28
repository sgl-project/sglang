"""CPU regression coverage for bounded model-weight inspection."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models.embedding_gemma2 import EmbeddingGemma2Model
from sglang.srt.models.exaone4 import Exaone4ForCausalLM
from sglang.srt.models.granite import GraniteForCausalLM
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.models.opt import OPTForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CAUSAL_MODELS = (
    LlamaForCausalLM,
    GraniteForCausalLM,
    OPTForCausalLM,
    Exaone4ForCausalLM,
)


def _model_with_weight(model_class, weight, *, tied=False):
    # Weight inspection only needs parameters and config, not inference layers.
    model = model_class.__new__(model_class)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(tie_word_embeddings=tied)
    model.stacked_params_mapping = []
    if tied:
        model.model = nn.Module()
        model.model.embed_tokens = nn.Module()
        model.model.embed_tokens.register_parameter(
            "weight", nn.Parameter(weight, requires_grad=False)
        )
    else:
        model.register_parameter("weight", nn.Parameter(weight, requires_grad=False))
    return model


class TestGetWeightsByName(CustomTestCase):
    def _read_with_transfer_sizes(self, model, name, truncate_size):
        transferred = []
        original_cpu = torch.Tensor.cpu

        def record_cpu(tensor, *args, **kwargs):
            transferred.append(tensor.numel())
            return original_cpu(tensor, *args, **kwargs)

        with patch.object(torch.Tensor, "cpu", record_cpu):
            result = model.get_weights_by_name(name, truncate_size=truncate_size)
        return result, transferred

    def test_truncation_bounds_materialization_and_preserves_rows(self):
        """Reading a few rows must not transfer/serialize the entire parameter."""
        for model_class in (*CAUSAL_MODELS, EmbeddingGemma2Model):
            for shape in ((12,), (12, 3)):
                weight = torch.arange(12 if len(shape) == 1 else 36).reshape(shape)
                weight = weight.to(torch.bfloat16)
                model = _model_with_weight(model_class, weight)
                for truncate_size in (0, 2, 100, -1):
                    with self.subTest(
                        model=model_class.__name__, shape=shape, limit=truncate_size
                    ):
                        result, transferred = self._read_with_transfer_sizes(
                            model, "weight", truncate_size
                        )
                        expected = weight[:truncate_size].float()
                        self.assertEqual(result, expected.tolist())
                        self.assertTrue(transferred)
                        self.assertLessEqual(max(transferred), expected.numel())

    def test_tied_lm_head_truncates_before_materialization(self):
        """The tied embedding shortcut must honor the same bounded read."""
        weight = torch.arange(36, dtype=torch.bfloat16).reshape(12, 3)
        for model_class in CAUSAL_MODELS:
            with self.subTest(model=model_class.__name__):
                model = _model_with_weight(model_class, weight, tied=True)
                result, transferred = self._read_with_transfer_sizes(
                    model, "lm_head.weight", 2
                )
                self.assertEqual(result, weight[:2].float().tolist())
                self.assertTrue(transferred)
                self.assertLessEqual(max(transferred), 6)

    def test_packed_projection_is_selected_before_truncation(self):
        """A read limit applies to the requested shard, not the packed parameter."""
        weight = torch.arange(54, dtype=torch.bfloat16).reshape(18, 3)
        for model_class in CAUSAL_MODELS:
            with self.subTest(model=model_class.__name__):
                model = _model_with_weight(model_class, weight)
                model.config.num_attention_heads = 2
                model.config.num_key_value_heads = 2
                model.config.hidden_size = 6
                model.stacked_params_mapping = [("weight", "k_proj", "k")]
                result, transferred = self._read_with_transfer_sizes(model, "k_proj", 2)
                self.assertEqual(result, weight[6:8].float().tolist())
                self.assertLessEqual(max(transferred), 6)

    def test_row_parallel_read_preserves_all_columns(self):
        """Truncation limits rows while TP shards must still join along columns."""
        weight = torch.arange(36, dtype=torch.bfloat16).reshape(12, 3)
        for model_class in CAUSAL_MODELS:
            with self.subTest(model=model_class.__name__):
                model = _model_with_weight(model_class, weight)
                parameter = model.weight
                del model.weight
                model.down_proj = nn.Module()
                model.down_proj.register_parameter("weight", parameter)
                gathered_sizes = []

                def gather(output_tensors, local_weight):
                    gathered_sizes.append(local_weight.numel())
                    for rank, output in enumerate(output_tensors):
                        output.copy_(local_weight + rank * 100)

                with patch("torch.distributed.all_gather", side_effect=gather):
                    result = model.get_weights_by_name(
                        "down_proj.weight", truncate_size=2, tp_size=2
                    )
                expected = torch.cat((weight[:2], weight[:2] + 100), dim=1)
                self.assertEqual(result, expected.float().tolist())
                self.assertEqual(gathered_sizes, [6])


if __name__ == "__main__":
    unittest.main()
