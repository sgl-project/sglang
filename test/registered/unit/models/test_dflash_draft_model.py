"""CPU regressions for DFlash draft checkpoint configuration."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models import dflash
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.speculative.dflash_utils import (
    get_dflash_attention_sliding_window_size,
)
from sglang.srt.speculative.dflash_worker_v2 import _get_dflash_embedding_module


class _FakeDraftLayer(nn.Module):
    calls = []

    def __init__(self, **kwargs):
        super().__init__()
        self.calls.append(kwargs)


class _FakeAttention(nn.Module):
    calls = []

    def __init__(self, **kwargs):
        super().__init__()
        self.calls.append(kwargs)


class _FakeMLP(nn.Module):
    calls = []

    def __init__(self, **kwargs):
        super().__init__()
        self.calls.append(kwargs)


class _FakeNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()


class _FakeEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs


class _FakeReplicatedLinear(nn.Module):
    calls = []

    def __init__(
        self,
        input_size,
        output_size,
        *,
        bias,
        quant_config,
        prefix,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.in_features = input_size
        self.calls.append(
            {
                "input_size": input_size,
                "output_size": output_size,
                "bias": bias,
                "quant_config": quant_config,
                "prefix": prefix,
            }
        )


class _MinimalDraftModel(DFlashDraftModel):
    decoder_layer_cls = _FakeDraftLayer


class _MinimalDecoderLayer(dflash.DFlashDecoderLayer):
    attention_cls = _FakeAttention


def _draft_config(*, has_embed_tokens):
    return SimpleNamespace(
        hidden_size=4,
        num_hidden_layers=6,
        rms_norm_eps=1e-6,
        vocab_size=128,
        has_embed_tokens=has_embed_tokens,
        dflash_config={
            "block_size": 8,
            "target_layer_ids": [1, 5, 19, 29, 41, 51],
        },
    )


class TestDFlashDraftModel(unittest.TestCase):
    @patch.object(dflash, "VocabParallelEmbedding", _FakeEmbedding)
    @patch.object(dflash, "RMSNorm", _FakeNorm)
    @patch.object(dflash, "ReplicatedLinear", _FakeReplicatedLinear)
    def test_explicit_target_ids_are_not_validated_against_draft_depth(self):
        _FakeDraftLayer.calls.clear()
        _FakeReplicatedLinear.calls.clear()
        quant_config = object()
        model = _MinimalDraftModel(
            _draft_config(has_embed_tokens=True),
            quant_config=quant_config,
            prefix="draft",
        )

        self.assertEqual(model.num_context_features, 6)
        self.assertEqual(model.fc.in_features, 24)
        self.assertEqual(model.block_size, 8)
        self.assertIs(model.get_input_embeddings(), model.embed_tokens)
        self.assertEqual(model.embed_tokens.kwargs["prefix"], "draft.embed_tokens")
        self.assertEqual(
            [call["prefix"] for call in _FakeDraftLayer.calls],
            [f"draft.layers.{i}" for i in range(6)],
        )
        self.assertEqual(_FakeReplicatedLinear.calls[0]["prefix"], "draft.fc")
        self.assertIs(_FakeReplicatedLinear.calls[0]["quant_config"], quant_config)

    @patch.object(dflash, "RMSNorm", _FakeNorm)
    @patch.object(dflash, "ReplicatedLinear", _FakeReplicatedLinear)
    def test_checkpoint_without_embedding_uses_fallback_contract(self):
        model = _MinimalDraftModel(_draft_config(has_embed_tokens=False))

        self.assertIsNone(model.get_input_embeddings())


class TestDFlashDecoderLayer(unittest.TestCase):
    def test_sliding_window_falls_back_to_published_dflash_config(self):
        config = SimpleNamespace(
            layer_types=["sliding_attention"],
            dflash_config={"swa_window_size": 1024},
        )

        self.assertEqual(get_dflash_attention_sliding_window_size(config), 1023)

    @patch.object(dflash, "DFlashMLP", _FakeMLP)
    @patch.object(dflash, "RMSNorm", _FakeNorm)
    def test_attention_receives_quant_config_and_full_prefix(self):
        _FakeAttention.calls.clear()
        _FakeMLP.calls.clear()
        quant_config = object()

        _MinimalDecoderLayer(
            SimpleNamespace(hidden_size=4, rms_norm_eps=1e-6),
            layer_id=2,
            quant_config=quant_config,
            prefix="draft.layers.2",
        )

        self.assertEqual(_FakeAttention.calls[0]["prefix"], "draft.layers.2.self_attn")
        self.assertIs(_FakeAttention.calls[0]["quant_config"], quant_config)
        self.assertEqual(_FakeMLP.calls[0]["prefix"], "draft.layers.2.mlp")


class TestDFlashEmbeddingSelection(unittest.TestCase):
    def test_draft_embedding_is_preferred(self):
        draft_embedding = object()
        target_embedding = object()
        draft_model = SimpleNamespace(get_input_embeddings=lambda: draft_embedding)
        target_model = SimpleNamespace(get_input_embeddings=lambda: target_embedding)

        self.assertIs(
            _get_dflash_embedding_module(draft_model, target_model), draft_embedding
        )

    def test_target_embedding_is_the_fallback(self):
        target_embedding = object()
        draft_model = SimpleNamespace(get_input_embeddings=lambda: None)
        target_model = SimpleNamespace(get_input_embeddings=lambda: target_embedding)

        self.assertIs(
            _get_dflash_embedding_module(draft_model, target_model), target_embedding
        )


class _FakeW4A16HeadMethod:
    def __init__(self, dense_weight):
        self.dense_weight = dense_weight

    def apply(self, _layer, hidden_states, _bias):
        return torch.matmul(hidden_states, self.dense_weight.T)


_FakeW4A16HeadMethod.__name__ = "ModelOptNvFp4A16LinearMethod"


class TestDFlashGreedyHead(unittest.TestCase):
    def test_quantized_head_uses_quant_method_instead_of_packed_weight(self):
        from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

        hidden_states = torch.tensor([[1.0, -2.0, 0.5, 3.0]])
        dense_weight = torch.tensor(
            [
                [0.5, 1.0, -1.0, 0.0],
                [1.0, 0.0, 1.0, 1.0],
                [-1.0, 0.0, 0.0, -1.0],
            ]
        )
        lm_head = SimpleNamespace(
            weight=torch.zeros((3, 1), dtype=torch.int32),
            quant_method=_FakeW4A16HeadMethod(dense_weight),
            weight_scale=torch.ones(1),
            weight_global_scale=torch.ones(1),
            workspace=torch.empty(1),
            input_size_per_partition=4,
            output_size_per_partition=3,
        )

        actual = DFlashWorkerV2._greedy_sample_from_vocab_parallel_head(
            SimpleNamespace(), hidden_states=hidden_states, lm_head=lm_head
        )
        expected = torch.argmax(hidden_states @ dense_weight.T, dim=-1)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
