"""CPU regressions for DSpark draft checkpoint construction."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models import dspark
from sglang.srt.models.dspark import DSparkDraftMixin, build_markov_head


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
        self.calls.append(
            {
                "input_size": input_size,
                "output_size": output_size,
                "bias": bias,
                "quant_config": quant_config,
                "prefix": prefix,
            }
        )


class _FakeW4A16HeadMethod:
    def __init__(self, dense_weight):
        self.dense_weight = dense_weight

    def apply(self, _layer, hidden_states, _bias):
        return torch.matmul(hidden_states, self.dense_weight.T)


_FakeW4A16HeadMethod.__name__ = "ModelOptNvFp4A16LinearMethod"


class _LoadWeightsBase(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.forwarded_weights = []

    def load_weights(self, weights):
        self.forwarded_weights.extend(weights)


class _LoadWeightsDraft(DSparkDraftMixin, _LoadWeightsBase):
    def __init__(self, *, has_embed_tokens):
        _LoadWeightsBase.__init__(self)
        self.has_embed_tokens = has_embed_tokens
        self.confidence_head = None
        self.markov_head = nn.Module()


class TestDSparkDraftModel(unittest.TestCase):
    @patch.object(dspark, "ReplicatedLinear", _FakeReplicatedLinear)
    def test_markov_projection_receives_quant_config_and_full_prefix(self):
        _FakeReplicatedLinear.calls.clear()
        quant_config = object()
        config = SimpleNamespace(
            vocab_size=131072,
            hidden_size=2688,
            markov_rank=256,
            markov_head_type="vanilla",
        )

        head = build_markov_head(config, quant_config=quant_config, prefix="draft")

        self.assertIsInstance(head, dspark.VanillaMarkov)
        self.assertEqual(len(_FakeReplicatedLinear.calls), 1)
        call = _FakeReplicatedLinear.calls[0]
        self.assertEqual(call["input_size"], 256)
        self.assertEqual(call["output_size"], 131072)
        self.assertFalse(call["bias"])
        self.assertIs(call["quant_config"], quant_config)
        self.assertEqual(call["prefix"], "draft.markov_head.markov_w2")

    def test_checkpoint_embedding_is_preserved_when_attaching_target_modules(self):
        draft_embedding = object()
        target_embedding = object()
        target_lm_head = object()
        model = SimpleNamespace(
            embed_tokens=draft_embedding,
            get_input_embeddings=lambda: draft_embedding,
        )

        DSparkDraftMixin.attach_shared_modules(
            model, embed_tokens=target_embedding, lm_head=target_lm_head
        )

        self.assertIs(model.embed_tokens, draft_embedding)
        self.assertIs(model.lm_head, target_lm_head)

    def test_target_embedding_is_used_when_checkpoint_omits_one(self):
        target_embedding = object()
        target_lm_head = object()
        model = SimpleNamespace(embed_tokens=None)

        DSparkDraftMixin.attach_shared_modules(
            model, embed_tokens=target_embedding, lm_head=target_lm_head
        )

        self.assertIs(model.embed_tokens, target_embedding)
        self.assertIs(model.lm_head, target_lm_head)

    def test_checkpoint_embedding_requires_matching_config_declaration(self):
        for name in ("embed_tokens.weight", "model.embed_tokens.weight"):
            with self.subTest(name=name):
                model = _LoadWeightsDraft(has_embed_tokens=False)
                with self.assertRaisesRegex(
                    ValueError, "contains embed_tokens weights"
                ):
                    model.load_weights([(name, torch.empty((2, 2)))])

    def test_declared_checkpoint_embedding_is_forwarded_to_backbone_loader(self):
        model = _LoadWeightsDraft(has_embed_tokens=True)
        weight = torch.empty((2, 2))

        model.load_weights([("embed_tokens.weight", weight)])

        self.assertEqual(len(model.forwarded_weights), 1)
        self.assertEqual(model.forwarded_weights[0][0], "embed_tokens.weight")
        self.assertIs(model.forwarded_weights[0][1], weight)

    @patch.object(dspark, "gather_and_crop_vocab", side_effect=lambda logits, _: logits)
    def test_quantized_target_head_uses_quant_method(self, _gather):
        hidden = torch.tensor([[1.0, -2.0, 0.5, 3.0]])
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
        draft = SimpleNamespace(
            lm_head=lm_head,
            logits_mup_width_multiplier=None,
        )

        actual, confidence = DSparkDraftMixin.compute_base_logits(draft, hidden)

        torch.testing.assert_close(actual, hidden @ dense_weight.T)
        self.assertIsNone(confidence)


if __name__ == "__main__":
    unittest.main()
