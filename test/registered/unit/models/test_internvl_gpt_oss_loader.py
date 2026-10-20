"""Regression tests for InternVL weight loading with a GPT-OSS backbone."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from sglang.srt.models.internvl import InternVLChatModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(4, 2)
        self.loaded_names = []

    def load_weights(self, weights):
        for name, _ in weights:
            self.loaded_names.append(name)


class _LazyLanguageModel(_LanguageModel):
    def load_weights(self, weights):
        self.loaded_names.append(next(iter(weights))[0])


def _make_model(language_model):
    model = InternVLChatModel.__new__(InternVLChatModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        llm_config=SimpleNamespace(architectures=["GptOssForCausalLM"]),
        vision_config=SimpleNamespace(num_dummy_heads=0),
    )
    model.language_model = language_model
    model.vision_model = nn.Module()
    model.vision_model.norm = nn.LayerNorm(2)
    model.mlp1 = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, 2))
    return model


language_names = [
    "language_model.model.embed_tokens.weight",
    "language_model.model.layers.0.self_attn.q_proj.weight",
    "language_model.model.layers.0.mlp.experts.gate_up_proj",
    "language_model.lm_head.weight",
]


def _weights():
    return [(name, torch.zeros(1)) for name in language_names] + [
        ("vision_model.norm.weight", torch.full((2,), 2.0)),
        ("mlp1.1.bias", torch.full((2,), 3.0)),
    ]


class TestInternVLGptOssWeightLoading(CustomTestCase):
    def test_gpt_oss_backbone_weights_go_to_its_own_loader(self):
        """Language weights must reach GptOssForCausalLM.load_weights unchanged; vision and projector weights load here."""
        model = _make_model(_LanguageModel())

        model.load_weights(iter(_weights()))

        self.assertEqual(
            model.language_model.loaded_names,
            [name[len("language_model.") :] for name in language_names],
        )
        torch.testing.assert_close(
            model.vision_model.norm.weight, torch.full((2,), 2.0)
        )
        torch.testing.assert_close(model.mlp1[1].bias, torch.full((2,), 3.0))

    def test_partially_consumed_language_weights_fail_loudly(self):
        """A language loader that stops early must raise, not silently drop weights."""
        model = _make_model(_LazyLanguageModel())

        with self.assertRaises(RuntimeError):
            model.load_weights(iter(_weights()))


if __name__ == "__main__":
    unittest.main()
