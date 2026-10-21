"""Regression tests for InternVL weight loading with a GPT-OSS backbone."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from sglang.srt.models.internvl import InternVLChatModel
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from torch import nn

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_model():
    model = InternVLChatModel.__new__(InternVLChatModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        llm_config=SimpleNamespace(architectures=["GptOssForCausalLM"]),
        vision_config=SimpleNamespace(num_dummy_heads=0),
    )
    model.language_model = Mock()
    model.vision_model = nn.Module()
    model.vision_model.norm = nn.LayerNorm(2)
    model.mlp1 = nn.Sequential(nn.LayerNorm(2), nn.Linear(2, 2))
    return model


class TestInternVLGptOssWeightLoading(CustomTestCase):
    def test_gpt_oss_backbone_weights_go_to_its_own_loader(self):
        model = _make_model()
        weights = [
            ("language_model.model.embed_tokens.weight", torch.zeros(1)),
            ("vision_model.norm.weight", torch.full((2,), 2.0)),
            ("mlp1.1.bias", torch.full((2,), 3.0)),
        ]

        model.load_weights(iter(weights))

        model.language_model.load_weights.assert_called_once()
        self.assertEqual(
            list(model.language_model.load_weights.call_args.args[0]),
            [("model.embed_tokens.weight", weights[0][1])],
        )
        torch.testing.assert_close(
            model.vision_model.norm.weight, torch.full((2,), 2.0)
        )
        torch.testing.assert_close(model.mlp1[1].bias, torch.full((2,), 3.0))


if __name__ == "__main__":
    unittest.main()
