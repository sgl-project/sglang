"""CPU checks for OPF checkpoint layout, routing precision, and token readout."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.configs.embedding_model_spec import resolve_embedding_model_spec
from sglang.srt.entrypoints.openai.serving_classify import OpenAIServingClassify
from sglang.srt.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from sglang.srt.models.openai_privacy_filter import (
    OpenAIPrivacyFilterForTokenClassification,
    _FP32Router,
    _interleave_gate_up,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOpenAIPrivacyFilter(unittest.TestCase):
    def test_vector_endpoints_reject_token_classification(self):
        spec = resolve_embedding_model_spec(
            ["OpenAIPrivacyFilterForTokenClassification"],
            is_embedding_requested=False,
            is_embedding_gemma=False,
        )
        serving = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(
                model_config=SimpleNamespace(embedding_model_spec=spec)
            )
        )
        for endpoint in (OpenAIServingClassify, OpenAIServingEmbedding):
            self.assertIn("/encode", endpoint._validate_request(serving, None))

    def test_expert_conversion_preserves_gate_and_up_projections(self):
        generator = torch.Generator().manual_seed(7)
        x = torch.randn(5, 8, generator=generator)
        weight = torch.randn(3, 8, 12, generator=generator)
        bias = torch.randn(3, 12, generator=generator)
        down = torch.randn(3, 6, 8, generator=generator)
        converted = dict(
            _interleave_gate_up(
                [
                    ("experts.gate_up_proj", weight),
                    ("experts.gate_up_proj_bias", bias),
                    ("experts.down_proj", down),
                ]
            )
        )
        source = x @ weight + bias[:, None, :]
        actual = (
            x @ converted["experts.gate_up_proj"]
            + converted["experts.gate_up_proj_bias"][:, None, :]
        )
        torch.testing.assert_close(actual[..., ::2], source[..., :6])
        torch.testing.assert_close(actual[..., 1::2], source[..., 6:])
        self.assertIs(converted["experts.down_proj"], down)

    def test_router_preserves_close_expert_scores_in_fp32(self):
        router = _FP32Router(2, 2, dtype=torch.float32)
        with torch.no_grad():
            router.weight.copy_(torch.tensor([[1.0, 0.0], [1.001, 0.0]]))
            router.bias.zero_()
        logits, _ = router(torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16))
        self.assertEqual(logits.dtype, torch.float32)
        self.assertEqual(logits.argmax(-1).item(), 1)

    def test_token_readout_keeps_bias_ragged_lengths_and_ignores_padding(self):
        model = OpenAIPrivacyFilterForTokenClassification.__new__(
            OpenAIPrivacyFilterForTokenClassification
        )
        nn.Module.__init__(model)

        class Backbone(nn.Module):
            def forward(self, *args):
                return torch.arange(24, dtype=torch.float32).reshape(6, 4)

        model.model = Backbone()
        model.score = nn.Linear(4, 3)
        with torch.no_grad():
            model.score.weight.zero_()
            model.score.bias.copy_(torch.tensor([2.0, -3.0, 1.0]))
        batch = SimpleNamespace(extend_seq_lens_cpu=[2, 3])
        result = model(None, None, batch).embeddings
        self.assertEqual([tuple(x.shape) for x in result], [(2, 3), (3, 3)])
        for logits in result:
            torch.testing.assert_close(logits, model.score.bias.expand_as(logits))


if __name__ == "__main__":
    unittest.main()
