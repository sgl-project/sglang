"""Unit tests for MiMo audio tokenizer attention."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.models import mimo_audio
from sglang.srt.models.mimo_audio import AudioEncoderAttention
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _StubVisionAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.forward_kwargs = None

    def forward(self, hidden_states, **kwargs):
        self.forward_kwargs = kwargs
        return hidden_states.unsqueeze(0)


class TestAudioEncoderAttention(CustomTestCase):
    def test_configures_vision_attention_with_causal_window(self):
        stub = _StubVisionAttention()
        with patch.object(
            mimo_audio, "VisionAttention", return_value=stub
        ) as vision_attention:
            attention = AudioEncoderAttention(
                embed_dim=8,
                num_heads=2,
                window_size=(128, 0),
                causal=True,
            )

        constructor_kwargs = vision_attention.call_args.kwargs
        self.assertTrue(constructor_kwargs["causal"])
        self.assertEqual(constructor_kwargs["window_size"], (128, 0))

        hidden_states = torch.randn(3, attention.embed_dim)
        cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
        output = attention(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=3,
        )

        torch.testing.assert_close(output, hidden_states)
        self.assertIs(stub.forward_kwargs["cu_seqlens"], cu_seqlens)
        self.assertEqual(stub.forward_kwargs["max_seqlen"], 3)


if __name__ == "__main__":
    unittest.main()
