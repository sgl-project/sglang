"""Checkpoint parity for the Qwen3 sequence-classification head."""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import Qwen3Config

from sglang.srt.models.qwen3_classification import Qwen3ForSequenceClassification
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQwen3ClassificationBias(CustomTestCase):
    def test_checkpoint_bias(self):
        """Missing bias must be neutral; an explicit checkpoint bias must survive."""
        config = Qwen3Config(hidden_size=8, num_labels=3)
        weight = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 10
        hidden_states = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 10
        for bias in (None, torch.tensor([0.1, -0.2, 0.3])):
            with self.subTest(bias=bias):
                # The backbone is unrelated to head initialization and loading.
                with patch(
                    "sglang.srt.models.qwen3_classification.Qwen3Model",
                    return_value=nn.Identity(),
                ):
                    model = Qwen3ForSequenceClassification(config)
                checkpoint = [("score.weight", weight)]
                if bias is not None:
                    checkpoint.append(("score.bias", bias))
                model.load_weights(iter(checkpoint))
                torch.testing.assert_close(
                    model.score(hidden_states),
                    nn.functional.linear(hidden_states, weight, bias),
                    rtol=0,
                    atol=0,
                )


if __name__ == "__main__":
    unittest.main()
