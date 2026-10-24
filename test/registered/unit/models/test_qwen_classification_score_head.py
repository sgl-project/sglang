"""Qwen2/Qwen3 sequence-classification score head matches HF: no bias."""

import unittest
from unittest.mock import patch

import torch
from transformers import Qwen2Config

from sglang.srt.models import qwen2_classification, qwen3_classification
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_CASES = (
    (qwen2_classification, "Qwen2ForSequenceClassification", "Qwen2Model"),
    (qwen3_classification, "Qwen3ForSequenceClassification", "Qwen3Model"),
)


def _build(module, cls_name, backbone_name):
    config = Qwen2Config(hidden_size=8, num_labels=2)
    # The backbone is irrelevant here and needs distributed state to build.
    with patch.object(module, backbone_name):
        return getattr(module, cls_name)(config)


class TestQwenClassificationScoreHead(CustomTestCase):
    def test_score_head_has_no_bias(self):
        for case in _CASES:
            with self.subTest(cls=case[1]):
                self.assertIsNone(_build(*case).score.bias)

    def test_checkpoint_score_bias_is_rejected(self):
        for case in _CASES:
            with self.subTest(cls=case[1]):
                model = _build(*case)
                with self.assertRaisesRegex(ValueError, "score.bias"):
                    model.load_weights([("score.bias", torch.zeros(2))])


if __name__ == "__main__":
    unittest.main()
