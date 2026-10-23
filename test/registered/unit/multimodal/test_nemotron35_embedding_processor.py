"""Unit tests for the Nemotron-3.5-Embed multimodal processor."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.multimodal.processors.nemotron35_embedding import (
    Nemotron35EmbeddingProcessor,
)
from sglang.srt.multimodal.processors.pixtral import PixtralProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestNemotron35EmbeddingProcessor(CustomTestCase):
    def test_tokenizer_padding_is_dropped(self):
        """pad_to_multiple_of pads would otherwise be mean-pooled."""
        processor = object.__new__(Nemotron35EmbeddingProcessor)
        padded = {
            "input_ids": torch.tensor([[5, 10, 12, 10, 13, 7, 11, 11]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]),
            "pixel_values": torch.zeros(1, 3, 28, 28),
        }
        with patch.object(PixtralProcessor, "process_mm_data", return_value=padded):
            result = processor.process_mm_data("passage: [IMG] text")

        self.assertEqual(result["input_ids"].tolist(), [[5, 10, 12, 10, 13, 7]])
        self.assertEqual(result["attention_mask"].tolist(), [[1, 1, 1, 1, 1, 1]])
        self.assertEqual(tuple(result["pixel_values"].shape), (1, 3, 28, 28))


if __name__ == "__main__":
    unittest.main()
