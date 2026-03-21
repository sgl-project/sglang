"""Unit tests for tiktoken_tokenizer — no server, no model loading."""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../python'))

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")

from sglang.srt.tokenizer.tiktoken_tokenizer import (
    CONTROL_TOKEN_TEXTS,
    DEFAULT_CONTROL_TOKENS,
    DEFAULT_SPECIAL_TOKENS,
    EOS,
    PAD,
    RESERVED_TOKEN_TEXTS,
    SEP,
    TiktokenProcessor,
)


class TestConstants(unittest.TestCase):
    def test_reserved_token_count(self):
        self.assertEqual(len(RESERVED_TOKEN_TEXTS), 125)

    def test_reserved_token_format(self):
        self.assertEqual(RESERVED_TOKEN_TEXTS[0], "<|reserved_3|>")
        self.assertEqual(RESERVED_TOKEN_TEXTS[-1], "<|reserved_127|>")

    def test_control_token_count(self):
        self.assertEqual(len(CONTROL_TOKEN_TEXTS), 704)

    def test_control_token_format(self):
        self.assertEqual(CONTROL_TOKEN_TEXTS[0], "<|control1|>")
        self.assertEqual(CONTROL_TOKEN_TEXTS[-1], "<|control704|>")

    def test_special_token_values(self):
        self.assertEqual(PAD, "<|pad|>")
        self.assertEqual(EOS, "<|eos|>")
        self.assertEqual(SEP, "<|separator|>")

    def test_default_special_tokens_contains_all(self):
        self.assertIn(PAD, DEFAULT_SPECIAL_TOKENS)
        self.assertIn(EOS, DEFAULT_SPECIAL_TOKENS)
        self.assertIn(SEP, DEFAULT_SPECIAL_TOKENS)

    def test_default_control_tokens_keys(self):
        self.assertIn("pad", DEFAULT_CONTROL_TOKENS)
        self.assertIn("sep", DEFAULT_CONTROL_TOKENS)
        self.assertIn("eos", DEFAULT_CONTROL_TOKENS)


class TestTiktokenProcessor(unittest.TestCase):
    def setUp(self):
        with patch(
            "sglang.srt.tokenizer.tiktoken_tokenizer.TiktokenTokenizer"
        ):
            self.processor = TiktokenProcessor.__new__(TiktokenProcessor)
            self.processor.tokenizer = MagicMock()

    def test_image_processor_returns_dict(self):
        result = self.processor.image_processor("fake_image")
        self.assertIsInstance(result, dict)

    def test_image_processor_has_pixel_values_key(self):
        result = self.processor.image_processor("fake_image")
        self.assertIn("pixel_values", result)

    def test_image_processor_wraps_image_in_list(self):
        image = "fake_image_data"
        result = self.processor.image_processor(image)
        self.assertEqual(result["pixel_values"], [image])

    def test_image_processor_with_none(self):
        result = self.processor.image_processor(None)
        self.assertEqual(result["pixel_values"], [None])


if __name__ == "__main__":
    unittest.main()
