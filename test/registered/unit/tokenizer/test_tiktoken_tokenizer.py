"""Unit tests for tiktoken_tokenizer — no server, no model loading."""

import unittest
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="stage-a-test-cpu")

from sglang.srt.tokenizer.tiktoken_tokenizer import (
    CONTROL_TOKEN_TEXTS,
    DEFAULT_CONTROL_TOKENS,
    DEFAULT_SPECIAL_TOKENS,
    EOS,
    PAD,
    RESERVED_TOKEN_TEXTS,
    SEP,
    TiktokenProcessor,
    TiktokenTokenizer,
)


class TestConstants(CustomTestCase):
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

    def test_default_control_tokens_values(self):
        # Note: "sep" maps to EOS and "eos" maps to SEP in the source code
        self.assertEqual(DEFAULT_CONTROL_TOKENS["pad"], PAD)
        self.assertEqual(DEFAULT_CONTROL_TOKENS["sep"], EOS)
        self.assertEqual(DEFAULT_CONTROL_TOKENS["eos"], SEP)


class TestTiktokenProcessor(CustomTestCase):
    def setUp(self):
        tokenizer_patcher = patch(
            "sglang.srt.tokenizer.tiktoken_tokenizer.TiktokenTokenizer"
        )
        tokenizer_patcher.start()
        self.addCleanup(tokenizer_patcher.stop)
        self.processor = TiktokenProcessor(name="dummy")

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


class TestTiktokenTokenizer(CustomTestCase):
    def setUp(self):
        from jinja2 import Template

        self.tok = TiktokenTokenizer.__new__(TiktokenTokenizer)
        self.mock_tokenizer = MagicMock()
        self.tok.tokenizer = self.mock_tokenizer
        self.tok.chat_template = "dummy"
        self.tok.chat_template_jinja = Template(
            "{% for message in messages %}"
            "{{ message['role'] }}: {{ message['content'] }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}assistant:{% endif %}"
        )

    def test_encode_delegates_to_tokenizer(self):
        self.mock_tokenizer.encode.return_value = [1, 2, 3]
        result = self.tok.encode("hello")
        self.mock_tokenizer.encode.assert_called_once_with("hello")
        self.assertEqual(result, [1, 2, 3])

    def test_decode_delegates_to_tokenizer(self):
        self.mock_tokenizer.decode.return_value = "hello"
        result = self.tok.decode([1, 2, 3])
        self.mock_tokenizer.decode.assert_called_once_with([1, 2, 3])
        self.assertEqual(result, "hello")

    def test_batch_decode_list_of_lists(self):
        self.mock_tokenizer.decode_batch.return_value = ["hello", "world"]
        result = self.tok.batch_decode([[1, 2], [3, 4]])
        self.mock_tokenizer.decode_batch.assert_called_once_with([[1, 2], [3, 4]])
        self.assertEqual(result, ["hello", "world"])

    def test_batch_decode_flat_list_wraps_each(self):
        self.mock_tokenizer.decode_batch.return_value = ["a", "b"]
        self.tok.batch_decode([1, 2])
        self.mock_tokenizer.decode_batch.assert_called_once_with([[1], [2]])

    def test_call_returns_input_ids(self):
        self.mock_tokenizer.encode.return_value = [1, 2, 3]
        result = self.tok(["hello", "world"])
        self.assertIn("input_ids", result)
        self.assertEqual(len(result["input_ids"]), 2)

    def test_apply_chat_template_no_tokenize(self):
        messages = [{"role": "user", "content": "hello"}]
        result = self.tok.apply_chat_template(
            messages=messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        self.assertIsInstance(result, str)
        self.assertIn("hello", result)

    def test_apply_chat_template_with_tokenize(self):
        self.mock_tokenizer.encode.return_value = [1, 2, 3]
        messages = [{"role": "user", "content": "hello"}]
        result = self.tok.apply_chat_template(
            messages=messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
