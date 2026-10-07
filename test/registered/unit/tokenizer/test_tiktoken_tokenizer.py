"""Unit tests for tiktoken_tokenizer — no server, no model loading."""

import unittest
from unittest.mock import MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

from sglang.srt.tokenizer.tiktoken_tokenizer import (
    TiktokenTokenizer,
)


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
