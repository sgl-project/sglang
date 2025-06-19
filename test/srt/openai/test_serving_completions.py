"""
Unit-tests for the refactored completions-serving handler (no pytest).
Run with:
    python -m unittest tests.test_serving_completions_unit -v
"""

import unittest
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.entrypoints.openai.protocol import CompletionRequest
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.tokenizer_manager import TokenizerManager


class ServingCompletionTestCase(unittest.TestCase):
    """Bundle all prompt/echo tests in one TestCase."""

    # ---------- shared test fixtures ----------
    def setUp(self):
        # build the mock TokenizerManager once for every test
        tm = Mock(spec=TokenizerManager)

        tm.tokenizer = Mock()
        tm.tokenizer.encode.return_value = [1, 2, 3, 4]
        tm.tokenizer.decode.return_value = "decoded text"
        tm.tokenizer.bos_token_id = 1

        tm.model_config = Mock(is_multimodal=False)
        tm.server_args = Mock(enable_cache_report=False)

        tm.generate_request = AsyncMock()
        tm.create_abort_task = Mock()

        self.sc = OpenAIServingCompletion(tm)

    # ---------- prompt-handling ----------
    def test_single_string_prompt(self):
        req = CompletionRequest(model="x", prompt="Hello world", max_tokens=100)
        internal, _ = self.sc._convert_to_internal_request([req], ["id"])
        self.assertEqual(internal.text, "Hello world")

    def test_single_token_ids_prompt(self):
        req = CompletionRequest(model="x", prompt=[1, 2, 3, 4], max_tokens=100)
        internal, _ = self.sc._convert_to_internal_request([req], ["id"])
        self.assertEqual(internal.input_ids, [1, 2, 3, 4])

    def test_completion_template_handling(self):
        req = CompletionRequest(
            model="x", prompt="def f():", suffix="return 1", max_tokens=100
        )
        with patch(
            "sglang.srt.entrypoints.openai.serving_completions.is_completion_template_defined",
            return_value=True,
        ), patch(
            "sglang.srt.entrypoints.openai.serving_completions.generate_completion_prompt_from_request",
            return_value="processed_prompt",
        ):
            internal, _ = self.sc._convert_to_internal_request([req], ["id"])
            self.assertEqual(internal.text, "processed_prompt")

    # ---------- echo-handling ----------
    def test_echo_with_string_prompt_streaming(self):
        req = CompletionRequest(model="x", prompt="Hello", max_tokens=1, echo=True)
        self.assertEqual(self.sc._get_echo_text(req, 0), "Hello")

    def test_echo_with_list_of_strings_streaming(self):
        req = CompletionRequest(
            model="x", prompt=["A", "B"], max_tokens=1, echo=True, n=1
        )
        self.assertEqual(self.sc._get_echo_text(req, 0), "A")
        self.assertEqual(self.sc._get_echo_text(req, 1), "B")

    def test_echo_with_token_ids_streaming(self):
        req = CompletionRequest(model="x", prompt=[1, 2, 3], max_tokens=1, echo=True)
        self.sc.tokenizer_manager.tokenizer.decode.return_value = "decoded_prompt"
        self.assertEqual(self.sc._get_echo_text(req, 0), "decoded_prompt")

    def test_echo_with_multiple_token_ids_streaming(self):
        req = CompletionRequest(
            model="x", prompt=[[1, 2], [3, 4]], max_tokens=1, echo=True, n=1
        )
        self.sc.tokenizer_manager.tokenizer.decode.return_value = "decoded"
        self.assertEqual(self.sc._get_echo_text(req, 0), "decoded")

    def test_prepare_echo_prompts_non_streaming(self):
        # single string
        req = CompletionRequest(model="x", prompt="Hi", echo=True)
        self.assertEqual(self.sc._prepare_echo_prompts(req), ["Hi"])

        # list of strings
        req = CompletionRequest(model="x", prompt=["Hi", "Yo"], echo=True)
        self.assertEqual(self.sc._prepare_echo_prompts(req), ["Hi", "Yo"])

        # token IDs
        req = CompletionRequest(model="x", prompt=[1, 2, 3], echo=True)
        self.sc.tokenizer_manager.tokenizer.decode.return_value = "decoded"
        self.assertEqual(self.sc._prepare_echo_prompts(req), ["decoded"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
