"""
Unit-tests for the refactored completions-serving handler (no pytest).
Run with:
    python -m unittest tests.test_serving_completions_unit -v
"""

import unittest
from typing import Optional
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.entrypoints.openai.protocol import CompletionRequest
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.tokenizer_manager import TokenizerManager


class _MockTemplateManager:
    """Minimal mock for TemplateManager."""

    def __init__(self):
        self.chat_template_name: Optional[str] = None
        self.jinja_template_content_format: Optional[str] = None
        self.completion_template_name: Optional[str] = (
            None  # Set to None to avoid template processing
        )


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

        self.template_manager = _MockTemplateManager()
        self.sc = OpenAIServingCompletion(tm, self.template_manager)

    # ---------- prompt-handling ----------
    def test_single_string_prompt(self):
        req = CompletionRequest(model="x", prompt="Hello world", max_tokens=100)
        internal, _ = self.sc._convert_to_internal_request(req)
        self.assertEqual(internal.text, "Hello world")

    def test_single_token_ids_prompt(self):
        req = CompletionRequest(model="x", prompt=[1, 2, 3, 4], max_tokens=100)
        internal, _ = self.sc._convert_to_internal_request(req)
        self.assertEqual(internal.input_ids, [1, 2, 3, 4])

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

    # ---------- response_format handling ----------
    def test_response_format_json_object(self):
        """Test that response_format json_object is correctly processed in sampling params."""
        req = CompletionRequest(
            model="x",
            prompt="Generate a JSON object:",
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        sampling_params = self.sc._build_sampling_params(req)
        self.assertEqual(sampling_params["json_schema"], '{"type": "object"}')

    def test_response_format_json_schema(self):
        """Test that response_format json_schema is correctly processed in sampling params."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        req = CompletionRequest(
            model="x",
            prompt="Generate a JSON object:",
            max_tokens=100,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "person", "schema": schema},
            },
        )
        sampling_params = self.sc._build_sampling_params(req)
        # The schema should be converted to string by convert_json_schema_to_str
        self.assertIn("json_schema", sampling_params)
        self.assertIsInstance(sampling_params["json_schema"], str)

    def test_response_format_structural_tag(self):
        """Test that response_format structural_tag is correctly processed in sampling params."""
        req = CompletionRequest(
            model="x",
            prompt="Generate structured output:",
            max_tokens=100,
            response_format={
                "type": "structural_tag",
                "structures": [{"begin": "<data>", "end": "</data>"}],
                "triggers": ["<data>"],
            },
        )
        sampling_params = self.sc._build_sampling_params(req)
        # The structural_tag should be processed
        self.assertIn("structural_tag", sampling_params)
        self.assertIsInstance(sampling_params["structural_tag"], str)

    def test_response_format_none(self):
        """Test that no response_format doesn't add extra constraints."""
        req = CompletionRequest(model="x", prompt="Generate text:", max_tokens=100)
        sampling_params = self.sc._build_sampling_params(req)
        # Should not have json_schema or structural_tag from response_format
        # (but might have json_schema from the legacy json_schema field)
        self.assertIsNone(sampling_params.get("structural_tag"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
