"""
Unit tests for the OpenAIServingEmbedding class from serving_embedding.py.
"""

import importlib
import importlib.abc
import importlib.machinery
import sys
import types
import unittest
import uuid
from unittest.mock import MagicMock, Mock

import jinja2


# Stub out sgl_kernel (and all submodules) before any sglang import so
# the test runs on CPU-only runners without the real CUDA library.
class _SglKernelMockLoader(importlib.abc.Loader):
    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []
        mod.__package__ = spec.name
        mod.__loader__ = self
        mod.__getattr__ = lambda name: MagicMock()
        return mod

    def exec_module(self, module):
        pass


class _SglKernelMockFinder(importlib.abc.MetaPathFinder):
    """Import hook that intercepts all sgl_kernel.* imports and returns mocks."""

    _PREFIX = "sgl_kernel"
    _loader = _SglKernelMockLoader()

    def find_spec(self, fullname, path, target=None):
        if fullname == self._PREFIX or fullname.startswith(self._PREFIX + "."):
            return importlib.machinery.ModuleSpec(
                fullname, self._loader, is_package=True
            )
        return None


if "sgl_kernel" not in sys.modules:
    sys.meta_path.insert(0, _SglKernelMockFinder())

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import (
    EmbeddingRequest,
    MultimodalEmbeddingInput,
)
from sglang.srt.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from sglang.srt.managers.io_struct import EmbeddingReqInput
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


# Mock TokenizerManager for embedding tests
class _MockTokenizerManager:
    def __init__(self):
        self.model_config = Mock()
        self.model_config.is_multimodal = False
        self.server_args = Mock()
        self.server_args.enable_cache_report = False
        self.model_path = "test-model"

        # Mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        self.tokenizer.decode = Mock(return_value="Test embedding input")
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1

        # Mock generate_request method for embeddings
        async def mock_generate_embedding():
            yield {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 20,  # 100-dim embedding
                "meta_info": {
                    "id": f"embd-{uuid.uuid4()}",
                    "prompt_tokens": 5,
                },
            }

        self.generate_request = Mock(return_value=mock_generate_embedding())


# Mock TemplateManager for embedding tests
class _MockTemplateManager:
    def __init__(self):
        self.chat_template_name = None  # None for embeddings usually
        self.jinja_template_content_format = "openai"
        self.completion_template_name = None


class ServingEmbeddingTestCase(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer_manager = _MockTokenizerManager()
        self.template_manager = _MockTemplateManager()
        self.serving_embedding = OpenAIServingEmbedding(
            self.tokenizer_manager, self.template_manager
        )

        self.request = Mock(spec=Request)
        self.request.headers = {}

        self.basic_req = EmbeddingRequest(
            model="test-model",
            input="Hello, how are you?",
            encoding_format="float",
        )
        self.list_req = EmbeddingRequest(
            model="test-model",
            input=["Hello, how are you?", "I am fine, thank you!"],
            encoding_format="float",
        )
        self.multimodal_req = EmbeddingRequest(
            model="test-model",
            input=[
                MultimodalEmbeddingInput(text="Hello", image="base64_image_data"),
                MultimodalEmbeddingInput(text="World", image=None),
            ],
            encoding_format="float",
        )
        self.image_only_multimodal_req = EmbeddingRequest(
            model="test-model",
            input=[
                MultimodalEmbeddingInput(text=None, image="base64_image_data"),
            ],
            encoding_format="float",
        )
        self.video_multimodal_req = EmbeddingRequest(
            model="test-model",
            input=[
                MultimodalEmbeddingInput(
                    text="Describe", image=None, video="base64_video_data"
                ),
            ],
            encoding_format="float",
        )
        self.token_ids_req = EmbeddingRequest(
            model="test-model",
            input=[1, 2, 3, 4, 5],
            encoding_format="float",
        )

    def test_convert_single_string_request(self):
        """Test converting single string request to internal format."""
        adapted_request, processed_request = (
            self.serving_embedding._convert_to_internal_request(self.basic_req)
        )

        self.assertIsInstance(adapted_request, EmbeddingReqInput)
        self.assertEqual(adapted_request.text, "Hello, how are you?")
        # self.assertEqual(adapted_request.rid, "test-id")
        self.assertEqual(processed_request, self.basic_req)

    def test_convert_list_string_request(self):
        """Test converting list of strings request to internal format."""
        adapted_request, processed_request = (
            self.serving_embedding._convert_to_internal_request(self.list_req)
        )

        self.assertIsInstance(adapted_request, EmbeddingReqInput)
        self.assertEqual(
            adapted_request.text, ["Hello, how are you?", "I am fine, thank you!"]
        )
        # self.assertEqual(adapted_request.rid, "test-id")
        self.assertEqual(processed_request, self.list_req)

    def test_convert_token_ids_request(self):
        """Test converting token IDs request to internal format."""
        adapted_request, processed_request = (
            self.serving_embedding._convert_to_internal_request(self.token_ids_req)
        )

        self.assertIsInstance(adapted_request, EmbeddingReqInput)
        self.assertEqual(adapted_request.input_ids, [1, 2, 3, 4, 5])
        # self.assertEqual(adapted_request.rid, "test-id")
        self.assertEqual(processed_request, self.token_ids_req)

    def test_convert_multimodal_request(self):
        """Test converting multimodal request to internal format."""
        adapted_request, processed_request = (
            self.serving_embedding._convert_to_internal_request(self.multimodal_req)
        )

        self.assertIsInstance(adapted_request, EmbeddingReqInput)
        # Should extract text and images separately
        self.assertEqual(len(adapted_request.text), 2)
        self.assertIn("Hello", adapted_request.text)
        self.assertIn("World", adapted_request.text)
        self.assertEqual(adapted_request.image_data[0], "base64_image_data")
        self.assertIsNone(adapted_request.image_data[1])
        # self.assertEqual(adapted_request.rid, "test-id")

    def test_convert_multimodal_request_with_jinja_chat_template(self):
        """Multimodal embeddings should apply explicit/HF Jinja chat templates."""
        self.tokenizer_manager.tokenizer.chat_template = "mock-template"
        self.tokenizer_manager.tokenizer.apply_chat_template = Mock(
            side_effect=[
                "<prompt>Hello<image></prompt>",
                "<prompt>World</prompt>",
            ]
        )

        adapted_request, _ = self.serving_embedding._convert_to_internal_request(
            self.multimodal_req
        )

        self.assertEqual(
            adapted_request.text,
            ["<prompt>Hello<image></prompt>", "<prompt>World</prompt>"],
        )
        self.assertEqual(adapted_request.image_data[0], "base64_image_data")
        self.assertIsNone(adapted_request.image_data[1])
        self.assertEqual(
            self.tokenizer_manager.tokenizer.apply_chat_template.call_count, 2
        )
        first_call = (
            self.tokenizer_manager.tokenizer.apply_chat_template.call_args_list[0]
        )
        first_messages = first_call.args[0]
        self.assertEqual(first_messages[0]["role"], "user")
        self.assertEqual(first_messages[0]["content"][0]["type"], "image")
        self.assertEqual(first_messages[0]["content"][1]["type"], "text")
        self.assertEqual(first_messages[0]["content"][1]["text"], "Hello")
        self.assertEqual(first_call.kwargs["tokenize"], False)
        self.assertEqual(first_call.kwargs["add_generation_prompt"], True)

        second_call = (
            self.tokenizer_manager.tokenizer.apply_chat_template.call_args_list[1]
        )
        second_messages = second_call.args[0]
        self.assertEqual(len(second_messages[0]["content"]), 1)
        self.assertEqual(second_messages[0]["content"][0]["type"], "text")
        self.assertEqual(second_messages[0]["content"][0]["text"], "World")

    def test_convert_image_only_multimodal_request_with_jinja_chat_template(self):
        """Image-only requests should not inject literal padding into Jinja prompts."""
        self.tokenizer_manager.tokenizer.chat_template = "mock-template"
        self.tokenizer_manager.tokenizer.apply_chat_template = Mock(
            return_value="<prompt><image></prompt>"
        )

        adapted_request, _ = self.serving_embedding._convert_to_internal_request(
            self.image_only_multimodal_req
        )

        self.assertEqual(adapted_request.text, "<prompt><image></prompt>")
        first_call = self.tokenizer_manager.tokenizer.apply_chat_template.call_args
        first_messages = first_call.args[0]
        self.assertEqual(first_messages[0]["role"], "user")
        self.assertEqual(len(first_messages[0]["content"]), 1)
        self.assertEqual(first_messages[0]["content"][0]["type"], "image")

    def test_convert_video_multimodal_request_with_jinja_chat_template(self):
        """Video inputs should land in video_data and flow through the Jinja branch."""
        self.tokenizer_manager.tokenizer.chat_template = "mock-template"
        self.tokenizer_manager.tokenizer.apply_chat_template = Mock(
            return_value="<prompt>Describe<video></prompt>"
        )

        adapted_request, _ = self.serving_embedding._convert_to_internal_request(
            self.video_multimodal_req
        )

        self.assertEqual(adapted_request.text, "<prompt>Describe<video></prompt>")
        self.assertEqual(adapted_request.video_data, "base64_video_data")
        self.assertIsNone(adapted_request.image_data)
        first_messages = (
            self.tokenizer_manager.tokenizer.apply_chat_template.call_args.args[0]
        )
        content = first_messages[0]["content"]
        self.assertEqual([c["type"] for c in content], ["video", "text"])

    def test_multimodal_request_falls_back_when_no_chat_template(self):
        """Without any chat template the raw-text fallback must run without raising."""
        self.tokenizer_manager.tokenizer.chat_template = None

        adapted_request, _ = self.serving_embedding._convert_to_internal_request(
            self.image_only_multimodal_req
        )

        # text=None on an image-only input falls back to the "padding" literal.
        self.assertEqual(adapted_request.text, "padding")
        self.assertEqual(adapted_request.image_data, "base64_image_data")

    def test_multimodal_request_with_no_tokenizer_uses_fallback(self):
        """Missing tokenizer should not crash the Jinja branch check."""
        self.tokenizer_manager.tokenizer = None

        adapted_request, _ = self.serving_embedding._convert_to_internal_request(
            self.multimodal_req
        )

        self.assertEqual(adapted_request.text, ["Hello", "World"])

    def test_jinja_template_errors_are_raised_as_value_error(self):
        """Template failures should be converted to ValueError for a 400 response."""
        self.tokenizer_manager.tokenizer.chat_template = "mock-template"
        self.tokenizer_manager.tokenizer.apply_chat_template = Mock(
            side_effect=jinja2.TemplateError("bad template")
        )

        with self.assertRaisesRegex(ValueError, "bad template"):
            self.serving_embedding._convert_to_internal_request(
                self.image_only_multimodal_req
            )

    def test_jinja_template_syntax_error_includes_location(self):
        """TemplateSyntaxError should surface template name and line number."""
        err = jinja2.TemplateSyntaxError("unexpected end", lineno=7, name="mock.jinja")
        self.tokenizer_manager.tokenizer.chat_template = "mock-template"
        self.tokenizer_manager.tokenizer.apply_chat_template = Mock(side_effect=err)

        with self.assertRaises(ValueError) as ctx:
            self.serving_embedding._convert_to_internal_request(
                self.image_only_multimodal_req
            )
        message = str(ctx.exception)
        self.assertIn("mock.jinja", message)
        self.assertIn("line=7", message)

    def test_non_jinja_template_errors_are_raised_as_value_error(self):
        """TypeError / KeyError from apply_chat_template should map to 400, not 500."""
        self.tokenizer_manager.tokenizer.chat_template = "mock-template"
        self.tokenizer_manager.tokenizer.apply_chat_template = Mock(
            side_effect=KeyError("missing_field")
        )

        with self.assertRaisesRegex(ValueError, "missing_field"):
            self.serving_embedding._convert_to_internal_request(
                self.image_only_multimodal_req
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
