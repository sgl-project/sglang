"""
Unit tests for the OpenAIServingEmbedding class from serving_embedding.py.
"""

import unittest
import uuid
from unittest.mock import Mock

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import (
    EmbeddingRequest,
    EmbeddingResponse,
    MultimodalEmbeddingInput,
)
from sglang.srt.entrypoints.openai.serving_embedding import OpenAIServingEmbedding
from sglang.srt.managers.io_struct import EmbeddingReqInput


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
        self.jinja_template_content_format = None
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
