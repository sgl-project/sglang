"""
Unit tests for the OpenAIServingEmbedding class from serving_embedding.py.

These tests ensure that the embedding serving implementation maintains compatibility
with the original adapter.py functionality and follows OpenAI API specifications.
"""

import asyncio
import json
import time
import unittest
import uuid
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

from fastapi import Request
from fastapi.responses import ORJSONResponse
from pydantic_core import ValidationError

from sglang.srt.entrypoints.openai.protocol import (
    EmbeddingObject,
    EmbeddingRequest,
    EmbeddingResponse,
    ErrorResponse,
    MultimodalEmbeddingInput,
    UsageInfo,
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


class ServingEmbeddingTestCase(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.tokenizer_manager = _MockTokenizerManager()
        self.serving_embedding = OpenAIServingEmbedding(self.tokenizer_manager)

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
            self.serving_embedding._convert_to_internal_request(
                [self.basic_req], ["test-id"]
            )
        )

        self.assertIsInstance(adapted_request, EmbeddingReqInput)
        self.assertEqual(adapted_request.text, "Hello, how are you?")
        self.assertEqual(adapted_request.rid, "test-id")
        self.assertEqual(processed_request, self.basic_req)

    def test_convert_list_string_request(self):
        """Test converting list of strings request to internal format."""
        adapted_request, processed_request = (
            self.serving_embedding._convert_to_internal_request(
                [self.list_req], ["test-id"]
            )
        )

        self.assertIsInstance(adapted_request, EmbeddingReqInput)
        self.assertEqual(
            adapted_request.text, ["Hello, how are you?", "I am fine, thank you!"]
        )
        self.assertEqual(adapted_request.rid, "test-id")
        self.assertEqual(processed_request, self.list_req)

    def test_convert_token_ids_request(self):
        """Test converting token IDs request to internal format."""
        adapted_request, processed_request = (
            self.serving_embedding._convert_to_internal_request(
                [self.token_ids_req], ["test-id"]
            )
        )

        self.assertIsInstance(adapted_request, EmbeddingReqInput)
        self.assertEqual(adapted_request.input_ids, [1, 2, 3, 4, 5])
        self.assertEqual(adapted_request.rid, "test-id")
        self.assertEqual(processed_request, self.token_ids_req)

    def test_convert_multimodal_request(self):
        """Test converting multimodal request to internal format."""
        adapted_request, processed_request = (
            self.serving_embedding._convert_to_internal_request(
                [self.multimodal_req], ["test-id"]
            )
        )

        self.assertIsInstance(adapted_request, EmbeddingReqInput)
        # Should extract text and images separately
        self.assertEqual(len(adapted_request.text), 2)
        self.assertIn("Hello", adapted_request.text)
        self.assertIn("World", adapted_request.text)
        self.assertEqual(adapted_request.image_data[0], "base64_image_data")
        self.assertIsNone(adapted_request.image_data[1])
        self.assertEqual(adapted_request.rid, "test-id")

    def test_build_single_embedding_response(self):
        """Test building response for single embedding."""
        ret_data = [
            {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "meta_info": {"prompt_tokens": 5},
            }
        ]

        response = self.serving_embedding._build_embedding_response(
            ret_data, "test-model"
        )

        self.assertIsInstance(response, EmbeddingResponse)
        self.assertEqual(response.model, "test-model")
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0].embedding, [0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertEqual(response.data[0].index, 0)
        self.assertEqual(response.data[0].object, "embedding")
        self.assertEqual(response.usage.prompt_tokens, 5)
        self.assertEqual(response.usage.total_tokens, 5)
        self.assertEqual(response.usage.completion_tokens, 0)

    def test_build_multiple_embedding_response(self):
        """Test building response for multiple embeddings."""
        ret_data = [
            {
                "embedding": [0.1, 0.2, 0.3],
                "meta_info": {"prompt_tokens": 3},
            },
            {
                "embedding": [0.4, 0.5, 0.6],
                "meta_info": {"prompt_tokens": 4},
            },
        ]

        response = self.serving_embedding._build_embedding_response(
            ret_data, "test-model"
        )

        self.assertIsInstance(response, EmbeddingResponse)
        self.assertEqual(len(response.data), 2)
        self.assertEqual(response.data[0].embedding, [0.1, 0.2, 0.3])
        self.assertEqual(response.data[0].index, 0)
        self.assertEqual(response.data[1].embedding, [0.4, 0.5, 0.6])
        self.assertEqual(response.data[1].index, 1)
        self.assertEqual(response.usage.prompt_tokens, 7)  # 3 + 4
        self.assertEqual(response.usage.total_tokens, 7)

    async def test_handle_request_success(self):
        """Test successful embedding request handling."""

        # Mock the generate_request to return expected data
        async def mock_generate():
            yield {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "meta_info": {"prompt_tokens": 5},
            }

        self.serving_embedding.tokenizer_manager.generate_request = Mock(
            return_value=mock_generate()
        )

        response = await self.serving_embedding.handle_request(
            self.basic_req, self.request
        )

        self.assertIsInstance(response, EmbeddingResponse)
        self.assertEqual(len(response.data), 1)
        self.assertEqual(response.data[0].embedding, [0.1, 0.2, 0.3, 0.4, 0.5])

    async def test_handle_request_validation_error(self):
        """Test handling request with validation error."""
        invalid_request = EmbeddingRequest(model="test-model", input="")

        response = await self.serving_embedding.handle_request(
            invalid_request, self.request
        )

        self.assertIsInstance(response, ORJSONResponse)
        self.assertEqual(response.status_code, 400)

    async def test_handle_request_generation_error(self):
        """Test handling request with generation error."""

        # Mock generate_request to raise an error
        async def mock_generate_error():
            raise ValueError("Generation failed")
            yield  # This won't be reached but needed for async generator

        self.serving_embedding.tokenizer_manager.generate_request = Mock(
            return_value=mock_generate_error()
        )

        response = await self.serving_embedding.handle_request(
            self.basic_req, self.request
        )

        self.assertIsInstance(response, ORJSONResponse)
        self.assertEqual(response.status_code, 400)

    async def test_handle_request_internal_error(self):
        """Test handling request with internal server error."""
        # Mock _convert_to_internal_request to raise an exception
        with patch.object(
            self.serving_embedding,
            "_convert_to_internal_request",
            side_effect=Exception("Internal error"),
        ):
            response = await self.serving_embedding.handle_request(
                self.basic_req, self.request
            )

            self.assertIsInstance(response, ORJSONResponse)
            self.assertEqual(response.status_code, 500)


if __name__ == "__main__":
    unittest.main(verbosity=2)
