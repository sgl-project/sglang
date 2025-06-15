"""
Unit tests for the OpenAIServingEmbedding class from serving_embedding.py.

These tests ensure that the embedding serving implementation maintains compatibility
with the original adapter.py functionality and follows OpenAI API specifications.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import Request
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
class MockTokenizerManager:
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


@pytest.fixture
def mock_tokenizer_manager():
    """Create a mock tokenizer manager for testing."""
    return MockTokenizerManager()


@pytest.fixture
def embedding_handler(mock_tokenizer_manager):
    """Create an OpenAIServingEmbedding instance for testing."""
    return OpenAIServingEmbedding(mock_tokenizer_manager)


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request."""
    request = Mock(spec=Request)
    request.headers = {}
    return request


@pytest.fixture
def basic_embedding_request():
    """Create a basic embedding request."""
    return EmbeddingRequest(
        model="test-model",
        input="Hello, how are you?",
        encoding_format="float",
    )


@pytest.fixture
def list_embedding_request():
    """Create an embedding request with list input."""
    return EmbeddingRequest(
        model="test-model",
        input=["Hello, how are you?", "I am fine, thank you!"],
        encoding_format="float",
    )


@pytest.fixture
def multimodal_embedding_request():
    """Create a multimodal embedding request."""
    return EmbeddingRequest(
        model="test-model",
        input=[
            MultimodalEmbeddingInput(text="Hello", image="base64_image_data"),
            MultimodalEmbeddingInput(text="World", image=None),
        ],
        encoding_format="float",
    )


@pytest.fixture
def token_ids_embedding_request():
    """Create an embedding request with token IDs."""
    return EmbeddingRequest(
        model="test-model",
        input=[1, 2, 3, 4, 5],
        encoding_format="float",
    )


class TestOpenAIServingEmbeddingConversion:
    """Test request conversion methods."""

    def test_convert_single_string_request(
        self, embedding_handler, basic_embedding_request
    ):
        """Test converting single string request to internal format."""
        adapted_request, processed_request = (
            embedding_handler._convert_to_internal_request(
                [basic_embedding_request], ["test-id"]
            )
        )

        assert isinstance(adapted_request, EmbeddingReqInput)
        assert adapted_request.text == "Hello, how are you?"
        assert adapted_request.rid == "test-id"
        assert processed_request == basic_embedding_request

    def test_convert_list_string_request(
        self, embedding_handler, list_embedding_request
    ):
        """Test converting list of strings request to internal format."""
        adapted_request, processed_request = (
            embedding_handler._convert_to_internal_request(
                [list_embedding_request], ["test-id"]
            )
        )

        assert isinstance(adapted_request, EmbeddingReqInput)
        assert adapted_request.text == ["Hello, how are you?", "I am fine, thank you!"]
        assert adapted_request.rid == "test-id"
        assert processed_request == list_embedding_request

    def test_convert_token_ids_request(
        self, embedding_handler, token_ids_embedding_request
    ):
        """Test converting token IDs request to internal format."""
        adapted_request, processed_request = (
            embedding_handler._convert_to_internal_request(
                [token_ids_embedding_request], ["test-id"]
            )
        )

        assert isinstance(adapted_request, EmbeddingReqInput)
        assert adapted_request.input_ids == [1, 2, 3, 4, 5]
        assert adapted_request.rid == "test-id"
        assert processed_request == token_ids_embedding_request

    def test_convert_multimodal_request(
        self, embedding_handler, multimodal_embedding_request
    ):
        """Test converting multimodal request to internal format."""
        adapted_request, processed_request = (
            embedding_handler._convert_to_internal_request(
                [multimodal_embedding_request], ["test-id"]
            )
        )

        assert isinstance(adapted_request, EmbeddingReqInput)
        # Should extract text and images separately
        assert len(adapted_request.text) == 2
        assert "Hello" in adapted_request.text
        assert "World" in adapted_request.text
        assert adapted_request.image_data[0] == "base64_image_data"
        assert adapted_request.image_data[1] is None
        assert adapted_request.rid == "test-id"

    def test_convert_batch_requests(self, embedding_handler):
        """Test converting multiple requests (batch) to internal format."""
        request1 = EmbeddingRequest(model="test-model", input="First text")
        request2 = EmbeddingRequest(model="test-model", input="Second text")

        adapted_request, processed_requests = (
            embedding_handler._convert_to_internal_request(
                [request1, request2], ["id1", "id2"]
            )
        )

        assert isinstance(adapted_request, EmbeddingReqInput)
        assert adapted_request.text == ["First text", "Second text"]
        assert adapted_request.rid == ["id1", "id2"]
        assert processed_requests == [request1, request2]

    def test_convert_batch_requests_type_mismatch_error(self, embedding_handler):
        """Test that batch requests with different input types raise error."""
        request1 = EmbeddingRequest(model="test-model", input="String input")
        request2 = EmbeddingRequest(model="test-model", input=[1, 2, 3])  # Token IDs

        with pytest.raises(AssertionError, match="same type"):
            embedding_handler._convert_to_internal_request(
                [request1, request2], ["id1", "id2"]
            )


class TestEmbeddingResponseBuilding:
    """Test response building methods."""

    def test_build_single_embedding_response(self, embedding_handler):
        """Test building response for single embedding."""
        ret_data = [
            {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "meta_info": {"prompt_tokens": 5},
            }
        ]

        response = embedding_handler._build_embedding_response(ret_data, "test-model")

        assert isinstance(response, EmbeddingResponse)
        assert response.model == "test-model"
        assert len(response.data) == 1
        assert response.data[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert response.data[0].index == 0
        assert response.data[0].object == "embedding"
        assert response.usage.prompt_tokens == 5
        assert response.usage.total_tokens == 5
        assert response.usage.completion_tokens == 0

    def test_build_multiple_embedding_response(self, embedding_handler):
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

        response = embedding_handler._build_embedding_response(ret_data, "test-model")

        assert isinstance(response, EmbeddingResponse)
        assert len(response.data) == 2
        assert response.data[0].embedding == [0.1, 0.2, 0.3]
        assert response.data[0].index == 0
        assert response.data[1].embedding == [0.4, 0.5, 0.6]
        assert response.data[1].index == 1
        assert response.usage.prompt_tokens == 7  # 3 + 4
        assert response.usage.total_tokens == 7


@pytest.mark.asyncio
class TestOpenAIServingEmbeddingAsyncMethods:
    """Test async methods of OpenAIServingEmbedding."""

    async def test_handle_request_success(
        self, embedding_handler, basic_embedding_request, mock_request
    ):
        """Test successful embedding request handling."""

        # Mock the generate_request to return expected data
        async def mock_generate():
            yield {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "meta_info": {"prompt_tokens": 5},
            }

        embedding_handler.tokenizer_manager.generate_request = Mock(
            return_value=mock_generate()
        )

        response = await embedding_handler.handle_request(
            basic_embedding_request, mock_request
        )

        assert isinstance(response, EmbeddingResponse)
        assert len(response.data) == 1
        assert response.data[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    async def test_handle_request_validation_error(
        self, embedding_handler, mock_request
    ):
        """Test handling request with validation error."""
        invalid_request = EmbeddingRequest(model="test-model", input="")

        response = await embedding_handler.handle_request(invalid_request, mock_request)

        assert isinstance(response, ErrorResponse)
        assert "empty" in response.message.lower()

    async def test_handle_request_generation_error(
        self, embedding_handler, basic_embedding_request, mock_request
    ):
        """Test handling request with generation error."""

        # Mock generate_request to raise an error
        async def mock_generate_error():
            raise ValueError("Generation failed")
            yield  # This won't be reached but needed for async generator

        embedding_handler.tokenizer_manager.generate_request = Mock(
            return_value=mock_generate_error()
        )

        response = await embedding_handler.handle_request(
            basic_embedding_request, mock_request
        )

        assert isinstance(response, ErrorResponse)
        assert "Generation failed" in response.message

    async def test_handle_request_internal_error(
        self, embedding_handler, basic_embedding_request, mock_request
    ):
        """Test handling request with internal server error."""
        # Mock _convert_to_internal_request to raise an exception
        with patch.object(
            embedding_handler,
            "_convert_to_internal_request",
            side_effect=Exception("Internal error"),
        ):
            response = await embedding_handler.handle_request(
                basic_embedding_request, mock_request
            )

            assert isinstance(response, ErrorResponse)
            assert "Internal server error" in response.message
            assert response.code == 500


class TestCompatibilityWithAdapter:
    """Test compatibility with original adapter.py implementation."""

    def test_embedding_request_structure_matches_adapter(self, embedding_handler):
        """Test that EmbeddingReqInput structure matches adapter expectations."""
        request = EmbeddingRequest(model="test-model", input="Test text")

        adapted_request, _ = embedding_handler._convert_to_internal_request(
            [request], ["test-id"]
        )

        # Check that adapted_request has expected fields from adapter.py
        assert hasattr(adapted_request, "rid")
        assert hasattr(adapted_request, "text") or hasattr(adapted_request, "input_ids")
        assert adapted_request.rid == "test-id"

    def test_multimodal_embedding_processing_compatibility(self, embedding_handler):
        """Test multimodal processing matches adapter patterns."""
        multimodal_input = [
            MultimodalEmbeddingInput(text="Hello", image="image_data"),
            MultimodalEmbeddingInput(text="World", image=None),
        ]
        request = EmbeddingRequest(model="test-model", input=multimodal_input)

        adapted_request, _ = embedding_handler._convert_to_internal_request(
            [request], ["test-id"]
        )

        # Should have text and image_data fields like adapter
        assert hasattr(adapted_request, "text")
        assert hasattr(adapted_request, "image_data")
        assert len(adapted_request.text) == 2
        assert len(adapted_request.image_data) == 2

    def test_response_format_matches_adapter(self, embedding_handler):
        """Test response format matches adapter.py output."""
        ret_data = [
            {
                "embedding": [0.1, 0.2, 0.3],
                "meta_info": {"prompt_tokens": 3},
            }
        ]

        response = embedding_handler._build_embedding_response(ret_data, "test-model")

        # Check response structure matches adapter output
        assert response.object == "list"
        assert isinstance(response.data, list)
        assert len(response.data) == 1
        assert response.data[0].object == "embedding"
        assert isinstance(response.data[0].embedding, list)
        assert isinstance(response.data[0].index, int)
        assert isinstance(response.usage, UsageInfo)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_multimodal_batch_not_implemented(self, embedding_handler):
        """Test that multimodal batch requests raise NotImplementedError."""
        request1 = EmbeddingRequest(
            model="test-model",
            input=[MultimodalEmbeddingInput(text="Hello", image="img1")],
        )
        request2 = EmbeddingRequest(
            model="test-model",
            input=[MultimodalEmbeddingInput(text="World", image="img2")],
        )

        with pytest.raises(NotImplementedError, match="multimodal.*not supported"):
            embedding_handler._convert_to_internal_request(
                [request1, request2], ["id1", "id2"]
            )

    def test_empty_return_data_handling(self, embedding_handler):
        """Test handling of empty return data from generation."""
        # Test with empty list
        response = embedding_handler._build_embedding_response([], "test-model")
        assert len(response.data) == 0
        assert response.usage.prompt_tokens == 0
        assert response.usage.total_tokens == 0

    def test_missing_meta_info_handling(self, embedding_handler):
        """Test handling of missing meta_info in return data."""
        ret_data = [
            {
                "embedding": [0.1, 0.2, 0.3],
                "meta_info": {},  # Missing prompt_tokens
            }
        ]

        # Should handle missing prompt_tokens gracefully
        response = embedding_handler._build_embedding_response(ret_data, "test-model")
        assert len(response.data) == 1
        # Should default to 0 for missing prompt_tokens
        assert response.usage.prompt_tokens == 0
