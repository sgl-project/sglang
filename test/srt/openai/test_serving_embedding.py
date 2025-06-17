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
def serving_embedding(mock_tokenizer_manager):
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
        self, serving_embedding, basic_embedding_request
    ):
        """Test converting single string request to internal format."""
        adapted_request, processed_request = (
            serving_embedding._convert_to_internal_request(
                [basic_embedding_request], ["test-id"]
            )
        )

        assert isinstance(adapted_request, EmbeddingReqInput)
        assert adapted_request.text == "Hello, how are you?"
        assert adapted_request.rid == "test-id"
        assert processed_request == basic_embedding_request

    def test_convert_list_string_request(
        self, serving_embedding, list_embedding_request
    ):
        """Test converting list of strings request to internal format."""
        adapted_request, processed_request = (
            serving_embedding._convert_to_internal_request(
                [list_embedding_request], ["test-id"]
            )
        )

        assert isinstance(adapted_request, EmbeddingReqInput)
        assert adapted_request.text == ["Hello, how are you?", "I am fine, thank you!"]
        assert adapted_request.rid == "test-id"
        assert processed_request == list_embedding_request

    def test_convert_token_ids_request(
        self, serving_embedding, token_ids_embedding_request
    ):
        """Test converting token IDs request to internal format."""
        adapted_request, processed_request = (
            serving_embedding._convert_to_internal_request(
                [token_ids_embedding_request], ["test-id"]
            )
        )

        assert isinstance(adapted_request, EmbeddingReqInput)
        assert adapted_request.input_ids == [1, 2, 3, 4, 5]
        assert adapted_request.rid == "test-id"
        assert processed_request == token_ids_embedding_request

    def test_convert_multimodal_request(
        self, serving_embedding, multimodal_embedding_request
    ):
        """Test converting multimodal request to internal format."""
        adapted_request, processed_request = (
            serving_embedding._convert_to_internal_request(
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


class TestEmbeddingResponseBuilding:
    """Test response building methods."""

    def test_build_single_embedding_response(self, serving_embedding):
        """Test building response for single embedding."""
        ret_data = [
            {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "meta_info": {"prompt_tokens": 5},
            }
        ]

        response = serving_embedding._build_embedding_response(ret_data, "test-model")

        assert isinstance(response, EmbeddingResponse)
        assert response.model == "test-model"
        assert len(response.data) == 1
        assert response.data[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert response.data[0].index == 0
        assert response.data[0].object == "embedding"
        assert response.usage.prompt_tokens == 5
        assert response.usage.total_tokens == 5
        assert response.usage.completion_tokens == 0

    def test_build_multiple_embedding_response(self, serving_embedding):
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

        response = serving_embedding._build_embedding_response(ret_data, "test-model")

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
        self, serving_embedding, basic_embedding_request, mock_request
    ):
        """Test successful embedding request handling."""

        # Mock the generate_request to return expected data
        async def mock_generate():
            yield {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5],
                "meta_info": {"prompt_tokens": 5},
            }

        serving_embedding.tokenizer_manager.generate_request = Mock(
            return_value=mock_generate()
        )

        response = await serving_embedding.handle_request(
            basic_embedding_request, mock_request
        )

        assert isinstance(response, EmbeddingResponse)
        assert len(response.data) == 1
        assert response.data[0].embedding == [0.1, 0.2, 0.3, 0.4, 0.5]

    async def test_handle_request_validation_error(
        self, serving_embedding, mock_request
    ):
        """Test handling request with validation error."""
        invalid_request = EmbeddingRequest(model="test-model", input="")

        response = await serving_embedding.handle_request(invalid_request, mock_request)

        assert isinstance(response, ORJSONResponse)
        assert response.status_code == 400

    async def test_handle_request_generation_error(
        self, serving_embedding, basic_embedding_request, mock_request
    ):
        """Test handling request with generation error."""

        # Mock generate_request to raise an error
        async def mock_generate_error():
            raise ValueError("Generation failed")
            yield  # This won't be reached but needed for async generator

        serving_embedding.tokenizer_manager.generate_request = Mock(
            return_value=mock_generate_error()
        )

        response = await serving_embedding.handle_request(
            basic_embedding_request, mock_request
        )

        assert isinstance(response, ORJSONResponse)
        assert response.status_code == 400

    async def test_handle_request_internal_error(
        self, serving_embedding, basic_embedding_request, mock_request
    ):
        """Test handling request with internal server error."""
        # Mock _convert_to_internal_request to raise an exception
        with patch.object(
            serving_embedding,
            "_convert_to_internal_request",
            side_effect=Exception("Internal error"),
        ):
            response = await serving_embedding.handle_request(
                basic_embedding_request, mock_request
            )

            assert isinstance(response, ORJSONResponse)
            assert response.status_code == 500
