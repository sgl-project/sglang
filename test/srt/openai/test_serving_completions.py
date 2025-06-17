"""
Tests for the refactored completions serving handler
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from sglang.srt.entrypoints.openai.protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionStreamResponse,
    ErrorResponse,
)
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import TokenizerManager


@pytest.fixture
def mock_tokenizer_manager():
    """Create a mock tokenizer manager"""
    manager = Mock(spec=TokenizerManager)

    # Mock tokenizer
    manager.tokenizer = Mock()
    manager.tokenizer.encode = Mock(return_value=[1, 2, 3, 4])
    manager.tokenizer.decode = Mock(return_value="decoded text")
    manager.tokenizer.bos_token_id = 1

    # Mock model config
    manager.model_config = Mock()
    manager.model_config.is_multimodal = False

    # Mock server args
    manager.server_args = Mock()
    manager.server_args.enable_cache_report = False

    # Mock generation
    manager.generate_request = AsyncMock()
    manager.create_abort_task = Mock(return_value=None)

    return manager


@pytest.fixture
def serving_completion(mock_tokenizer_manager):
    """Create a OpenAIServingCompletion instance"""
    return OpenAIServingCompletion(mock_tokenizer_manager)


class TestPromptHandling:
    """Test different prompt types and formats from adapter.py"""

    def test_single_string_prompt(self, serving_completion):
        """Test handling single string prompt"""
        request = CompletionRequest(
            model="test-model", prompt="Hello world", max_tokens=100
        )

        adapted_request, _ = serving_completion._convert_to_internal_request(
            [request], ["test-id"]
        )

        assert adapted_request.text == "Hello world"

    def test_single_token_ids_prompt(self, serving_completion):
        """Test handling single token IDs prompt"""
        request = CompletionRequest(
            model="test-model", prompt=[1, 2, 3, 4], max_tokens=100
        )

        adapted_request, _ = serving_completion._convert_to_internal_request(
            [request], ["test-id"]
        )

        assert adapted_request.input_ids == [1, 2, 3, 4]

    def test_completion_template_handling(self, serving_completion):
        """Test completion template processing"""
        request = CompletionRequest(
            model="test-model",
            prompt="def hello():",
            suffix="return 'world'",
            max_tokens=100,
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_completions.is_completion_template_defined",
            return_value=True,
        ):
            with patch(
                "sglang.srt.entrypoints.openai.serving_completions.generate_completion_prompt_from_request",
                return_value="processed_prompt",
            ):
                adapted_request, _ = serving_completion._convert_to_internal_request(
                    [request], ["test-id"]
                )

                assert adapted_request.text == "processed_prompt"


class TestEchoHandling:
    """Test echo functionality from adapter.py"""

    def test_echo_with_string_prompt_streaming(self, serving_completion):
        """Test echo handling with string prompt in streaming"""
        request = CompletionRequest(
            model="test-model", prompt="Hello", max_tokens=100, echo=True
        )

        # Test _get_echo_text method
        echo_text = serving_completion._get_echo_text(request, 0)
        assert echo_text == "Hello"

    def test_echo_with_list_of_strings_streaming(self, serving_completion):
        """Test echo handling with list of strings in streaming"""
        request = CompletionRequest(
            model="test-model",
            prompt=["Hello", "World"],
            max_tokens=100,
            echo=True,
            n=1,
        )

        echo_text = serving_completion._get_echo_text(request, 0)
        assert echo_text == "Hello"

        echo_text = serving_completion._get_echo_text(request, 1)
        assert echo_text == "World"

    def test_echo_with_token_ids_streaming(self, serving_completion):
        """Test echo handling with token IDs in streaming"""
        request = CompletionRequest(
            model="test-model", prompt=[1, 2, 3], max_tokens=100, echo=True
        )

        serving_completion.tokenizer_manager.tokenizer.decode.return_value = (
            "decoded_prompt"
        )
        echo_text = serving_completion._get_echo_text(request, 0)
        assert echo_text == "decoded_prompt"

    def test_echo_with_multiple_token_ids_streaming(self, serving_completion):
        """Test echo handling with multiple token ID prompts in streaming"""
        request = CompletionRequest(
            model="test-model", prompt=[[1, 2], [3, 4]], max_tokens=100, echo=True, n=1
        )

        serving_completion.tokenizer_manager.tokenizer.decode.return_value = "decoded"
        echo_text = serving_completion._get_echo_text(request, 0)
        assert echo_text == "decoded"

    def test_prepare_echo_prompts_non_streaming(self, serving_completion):
        """Test prepare echo prompts for non-streaming response"""
        # Test with single string
        request = CompletionRequest(model="test-model", prompt="Hello", echo=True)

        echo_prompts = serving_completion._prepare_echo_prompts(request)
        assert echo_prompts == ["Hello"]

        # Test with list of strings
        request = CompletionRequest(
            model="test-model", prompt=["Hello", "World"], echo=True
        )

        echo_prompts = serving_completion._prepare_echo_prompts(request)
        assert echo_prompts == ["Hello", "World"]

        # Test with token IDs
        request = CompletionRequest(model="test-model", prompt=[1, 2, 3], echo=True)

        serving_completion.tokenizer_manager.tokenizer.decode.return_value = "decoded"
        echo_prompts = serving_completion._prepare_echo_prompts(request)
        assert echo_prompts == ["decoded"]
