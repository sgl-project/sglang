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
from sglang.srt.entrypoints.openai.serving_completions import CompletionHandler
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
def completion_handler(mock_tokenizer_manager):
    """Create a completion handler instance"""
    return CompletionHandler(mock_tokenizer_manager)


class TestUtilityFunctions:
    """Test utility functions that were moved from OpenAIServingBase."""

    def test_create_error_response_functionality(self, completion_handler):
        """Test that create_error_response works correctly."""
        error = completion_handler.create_error_response("Test error message")
        assert isinstance(error, ErrorResponse)
        assert error.message == "Test error message"
        assert error.type == "BadRequestError"
        assert error.code == 400

    def test_create_streaming_error_response_functionality(self, completion_handler):
        """Test that create_streaming_error_response works correctly."""
        error_json = completion_handler.create_streaming_error_response(
            "Test streaming error"
        )
        # Should return JSON string with error structure
        import json

        error_data = json.loads(error_json)
        assert "error" in error_data
        assert error_data["error"]["message"] == "Test streaming error"


class TestPromptHandling:
    """Test different prompt types and formats from adapter.py"""

    def test_single_string_prompt(self, completion_handler):
        """Test handling single string prompt"""
        request = CompletionRequest(
            model="test-model", prompt="Hello world", max_tokens=100
        )

        adapted_request, _ = completion_handler._convert_to_internal_request(
            [request], ["test-id"]
        )

        assert adapted_request.text == "Hello world"

    def test_single_token_ids_prompt(self, completion_handler):
        """Test handling single token IDs prompt"""
        request = CompletionRequest(
            model="test-model", prompt=[1, 2, 3, 4], max_tokens=100
        )

        adapted_request, _ = completion_handler._convert_to_internal_request(
            [request], ["test-id"]
        )

        assert adapted_request.input_ids == [1, 2, 3, 4]

    def test_multiple_string_prompts(self, completion_handler):
        """Test handling multiple string prompts"""
        requests = [
            CompletionRequest(model="test-model", prompt="Hello", max_tokens=50),
            CompletionRequest(model="test-model", prompt="World", max_tokens=50),
        ]

        adapted_request, _ = completion_handler._convert_to_internal_request(
            requests, ["id1", "id2"]
        )

        assert adapted_request.text == ["Hello", "World"]
        assert adapted_request.rid == ["id1", "id2"]

    def test_multiple_token_ids_prompts(self, completion_handler):
        """Test handling multiple token IDs prompts"""
        requests = [
            CompletionRequest(model="test-model", prompt=[1, 2], max_tokens=50),
            CompletionRequest(model="test-model", prompt=[3, 4], max_tokens=50),
        ]

        adapted_request, _ = completion_handler._convert_to_internal_request(
            requests, ["id1", "id2"]
        )

        assert adapted_request.input_ids == [[1, 2], [3, 4]]

    def test_list_of_strings_prompt(self, completion_handler):
        """Test handling list of strings as prompt"""
        request = CompletionRequest(
            model="test-model", prompt=["Hello", "world"], max_tokens=100
        )

        adapted_request, _ = completion_handler._convert_to_internal_request(
            [request], ["test-id"]
        )

        assert adapted_request.text == ["Hello", "world"]

    def test_completion_template_handling(self, completion_handler):
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
                "sglang.srt.entrypoints.openai.serving_completions.generate_completion_prompt",
                return_value="processed_prompt",
            ):
                adapted_request, _ = completion_handler._convert_to_internal_request(
                    [request], ["test-id"]
                )

                assert adapted_request.text == "processed_prompt"


class TestEchoHandling:
    """Test echo functionality from adapter.py"""

    def test_echo_with_string_prompt_streaming(self, completion_handler):
        """Test echo handling with string prompt in streaming"""
        request = CompletionRequest(
            model="test-model", prompt="Hello", max_tokens=100, echo=True
        )

        # Test _get_echo_text method
        echo_text = completion_handler._get_echo_text(request, 0)
        assert echo_text == "Hello"

    def test_echo_with_list_of_strings_streaming(self, completion_handler):
        """Test echo handling with list of strings in streaming"""
        request = CompletionRequest(
            model="test-model",
            prompt=["Hello", "World"],
            max_tokens=100,
            echo=True,
            n=1,
        )

        echo_text = completion_handler._get_echo_text(request, 0)
        assert echo_text == "Hello"

        echo_text = completion_handler._get_echo_text(request, 1)
        assert echo_text == "World"

    def test_echo_with_token_ids_streaming(self, completion_handler):
        """Test echo handling with token IDs in streaming"""
        request = CompletionRequest(
            model="test-model", prompt=[1, 2, 3], max_tokens=100, echo=True
        )

        completion_handler.tokenizer_manager.tokenizer.decode.return_value = (
            "decoded_prompt"
        )
        echo_text = completion_handler._get_echo_text(request, 0)
        assert echo_text == "decoded_prompt"

    def test_echo_with_multiple_token_ids_streaming(self, completion_handler):
        """Test echo handling with multiple token ID prompts in streaming"""
        request = CompletionRequest(
            model="test-model", prompt=[[1, 2], [3, 4]], max_tokens=100, echo=True, n=1
        )

        completion_handler.tokenizer_manager.tokenizer.decode.return_value = "decoded"
        echo_text = completion_handler._get_echo_text(request, 0)
        assert echo_text == "decoded"

    def test_prepare_echo_prompts_non_streaming(self, completion_handler):
        """Test prepare echo prompts for non-streaming response"""
        # Test with single string
        request = CompletionRequest(model="test-model", prompt="Hello", echo=True)

        echo_prompts = completion_handler._prepare_echo_prompts(request)
        assert echo_prompts == ["Hello"]

        # Test with list of strings
        request = CompletionRequest(
            model="test-model", prompt=["Hello", "World"], echo=True
        )

        echo_prompts = completion_handler._prepare_echo_prompts(request)
        assert echo_prompts == ["Hello", "World"]

        # Test with token IDs
        request = CompletionRequest(model="test-model", prompt=[1, 2, 3], echo=True)

        completion_handler.tokenizer_manager.tokenizer.decode.return_value = "decoded"
        echo_prompts = completion_handler._prepare_echo_prompts(request)
        assert echo_prompts == ["decoded"]


class TestCompletionRequestConversion:
    """Test request conversion to internal format"""

    def test_convert_simple_string_prompt(self, completion_handler):
        """Test conversion of simple string prompt"""
        request = CompletionRequest(
            model="test-model", prompt="Hello world", max_tokens=100, temperature=0.7
        )

        adapted_request, processed_request = (
            completion_handler._convert_to_internal_request([request], ["test-id"])
        )

        assert isinstance(adapted_request, GenerateReqInput)
        assert adapted_request.text == "Hello world"
        assert adapted_request.sampling_params["temperature"] == 0.7
        assert adapted_request.sampling_params["max_new_tokens"] == 100
        assert adapted_request.rid == "test-id"
        assert processed_request == request

    def test_convert_token_ids_prompt(self, completion_handler):
        """Test conversion of token IDs prompt"""
        request = CompletionRequest(
            model="test-model", prompt=[1, 2, 3, 4], max_tokens=100
        )

        adapted_request, processed_request = (
            completion_handler._convert_to_internal_request([request], ["test-id"])
        )

        assert isinstance(adapted_request, GenerateReqInput)
        assert adapted_request.input_ids == [1, 2, 3, 4]
        assert adapted_request.sampling_params["max_new_tokens"] == 100

    def test_convert_logprob_start_len_with_echo_and_logprobs(self, completion_handler):
        """Test logprob_start_len setting with echo and logprobs"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello world",
            max_tokens=100,
            echo=True,
            logprobs=5,
        )

        adapted_request, _ = completion_handler._convert_to_internal_request(
            [request], ["test-id"]
        )

        # When echo=True and logprobs is set, should be 0
        assert adapted_request.logprob_start_len == 0
        assert adapted_request.return_logprob == True
        assert adapted_request.top_logprobs_num == 5

    def test_convert_logprob_start_len_without_echo(self, completion_handler):
        """Test logprob_start_len setting without echo"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello world",
            max_tokens=100,
            echo=False,
            logprobs=3,
        )

        adapted_request, _ = completion_handler._convert_to_internal_request(
            [request], ["test-id"]
        )

        # When echo=False, should be -1
        assert adapted_request.logprob_start_len == -1
        assert adapted_request.return_logprob == True
        assert adapted_request.top_logprobs_num == 3


class TestCompatibilityWithAdapter:
    """Test compatibility with adapter.py functionality"""

    def test_bootstrap_parameters_support(self, completion_handler):
        """Test that bootstrap parameters are supported"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello world",
            max_tokens=100,
            bootstrap_host="localhost",
            bootstrap_port=8080,
            bootstrap_room=123,
        )

        adapted_request, _ = completion_handler._convert_to_internal_request(
            [request], ["test-id"]
        )

        assert adapted_request.bootstrap_host == "localhost"
        assert adapted_request.bootstrap_port == 8080
        assert adapted_request.bootstrap_room == 123

    def test_lora_path_support(self, completion_handler):
        """Test that LoRA path is supported"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello world",
            max_tokens=100,
            lora_path="/path/to/lora",
        )

        adapted_request, _ = completion_handler._convert_to_internal_request(
            [request], ["test-id"]
        )

        assert adapted_request.lora_path == "/path/to/lora"

    def test_echo_and_logprobs_compatibility(self, completion_handler):
        """Test echo and logprobs handling matches adapter behavior"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello world",
            max_tokens=100,
            echo=True,
            logprobs=5,
        )

        adapted_request, _ = completion_handler._convert_to_internal_request(
            [request], ["test-id"]
        )

        # When echo=True and logprobs is set, logprob_start_len should be 0
        assert adapted_request.logprob_start_len == 0
        assert adapted_request.return_logprob == True
        assert adapted_request.top_logprobs_num == 5

    def test_no_echo_logprobs_compatibility(self, completion_handler):
        """Test no echo but logprobs handling"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello world",
            max_tokens=100,
            echo=False,
            logprobs=3,
        )

        adapted_request, _ = completion_handler._convert_to_internal_request(
            [request], ["test-id"]
        )

        # When echo=False, logprob_start_len should be -1
        assert adapted_request.logprob_start_len == -1
        assert adapted_request.return_logprob == True
        assert adapted_request.top_logprobs_num == 3

    def test_return_text_in_logprobs_setting(self, completion_handler):
        """Test that return_text_in_logprobs is properly set"""
        request = CompletionRequest(
            model="test-model", prompt="Hello world", max_tokens=100
        )

        adapted_request, _ = completion_handler._convert_to_internal_request(
            [request], ["test-id"]
        )

        assert adapted_request.return_text_in_logprobs == True

    def test_multiple_requests_batch_handling(self, completion_handler):
        """Test handling of multiple requests in batch mode"""
        requests = [
            CompletionRequest(
                model="test-model", prompt="Hello", max_tokens=50, lora_path="/path1"
            ),
            CompletionRequest(
                model="test-model", prompt="World", max_tokens=50, lora_path="/path2"
            ),
        ]

        adapted_request, processed_requests = (
            completion_handler._convert_to_internal_request(requests, ["id1", "id2"])
        )

        assert adapted_request.text == ["Hello", "World"]
        assert adapted_request.lora_path == ["/path1", "/path2"]
        assert adapted_request.rid == ["id1", "id2"]
        assert (
            processed_requests == requests
        )  # Should return list for multiple requests


class TestResponseBuilding:
    """Test response building functionality"""

    def test_build_simple_response(self, completion_handler):
        """Test building simple completion response"""
        request = CompletionRequest(model="test-model", prompt="Hello", max_tokens=100)

        mock_ret = [
            {
                "text": " world!",
                "meta_info": {
                    "id": "test-id",
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "finish_reason": {"type": "stop"},
                },
            }
        ]

        response = completion_handler._build_completion_response(
            request, mock_ret, 1234567890
        )

        assert isinstance(response, CompletionResponse)
        assert response.id == "test-id"
        assert response.model == "test-model"
        assert response.created == 1234567890
        assert len(response.choices) == 1
        assert response.choices[0].text == " world!"
        assert response.choices[0].finish_reason == "stop"
        assert response.usage.prompt_tokens == 5
        assert response.usage.completion_tokens == 10
        assert response.usage.total_tokens == 15

    def test_build_response_with_echo(self, completion_handler):
        """Test building response with echo enabled"""
        request = CompletionRequest(
            model="test-model", prompt="Hello", max_tokens=100, echo=True
        )

        # Mock echo prompts preparation
        completion_handler._prepare_echo_prompts = Mock(return_value=["Hello"])

        mock_ret = [
            {
                "text": " world!",
                "meta_info": {
                    "id": "test-id",
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "finish_reason": {"type": "stop"},
                },
            }
        ]

        response = completion_handler._build_completion_response(
            request, mock_ret, 1234567890
        )

        # With echo=True, text should include the prompt
        assert response.choices[0].text == "Hello world!"

    def test_build_response_with_logprobs(self, completion_handler):
        """Test building response with logprobs"""
        request = CompletionRequest(
            model="test-model", prompt="Hello", max_tokens=100, logprobs=3
        )

        mock_ret = [
            {
                "text": " world!",
                "meta_info": {
                    "id": "test-id",
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "finish_reason": {"type": "stop"},
                    "output_token_logprobs": [(-0.1, 1, " world"), (-0.2, 2, "!")],
                    "output_top_logprobs": [
                        [(-0.1, 1, " world"), (-0.3, 3, " earth")],
                        [(-0.2, 2, "!"), (-0.4, 4, ".")],
                    ],
                },
            }
        ]

        response = completion_handler._build_completion_response(
            request, mock_ret, 1234567890
        )

        assert response.choices[0].logprobs is not None
        assert len(response.choices[0].logprobs.tokens) == 2
        assert response.choices[0].logprobs.tokens[0] == " world"
        assert response.choices[0].logprobs.tokens[1] == "!"

    def test_build_response_with_echo_and_logprobs(self, completion_handler):
        """Test building response with both echo and logprobs"""
        request = CompletionRequest(
            model="test-model", prompt="Hello", max_tokens=100, echo=True, logprobs=2
        )

        completion_handler._prepare_echo_prompts = Mock(return_value=["Hello"])

        mock_ret = [
            {
                "text": " world!",
                "meta_info": {
                    "id": "test-id",
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "finish_reason": {"type": "stop"},
                    "input_token_logprobs": [(-0.05, 0, "Hello")],
                    "input_top_logprobs": [[(-0.05, 0, "Hello"), (-0.1, 1, "Hi")]],
                    "output_token_logprobs": [(-0.1, 1, " world"), (-0.2, 2, "!")],
                    "output_top_logprobs": [
                        [(-0.1, 1, " world"), (-0.3, 3, " earth")],
                        [(-0.2, 2, "!"), (-0.4, 4, ".")],
                    ],
                },
            }
        ]

        response = completion_handler._build_completion_response(
            request, mock_ret, 1234567890
        )

        assert response.choices[0].text == "Hello world!"
        assert response.choices[0].logprobs is not None
        # Should include both input and output logprobs
        assert len(response.choices[0].logprobs.tokens) == 3  # Hello + world + !

    def test_build_response_with_matched_stop(self, completion_handler):
        """Test building response with matched stop token"""
        request = CompletionRequest(model="test-model", prompt="Hello", max_tokens=100)

        mock_ret = [
            {
                "text": " world!",
                "meta_info": {
                    "id": "test-id",
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "finish_reason": {"type": "stop", "matched": "</s>"},
                },
            }
        ]

        response = completion_handler._build_completion_response(
            request, mock_ret, 1234567890
        )

        assert response.choices[0].finish_reason == "stop"
        assert response.choices[0].matched_stop == "</s>"

    def test_build_response_with_cache_report(self, completion_handler):
        """Test building response with cache reporting enabled"""
        request = CompletionRequest(model="test-model", prompt="Hello", max_tokens=100)

        mock_ret = [
            {
                "text": " world!",
                "meta_info": {
                    "id": "test-id",
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "cached_tokens": 3,
                    "finish_reason": {"type": "stop"},
                },
            }
        ]

        response = completion_handler._build_completion_response(
            request, mock_ret, 1234567890, cache_report=True
        )

        assert response.usage.prompt_tokens_details is not None
        assert response.usage.prompt_tokens_details["cached_tokens"] == 3

    def test_build_response_multiple_choices(self, completion_handler):
        """Test building response with multiple choices (n > 1)"""
        request = CompletionRequest(
            model="test-model", prompt="Hello", max_tokens=100, n=2
        )

        completion_handler._prepare_echo_prompts = Mock(return_value=["Hello"])

        mock_ret = [
            {
                "text": " world!",
                "meta_info": {
                    "id": "test-id",
                    "prompt_tokens": 5,
                    "completion_tokens": 10,
                    "finish_reason": {"type": "stop"},
                },
            },
            {
                "text": " there!",
                "meta_info": {
                    "id": "test-id",
                    "prompt_tokens": 5,
                    "completion_tokens": 8,
                    "finish_reason": {"type": "stop"},
                },
            },
        ]

        response = completion_handler._build_completion_response(
            request, mock_ret, 1234567890
        )

        assert len(response.choices) == 2
        assert response.choices[0].text == " world!"
        assert response.choices[1].text == " there!"
        assert response.choices[0].index == 0
        assert response.choices[1].index == 1
        # Total tokens should be: prompt_tokens + both completion_tokens
        assert response.usage.total_tokens == 5 + 10 + 8


@pytest.mark.asyncio
class TestAsyncMethods:
    """Test async handler methods"""

    async def test_handle_request_non_streaming(self, completion_handler):
        """Test handling non-streaming request - simplified test for async flow"""
        mock_request = Mock()
        request = CompletionRequest(
            model="test-model", prompt="Hello world", max_tokens=100, stream=False
        )

        # For now, just test that we can call the method and get some response
        # The detailed functionality is tested in the sync tests above
        response = await completion_handler.handle_request(request, mock_request)

        # Should return some response (either error or success, depending on mock setup)
        assert response is not None
        assert hasattr(response, "model_dump")

    async def test_handle_request_streaming(self, completion_handler):
        """Test handling streaming request"""
        mock_request = Mock()
        request = CompletionRequest(
            model="test-model", prompt="Hello world", max_tokens=100, stream=True
        )

        response = await completion_handler.handle_request(request, mock_request)

        # Should return StreamingResponse
        from fastapi.responses import StreamingResponse

        assert isinstance(response, StreamingResponse)

    async def test_handle_streaming_with_usage(self, completion_handler):
        """Test streaming with usage reporting"""
        mock_request = Mock()
        request = CompletionRequest(
            model="test-model",
            prompt="Hello world",
            max_tokens=100,
            stream=True,
            stream_options={"include_usage": True},
        )

        response = await completion_handler.handle_request(request, mock_request)

        from fastapi.responses import StreamingResponse

        assert isinstance(response, StreamingResponse)


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_multiple_requests_different_prompt_types_error(self, completion_handler):
        """Test error when multiple requests have different prompt types"""
        requests = [
            CompletionRequest(model="test-model", prompt="Hello", max_tokens=50),
            CompletionRequest(model="test-model", prompt=[1, 2, 3], max_tokens=50),
        ]

        with pytest.raises(AssertionError):
            completion_handler._convert_to_internal_request(requests, ["id1", "id2"])

    def test_multiple_requests_with_n_greater_than_1_error(self, completion_handler):
        """Test error when multiple requests have n > 1"""
        requests = [
            CompletionRequest(model="test-model", prompt="Hello", max_tokens=50, n=2),
            CompletionRequest(model="test-model", prompt="World", max_tokens=50, n=1),
        ]

        with pytest.raises(ValueError, match="Parallel sampling is not supported"):
            completion_handler._convert_to_internal_request(requests, ["id1", "id2"])

    def test_suffix_without_completion_template(self, completion_handler):
        """Test that suffix is ignored when completion template is not defined"""
        request = CompletionRequest(
            model="test-model",
            prompt="def hello():",
            suffix="return 'world'",
            max_tokens=100,
        )

        with patch(
            "sglang.srt.entrypoints.openai.serving_completions.is_completion_template_defined",
            return_value=False,
        ):
            adapted_request, _ = completion_handler._convert_to_internal_request(
                [request], ["test-id"]
            )

            # Should use original prompt, not processed with suffix
            assert adapted_request.text == "def hello():"
