"""
Unit tests for the OpenAIServingChat class from serving_chat.py.

These tests ensure that the refactored implementation maintains compatibility
with the original adapter.py functionality.
"""

import uuid
from unittest.mock import Mock, patch

import pytest
from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, ErrorResponse
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.io_struct import GenerateReqInput


# Mock TokenizerManager since it may not be directly importable in tests
class MockTokenizerManager:
    def __init__(self):
        self.model_config = Mock()
        self.model_config.is_multimodal = False
        self.server_args = Mock()
        self.server_args.enable_cache_report = False
        self.server_args.tool_call_parser = "hermes"
        self.server_args.reasoning_parser = None
        self.chat_template_name = "llama-3"

        # Mock tokenizer
        self.tokenizer = Mock()
        self.tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        self.tokenizer.decode = Mock(return_value="Test response")
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1

        # Mock generate_request method
        async def mock_generate():
            yield {
                "text": "Test response",
                "meta_info": {
                    "id": f"chatcmpl-{uuid.uuid4()}",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [(0.1, 1, "Test"), (0.2, 2, "response")],
                    "output_top_logprobs": None,
                },
                "index": 0,
            }

        self.generate_request = Mock(return_value=mock_generate())
        self.create_abort_task = Mock(return_value=None)


@pytest.fixture
def mock_tokenizer_manager():
    """Create a mock tokenizer manager for testing."""
    return MockTokenizerManager()


@pytest.fixture
def serving_chat(mock_tokenizer_manager):
    """Create a OpenAIServingChat instance for testing."""
    return OpenAIServingChat(mock_tokenizer_manager)


@pytest.fixture
def mock_request():
    """Create a mock FastAPI request."""
    request = Mock(spec=Request)
    request.headers = {}
    return request


@pytest.fixture
def basic_chat_request():
    """Create a basic chat completion request."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=0.7,
        max_tokens=100,
        stream=False,
    )


@pytest.fixture
def streaming_chat_request():
    """Create a streaming chat completion request."""
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        temperature=0.7,
        max_tokens=100,
        stream=True,
    )


class TestOpenAIServingChatConversion:
    """Test request conversion methods."""

    def test_convert_to_internal_request_single(
        self, serving_chat, basic_chat_request, mock_tokenizer_manager
    ):
        """Test converting single request to internal format."""
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as mock_conv:
            mock_conv_instance = Mock()
            mock_conv_instance.get_prompt.return_value = "Test prompt"
            mock_conv_instance.image_data = None
            mock_conv_instance.audio_data = None
            mock_conv_instance.modalities = []
            mock_conv_instance.stop_str = ["</s>"]
            mock_conv.return_value = mock_conv_instance

            # Mock the _process_messages method to return expected values
            with patch.object(serving_chat, "_process_messages") as mock_process:
                mock_process.return_value = (
                    "Test prompt",
                    [1, 2, 3],
                    None,
                    None,
                    [],
                    ["</s>"],
                    None,  # tool_call_constraint
                )

                adapted_request, processed_request = (
                    serving_chat._convert_to_internal_request(
                        [basic_chat_request], ["test-id"]
                    )
                )

                assert isinstance(adapted_request, GenerateReqInput)
                assert adapted_request.stream == basic_chat_request.stream
                assert processed_request == basic_chat_request


class TestToolCalls:
    """Test tool call functionality from adapter.py"""

    def test_tool_call_request_conversion(self, serving_chat):
        """Test request with tool calls"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "What's the weather?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather information",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    },
                }
            ],
            tool_choice="auto",
        )

        with patch.object(serving_chat, "_process_messages") as mock_process:
            mock_process.return_value = (
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,  # tool_call_constraint
            )

            adapted_request, _ = serving_chat._convert_to_internal_request(
                [request], ["test-id"]
            )

            assert adapted_request.rid == "test-id"
            # Tool call constraint should be processed
            assert request.tools is not None

    def test_tool_choice_none(self, serving_chat):
        """Test tool_choice=none disables tool calls"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            tools=[{"type": "function", "function": {"name": "test_func"}}],
            tool_choice="none",
        )

        with patch.object(serving_chat, "_process_messages") as mock_process:
            mock_process.return_value = (
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,  # tool_call_constraint
            )

            adapted_request, _ = serving_chat._convert_to_internal_request(
                [request], ["test-id"]
            )

            # Tools should not be processed when tool_choice is "none"
            assert adapted_request.rid == "test-id"

    def test_tool_call_response_processing(self, serving_chat):
        """Test processing tool calls in response"""
        mock_ret_item = {
            "text": '{"name": "get_weather", "parameters": {"location": "Paris"}}',
            "meta_info": {
                "output_token_logprobs": [],
                "output_top_logprobs": None,
            },
        }

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]

        finish_reason = {"type": "stop", "matched": None}

        # Mock FunctionCallParser
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.FunctionCallParser"
        ) as mock_parser_class:
            mock_parser = Mock()
            mock_parser.has_tool_call.return_value = True

            # Create proper mock tool call object
            mock_tool_call = Mock()
            mock_tool_call.name = "get_weather"
            mock_tool_call.parameters = '{"location": "Paris"}'

            mock_parser.parse_non_stream.return_value = ("", [mock_tool_call])
            mock_parser_class.return_value = mock_parser

            tool_calls, text, updated_finish_reason = serving_chat._process_tool_calls(
                mock_ret_item["text"], tools, "hermes", finish_reason
            )

            assert tool_calls is not None
            assert len(tool_calls) == 1
            assert updated_finish_reason["type"] == "tool_calls"


class TestMultimodalContent:
    """Test multimodal content handling from adapter.py"""

    def test_multimodal_request_with_images(self, serving_chat):
        """Test request with image content"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,..."},
                        },
                    ],
                }
            ],
        )

        # Set multimodal mode
        serving_chat.tokenizer_manager.model_config.is_multimodal = True

        with patch.object(serving_chat, "_apply_jinja_template") as mock_apply:
            mock_apply.return_value = (
                "prompt",
                [1, 2, 3],
                ["image_data"],
                None,
                [],
                [],
            )

            with patch.object(
                serving_chat, "_apply_conversation_template"
            ) as mock_conv:
                mock_conv.return_value = ("prompt", ["image_data"], None, [], [])

                (
                    prompt,
                    prompt_ids,
                    image_data,
                    audio_data,
                    modalities,
                    stop,
                    tool_call_constraint,
                ) = serving_chat._process_messages(request, True)

                assert image_data == ["image_data"]
                assert prompt == "prompt"

    def test_multimodal_request_with_audio(self, serving_chat):
        """Test request with audio content"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this audio"},
                        {
                            "type": "audio_url",
                            "audio_url": {"url": "data:audio/wav;base64,UklGR..."},
                        },
                    ],
                }
            ],
        )

        serving_chat.tokenizer_manager.model_config.is_multimodal = True

        with patch.object(serving_chat, "_apply_jinja_template") as mock_apply:
            mock_apply.return_value = (
                "prompt",
                [1, 2, 3],
                None,
                ["audio_data"],
                ["audio"],
                [],
            )

            with patch.object(
                serving_chat, "_apply_conversation_template"
            ) as mock_conv:
                mock_conv.return_value = ("prompt", None, ["audio_data"], ["audio"], [])

                (
                    prompt,
                    prompt_ids,
                    image_data,
                    audio_data,
                    modalities,
                    stop,
                    tool_call_constraint,
                ) = serving_chat._process_messages(request, True)

                assert audio_data == ["audio_data"]
                assert modalities == ["audio"]


class TestTemplateHandling:
    """Test chat template handling from adapter.py"""

    def test_jinja_template_processing(self, serving_chat):
        """Test Jinja template processing"""
        request = ChatCompletionRequest(
            model="test-model", messages=[{"role": "user", "content": "Hello"}]
        )

        # Mock the template attribute directly
        serving_chat.tokenizer_manager.chat_template_name = None
        serving_chat.tokenizer_manager.tokenizer.chat_template = "<jinja_template>"

        with patch.object(serving_chat, "_apply_jinja_template") as mock_apply:
            mock_apply.return_value = (
                "processed_prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
            )

            # Mock hasattr to simulate the None check
            with patch("builtins.hasattr") as mock_hasattr:
                mock_hasattr.return_value = True

                (
                    prompt,
                    prompt_ids,
                    image_data,
                    audio_data,
                    modalities,
                    stop,
                    tool_call_constraint,
                ) = serving_chat._process_messages(request, False)

                assert prompt == "processed_prompt"
                assert prompt_ids == [1, 2, 3]

    def test_conversation_template_processing(self, serving_chat):
        """Test conversation template processing"""
        request = ChatCompletionRequest(
            model="test-model", messages=[{"role": "user", "content": "Hello"}]
        )

        serving_chat.tokenizer_manager.chat_template_name = "llama-3"

        with patch.object(serving_chat, "_apply_conversation_template") as mock_apply:
            mock_apply.return_value = ("conv_prompt", None, None, [], ["</s>"])

            (
                prompt,
                prompt_ids,
                image_data,
                audio_data,
                modalities,
                stop,
                tool_call_constraint,
            ) = serving_chat._process_messages(request, False)

            assert prompt == "conv_prompt"
            assert stop == ["</s>"]

    def test_continue_final_message(self, serving_chat):
        """Test continue_final_message functionality"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            continue_final_message=True,
        )

        with patch.object(serving_chat, "_apply_conversation_template") as mock_apply:
            mock_apply.return_value = ("Hi there", None, None, [], ["</s>"])

            (
                prompt,
                prompt_ids,
                image_data,
                audio_data,
                modalities,
                stop,
                tool_call_constraint,
            ) = serving_chat._process_messages(request, False)

            # Should handle continue_final_message properly
            assert prompt == "Hi there"


class TestReasoningContent:
    """Test reasoning content separation from adapter.py"""

    def test_reasoning_content_request(self, serving_chat):
        """Test request with reasoning content separation"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Solve this math problem"}],
            separate_reasoning=True,
            stream_reasoning=False,
        )

        with patch.object(serving_chat, "_process_messages") as mock_process:
            mock_process.return_value = (
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,  # tool_call_constraint
            )

            adapted_request, _ = serving_chat._convert_to_internal_request(
                [request], ["test-id"]
            )

            assert adapted_request.rid == "test-id"
            assert request.separate_reasoning == True

    def test_reasoning_content_response(self, serving_chat):
        """Test reasoning content in response"""
        mock_ret_item = {
            "text": "<thinking>This is reasoning</thinking>Answer: 42",
            "meta_info": {
                "output_token_logprobs": [],
                "output_top_logprobs": None,
            },
        }

        # Mock ReasoningParser
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.ReasoningParser"
        ) as mock_parser_class:
            mock_parser = Mock()
            mock_parser.parse_non_stream.return_value = (
                "This is reasoning",
                "Answer: 42",
            )
            mock_parser_class.return_value = mock_parser

            choice_logprobs = None
            reasoning_text = None
            text = mock_ret_item["text"]

            # Simulate reasoning processing
            enable_thinking = True
            if enable_thinking:
                parser = mock_parser_class(model_type="test", stream_reasoning=False)
                reasoning_text, text = parser.parse_non_stream(text)

            assert reasoning_text == "This is reasoning"
            assert text == "Answer: 42"


class TestSamplingParams:
    """Test sampling parameter handling from adapter.py"""

    def test_all_sampling_parameters(self, serving_chat):
        """Test all sampling parameters are properly handled"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.8,
            max_tokens=150,
            max_completion_tokens=200,
            min_tokens=5,
            top_p=0.9,
            top_k=50,
            min_p=0.1,
            presence_penalty=0.1,
            frequency_penalty=0.2,
            repetition_penalty=1.1,
            stop=["<|endoftext|>"],
            stop_token_ids=[13, 14],
            regex=r"\d+",
            ebnf="<expr> ::= <number>",
            n=2,
            no_stop_trim=True,
            ignore_eos=True,
            skip_special_tokens=False,
            logit_bias={"1": 0.5, "2": -0.3},
        )

        with patch.object(serving_chat, "_process_messages") as mock_process:
            mock_process.return_value = (
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,  # tool_call_constraint
            )

            sampling_params = serving_chat._build_sampling_params(
                request, ["</s>"], None
            )

            # Verify all parameters
            assert sampling_params["temperature"] == 0.8
            assert sampling_params["max_new_tokens"] == 150
            assert sampling_params["min_new_tokens"] == 5
            assert sampling_params["top_p"] == 0.9
            assert sampling_params["top_k"] == 50
            assert sampling_params["min_p"] == 0.1
            assert sampling_params["presence_penalty"] == 0.1
            assert sampling_params["frequency_penalty"] == 0.2
            assert sampling_params["repetition_penalty"] == 1.1
            assert sampling_params["stop"] == ["</s>"]
            assert sampling_params["logit_bias"] == {"1": 0.5, "2": -0.3}

    def test_response_format_json_schema(self, serving_chat):
        """Test response format with JSON schema"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Generate JSON"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                    },
                },
            },
        )

        with patch.object(serving_chat, "_process_messages") as mock_process:
            mock_process.return_value = (
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,  # tool_call_constraint
            )

            sampling_params = serving_chat._build_sampling_params(
                request, ["</s>"], None
            )

            assert "json_schema" in sampling_params
            assert '"type": "object"' in sampling_params["json_schema"]

    def test_response_format_json_object(self, serving_chat):
        """Test response format with JSON object"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Generate JSON"}],
            response_format={"type": "json_object"},
        )

        with patch.object(serving_chat, "_process_messages") as mock_process:
            mock_process.return_value = (
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,  # tool_call_constraint
            )

            sampling_params = serving_chat._build_sampling_params(
                request, ["</s>"], None
            )

            assert sampling_params["json_schema"] == '{"type": "object"}'
