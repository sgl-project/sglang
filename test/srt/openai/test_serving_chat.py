"""
Unit-tests for OpenAIServingChat — rewritten to use only the std-lib 'unittest'.
Run with either:
    python tests/test_serving_chat_unit.py -v
or
    python -m unittest discover -s tests -p "test_*unit.py" -v
"""

import unittest
import uuid
from typing import Optional
from unittest.mock import Mock, patch

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.managers.io_struct import GenerateReqInput


class _MockTokenizerManager:
    """Minimal mock that satisfies OpenAIServingChat."""

    def __init__(self):
        self.model_config = Mock(is_multimodal=False)
        self.server_args = Mock(
            enable_cache_report=False,
            tool_call_parser="hermes",
            reasoning_parser=None,
        )
        self.chat_template_name: Optional[str] = "llama-3"

        # tokenizer stub
        self.tokenizer = Mock()
        self.tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        self.tokenizer.decode.return_value = "Test response"
        self.tokenizer.chat_template = None
        self.tokenizer.bos_token_id = 1

        # async generator stub for generate_request
        async def _mock_generate():
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

        self.generate_request = Mock(return_value=_mock_generate())
        self.create_abort_task = Mock()


class ServingChatTestCase(unittest.TestCase):
    # ------------- common fixtures -------------
    def setUp(self):
        self.tm = _MockTokenizerManager()
        self.chat = OpenAIServingChat(self.tm)

        # frequently reused requests
        self.basic_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=False,
        )
        self.stream_req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi?"}],
            temperature=0.7,
            max_tokens=100,
            stream=True,
        )

        self.fastapi_request = Mock(spec=Request)
        self.fastapi_request.headers = {}

    # ------------- conversion tests -------------
    def test_convert_to_internal_request_single(self):
        with patch(
            "sglang.srt.entrypoints.openai.serving_chat.generate_chat_conv"
        ) as conv_mock, patch.object(self.chat, "_process_messages") as proc_mock:
            conv_ins = Mock()
            conv_ins.get_prompt.return_value = "Test prompt"
            conv_ins.image_data = conv_ins.audio_data = None
            conv_ins.modalities = []
            conv_ins.stop_str = ["</s>"]
            conv_mock.return_value = conv_ins

            proc_mock.return_value = (
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,
            )

            adapted, processed = self.chat._convert_to_internal_request(self.basic_req)
            self.assertIsInstance(adapted, GenerateReqInput)
            self.assertFalse(adapted.stream)
            self.assertEqual(processed, self.basic_req)

    # # ------------- tool-call branch -------------
    # def test_tool_call_request_conversion(self):
    #     req = ChatCompletionRequest(
    #         model="x",
    #         messages=[{"role": "user", "content": "Weather?"}],
    #         tools=[
    #             {
    #                 "type": "function",
    #                 "function": {
    #                     "name": "get_weather",
    #                     "parameters": {"type": "object", "properties": {}},
    #                 },
    #             }
    #         ],
    #         tool_choice="auto",
    #     )

    #     with patch.object(
    #         self.chat,
    #         "_process_messages",
    #         return_value=("Prompt", [1, 2, 3], None, None, [], ["</s>"], None),
    #     ):
    #         adapted, _ = self.chat._convert_to_internal_request(req, "rid")
    #         self.assertEqual(adapted.rid, "rid")

    # def test_tool_choice_none(self):
    #     req = ChatCompletionRequest(
    #         model="x",
    #         messages=[{"role": "user", "content": "Hi"}],
    #         tools=[{"type": "function", "function": {"name": "noop"}}],
    #         tool_choice="none",
    #     )
    #     with patch.object(
    #         self.chat,
    #         "_process_messages",
    #         return_value=("Prompt", [1, 2, 3], None, None, [], ["</s>"], None),
    #     ):
    #         adapted, _ = self.chat._convert_to_internal_request(req, "rid")
    #         self.assertEqual(adapted.rid, "rid")

    # ------------- multimodal branch -------------
    def test_multimodal_request_with_images(self):
        self.tm.model_config.is_multimodal = True

        req = ChatCompletionRequest(
            model="x",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in the image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,"},
                        },
                    ],
                }
            ],
        )

        with patch.object(
            self.chat,
            "_apply_jinja_template",
            return_value=("prompt", [1, 2], ["img"], None, [], []),
        ), patch.object(
            self.chat,
            "_apply_conversation_template",
            return_value=("prompt", ["img"], None, [], []),
        ):
            out = self.chat._process_messages(req, True)
            _, _, image_data, *_ = out
            self.assertEqual(image_data, ["img"])

    # ------------- template handling -------------
    def test_jinja_template_processing(self):
        req = ChatCompletionRequest(
            model="x", messages=[{"role": "user", "content": "Hello"}]
        )
        self.tm.chat_template_name = None
        self.tm.tokenizer.chat_template = "<jinja>"

        with patch.object(
            self.chat,
            "_apply_jinja_template",
            return_value=("processed", [1], None, None, [], ["</s>"]),
        ), patch("builtins.hasattr", return_value=True):
            prompt, prompt_ids, *_ = self.chat._process_messages(req, False)
            self.assertEqual(prompt, "processed")
            self.assertEqual(prompt_ids, [1])

    # ------------- sampling-params -------------
    def test_sampling_param_build(self):
        req = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "Hi"}],
            temperature=0.8,
            max_tokens=150,
            min_tokens=5,
            top_p=0.9,
            stop=["</s>"],
        )
        with patch.object(
            self.chat,
            "_process_messages",
            return_value=("Prompt", [1], None, None, [], ["</s>"], None),
        ):
            params = self.chat._build_sampling_params(req, ["</s>"], None)
            self.assertEqual(params["temperature"], 0.8)
            self.assertEqual(params["max_new_tokens"], 150)
            self.assertEqual(params["min_new_tokens"], 5)
            self.assertEqual(params["stop"], ["</s>"])


<<<<<<< HEAD
if __name__ == "__main__":
    unittest.main(verbosity=2)
=======
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


class TestHiddenStates:
    """Test hidden states functionality"""

    def test_hidden_states_request_conversion_single(self, serving_chat):
        """Test request conversion with return_hidden_states=True for single request"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            return_hidden_states=True,
        )

        with patch.object(serving_chat, "_process_messages") as mock_process:
            mock_process.return_value = (
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,
            )

            adapted_request, _ = serving_chat._convert_to_internal_request(
                [request], ["test-id"]
            )

            assert adapted_request.return_hidden_states is True

    def test_hidden_states_request_conversion_multiple(self, serving_chat):
        """Test request conversion with return_hidden_states=True for multiple requests"""
        requests = [
            ChatCompletionRequest(
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
                return_hidden_states=True,
            ),
            ChatCompletionRequest(
                model="test-model",
                messages=[{"role": "user", "content": "World"}],
                return_hidden_states=False,
            ),
        ]

        with patch.object(serving_chat, "_process_messages") as mock_process:
            mock_process.return_value = (
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,
            )

            adapted_request, _ = serving_chat._convert_to_internal_request(
                requests, ["test-id-1", "test-id-2"]
            )

            assert adapted_request.return_hidden_states == [True, False]

    def test_hidden_states_non_streaming_response(self, serving_chat, mock_request):
        """Test hidden states in non-streaming response"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            return_hidden_states=True,
            chat_template_kwargs={"enable_thinking": True},
        )

        # Mock the generate_request to return hidden states
        async def mock_generate():
            yield {
                "text": "Test",
                "meta_info": {
                    "id": "chatcmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "finish_reason": None,
                    "output_token_logprobs": [],
                    "output_top_logprobs": None,
                    "hidden_states": [[0.1, 0.2, 0.3]],  # First token only
                },
                "index": 0,
            }
            yield {
                "text": " response",
                "meta_info": {
                    "id": "chatcmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [],
                    "output_top_logprobs": None,
                    "hidden_states": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # Both tokens
                },
                "index": 0,
            }

        serving_chat.tokenizer_manager.generate_request = Mock(return_value=mock_generate())

        with patch.object(serving_chat, "_process_messages") as mock_process:
            mock_process.return_value = (
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,
            )

            adapted_request, _ = serving_chat._convert_to_internal_request(
                [request], ["test-id"]
            )

            # Test the _build_chat_response method
            ret = [{
                "text": "Test response",
                "meta_info": {
                    "id": "chatcmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [],
                    "output_top_logprobs": None,
                    "hidden_states": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                },
            }]

            response = serving_chat._build_chat_response(
                request, ret, 1234567890
            )

            assert len(response.choices) == 1
            choice = response.choices[0]
            assert choice.hidden_states == [0.4, 0.5, 0.6]  # Should return last token's hidden states

    def test_hidden_states_non_streaming_response_no_hidden_states(self, serving_chat):
        """Test response when return_hidden_states=False"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            return_hidden_states=False,
            chat_template_kwargs={"enable_thinking": True},
        )

        ret = [{
            "text": "Test response",
            "meta_info": {
                "id": "chatcmpl-test",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cached_tokens": 0,
                "finish_reason": {"type": "stop", "matched": None},
                "output_token_logprobs": [],
                "output_top_logprobs": None,
            },
        }]

        response = serving_chat._build_chat_response(
            request, ret, 1234567890
        )

        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.hidden_states is None

    @pytest.mark.asyncio
    async def test_hidden_states_streaming_response(self, serving_chat, mock_request):
        """Test hidden states in streaming response"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            return_hidden_states=True,
            stream=True,
            chat_template_kwargs={"enable_thinking": True},
        )

        # Mock the generate_request to return hidden states
        async def mock_generate():
            yield {
                "text": "Test",
                "meta_info": {
                    "id": "chatcmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "finish_reason": None,
                    "output_token_logprobs": [],
                    "output_top_logprobs": None,
                    "hidden_states": [[0.1, 0.2, 0.3]],  # First token only
                },
                "index": 0,
            }
            yield {
                "text": " response",
                "meta_info": {
                    "id": "chatcmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [],
                    "output_top_logprobs": None,
                    "hidden_states": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # Both tokens
                },
                "index": 0,
            }

        serving_chat.tokenizer_manager.generate_request = Mock(return_value=mock_generate())

        with patch.object(serving_chat, "_process_messages") as mock_process:
            mock_process.return_value = (
                "Test prompt",
                [1, 2, 3],
                None,
                None,
                [],
                ["</s>"],
                None,
            )

            adapted_request, _ = serving_chat._convert_to_internal_request(
                [request], ["test-id"]
            )

            response = await serving_chat._handle_streaming_request(
                adapted_request, request, mock_request
            )

            # Collect all chunks from the streaming response
            chunks = []
            async for chunk in response.body_iterator:
                chunks.append(chunk)

            # Parse and validate chunks
            import json
            parsed_chunks = []
            for chunk in chunks:
                if chunk.startswith("data:") and chunk.strip() != "data: [DONE]":
                    try:
                        chunk_data = json.loads(chunk[6:])  # Remove "data: " prefix
                        parsed_chunks.append(chunk_data)
                    except json.JSONDecodeError:
                        # Skip chunks that can't be parsed as JSON
                        continue

            # Should have at least 3 chunks: role, hidden_states, and final content
            assert len(parsed_chunks) >= 3
            
            # First chunk should contain role
            assert parsed_chunks[0]["choices"][0]["delta"]["role"] == "assistant"
            
            # Find hidden states chunk
            hidden_states_found = False
            for chunk_data in parsed_chunks:
                delta = chunk_data["choices"][0]["delta"]
                if delta.get("hidden_states") is not None:
                    assert delta["hidden_states"] == [0.4, 0.5, 0.6]  # Last token hidden states
                    hidden_states_found = True
                    break
            
            assert hidden_states_found, "Hidden states should be present in streaming response"

    def test_hidden_states_multiple_choices(self, serving_chat):
        """Test hidden states with multiple choices (n > 1)"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            return_hidden_states=True,
            n=2,
            chat_template_kwargs={"enable_thinking": True},
        )

        ret = [
            {
                "text": "Response 1",
                "meta_info": {
                    "id": "chatcmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [],
                    "output_top_logprobs": None,
                    "hidden_states": [[0.1, 0.2], [0.3, 0.4]],
                },
            },
            {
                "text": "Response 2",
                "meta_info": {
                    "id": "chatcmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [],
                    "output_top_logprobs": None,
                    "hidden_states": [[0.5, 0.6], [0.7, 0.8]],
                },
            }
        ]

        response = serving_chat._build_chat_response(
            request, ret, 1234567890
        )

        assert len(response.choices) == 2
        assert response.choices[0].hidden_states == [0.3, 0.4]  # Last token for choice 0
        assert response.choices[1].hidden_states == [0.7, 0.8]  # Last token for choice 1

    def test_hidden_states_empty_list(self, serving_chat):
        """Test handling of empty hidden states list"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            return_hidden_states=True,
            chat_template_kwargs={"enable_thinking": True},
        )

        ret = [{
            "text": "Test response",
            "meta_info": {
                "id": "chatcmpl-test",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "cached_tokens": 0,
                "finish_reason": {"type": "stop", "matched": None},
                "output_token_logprobs": [],
                "output_top_logprobs": None,
                "hidden_states": [],  # Empty hidden states
            },
        }]

        response = serving_chat._build_chat_response(
            request, ret, 1234567890
        )

        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.hidden_states == []

    def test_hidden_states_single_token(self, serving_chat):
        """Test handling of hidden states with single token"""
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "Hello"}],
            return_hidden_states=True,
            chat_template_kwargs={"enable_thinking": True},
        )

        ret = [{
            "text": "Test",
            "meta_info": {
                "id": "chatcmpl-test",
                "prompt_tokens": 10,
                "completion_tokens": 1,
                "cached_tokens": 0,
                "finish_reason": {"type": "stop", "matched": None},
                "output_token_logprobs": [],
                "output_top_logprobs": None,
                "hidden_states": [[0.1, 0.2, 0.3]],  # Single token hidden states
            },
        }]

        response = serving_chat._build_chat_response(
            request, ret, 1234567890
        )

        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.hidden_states == []  # Should return empty list for single token (as per current logic)
>>>>>>> c4f0693 (Add hidden state support)
