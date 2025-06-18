"""
Unit-tests for the refactored completions-serving handler (no pytest).
Run with:
    python -m unittest tests.test_serving_completions_unit -v
"""

import unittest
from unittest.mock import AsyncMock, Mock, patch

from sglang.srt.entrypoints.openai.protocol import CompletionRequest
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.srt.managers.tokenizer_manager import TokenizerManager


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

        self.sc = OpenAIServingCompletion(tm)
        self.mock_request = Mock()

    # ---------- prompt-handling ----------
    def test_single_string_prompt(self):
        req = CompletionRequest(model="x", prompt="Hello world", max_tokens=100)
        internal, _ = self.sc._convert_to_internal_request(req)
        self.assertEqual(internal.text, "Hello world")

    def test_single_token_ids_prompt(self):
        req = CompletionRequest(model="x", prompt=[1, 2, 3, 4], max_tokens=100)
        internal, _ = self.sc._convert_to_internal_request(req)
        self.assertEqual(internal.input_ids, [1, 2, 3, 4])

    def test_completion_template_handling(self):
        req = CompletionRequest(
            model="x", prompt="def f():", suffix="return 1", max_tokens=100
        )
        with patch(
            "sglang.srt.entrypoints.openai.serving_completions.is_completion_template_defined",
            return_value=True,
        ), patch(
            "sglang.srt.entrypoints.openai.serving_completions.generate_completion_prompt_from_request",
            return_value="processed_prompt",
        ):
            internal, _ = self.sc._convert_to_internal_request(req)
            self.assertEqual(internal.text, "processed_prompt")

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

    # ---------- hidden states tests ----------
    def test_hidden_states_request_conversion(self):
        """Test request conversion with return_hidden_states=True"""
        # Test single request
        request = CompletionRequest(
            model="test-model",
            prompt="Hello world",
            return_hidden_states=True,
        )
        adapted_request, _ = self.sc._convert_to_internal_request(
            [request], ["test-id"]
        )
        assert adapted_request.return_hidden_states is True

        # Test multiple requests
        requests = [
            CompletionRequest(
                model="test-model",
                prompt="Hello",
                return_hidden_states=True,
            ),
            CompletionRequest(
                model="test-model",
                prompt="World",
                return_hidden_states=False,
            ),
        ]
        adapted_request, _ = self.sc._convert_to_internal_request(
            requests, ["test-id-1", "test-id-2"]
        )
        assert adapted_request.return_hidden_states == [True, False]

    def test_hidden_states_response_handling(self):
        """Test hidden states in response handling"""
        # Test non-streaming response with hidden states
        request = CompletionRequest(
            model="test-model",
            prompt="Hello world",
            return_hidden_states=True,
        )
        ret = [
            {
                "text": "Test response",
                "meta_info": {
                    "id": "cmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "cached_tokens": 0,
                    "finish_reason": {"type": "stop", "matched": None},
                    "output_token_logprobs": [],
                    "output_top_logprobs": None,
                    "hidden_states": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                },
            }
        ]
        response = self.sc._build_completion_response(request, ret, 1234567890)
        assert len(response.choices) == 1
        assert response.choices[0].hidden_states == [0.4, 0.5, 0.6]

        # Test response without hidden states
        request = CompletionRequest(
            model="test-model",
            prompt="Hello world",
            return_hidden_states=False,
        )
        ret[0]["meta_info"].pop("hidden_states")
        response = self.sc._build_completion_response(request, ret, 1234567890)
        assert len(response.choices) == 1
        assert response.choices[0].hidden_states is None

    async def test_hidden_states_streaming(self):
        """Test hidden states in streaming response"""
        request = CompletionRequest(
            model="test-model",
            prompt="Hello world",
            return_hidden_states=True,
            stream=True,
        )

        async def mock_generate():
            yield {
                "text": "Test",
                "meta_info": {
                    "id": "cmpl-test",
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "cached_tokens": 0,
                    "finish_reason": None,
                    "output_token_logprobs": [],
                    "output_top_logprobs": None,
                    "hidden_states": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                },
                "index": 0,
            }

        self.sc.tokenizer_manager.generate_request = Mock(return_value=mock_generate())
        adapted_request, _ = self.sc._convert_to_internal_request(
            [request], ["test-id"]
        )
        response = await self.sc._handle_streaming_request(
            adapted_request, request, self.mock_request
        )

        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk)

        import json

        parsed_chunks = []
        for chunk in chunks:
            if chunk.startswith("data:") and chunk.strip() != "data: [DONE]":
                try:
                    chunk_data = json.loads(chunk[6:])
                    parsed_chunks.append(chunk_data)
                except json.JSONDecodeError:
                    continue

        assert len(parsed_chunks) >= 1
        hidden_states_found = False
        for chunk_data in parsed_chunks:
            choice = chunk_data["choices"][0]
            if choice.get("hidden_states") is not None:
                assert choice["hidden_states"] == [0.4, 0.5, 0.6]
                hidden_states_found = True
                break
        assert (
            hidden_states_found
        ), "Hidden states should be present in streaming response"


if __name__ == "__main__":
    unittest.main(verbosity=2)
