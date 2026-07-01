"""Unit tests for OpenAI-compatible tokenize serving."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from fastapi import Request

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
)
from sglang.srt.entrypoints.openai.serving_tokenize import OpenAIServingTokenize
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class _MockTokenizerManager:
    def __init__(self, token_ids):
        self.server_args = SimpleNamespace(tokenizer_metrics_allowed_custom_labels=None)
        self.tokenizer = SimpleNamespace(model_max_length=4096)
        self.request_logger = SimpleNamespace(
            log_requests=False,
            log_requests_level=0,
            log_openai_received_request=Mock(),
        )
        self.tokenize_request = AsyncMock(return_value=token_ids)


def _raw_request():
    raw = Mock(spec=Request)
    raw.headers = {}
    return raw


def _generate_req(input_ids=None):
    return GenerateReqInput(input_ids=input_ids or [1, 2], sampling_params={})


class TestOpenAIServingTokenize(CustomTestCase):
    def _make_serving(self, token_ids):
        tokenizer_manager = _MockTokenizerManager(token_ids)
        chat_serving = Mock()
        completion_serving = Mock()
        chat_serving._validate_request.return_value = None
        completion_serving._validate_request.return_value = None
        chat_serving._convert_to_internal_request.return_value = (
            _generate_req([1, 2]),
            None,
        )
        completion_serving._convert_to_internal_request.return_value = (
            _generate_req([3, 4]),
            None,
        )
        serving = OpenAIServingTokenize(
            tokenizer_manager=tokenizer_manager,
            chat_serving=chat_serving,
            completion_serving=completion_serving,
        )
        return serving, tokenizer_manager, chat_serving, completion_serving

    def test_chat_tokenize_uses_chat_conversion_path(self):
        serving, tokenizer_manager, chat_serving, completion_serving = (
            self._make_serving([11, 12])
        )
        raw = _raw_request()
        request = ChatCompletionRequest(
            model="x",
            messages=[{"role": "user", "content": "hi"}],
        )

        response = asyncio.run(serving.handle_chat_request(request, raw))

        self.assertEqual(response.tokens, [11, 12])
        self.assertEqual(response.count, 2)
        self.assertEqual(response.max_model_len, 4096)
        chat_serving._validate_request.assert_called_once_with(request)
        chat_serving._convert_to_internal_request.assert_called_once_with(request, raw)
        tokenizer_manager.tokenize_request.assert_awaited_once_with(
            chat_serving._convert_to_internal_request.return_value[0], raw
        )
        completion_serving._convert_to_internal_request.assert_not_called()

    def test_completions_tokenize_returns_batch_counts(self):
        serving, tokenizer_manager, chat_serving, completion_serving = (
            self._make_serving([[21], [22, 23]])
        )
        raw = _raw_request()
        request = CompletionRequest(model="x", prompt=["a", "b"])

        response = asyncio.run(serving.handle_completions_request(request, raw))

        self.assertEqual(response.tokens, [[21], [22, 23]])
        self.assertEqual(response.count, [1, 2])
        completion_serving._validate_request.assert_called_once_with(request)
        completion_serving._convert_to_internal_request.assert_called_once_with(
            request, raw
        )
        tokenizer_manager.tokenize_request.assert_awaited_once_with(
            completion_serving._convert_to_internal_request.return_value[0], raw
        )
        chat_serving._convert_to_internal_request.assert_not_called()

    def test_generic_tokenize_dispatches_messages_to_chat(self):
        serving, _, _, _ = self._make_serving([31])
        raw = _raw_request()
        serving.handle_chat_request = AsyncMock(return_value="chat-response")
        serving.handle_completions_request = AsyncMock()

        response = asyncio.run(
            serving.handle_request(
                {"model": "x", "messages": [{"role": "user", "content": "hi"}]},
                raw,
            )
        )

        self.assertEqual(response, "chat-response")
        serving.handle_chat_request.assert_awaited_once()
        chat_request = serving.handle_chat_request.await_args.args[0]
        self.assertIsInstance(chat_request, ChatCompletionRequest)
        serving.handle_completions_request.assert_not_called()

    def test_generic_tokenize_dispatches_prompt_to_completions(self):
        serving, _, _, _ = self._make_serving([41])
        raw = _raw_request()
        serving.handle_chat_request = AsyncMock()
        serving.handle_completions_request = AsyncMock(
            return_value="completion-response"
        )

        response = asyncio.run(
            serving.handle_request({"model": "x", "prompt": "hello"}, raw)
        )

        self.assertEqual(response, "completion-response")
        serving.handle_completions_request.assert_awaited_once()
        completion_request = serving.handle_completions_request.await_args.args[0]
        self.assertIsInstance(completion_request, CompletionRequest)
        serving.handle_chat_request.assert_not_called()

    def test_generic_tokenize_rejects_ambiguous_body(self):
        serving, _, _, _ = self._make_serving([51])
        raw = _raw_request()

        response = asyncio.run(
            serving.handle_request(
                {
                    "model": "x",
                    "prompt": "hello",
                    "messages": [{"role": "user", "content": "hi"}],
                },
                raw,
            )
        )

        self.assertEqual(response.status_code, 400)
        body = json.loads(response.body)
        self.assertIn("Exactly one", body["message"])

    def test_generic_tokenize_rejects_missing_input(self):
        serving, _, _, _ = self._make_serving([61])
        raw = _raw_request()

        response = asyncio.run(serving.handle_request({"model": "x"}, raw))

        self.assertEqual(response.status_code, 400)
        body = json.loads(response.body)
        self.assertIn("Exactly one", body["message"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
