"""Unit tests for the OpenAI tokenize and detokenize serving handlers."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may transitively load sgl_kernel

import json
import unittest
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock, call

from sglang.srt.entrypoints.openai.protocol import (
    DetokenizeRequest,
    DetokenizeResponse,
    TokenizeRequest,
    TokenizeResponse,
)
from sglang.srt.entrypoints.openai.serving_tokenize import (
    OpenAIServingDetokenize,
    OpenAIServingTokenize,
)
from sglang.srt.utils import get_or_create_event_loop
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class _FakeTokenizer:
    model_max_length = 4096

    def __init__(self):
        self.encode = Mock(side_effect=self._encode)
        self.decode = Mock(side_effect=self._decode)

    def _encode(self, text: str, add_special_tokens: bool = True):
        token_ids = [ord(ch) for ch in text]
        if add_special_tokens:
            return [101] + token_ids + [102]
        return token_ids

    def _decode(self, tokens, skip_special_tokens: bool = True):
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in (101, 102)]
        return "|".join(str(t) for t in tokens)


class _TokenizerWithoutMaxLength:
    def __init__(self):
        self.encode = Mock(return_value=[7, 8])


def _tokenizer_manager(tokenizer):
    return SimpleNamespace(tokenizer=tokenizer, server_args=object())


def _run(coro):
    return get_or_create_event_loop().run_until_complete(coro)


def _error_payload(response):
    return json.loads(response.body)


class TestOpenAIServingTokenize(CustomTestCase):
    def setUp(self):
        self.tokenizer = _FakeTokenizer()
        self.serving = OpenAIServingTokenize(_tokenizer_manager(self.tokenizer))

    def _handle(self, request):
        return _run(self.serving._handle_non_streaming_request(request, request, None))

    def test_tokenize_string_prompt(self):
        request = TokenizeRequest(
            model="test-model", prompt="hi", add_special_tokens=True
        )

        response = self._handle(request)

        self.assertIsInstance(response, TokenizeResponse)
        self.assertEqual(response.tokens, [101, 104, 105, 102])
        self.assertEqual(response.count, 4)
        self.assertEqual(response.max_model_len, 4096)
        self.tokenizer.encode.assert_called_once_with("hi", add_special_tokens=True)

    def test_tokenize_convert_to_internal_request_is_identity(self):
        request = TokenizeRequest(model="test-model", prompt="hi")

        adapted_request, processed_request = self.serving._convert_to_internal_request(
            request, None
        )

        self.assertIs(adapted_request, request)
        self.assertIs(processed_request, request)

    def test_tokenize_list_prompt(self):
        request = TokenizeRequest(
            model="test-model", prompt=["a", "bc"], add_special_tokens=False
        )

        response = self._handle(request)

        self.assertIsInstance(response, TokenizeResponse)
        self.assertEqual(response.tokens, [[97], [98, 99]])
        self.assertEqual(response.count, [1, 2])
        self.assertEqual(response.max_model_len, 4096)
        self.tokenizer.encode.assert_has_calls(
            [
                call("a", add_special_tokens=False),
                call("bc", add_special_tokens=False),
            ]
        )
        self.assertEqual(self.tokenizer.encode.call_count, 2)

    def test_tokenize_empty_string_without_special_tokens(self):
        request = TokenizeRequest(
            model="test-model", prompt="", add_special_tokens=False
        )

        response = self._handle(request)

        self.assertIsInstance(response, TokenizeResponse)
        self.assertEqual(response.tokens, [])
        self.assertEqual(response.count, 0)
        self.assertEqual(response.max_model_len, 4096)
        self.tokenizer.encode.assert_called_once_with("", add_special_tokens=False)

    def test_tokenize_empty_prompt_list(self):
        request = TokenizeRequest(
            model="test-model", prompt=[], add_special_tokens=False
        )

        response = self._handle(request)

        self.assertIsInstance(response, TokenizeResponse)
        self.assertEqual(response.tokens, [])
        self.assertEqual(response.count, [])
        self.assertEqual(response.max_model_len, 4096)
        self.tokenizer.encode.assert_not_called()

    def test_tokenize_missing_model_max_length(self):
        tokenizer = _TokenizerWithoutMaxLength()
        serving = OpenAIServingTokenize(_tokenizer_manager(tokenizer))
        request = TokenizeRequest(model="test-model", prompt="x")

        response = _run(serving._handle_non_streaming_request(request, request, None))

        self.assertIsInstance(response, TokenizeResponse)
        self.assertEqual(response.tokens, [7, 8])
        self.assertEqual(response.count, 2)
        self.assertEqual(response.max_model_len, -1)

    def test_tokenize_rejects_invalid_prompt_type(self):
        request = TokenizeRequest.model_construct(
            model="test-model", prompt=123, add_special_tokens=True
        )

        response = self._handle(request)
        payload = _error_payload(response)

        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(payload["type"], "BadRequestError")
        self.assertIn("Invalid prompt type", payload["message"])

    def test_tokenize_encode_exception_returns_internal_error(self):
        tokenizer = _FakeTokenizer()
        tokenizer.encode = Mock(side_effect=RuntimeError("tokenizer exploded"))
        serving = OpenAIServingTokenize(_tokenizer_manager(tokenizer))
        request = TokenizeRequest(model="test-model", prompt="hi")

        response = _run(serving._handle_non_streaming_request(request, request, None))
        payload = _error_payload(response)

        self.assertEqual(response.status_code, HTTPStatus.INTERNAL_SERVER_ERROR)
        self.assertEqual(payload["type"], "InternalServerError")
        self.assertIn("tokenizer exploded", payload["message"])


class TestOpenAIServingDetokenize(CustomTestCase):
    def setUp(self):
        self.tokenizer = _FakeTokenizer()
        self.serving = OpenAIServingDetokenize(_tokenizer_manager(self.tokenizer))

    def _handle(self, request):
        return _run(self.serving._handle_non_streaming_request(request, request, None))

    def test_detokenize_single_token_list(self):
        request = DetokenizeRequest(
            model="test-model", tokens=[101, 7, 102], skip_special_tokens=True
        )

        response = self._handle(request)

        self.assertIsInstance(response, DetokenizeResponse)
        self.assertEqual(response.text, "7")
        self.tokenizer.decode.assert_called_once_with(
            [101, 7, 102], skip_special_tokens=True
        )

    def test_detokenize_convert_to_internal_request_is_identity(self):
        request = DetokenizeRequest(model="test-model", tokens=[1, 2])

        adapted_request, processed_request = self.serving._convert_to_internal_request(
            request, None
        )

        self.assertIs(adapted_request, request)
        self.assertIs(processed_request, request)

    def test_detokenize_nested_token_lists(self):
        request = DetokenizeRequest(
            model="test-model",
            tokens=[[1, 2], [101, 3, 102]],
            skip_special_tokens=False,
        )

        response = self._handle(request)

        self.assertIsInstance(response, DetokenizeResponse)
        self.assertEqual(response.text, ["1|2", "101|3|102"])
        self.tokenizer.decode.assert_has_calls(
            [
                call([1, 2], skip_special_tokens=False),
                call([101, 3, 102], skip_special_tokens=False),
            ]
        )
        self.assertEqual(self.tokenizer.decode.call_count, 2)

    def test_detokenize_nested_empty_token_list(self):
        request = DetokenizeRequest(
            model="test-model", tokens=[[]], skip_special_tokens=True
        )

        response = self._handle(request)

        self.assertIsInstance(response, DetokenizeResponse)
        self.assertEqual(response.text, [""])
        self.tokenizer.decode.assert_called_once_with([], skip_special_tokens=True)

    def test_detokenize_empty_token_list(self):
        request = DetokenizeRequest(model="test-model", tokens=[])

        response = self._handle(request)

        self.assertIsInstance(response, DetokenizeResponse)
        self.assertEqual(response.text, "")
        self.tokenizer.decode.assert_not_called()

    def test_detokenize_rejects_mixed_flat_token_list(self):
        request = DetokenizeRequest.model_construct(
            model="test-model", tokens=[1, "bad"], skip_special_tokens=True
        )

        response = self._handle(request)
        payload = _error_payload(response)

        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(payload["type"], "BadRequestError")
        self.assertIn("list of integers", payload["message"])
        self.tokenizer.decode.assert_not_called()

    def test_detokenize_rejects_mixed_nested_token_list(self):
        request = DetokenizeRequest.model_construct(
            model="test-model", tokens=[["bad"]], skip_special_tokens=True
        )

        response = self._handle(request)
        payload = _error_payload(response)

        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(payload["type"], "BadRequestError")
        self.assertIn("Sublist", payload["message"])
        self.tokenizer.decode.assert_not_called()

    def test_detokenize_rejects_invalid_tokens_type(self):
        request = DetokenizeRequest.model_construct(
            model="test-model", tokens="bad", skip_special_tokens=True
        )

        response = self._handle(request)
        payload = _error_payload(response)

        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(payload["type"], "BadRequestError")
        self.assertIn("Invalid tokens type", payload["message"])
        self.tokenizer.decode.assert_not_called()

    def test_detokenize_decode_exception_returns_decode_error(self):
        tokenizer = _FakeTokenizer()
        tokenizer.decode = Mock(side_effect=ValueError("decode failed"))
        serving = OpenAIServingDetokenize(_tokenizer_manager(tokenizer))
        request = DetokenizeRequest(model="test-model", tokens=[1, 2])

        response = _run(serving._handle_non_streaming_request(request, request, None))
        payload = _error_payload(response)

        self.assertEqual(response.status_code, HTTPStatus.BAD_REQUEST)
        self.assertEqual(payload["type"], "DecodeError")
        self.assertIn("decode failed", payload["message"])

    def test_detokenize_other_exception_returns_internal_error(self):
        tokenizer = _FakeTokenizer()
        tokenizer.decode = Mock(side_effect=RuntimeError("backend failed"))
        serving = OpenAIServingDetokenize(_tokenizer_manager(tokenizer))
        request = DetokenizeRequest(model="test-model", tokens=[1, 2])

        response = _run(serving._handle_non_streaming_request(request, request, None))
        payload = _error_payload(response)

        self.assertEqual(response.status_code, HTTPStatus.INTERNAL_SERVER_ERROR)
        self.assertEqual(payload["type"], "InternalServerError")
        self.assertIn("backend failed", payload["message"])


if __name__ == "__main__":
    unittest.main()
