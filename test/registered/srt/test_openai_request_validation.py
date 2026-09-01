"""
Unit tests for three OpenAI API compliance validation fixes:

1. n > 1 combined with stream=true is rejected (chat and completions).
   OpenAI's API does not support streaming multiple candidates simultaneously.

2. continue_final_message=true with only a single assistant message is
   rejected. Stripping the only message would produce an empty list and
   cause an IndexError downstream.

3. logit_bias values outside [-100, 100] are rejected at construction time
   via Pydantic validation, matching OpenAI's documented constraint.

All tests exercise request validation directly without a live server.
"""

import unittest
from unittest.mock import MagicMock

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionMessageUserParam,
    ChatCompletionRequest,
    CompletionRequest,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "base-a-test-cpu")

_USER_MSG = ChatCompletionMessageUserParam(role="user", content="hi")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_serving():
    serving = MagicMock(spec=OpenAIServingChat)
    serving._validate_request = OpenAIServingChat._validate_request.__get__(
        serving, OpenAIServingChat
    )
    serving.tokenizer_manager = MagicMock()
    serving.tokenizer_manager.server_args.context_length = None
    serving.tokenizer_manager.server_args.allow_auto_truncate = False
    return serving


def _make_completion_serving():
    serving = MagicMock(spec=OpenAIServingCompletion)
    serving._validate_request = OpenAIServingCompletion._validate_request.__get__(
        serving, OpenAIServingCompletion
    )
    return serving


def _chat(**kwargs):
    defaults = dict(model="test-model", messages=[_USER_MSG])
    defaults.update(kwargs)
    return ChatCompletionRequest(**defaults)


def _completion(**kwargs):
    defaults = dict(model="test-model", prompt="hello")
    defaults.update(kwargs)
    return CompletionRequest(**defaults)


# ---------------------------------------------------------------------------
# n > 1 + stream
# ---------------------------------------------------------------------------


class TestStreamingWithMultipleCandidates(unittest.TestCase):
    def setUp(self):
        self.chat = _make_chat_serving()
        self.completion = _make_completion_serving()

    # Chat — rejected

    def test_chat_n2_stream_returns_error(self):
        err = self.chat._validate_request(_chat(n=2, stream=True))
        self.assertIsNotNone(err)
        self.assertIn("stream", err.lower())

    def test_chat_n5_stream_returns_error(self):
        err = self.chat._validate_request(_chat(n=5, stream=True))
        self.assertIsNotNone(err)

    # Chat — allowed

    def test_chat_n2_no_stream_is_allowed(self):
        self.assertIsNone(self.chat._validate_request(_chat(n=2, stream=False)))

    def test_chat_n1_stream_is_allowed(self):
        self.assertIsNone(self.chat._validate_request(_chat(n=1, stream=True)))

    def test_chat_default_n_stream_is_allowed(self):
        self.assertIsNone(self.chat._validate_request(_chat(stream=True)))

    # Completions — rejected

    def test_completion_n2_stream_returns_error(self):
        err = self.completion._validate_request(_completion(n=2, stream=True))
        self.assertIsNotNone(err)
        self.assertIn("stream", err.lower())

    def test_completion_n3_stream_returns_error(self):
        err = self.completion._validate_request(_completion(n=3, stream=True))
        self.assertIsNotNone(err)

    # Completions — allowed

    def test_completion_n2_no_stream_is_allowed(self):
        self.assertIsNone(
            self.completion._validate_request(_completion(n=2, stream=False))
        )

    def test_completion_n1_stream_is_allowed(self):
        self.assertIsNone(
            self.completion._validate_request(_completion(n=1, stream=True))
        )


# ---------------------------------------------------------------------------
# continue_final_message with only-assistant message
# ---------------------------------------------------------------------------


class TestContinueFinalMessageValidation(unittest.TestCase):
    def setUp(self):
        self.chat = _make_chat_serving()

    def _assistant_msg(self):
        from sglang.srt.entrypoints.openai.protocol import (
            ChatCompletionMessageGenericParam,
        )

        return ChatCompletionMessageGenericParam(role="assistant", content="sure")

    def test_single_assistant_with_continue_returns_error(self):
        req = _chat(
            messages=[self._assistant_msg()],
            continue_final_message=True,
        )
        err = self.chat._validate_request(req)
        self.assertIsNotNone(err)
        self.assertIn("continue_final_message", err)

    def test_user_then_assistant_with_continue_is_allowed(self):
        req = _chat(
            messages=[_USER_MSG, self._assistant_msg()],
            continue_final_message=True,
        )
        self.assertIsNone(self.chat._validate_request(req))

    def test_single_user_message_no_continue_is_allowed(self):
        self.assertIsNone(self.chat._validate_request(_chat(messages=[_USER_MSG])))

    def test_single_assistant_no_continue_is_allowed(self):
        req = _chat(
            messages=[self._assistant_msg()],
            continue_final_message=False,
        )
        self.assertIsNone(self.chat._validate_request(req))


# ---------------------------------------------------------------------------
# logit_bias range validation
# ---------------------------------------------------------------------------


class TestLogitBiasRangeValidation(unittest.TestCase):
    # CompletionRequest

    def test_completion_bias_above_100_raises(self):
        with self.assertRaises(Exception):
            _completion(logit_bias={"1234": 101.0})

    def test_completion_bias_below_minus_100_raises(self):
        with self.assertRaises(Exception):
            _completion(logit_bias={"1234": -101.0})

    def test_completion_bias_at_100_is_allowed(self):
        self.assertEqual(_completion(logit_bias={"1": 100.0}).logit_bias["1"], 100.0)

    def test_completion_bias_at_minus_100_is_allowed(self):
        self.assertEqual(_completion(logit_bias={"1": -100.0}).logit_bias["1"], -100.0)

    def test_completion_no_logit_bias_is_allowed(self):
        self.assertIsNone(_completion().logit_bias)

    # ChatCompletionRequest

    def test_chat_bias_above_100_raises(self):
        with self.assertRaises(Exception):
            _chat(logit_bias={"42": 150.0})

    def test_chat_bias_below_minus_100_raises(self):
        with self.assertRaises(Exception):
            _chat(logit_bias={"42": -200.0})

    def test_chat_bias_in_range_is_allowed(self):
        req = _chat(logit_bias={"42": 50.0, "99": -75.0})
        self.assertEqual(req.logit_bias["42"], 50.0)

    def test_chat_no_logit_bias_is_allowed(self):
        self.assertIsNone(_chat().logit_bias)

    def test_chat_multiple_tokens_one_out_of_range_raises(self):
        with self.assertRaises(Exception):
            _chat(logit_bias={"1": 10.0, "2": 50.0, "3": 101.0})


if __name__ == "__main__":
    unittest.main()
