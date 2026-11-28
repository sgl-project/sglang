"""OpenAI frontend streaming tests with an in-memory HTTP transport."""

import json
import unittest

import httpx
import openai

from sglang.lang.backend.openai import TokenUsage, openai_completion_stream
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestOpenAIBackendStream(CustomTestCase):
    def _chunk(self, is_chat, text=None, usage=None):
        choices = []
        if text is not None:
            choice = {"index": 0, "finish_reason": None}
            choice["delta" if is_chat else "text"] = (
                {"content": text} if is_chat else text
            )
            choices.append(choice)
        return {
            "id": "test-stream",
            "object": "chat.completion.chunk" if is_chat else "text_completion",
            "created": 0,
            "model": "test-model",
            "choices": choices,
            "usage": usage,
        }

    def _stream(self, is_chat, chunks):
        body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
        body += "data: [DONE]\n\n"
        transport = httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, text=body
            )
        )
        usage = TokenUsage(11, 13)
        with openai.OpenAI(
            api_key="test-key",
            base_url="http://test.invalid/v1",
            http_client=httpx.Client(transport=transport),
        ) as client:
            output = list(
                openai_completion_stream(
                    client,
                    usage,
                    is_chat=is_chat,
                    prompt=[{"role": "user", "content": "hello"}]
                    if is_chat
                    else "hello",
                    model="test-model",
                )
            )
        return output, usage

    def test_stream_without_usage(self):
        for is_chat in (False, True):
            with self.subTest(is_chat=is_chat):
                output, usage = self._stream(
                    is_chat, [self._chunk(is_chat, "hello"), self._chunk(is_chat, "!")]
                )
                self.assertEqual(output, [("hello", {}), ("!", {})])
                self.assertEqual(usage, TokenUsage(11, 13))

    def test_empty_stream(self):
        for is_chat in (False, True):
            with self.subTest(is_chat=is_chat):
                output, usage = self._stream(is_chat, [])
                self.assertEqual(output, [])
                self.assertEqual(usage, TokenUsage(11, 13))

    def test_usage_is_preserved_after_a_chunk_without_usage(self):
        for is_chat in (False, True):
            with self.subTest(is_chat=is_chat):
                output, usage = self._stream(
                    is_chat,
                    [
                        self._chunk(is_chat, "hello"),
                        self._chunk(
                            is_chat,
                            usage={
                                "prompt_tokens": 3,
                                "completion_tokens": 2,
                                "total_tokens": 5,
                            },
                        ),
                        self._chunk(is_chat, "!"),
                    ],
                )
                self.assertEqual(output, [("hello", {}), ("!", {})])
                self.assertEqual(usage, TokenUsage(14, 15))

    def test_usage_summary_is_added_once(self):
        for is_chat in (False, True):
            with self.subTest(is_chat=is_chat):
                summary = {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                }
                output, usage = self._stream(
                    is_chat,
                    [
                        self._chunk(is_chat, "hello", summary),
                        self._chunk(is_chat, usage=summary),
                    ],
                )
                self.assertEqual(output, [("hello", {})])
                self.assertEqual(usage, TokenUsage(14, 15))


if __name__ == "__main__":
    unittest.main()
