# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Unit tests for OpenAI SSE chunk utilities."""

import json
import unittest

from sglang.srt.entrypoints.openai.sse_utils import build_sse_content
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestBuildSseContent(CustomTestCase):
    def _decode_sse_payload(self, sse_content: str) -> dict:
        self.assertTrue(sse_content.startswith("data: "))
        self.assertTrue(sse_content.endswith("\n\n"))
        return json.loads(sse_content[len("data: ") : -2])

    def test_builds_role_chunk_with_sse_framing(self):
        payload = self._decode_sse_payload(
            build_sse_content(
                chunk_id="chatcmpl-test",
                created=123,
                model="test-model",
                index=0,
                role="assistant",
                content="",
            )
        )

        self.assertEqual(payload["id"], "chatcmpl-test")
        self.assertEqual(payload["object"], "chat.completion.chunk")
        self.assertEqual(payload["created"], 123)
        self.assertEqual(payload["model"], "test-model")
        self.assertNotIn("usage", payload)

        self.assertEqual(len(payload["choices"]), 1)
        choice = payload["choices"][0]
        self.assertEqual(choice["index"], 0)
        self.assertIsNone(choice["finish_reason"])
        self.assertIsNone(choice["matched_stop"])
        self.assertIsNone(choice["logprobs"])
        self.assertEqual(
            choice["delta"],
            {
                "reasoning_content": None,
                "role": "assistant",
                "content": "",
            },
        )

    def test_reasoning_content_key_is_preserved_when_none(self):
        payload = self._decode_sse_payload(
            build_sse_content(
                chunk_id="chatcmpl-test",
                created=123,
                model="test-model",
                index=0,
                content="visible text",
            )
        )

        delta = payload["choices"][0]["delta"]
        self.assertIn("reasoning_content", delta)
        self.assertIsNone(delta["reasoning_content"])
        self.assertEqual(delta["content"], "visible text")
        self.assertNotIn("role", delta)

    def test_reasoning_delta_omits_content_when_absent(self):
        payload = self._decode_sse_payload(
            build_sse_content(
                chunk_id="chatcmpl-test",
                created=123,
                model="test-model",
                index=1,
                reasoning_content="thinking",
            )
        )

        self.assertEqual(
            payload["choices"][0]["delta"],
            {"reasoning_content": "thinking"},
        )

    def test_finish_chunk_includes_string_matched_stop(self):
        payload = self._decode_sse_payload(
            build_sse_content(
                chunk_id="chatcmpl-test",
                created=123,
                model="test-model",
                index=2,
                finish_reason="stop",
                matched_stop="</s>",
            )
        )

        choice = payload["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(choice["matched_stop"], "</s>")
        self.assertEqual(choice["delta"], {"reasoning_content": None})

    def test_finish_chunk_accepts_integer_matched_stop(self):
        payload = self._decode_sse_payload(
            build_sse_content(
                chunk_id="chatcmpl-test",
                created=123,
                model="test-model",
                index=3,
                finish_reason="length",
                matched_stop=151643,
            )
        )

        choice = payload["choices"][0]
        self.assertEqual(choice["finish_reason"], "length")
        self.assertEqual(choice["matched_stop"], 151643)

    def test_optional_usage_and_logprobs_are_serialized(self):
        usage = {
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "total_tokens": 6,
        }
        logprobs = {
            "content": [
                {
                    "token": "hello",
                    "logprob": -0.1,
                    "bytes": [104, 101, 108, 108, 111],
                    "top_logprobs": [],
                }
            ]
        }

        payload = self._decode_sse_payload(
            build_sse_content(
                chunk_id="chatcmpl-test",
                created=123,
                model="test-model",
                index=0,
                content="hello",
                logprobs=logprobs,
                usage=usage,
            )
        )

        self.assertEqual(payload["usage"], usage)
        self.assertEqual(payload["choices"][0]["logprobs"], logprobs)


if __name__ == "__main__":
    unittest.main()
