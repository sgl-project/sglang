"""Unit tests for OpenAI chat-completion SSE serialization."""

import json
import unittest

from sglang.srt.entrypoints.openai.sse_utils import build_sse_content
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _decode_sse_frame(frame: str) -> dict:
    """Decode one data-only SSE frame into its JSON payload."""
    prefix = "data: "
    suffix = "\n\n"
    if not frame.startswith(prefix) or not frame.endswith(suffix):
        raise ValueError(f"invalid SSE frame: {frame!r}")
    return json.loads(frame[len(prefix) : -len(suffix)])


class TestBuildSSEContent(CustomTestCase):
    def test_null_reasoning_content_remains_client_accessible(self):
        """A content-only delta must expose reasoning_content as JSON null."""
        payload = _decode_sse_frame(
            build_sse_content(
                chunk_id="chatcmpl-1",
                created=123,
                model="test-model",
                index=0,
                role="assistant",
                content="hello",
            )
        )

        delta = payload["choices"][0]["delta"]
        self.assertIn("reasoning_content", delta)
        self.assertIsNone(delta["reasoning_content"])
        self.assertEqual(delta["role"], "assistant")
        self.assertEqual(delta["content"], "hello")
        self.assertNotIn("usage", payload)

    def test_stream_metadata_and_text_round_trip(self):
        """SSE framing must preserve nested metadata and escaped text."""
        logprobs = {"content": [{"token": "你\n好", "logprob": -0.25}]}
        usage = {
            "prompt_tokens": 2,
            "completion_tokens": 1,
            "total_tokens": 3,
        }
        payload = _decode_sse_frame(
            build_sse_content(
                chunk_id="chatcmpl-2",
                created=456,
                model="test-model",
                index=1,
                content='quoted "text"\n下一行',
                reasoning_content="思考",
                finish_reason="stop",
                logprobs=logprobs,
                matched_stop="</s>",
                usage=usage,
            )
        )

        self.assertEqual(payload["id"], "chatcmpl-2")
        self.assertEqual(payload["object"], "chat.completion.chunk")
        self.assertEqual(payload["created"], 456)
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["usage"], usage)

        choice = payload["choices"][0]
        self.assertEqual(choice["index"], 1)
        self.assertEqual(choice["logprobs"], logprobs)
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(choice["matched_stop"], "</s>")
        self.assertEqual(choice["delta"]["content"], 'quoted "text"\n下一行')
        self.assertEqual(choice["delta"]["reasoning_content"], "思考")


if __name__ == "__main__":
    unittest.main()
