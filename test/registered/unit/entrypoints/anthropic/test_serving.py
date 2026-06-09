import asyncio
import json
import sys
import time
import types
import unittest
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

req_time_stats = types.ModuleType("sglang.srt.observability.req_time_stats")
req_time_stats.monotonic_time = time.monotonic
sys.modules.setdefault("sglang.srt.observability.req_time_stats", req_time_stats)

from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    UsageInfo,
)
from sglang.srt.entrypoints.openai.sse_utils import build_sse_content
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAnthropicServingThinking(unittest.TestCase):
    def test_response_conversion_includes_thinking_block(self):
        serving = AnthropicServing(openai_serving_chat=cast(Any, object()))
        response = ChatCompletionResponse(
            id="chatcmpl-test",
            model="test-model",
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content="Final answer",
                        reasoning_content="Internal reasoning",
                    ),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(prompt_tokens=3, completion_tokens=5, total_tokens=8),
        )

        anthropic = serving._convert_response(response)

        self.assertEqual(
            [block.model_dump(exclude_none=True) for block in anthropic.content],
            [
                {
                    "type": "thinking",
                    "thinking": "Internal reasoning",
                    "signature": "",
                },
                {"type": "text", "text": "Final answer"},
            ],
        )

    def test_stream_emits_thinking_blocks(self):
        events = asyncio.run(self._collect_stream_events())
        content_events = [
            event
            for event in events
            if event["type"] in {"content_block_start", "content_block_delta"}
        ]

        self.assertEqual(
            content_events,
            [
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "thinking",
                        "thinking": "",
                        "signature": "",
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "thinking_delta",
                        "thinking": "Need to inspect files.",
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "Final answer"},
                },
            ],
        )

    async def _collect_stream_events(self) -> list[dict[str, Any]]:
        created = int(time.time())

        async def fake_openai_stream(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
            yield build_sse_content(
                chunk_id="chatcmpl-test",
                created=created,
                model="test-model",
                index=0,
                role="assistant",
                content="",
            )
            yield build_sse_content(
                chunk_id="chatcmpl-test",
                created=created,
                model="test-model",
                index=0,
                reasoning_content="Need to inspect files.",
            )
            yield build_sse_content(
                chunk_id="chatcmpl-test",
                created=created,
                model="test-model",
                index=0,
                content="Final answer",
            )
            yield build_sse_content(
                chunk_id="chatcmpl-test",
                created=created,
                model="test-model",
                index=0,
                finish_reason="stop",
            )
            yield "data: [DONE]\n\n"

        openai_serving_chat = SimpleNamespace(
            _generate_chat_stream=fake_openai_stream,
            tokenizer_manager=SimpleNamespace(create_abort_task=lambda _: None),
        )
        serving = AnthropicServing(openai_serving_chat=cast(Any, openai_serving_chat))
        request = AnthropicMessagesRequest(
            model="test-model",
            max_tokens=64,
            stream=True,
            messages=cast(Any, [{"role": "user", "content": "Hi"}]),
        )
        processed_request = ChatCompletionRequest(
            model="test-model",
            messages=cast(Any, [{"role": "user", "content": "Hi"}]),
            stream=True,
        )

        events = []
        async for event in serving._generate_anthropic_stream(
            adapted_request=object(),
            processed_request=processed_request,
            anthropic_request=request,
            raw_request=cast(Any, None),
        ):
            if not event.startswith("event: "):
                continue
            data = event.split("\ndata: ", 1)[1].strip()
            events.append(json.loads(data))
        return events


if __name__ == "__main__":
    unittest.main()
