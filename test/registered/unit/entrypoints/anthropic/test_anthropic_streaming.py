# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Streaming tests for the Anthropic-compatible serving layer.

These tests do not boot a real model. They drive ``AnthropicServing.
_generate_anthropic_stream`` with a fake OpenAI SSE source and verify the
emitted Anthropic event stream.

Coverage:

* Pure text streaming (no thinking, no tools): message_start / one text
  content block / message_delta / message_stop ordering.
* Streaming with thinking enabled (adaptive + summarized): a thinking block
  precedes the text block; ``thinking_delta`` events are emitted.
* Tool calling with thinking enabled: thinking block, then a ``tool_use``
  content block with ``input_json_delta`` events; ``stop_reason`` becomes
  ``tool_use``.
* ``thinking.display = "omitted"``: ``content_block_start`` / stop are
  emitted for the thinking block but no ``thinking_delta`` events.
"""

import asyncio
import json
import unittest
from typing import AsyncIterator, List

from sglang.srt.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _data_line(payload: dict) -> str:
    """Serialize one OpenAI-style SSE ``data:`` line."""
    return "data: " + json.dumps(payload) + "\n\n"


def _done_line() -> str:
    return "data: [DONE]\n\n"


class _FakeOpenAIChatHandler:
    """Minimal stub for OpenAIServingChat used by streaming.

    Only ``_generate_chat_stream`` is exercised here.
    """

    def __init__(self, sse_lines: List[str]):
        self._sse_lines = sse_lines

    def _generate_chat_stream(
        self, adapted_request, processed_request, raw_request
    ) -> AsyncIterator[str]:
        sse_lines = self._sse_lines

        async def _iter() -> AsyncIterator[str]:
            for line in sse_lines:
                yield line

        return _iter()


def _collect_events(sse_text: str) -> List[dict]:
    """Parse the Anthropic SSE output into a list of events.

    Each event is represented as ``{"event": <type>, "data": <parsed json>}``.
    Whitespace-only or malformed blocks are skipped.
    """
    events: List[dict] = []
    for raw_block in sse_text.split("\n\n"):
        block = raw_block.strip()
        if not block:
            continue
        event_type = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event_type = line[len("event: ") :].strip()
            elif line.startswith("data: "):
                try:
                    data = json.loads(line[len("data: ") :].strip())
                except json.JSONDecodeError:
                    data = None
        if event_type is not None:
            events.append({"event": event_type, "data": data})
    return events


def _drive_stream(serving: AnthropicServing, anthropic_request) -> List[dict]:
    """Run the async generator to completion and return parsed events."""

    async def _run() -> str:
        chunks: List[str] = []
        async for piece in serving._generate_anthropic_stream(
            adapted_request=None,
            processed_request=None,
            anthropic_request=anthropic_request,
            raw_request=None,
        ):
            chunks.append(piece)
        return "".join(chunks)

    sse_text = asyncio.run(_run())
    return _collect_events(sse_text)


class TestAnthropicStreaming(unittest.TestCase):
    def _build(self, sse_lines: List[str]) -> AnthropicServing:
        return AnthropicServing(openai_serving_chat=_FakeOpenAIChatHandler(sse_lines))

    def test_streaming_text_only(self):
        """Non-thinking streaming: message_start, one text block, stop events."""
        sse_lines = [
            _data_line(
                {
                    "id": "chunk-1",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 0,
                        "total_tokens": 5,
                    },
                }
            ),
            _data_line(
                {
                    "id": "chunk-2",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [{"index": 0, "delta": {"content": "Hello"}}],
                }
            ),
            _data_line(
                {
                    "id": "chunk-3",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [{"index": 0, "delta": {"content": " world"}}],
                }
            ),
            _data_line(
                {
                    "id": "chunk-4",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            ),
            _data_line(
                {
                    "id": "chunk-5",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 2,
                        "total_tokens": 7,
                    },
                }
            ),
            _done_line(),
        ]
        serving = self._build(sse_lines)
        request = AnthropicMessagesRequest(
            model="claude-opus-4-7",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
        )

        events = _drive_stream(serving, request)
        types = [e["event"] for e in events]

        self.assertEqual(types[0], "message_start")
        self.assertIn("content_block_start", types)
        self.assertIn("content_block_delta", types)
        self.assertEqual(types[-1], "message_stop")
        self.assertEqual(types[-2], "message_delta")

        # First content block must be text.
        first_block_start = next(
            e for e in events if e["event"] == "content_block_start"
        )
        self.assertEqual(first_block_start["data"]["content_block"]["type"], "text")

        # Concatenated text deltas reproduce the model output.
        text_pieces = [
            e["data"]["delta"]["text"]
            for e in events
            if e["event"] == "content_block_delta"
            and e["data"]["delta"]["type"] == "text_delta"
        ]
        self.assertEqual("".join(text_pieces), "Hello world")

        # message_delta carries stop_reason and usage.
        msg_delta = next(e for e in events if e["event"] == "message_delta")
        self.assertEqual(msg_delta["data"]["delta"]["stop_reason"], "end_turn")
        self.assertEqual(msg_delta["data"]["usage"]["input_tokens"], 5)
        self.assertEqual(msg_delta["data"]["usage"]["output_tokens"], 2)

    def test_streaming_with_thinking_summarized(self):
        """Adaptive thinking with display=summarized: a thinking block comes
        first and emits thinking_delta events; a text block follows."""
        sse_lines = [
            _data_line(
                {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                        }
                    ],
                }
            ),
            _data_line(
                {
                    "id": "c2",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [
                        {"index": 0, "delta": {"reasoning_content": "Think A"}}
                    ],
                }
            ),
            _data_line(
                {
                    "id": "c3",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [{"index": 0, "delta": {"reasoning_content": " more"}}],
                }
            ),
            _data_line(
                {
                    "id": "c4",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [{"index": 0, "delta": {"content": "Answer"}}],
                }
            ),
            _data_line(
                {
                    "id": "c5",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            ),
            _done_line(),
        ]
        serving = self._build(sse_lines)
        request = AnthropicMessagesRequest(
            model="claude-opus-4-7",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "adaptive", "display": "summarized"},
        )

        events = _drive_stream(serving, request)

        # The first content block must be thinking.
        block_starts = [e for e in events if e["event"] == "content_block_start"]
        self.assertEqual(block_starts[0]["data"]["content_block"]["type"], "thinking")
        self.assertEqual(block_starts[1]["data"]["content_block"]["type"], "text")

        # thinking_delta events carry the streamed reasoning_content.
        thinking_pieces = [
            e["data"]["delta"]["thinking"]
            for e in events
            if e["event"] == "content_block_delta"
            and e["data"]["delta"]["type"] == "thinking_delta"
        ]
        self.assertEqual("".join(thinking_pieces), "Think A more")

        # text_delta still works after the thinking block was closed.
        text_pieces = [
            e["data"]["delta"]["text"]
            for e in events
            if e["event"] == "content_block_delta"
            and e["data"]["delta"]["type"] == "text_delta"
        ]
        self.assertEqual("".join(text_pieces), "Answer")

        msg_delta = next(e for e in events if e["event"] == "message_delta")
        self.assertEqual(msg_delta["data"]["delta"]["stop_reason"], "end_turn")

    def test_streaming_with_thinking_omitted(self):
        """display=omitted: thinking block start/stop are emitted but no
        thinking_delta is leaked.
        """
        sse_lines = [
            _data_line(
                {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                        }
                    ],
                }
            ),
            _data_line(
                {
                    "id": "c2",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [{"index": 0, "delta": {"reasoning_content": "secret"}}],
                }
            ),
            _data_line(
                {
                    "id": "c3",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [{"index": 0, "delta": {"content": "ok"}}],
                }
            ),
            _data_line(
                {
                    "id": "c4",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            ),
            _done_line(),
        ]
        serving = self._build(sse_lines)
        request = AnthropicMessagesRequest(
            model="claude-opus-4-7",
            max_tokens=64,
            messages=[{"role": "user", "content": "hi"}],
            thinking={"type": "adaptive", "display": "omitted"},
        )

        events = _drive_stream(serving, request)

        block_starts = [e for e in events if e["event"] == "content_block_start"]
        # A thinking block is still announced (so clients can render an
        # empty thinking placeholder), but no thinking_delta is emitted.
        self.assertEqual(block_starts[0]["data"]["content_block"]["type"], "thinking")

        thinking_deltas = [
            e
            for e in events
            if e["event"] == "content_block_delta"
            and e["data"]["delta"]["type"] == "thinking_delta"
        ]
        self.assertEqual(thinking_deltas, [])

        # Text streaming still works normally.
        text_pieces = [
            e["data"]["delta"]["text"]
            for e in events
            if e["event"] == "content_block_delta"
            and e["data"]["delta"]["type"] == "text_delta"
        ]
        self.assertEqual("".join(text_pieces), "ok")

    def test_streaming_tool_call_with_thinking_enabled(self):
        """Tool calling with adaptive thinking (summarized).

        Expected order:
          message_start
          content_block_start(thinking) ... thinking_delta ... content_block_stop
          content_block_start(tool_use) ... input_json_delta ... content_block_stop
          message_delta(stop_reason=tool_use)
          message_stop
        """
        sse_lines = [
            _data_line(
                {
                    "id": "c1",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                        }
                    ],
                }
            ),
            _data_line(
                {
                    "id": "c2",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"reasoning_content": "I should call the tool."},
                        }
                    ],
                }
            ),
            # Tool call start (with name) and partial arguments.
            _data_line(
                {
                    "id": "c3",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": '{"city":',
                                        },
                                    }
                                ]
                            },
                        }
                    ],
                }
            ),
            # Continuation of arguments.
            _data_line(
                {
                    "id": "c4",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "type": "function",
                                        "function": {"arguments": ' "Paris"}'},
                                    }
                                ]
                            },
                        }
                    ],
                }
            ),
            _data_line(
                {
                    "id": "c5",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                    ],
                }
            ),
            _data_line(
                {
                    "id": "c6",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "claude-opus-4-7",
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 4,
                        "total_tokens": 16,
                    },
                }
            ),
            _done_line(),
        ]
        serving = self._build(sse_lines)
        request = AnthropicMessagesRequest(
            model="claude-opus-4-7",
            max_tokens=64,
            messages=[{"role": "user", "content": "weather?"}],
            thinking={"type": "adaptive", "display": "summarized"},
            tools=[
                {
                    "name": "get_weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
            tool_choice={"type": "auto"},
        )

        events = _drive_stream(serving, request)

        block_starts = [e for e in events if e["event"] == "content_block_start"]
        self.assertEqual(len(block_starts), 2)
        self.assertEqual(block_starts[0]["data"]["content_block"]["type"], "thinking")
        self.assertEqual(block_starts[1]["data"]["content_block"]["type"], "tool_use")
        self.assertEqual(
            block_starts[1]["data"]["content_block"]["name"], "get_weather"
        )
        self.assertEqual(block_starts[1]["data"]["content_block"]["id"], "call_1")

        # Indices increment: thinking=0, tool_use=1.
        self.assertEqual(block_starts[0]["data"]["index"], 0)
        self.assertEqual(block_starts[1]["data"]["index"], 1)

        # Reassemble streamed thinking and tool arguments.
        thinking_pieces = [
            e["data"]["delta"]["thinking"]
            for e in events
            if e["event"] == "content_block_delta"
            and e["data"]["delta"]["type"] == "thinking_delta"
        ]
        self.assertEqual("".join(thinking_pieces), "I should call the tool.")

        json_pieces = [
            e["data"]["delta"]["partial_json"]
            for e in events
            if e["event"] == "content_block_delta"
            and e["data"]["delta"]["type"] == "input_json_delta"
        ]
        self.assertEqual(json.loads("".join(json_pieces)), {"city": "Paris"})

        msg_delta = next(e for e in events if e["event"] == "message_delta")
        self.assertEqual(msg_delta["data"]["delta"]["stop_reason"], "tool_use")
        self.assertEqual(msg_delta["data"]["usage"]["input_tokens"], 12)
        self.assertEqual(msg_delta["data"]["usage"]["output_tokens"], 4)


if __name__ == "__main__":
    unittest.main()
