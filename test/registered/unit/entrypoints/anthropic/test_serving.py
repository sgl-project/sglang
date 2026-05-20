import asyncio
import json
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from fastapi.responses import JSONResponse  # noqa: E402

from sglang.srt.entrypoints.anthropic.protocol import (  # noqa: E402
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from sglang.srt.entrypoints.openai.protocol import (  # noqa: E402
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeOpenAIServingChat:
    def __init__(self, stream_lines=None):
        self.stream_lines = stream_lines or []

    def _generate_chat_stream(self, adapted_request, processed_request, raw_request):
        async def _gen():
            for line in self.stream_lines:
                yield line

        return _gen()


class _FakeNonStreamingErrorOpenAI:
    """Returns a configurable error response from the OpenAI handler."""

    def __init__(self, status_code=400, body=None, content=None):
        self._status_code = status_code
        self._body = body
        self._content = content

    def _validate_request(self, chat_request):
        return None

    def _convert_to_internal_request(self, chat_request, raw_request):
        return SimpleNamespace(), chat_request

    async def _handle_non_streaming_request(
        self, adapted_request, processed_request, raw_request
    ):
        if self._body is not None:
            # Build a response object exposing raw bytes via `.body`.
            return SimpleNamespace(status_code=self._status_code, body=self._body)
        return JSONResponse(
            status_code=self._status_code,
            content=self._content
            or {
                "error": {
                    "type": "invalid_request_error",
                    "message": "context length exceeded",
                }
            },
        )


class _FakeNonStreamingOpenAI:
    """Returns a configurable ChatCompletionResponse from the OpenAI handler."""

    def __init__(self, response):
        self._response = response

    def _validate_request(self, chat_request):
        return None

    def _convert_to_internal_request(self, chat_request, raw_request):
        return SimpleNamespace(), chat_request

    async def _handle_non_streaming_request(
        self, adapted_request, processed_request, raw_request
    ):
        return self._response


def _chunk(choices=None, usage=None):
    data = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "test-model",
        "choices": choices or [],
    }
    if usage is not None:
        data["usage"] = usage
    return f"data: {json.dumps(data)}\n\n"


def _choice(delta, finish_reason=None):
    return {
        "index": 0,
        "delta": delta,
        "finish_reason": finish_reason,
    }


async def _collect_anthropic_events(serving, anthropic_request):
    events = []
    async for sse in serving._generate_anthropic_stream(
        adapted_request=object(),
        processed_request=object(),
        anthropic_request=anthropic_request,
        raw_request=object(),
    ):
        for line in sse.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


class TestAnthropicServing(unittest.TestCase):
    def _serving(self, stream_lines=None):
        return AnthropicServing(_FakeOpenAIServingChat(stream_lines))

    def _anthropic_request(self, **overrides):
        data = {
            "model": "test-model",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        }
        data.update(overrides)
        return AnthropicMessagesRequest.model_validate(data)

    def test_stream_closes_tool_block_before_text_delta(self):
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk(
                    [
                        _choice(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": '{"query"',
                                        },
                                    }
                                ]
                            }
                        )
                    ]
                ),
                _chunk(
                    [
                        _choice(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "type": "function",
                                        "function": {"arguments": ': "sglang"}'},
                                    }
                                ]
                            }
                        )
                    ]
                ),
                _chunk([_choice({"content": "done"})]),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )

        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        block_events = [
            (event["type"], event.get("content_block", {}).get("type"))
            for event in events
            if event["type"].startswith("content_block")
        ]

        self.assertEqual(
            block_events,
            [
                ("content_block_start", "tool_use"),
                ("content_block_delta", None),
                ("content_block_delta", None),
                ("content_block_stop", None),
                ("content_block_start", "text"),
                ("content_block_delta", None),
                ("content_block_stop", None),
            ],
        )

        text_delta = [
            event
            for event in events
            if event["type"] == "content_block_delta"
            and event["delta"].get("type") == "text_delta"
        ][0]
        self.assertEqual(text_delta["index"], 1)

    def test_stream_reasoning_content_uses_thinking_block(self):
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk([_choice({"reasoning_content": "think first"})]),
                _chunk([_choice({"content": "answer"})]),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )

        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        content_events = [
            event for event in events if event["type"].startswith("content_block")
        ]

        self.assertEqual(content_events[0]["content_block"]["type"], "thinking")
        # Signature is absent (None and excluded) — never emit empty
        # string, which would fail downstream Anthropic signature verifiers.
        self.assertNotIn("signature", content_events[0]["content_block"])
        self.assertEqual(content_events[1]["delta"]["type"], "thinking_delta")
        self.assertEqual(content_events[1]["delta"]["thinking"], "think first")
        # No empty signature_delta event between thinking_delta and content_block_stop.
        self.assertEqual(content_events[2]["type"], "content_block_stop")
        self.assertEqual(content_events[3]["content_block"]["type"], "text")
        # Confirm no signature_delta event was emitted in the entire stream.
        sig_deltas = [
            event
            for event in events
            if event["type"] == "content_block_delta"
            and event.get("delta", {}).get("type") == "signature_delta"
        ]
        self.assertEqual(sig_deltas, [])

    def test_stream_usage_subtracts_cache_read_and_omits_final_input_tokens(self):
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 0,
            "total_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
        final_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk([_choice({"content": "hi"})], usage=usage),
                _chunk([], usage=final_usage),
                "data: [DONE]\n\n",
            ]
        )

        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        message_start = [event for event in events if event["type"] == "message_start"][
            0
        ]
        message_delta = [event for event in events if event["type"] == "message_delta"][
            0
        ]

        self.assertEqual(message_start["message"]["usage"]["input_tokens"], 6)
        self.assertEqual(
            message_start["message"]["usage"]["cache_read_input_tokens"], 4
        )
        self.assertNotIn("input_tokens", message_delta["usage"])
        self.assertEqual(message_delta["usage"]["output_tokens"], 2)

    def test_non_streaming_usage_subtracts_cache_read_tokens(self):
        response = ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-test",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            }
        )

        anthropic_response = self._serving()._convert_response(response)

        self.assertEqual(anthropic_response.usage.input_tokens, 6)
        self.assertEqual(anthropic_response.usage.output_tokens, 2)
        self.assertEqual(anthropic_response.usage.cache_read_input_tokens, 4)

    def test_tool_result_search_result_content_is_flattened(self):
        request = AnthropicMessagesRequest.model_validate(
            {
                "model": "test-model",
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "call_1",
                                "content": [
                                    {
                                        "type": "search_result",
                                        "title": "SGLang docs",
                                        "source": "https://docs.sglang.ai",
                                        "content": [
                                            {
                                                "type": "text",
                                                "text": "Anthropic API notes",
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        )

        chat_request = self._serving()._convert_to_chat_completion_request(request)
        tool_message = [
            msg
            for msg in chat_request.model_dump()["messages"]
            if msg["role"] == "tool"
        ][0]

        self.assertIn("SGLang docs", tool_message["content"])
        self.assertIn("https://docs.sglang.ai", tool_message["content"])
        self.assertIn("Anthropic API notes", tool_message["content"])

    def test_builtin_web_search_tool_without_schema_is_skipped(self):
        request = AnthropicMessagesRequest.model_validate(
            {
                "model": "test-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "search sglang"}],
                "tools": [{"name": "web_search", "type": "web_search_20250305"}],
                "tool_choice": {"type": "auto"},
            }
        )

        chat_request = self._serving()._convert_to_chat_completion_request(request)

        self.assertIsNone(chat_request.tools)
        self.assertEqual(chat_request.tool_choice, "none")

    def test_custom_tool_without_schema_is_rejected(self):
        # With the discriminated union, an AnthropicCustomTool variant must
        # carry an input_schema. The check fires at request-parse time
        # (Pydantic raises ValidationError, a subclass of ValueError).
        with self.assertRaisesRegex(ValueError, "input_schema"):
            AnthropicMessagesRequest.model_validate(
                {
                    "model": "test-model",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "call a tool"}],
                    "tools": [{"name": "custom_without_schema"}],
                }
            )

    def test_non_streaming_openai_error_response_is_forwarded(self):
        serving = AnthropicServing(_FakeNonStreamingErrorOpenAI())
        chat_request = ChatCompletionRequest(
            model="test-model",
            max_tokens=16,
            messages=[{"role": "user", "content": "hello"}],
        )
        anthropic_request = self._anthropic_request(stream=False)

        response = asyncio.run(
            serving._handle_non_streaming(chat_request, anthropic_request, object())
        )
        payload = json.loads(response.body)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertEqual(payload["error"]["message"], "context length exceeded")

    # ------------------------------------------------------------------
    # Edge-case coverage added in the review-fix pass
    # ------------------------------------------------------------------

    def test_stream_text_then_tool_use_closes_text_block(self):
        """Text deltas followed by tool_use must close the text block before opening tool_use index."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": "Hello"})]),
                _chunk([_choice({"content": " world"})]),
                _chunk(
                    [
                        _choice(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "search",
                                            "arguments": '{"q":"hello"}',
                                        },
                                    }
                                ]
                            }
                        )
                    ]
                ),
                _chunk([_choice({}, finish_reason="tool_calls")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        block_events = [
            (event["type"], event.get("content_block", {}).get("type"))
            for event in events
            if event["type"].startswith("content_block")
        ]
        # text block (start, delta, delta, stop) then tool_use (start, delta, stop)
        self.assertEqual(block_events[0], ("content_block_start", "text"))
        text_stop_idx = next(
            i for i, ev in enumerate(block_events) if ev == ("content_block_stop", None)
        )
        tool_start_idx = next(
            i
            for i, ev in enumerate(block_events)
            if ev == ("content_block_start", "tool_use")
        )
        self.assertLess(text_stop_idx, tool_start_idx)

    def test_stream_no_usage_chunk_emits_error_event(self):
        """Stream that yields only [DONE] (no content delta) must surface as an error event."""
        serving = self._serving(["data: [DONE]\n\n"])
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        types = [event["type"] for event in events]
        # Sequence: message_start, error, message_stop
        self.assertEqual(types, ["message_start", "error", "message_stop"])
        error_event = events[1]
        self.assertEqual(error_event["error"]["type"], "api_error")
        self.assertIn("no content", error_event["error"]["message"].lower())

    def test_cache_read_exceeds_prompt_tokens_clamps_to_zero(self):
        """When cached_tokens > prompt_tokens, input_tokens clamps to 0 instead of going negative."""
        usage = {
            "prompt_tokens": 4,
            "completion_tokens": 0,
            "total_tokens": 4,
            "prompt_tokens_details": {"cached_tokens": 10},
        }
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk([_choice({"content": "ok"})], usage=usage),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        message_start = next(e for e in events if e["type"] == "message_start")
        usage_out = message_start["message"]["usage"]
        self.assertEqual(usage_out["input_tokens"], 0)
        self.assertEqual(usage_out["cache_read_input_tokens"], 10)

    def test_usage_without_prompt_tokens_details(self):
        """Usage object without prompt_tokens_details must omit cache_read_input_tokens cleanly."""
        usage = {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5}
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": ""})]),
                _chunk([_choice({"content": "ok"})], usage=usage),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        message_start = next(e for e in events if e["type"] == "message_start")
        usage_out = message_start["message"]["usage"]
        self.assertEqual(usage_out["input_tokens"], 5)
        self.assertNotIn("cache_read_input_tokens", usage_out)

    def test_non_streaming_error_with_non_json_body(self):
        """Non-JSON upstream error body falls back to body[:500] as the message (for 4xx)."""
        serving = AnthropicServing(
            _FakeNonStreamingErrorOpenAI(
                status_code=400,
                body=b"<html>upstream gateway rejected: bad payload</html>",
            )
        )
        chat_request = ChatCompletionRequest(
            model="test-model",
            max_tokens=16,
            messages=[{"role": "user", "content": "hello"}],
        )
        anthropic_request = self._anthropic_request(stream=False)
        response = asyncio.run(
            serving._handle_non_streaming(chat_request, anthropic_request, object())
        )
        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertIn("upstream gateway rejected", payload["error"]["message"])

    def test_non_streaming_error_5xx_scrubs_message(self):
        """5xx errors always return a generic message regardless of upstream payload."""
        for status_code, expected_type in [
            (500, "api_error"),
            (502, "api_error"),
            (503, "overloaded_error"),
            (504, "api_error"),
        ]:
            serving = AnthropicServing(
                _FakeNonStreamingErrorOpenAI(
                    status_code=status_code,
                    body=b'{"error":{"message":"sensitive internals: /opt/secret","type":"internal"}}',
                )
            )
            chat_request = ChatCompletionRequest(
                model="test-model",
                max_tokens=16,
                messages=[{"role": "user", "content": "hello"}],
            )
            anthropic_request = self._anthropic_request(stream=False)
            response = asyncio.run(
                serving._handle_non_streaming(chat_request, anthropic_request, object())
            )
            payload = json.loads(response.body)
            self.assertEqual(
                response.status_code,
                status_code,
                f"status {status_code} should be preserved",
            )
            self.assertEqual(
                payload["error"]["type"],
                expected_type,
                f"status {status_code} should map to {expected_type}",
            )
            self.assertEqual(
                payload["error"]["message"],
                "Internal server error",
                f"status {status_code} must scrub the message; got {payload['error']['message']!r}",
            )

    def test_non_streaming_response_includes_thinking_block(self):
        """When the OpenAI response carries reasoning_content, the Anthropic response has a thinking block first."""
        response = ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-test",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "the answer is 4",
                            "reasoning_content": "2 + 2 = 4",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 5,
                    "total_tokens": 8,
                },
            }
        )
        anthropic_response = self._serving()._convert_response(response)
        # thinking block first, then text block
        self.assertEqual(anthropic_response.content[0].type, "thinking")
        self.assertEqual(anthropic_response.content[0].thinking, "2 + 2 = 4")
        self.assertEqual(anthropic_response.content[1].type, "text")
        self.assertEqual(anthropic_response.content[1].text, "the answer is 4")

    def test_stream_text_then_thinking_closes_text_block(self):
        """Text deltas followed by reasoning_content must close the text block before opening thinking."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": "Direct"})]),
                _chunk([_choice({"reasoning_content": "but wait, let me think"})]),
                _chunk([_choice({}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        block_events = [
            (event["type"], event.get("content_block", {}).get("type"))
            for event in events
            if event["type"].startswith("content_block")
        ]
        # text (start, delta, stop) then thinking (start, delta, stop)
        self.assertEqual(block_events[0], ("content_block_start", "text"))
        text_stop_idx = next(
            i for i, ev in enumerate(block_events) if ev == ("content_block_stop", None)
        )
        thinking_start_idx = next(
            i
            for i, ev in enumerate(block_events)
            if ev == ("content_block_start", "thinking")
        )
        self.assertLess(text_stop_idx, thinking_start_idx)


if __name__ == "__main__":
    unittest.main()
