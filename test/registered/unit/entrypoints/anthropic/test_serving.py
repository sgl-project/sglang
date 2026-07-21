import asyncio
import json
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from fastapi.responses import JSONResponse  # noqa: E402

from sglang.srt.entrypoints.anthropic.protocol import (  # noqa: E402
    AnthropicMessage,
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from sglang.srt.entrypoints.openai.protocol import (  # noqa: E402
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from sglang.srt.parser.template_detection import (  # noqa: E402
    detect_inline_system_support,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeOpenAIServingChat:
    def __init__(self, stream_lines=None, chat_template=None):
        self.stream_lines = stream_lines or []
        self.apply_reasoning_calls: list[bool] = []
        self.tokenizer_manager = SimpleNamespace(
            tokenizer=SimpleNamespace(chat_template=chat_template)
        )

    def _generate_chat_stream(self, adapted_request, processed_request, raw_request):
        async def _gen():
            for line in self.stream_lines:
                yield line

        return _gen()

    def apply_reasoning_enabled(self, chat_request, enabled):
        self.apply_reasoning_calls.append(enabled)

    def wrap_reasoning_history(self, text):
        return f"<think>\n{text}\n</think>"


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
    # Renders system at any position (GLM/Kimi/Qwen3) → can pass through.
    INLINE_SYSTEM_TEMPLATE = (
        "{%- for message in messages %}"
        "{{- message.role }}: {{ message.content }}\n"
        "{%- endfor %}"
    )

    def _serving(self, stream_lines=None, chat_template=None):
        return AnthropicServing(_FakeOpenAIServingChat(stream_lines, chat_template))

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

    def test_stream_tool_use_without_arguments_is_not_empty_completion(self):
        """A zero-argument tool call is valid content even without input_json_delta."""
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
                                        "function": {"name": "ping", "arguments": ""},
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

        self.assertFalse(any(event["type"] == "error" for event in events))
        tool_start = next(
            event
            for event in events
            if event["type"] == "content_block_start"
            and event["content_block"]["type"] == "tool_use"
        )
        self.assertEqual(tool_start["content_block"]["name"], "ping")
        message_delta = next(
            event for event in events if event["type"] == "message_delta"
        )
        self.assertEqual(message_delta["delta"]["stop_reason"], "tool_use")

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

    def test_request_thinking_disabled_invokes_apply_reasoning_enabled(self):
        """``thinking={"type": "disabled"}`` must flip the reasoning toggle off."""
        serving = self._serving()
        request = self._anthropic_request(thinking={"type": "disabled"}, stream=False)
        serving._convert_to_chat_completion_request(request)
        self.assertEqual(serving.openai_serving_chat.apply_reasoning_calls, [False])

    def test_request_thinking_enabled_with_budget_tokens_logs_warning(self):
        """SDK shape: ``enabled`` requires ``budget_tokens``. We accept it
        (the SDK would), but log a WARNING because the local backend has
        no equivalent hard-cap knob — the budget is not enforced."""
        import logging

        serving = self._serving()
        request = self._anthropic_request(
            thinking={"type": "enabled", "budget_tokens": 2048}, stream=False
        )
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic.serving", level=logging.WARNING
        ) as log:
            serving._convert_to_chat_completion_request(request)
        self.assertEqual(serving.openai_serving_chat.apply_reasoning_calls, [True])
        self.assertTrue(
            any("budget_tokens=2048" in r and "not enforced" in r for r in log.output),
            f"expected unenforced-budget warning: {log.output}",
        )

    def test_request_thinking_enabled_requires_budget_tokens(self):
        """SDK requires ``budget_tokens`` for ``type=enabled`` — Pydantic 400."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            self._anthropic_request(thinking={"type": "enabled"}, stream=False)
        self.assertIn("budget_tokens", str(ctx.exception))

    def test_request_thinking_enabled_budget_below_min_is_rejected(self):
        """SDK doc: ``budget_tokens`` must be >= 1024."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            self._anthropic_request(
                thinking={"type": "enabled", "budget_tokens": 512}, stream=False
            )
        self.assertIn("1024", str(ctx.exception))

    def test_request_thinking_disabled_with_display_is_rejected(self):
        """SDK ``ThinkingConfigDisabledParam`` has no ``display`` field."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            self._anthropic_request(
                thinking={"type": "disabled", "display": "omitted"}, stream=False
            )
        self.assertIn("display", str(ctx.exception))

    def test_request_thinking_disabled_with_budget_is_rejected(self):
        """SDK ``ThinkingConfigDisabledParam`` has no ``budget_tokens`` field."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            self._anthropic_request(
                thinking={"type": "disabled", "budget_tokens": 2048}, stream=False
            )
        self.assertIn("budget_tokens", str(ctx.exception))

    def test_request_thinking_adaptive_with_budget_is_rejected(self):
        """SDK ``ThinkingConfigAdaptiveParam`` has no ``budget_tokens`` field."""
        from pydantic import ValidationError

        with self.assertRaises(ValidationError) as ctx:
            self._anthropic_request(
                thinking={"type": "adaptive", "budget_tokens": 2048}, stream=False
            )
        self.assertIn("budget_tokens", str(ctx.exception))

    def test_request_thinking_adaptive_is_treated_as_enabled(self):
        """Claude 4.7 ``thinking.type='adaptive'`` (the SDK default for
        unknown models) must be accepted and routed to ``apply_reasoning_enabled(True)``.
        """
        serving = self._serving()
        request = self._anthropic_request(thinking={"type": "adaptive"}, stream=False)
        serving._convert_to_chat_completion_request(request)
        self.assertEqual(serving.openai_serving_chat.apply_reasoning_calls, [True])

    def test_request_thinking_display_omitted_logs_warning_but_still_enables(self):
        """``thinking.display='omitted'`` is accepted; reasoning stays on
        because we cannot suppress reasoning text from the OpenAI stream.
        ``enabled`` requires ``budget_tokens`` per SDK shape."""
        import logging

        serving = self._serving()
        request = self._anthropic_request(
            thinking={
                "type": "enabled",
                "budget_tokens": 1024,
                "display": "omitted",
            },
            stream=False,
        )
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic.serving", level=logging.WARNING
        ) as log:
            serving._convert_to_chat_completion_request(request)
        self.assertEqual(serving.openai_serving_chat.apply_reasoning_calls, [True])
        self.assertTrue(any("omitted" in r for r in log.output))

    def test_request_output_config_effort_maps_to_reasoning_effort(self):
        """``output_config.effort`` rows map onto ``reasoning_effort``."""
        for anthropic_effort, openai_effort in [
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "max"),  # OpenAI Literal has no xhigh
            ("max", "max"),
        ]:
            with self.subTest(anthropic_effort=anthropic_effort):
                serving = self._serving()
                request = self._anthropic_request(
                    output_config={"effort": anthropic_effort}, stream=False
                )
                chat_request = serving._convert_to_chat_completion_request(request)
                self.assertEqual(chat_request.reasoning_effort, openai_effort)

    def test_request_output_config_task_budget_is_logged_not_enforced(self):
        """``task_budget`` is a soft hint; ``max_tokens`` is the hard cap."""
        import logging

        serving = self._serving()
        request = self._anthropic_request(
            output_config={"task_budget": {"type": "tokens", "total": 32768}},
            stream=False,
        )
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic.serving", level=logging.INFO
        ) as log:
            chat_request = serving._convert_to_chat_completion_request(request)
        # max_tokens is untouched
        self.assertEqual(chat_request.max_tokens, 16)
        self.assertTrue(any("task_budget" in r and "32768" in r for r in log.output))

    def test_request_betas_is_accepted_and_logged(self):
        """``betas`` is accepted and logged; the local backend has no beta system."""
        import logging

        serving = self._serving()
        request = self._anthropic_request(betas=["thinking-2025-08-04"], stream=False)
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic.serving", level=logging.INFO
        ) as log:
            serving._convert_to_chat_completion_request(request)
        self.assertTrue(any("thinking-2025-08-04" in r for r in log.output))

    def test_assistant_thinking_history_is_rewrapped_for_chat_template(self):
        """Past-turn thinking blocks get re-emitted via wrap_reasoning_history."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "ponder"},
                        {"type": "text", "text": "hello"},
                    ],
                },
                {"role": "user", "content": "again"},
            ],
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        # ``ChatCompletionRequest.messages`` is a list of Pydantic
        # ChatCompletionMessage*Param instances; access via attributes.
        assistant_msg = next(m for m in chat_request.messages if m.role == "assistant")
        content = assistant_msg.content
        # Reasoning history sits in front; the thinking block itself is dropped
        # from the prompt so its text is not duplicated.
        if isinstance(content, list):
            texts = []
            for part in content:
                if isinstance(part, dict):
                    texts.append(part.get("text", ""))
                else:
                    texts.append(getattr(part, "text", "") or "")
        else:
            texts = [content]
        joined = "\n".join(texts)
        self.assertIn("<think>", joined)
        self.assertIn("ponder", joined)
        self.assertNotIn("<think>\nponder\n</think>\nponder", joined)

    def test_redacted_thinking_history_is_rejected(self):
        """``redacted_thinking`` cannot be rendered by local parsers."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            messages=[
                {
                    "role": "assistant",
                    "content": [
                        {"type": "redacted_thinking", "data": "opaque"},
                    ],
                },
            ],
        )
        with self.assertRaises(ValueError):
            serving._convert_to_chat_completion_request(request)

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

    def test_stream_consecutive_tool_calls_get_separate_blocks(self):
        """Two tool_use calls in sequence must occupy distinct content_block indices."""
        serving = self._serving(
            [
                _chunk(
                    [
                        _choice(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_a",
                                        "function": {
                                            "name": "alpha",
                                            "arguments": '{"x":1}',
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
                                        "index": 1,
                                        "id": "call_b",
                                        "function": {
                                            "name": "beta",
                                            "arguments": '{"y":2}',
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
        starts = [
            (e["index"], e["content_block"]["name"])
            for e in events
            if e["type"] == "content_block_start"
        ]
        stops = [e["index"] for e in events if e["type"] == "content_block_stop"]
        deltas = [
            (e["index"], e["delta"].get("partial_json"))
            for e in events
            if e["type"] == "content_block_delta"
            and e["delta"].get("type") == "input_json_delta"
        ]
        # Each tool gets its own start, its own stop, and its own
        # argument delta — without the fix, beta's args were appended
        # to alpha's index 0 block.
        self.assertEqual(starts, [(0, "alpha"), (1, "beta")])
        self.assertEqual(stops, [0, 1])
        self.assertEqual(deltas, [(0, '{"x":1}'), (1, '{"y":2}')])

    def test_stream_finish_chunk_with_payload_emits_delta(self):
        """A chunk carrying both finish_reason and content must not drop the content."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant"})]),
                _chunk([_choice({"content": "last token"}, finish_reason="stop")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        text_deltas = [
            e["delta"]["text"]
            for e in events
            if e["type"] == "content_block_delta"
            and e["delta"].get("type") == "text_delta"
        ]
        self.assertEqual(text_deltas, ["last token"])
        # And stop_reason still travels via message_delta
        message_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(message_delta["delta"]["stop_reason"], "end_turn")

    def test_stream_empty_completion_with_finish_reason_emits_message_delta(self):
        """An empty stream with a finish_reason is a legitimate stop, not api_error."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant"})]),
                _chunk([_choice({}, finish_reason="length")]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        types = [e["type"] for e in events]
        self.assertIn("message_start", types)
        self.assertIn("message_delta", types)
        self.assertIn("message_stop", types)
        self.assertNotIn("error", types)
        message_delta = next(e for e in events if e["type"] == "message_delta")
        self.assertEqual(message_delta["delta"]["stop_reason"], "max_tokens")

    def test_stream_no_finish_no_content_still_emits_api_error(self):
        """Backend that drops both content and finish_reason is genuinely broken."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant"})]),
                "data: [DONE]\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        types = [e["type"] for e in events]
        self.assertIn("error", types)
        err = next(e for e in events if e["type"] == "error")
        self.assertEqual(err["error"]["type"], "api_error")

    def test_stream_upstream_error_envelope_is_forwarded(self):
        """OpenAI handler streaming-error JSON must surface real type/message."""
        upstream_error = {
            "error": {
                "object": "error",
                "message": "context length exceeded",
                "type": "BadRequestError",
                "code": 400,
            }
        }
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant"})]),
                f"data: {json.dumps(upstream_error)}\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        err = next(e for e in events if e["type"] == "error")
        self.assertEqual(err["error"]["type"], "invalid_request_error")
        self.assertEqual(err["error"]["message"], "context length exceeded")
        # message_stop must still close the stream
        self.assertEqual(events[-1]["type"], "message_stop")

    def test_stream_parse_failure_closes_open_content_block(self):
        """Unparsable mid-stream chunk must still close any open content_block."""
        serving = self._serving(
            [
                _chunk([_choice({"role": "assistant", "content": "first"})]),
                "data: {not-json\n\n",
            ]
        )
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        types = [e["type"] for e in events]
        # Sequence: message_start, content_block_start, content_block_delta,
        # content_block_stop, error, message_stop
        self.assertIn("content_block_start", types)
        self.assertEqual(
            types.count("content_block_stop"),
            types.count("content_block_start"),
            f"unbalanced block events: {types}",
        )
        self.assertIn("error", types)
        self.assertEqual(types[-1], "message_stop")

    def test_stream_pre_first_chunk_value_error_emits_envelope(self):
        """ValueError before any chunk must yield a clean Anthropic error sequence."""

        class _RaisingOpenAI(_FakeOpenAIServingChat):
            def _generate_chat_stream(
                self, adapted_request, processed_request, raw_request
            ):
                async def _gen():
                    raise ValueError("tokenization failed")
                    yield  # pragma: no cover

                return _gen()

        serving = AnthropicServing(_RaisingOpenAI())
        events = asyncio.run(
            _collect_anthropic_events(serving, self._anthropic_request())
        )
        types = [e["type"] for e in events]
        self.assertEqual(types[0], "message_start")
        self.assertIn("error", types)
        err = next(e for e in events if e["type"] == "error")
        self.assertEqual(err["error"]["type"], "invalid_request_error")
        self.assertIn("tokenization failed", err["error"]["message"])
        self.assertEqual(types[-1], "message_stop")

    def test_server_tool_only_with_tool_choice_any_raises_400(self):
        """A request with only server-side tools cannot honor tool_choice=any."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            tool_choice={"type": "any"},
        )
        with self.assertRaises(ValueError) as ctx:
            serving._convert_to_chat_completion_request(request)
        self.assertIn("tool_choice", str(ctx.exception))

    def test_tool_choice_named_custom_tool_is_resolved(self):
        """tool_choice={type:'tool', name:'X'} where X is a custom tool wires through."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            tools=[
                {
                    "type": "custom",
                    "name": "lookup",
                    "input_schema": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    },
                }
            ],
            tool_choice={"type": "tool", "name": "lookup"},
        )
        # Must not AttributeError: Tool.function is a Pydantic model, not a
        # dict — access must be via .name, never .get("name").
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(chat_request.tool_choice.type, "function")
        self.assertEqual(chat_request.tool_choice.function.name, "lookup")

    def test_tool_choice_named_unknown_tool_raises_400(self):
        """tool_choice={type:'tool', name:'X'} where X is missing must raise."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            tools=[
                {
                    "type": "custom",
                    "name": "lookup",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            tool_choice={"type": "tool", "name": "nonexistent"},
        )
        with self.assertRaises(ValueError) as ctx:
            serving._convert_to_chat_completion_request(request)
        self.assertIn("nonexistent", str(ctx.exception))

    def test_convert_response_non_streaming_empty_content_keeps_block(self):
        """Empty-string completion must still produce a content list of len 1."""
        response = ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": ""},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 0,
                    "total_tokens": 5,
                },
            }
        )
        serving = self._serving()
        anthropic_response = serving._convert_response(response)
        self.assertEqual(len(anthropic_response.content), 1)
        self.assertEqual(anthropic_response.content[0].type, "text")
        self.assertEqual(anthropic_response.content[0].text, "")

    def test_error_response_does_not_leak_exception_name(self):
        """``error.type`` must stay in Anthropic's documented literal set."""
        serving = self._serving()
        response = serving._error_response(
            status_code=500,
            error_type="api_error",
            message="Internal server error",
            exception_name="KeyError",
        )
        body = json.loads(bytes(response.body).decode())
        self.assertEqual(body["error"]["type"], "api_error")

    def test_user_message_text_tool_text_preserves_order(self):
        """User message [text, tool_result, text] must stay user→tool→user on the wire."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "first"},
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_x",
                            "content": "ok",
                        },
                        {"type": "text", "text": "second"},
                    ],
                }
            ],
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        # chat_request.messages items are Pydantic ChatCompletionMessage*Param
        # variants — use attribute access, not subscripts.
        roles = [m.role for m in chat_request.messages]
        self.assertEqual(roles, ["user", "tool", "user"])
        self.assertEqual(chat_request.messages[0].content, "first")
        self.assertEqual(chat_request.messages[2].content, "second")

    def test_empty_text_assistant_turn_preserves_role_alternation(self):
        """Assistant turn with only empty text must NOT vanish from the wire."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            messages=[
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": [{"type": "text", "text": ""}]},
                {"role": "user", "content": "u2"},
            ],
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        roles = [m.role for m in chat_request.messages]
        # Without the fix this collapses to ['user', 'user'] and breaks
        # strict role-alternation chat templates (qwen, llama, mistral).
        self.assertEqual(roles, ["user", "assistant", "user"])

    def test_in_messages_system_merged_when_template_requires_first(self):
        """When the chat template rejects mid-conversation ``role: "system"``
        (e.g. Qwen's system-first guard), the converter folds the inline
        system turn into the leading system block so the template doesn't
        400. The request object itself is no longer mutated — detection runs
        in the serving layer on conversion."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "Reply with exactly: OK"},
                {"role": "user", "content": "go"},
            ],
        )
        self.assertIsNone(request.system)
        self.assertEqual([m.role for m in request.messages], ["user", "system", "user"])
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(
            [m.role for m in chat_request.messages], ["system", "user", "user"]
        )
        self.assertEqual(chat_request.messages[0].content, "Reply with exactly: OK")

    def test_in_messages_system_merged_with_top_level_when_merge(self):
        """On the merge path, a top-level ``system`` field and a mid-conversation
        system turn are joined into the leading system block; top-level text
        comes first."""
        serving = self._serving()
        request = self._anthropic_request(
            stream=False,
            system="You are terse.",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "One word only."},
                {"role": "user", "content": "go"},
            ],
        )
        self.assertEqual(request.system, "You are terse.")
        self.assertEqual([m.role for m in request.messages], ["user", "system", "user"])
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(
            [m.role for m in chat_request.messages], ["system", "user", "user"]
        )
        self.assertEqual(
            chat_request.messages[0].content, "You are terse.\nOne word only."
        )

    def test_in_messages_system_passed_through_when_template_allows_inline(self):
        """When the chat template renders ``role: "system"`` at any position
        (GLM / Kimi / Qwen3), the inline system turn stays at its original
        position — preserving the prefix cache and the request's structure."""
        serving = self._serving(chat_template=self.INLINE_SYSTEM_TEMPLATE)
        self.assertFalse(serving._merge_inline_system)
        request = self._anthropic_request(
            stream=False,
            system="You are terse.",
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "Reply with exactly: OK"},
                {"role": "user", "content": "go"},
            ],
        )
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(
            [m.role for m in chat_request.messages],
            ["system", "user", "system", "user"],
        )
        self.assertEqual(chat_request.messages[0].content, "You are terse.")
        self.assertEqual(chat_request.messages[2].content, "Reply with exactly: OK")

    def test_top_level_system_only_is_unchanged(self):
        """A request with only the top-level ``system`` field (no in-messages
        system turn) is unaffected on both detection paths: the system field is
        preserved verbatim and the dialogue order is untouched. Guards the
        common multi-turn path against regressions."""
        for template in (None, self.INLINE_SYSTEM_TEMPLATE):
            serving = self._serving(chat_template=template)
            request = self._anthropic_request(
                stream=False,
                system="You are a helpful assistant.",
                messages=[
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Hello Alice!"},
                    {"role": "user", "content": "What is my name?"},
                ],
            )
            self.assertEqual(request.system, "You are a helpful assistant.")
            self.assertEqual(
                [m.role for m in request.messages], ["user", "assistant", "user"]
            )
            chat_request = serving._convert_to_chat_completion_request(request)
            self.assertEqual(
                [m.role for m in chat_request.messages],
                ["system", "user", "assistant", "user"],
            )
            self.assertEqual(
                chat_request.messages[0].content, "You are a helpful assistant."
            )

    def test_constructed_message_objects_merged_on_merge_path(self):
        """Requests built programmatically with ``AnthropicMessage`` objects
        (e.g. ``handle_count_tokens``) also get inline system folded into the
        leading block on the merge path."""
        serving = self._serving()
        request = AnthropicMessagesRequest(
            model="m",
            max_tokens=8,
            messages=[
                AnthropicMessage(role="user", content="hi"),
                AnthropicMessage(role="system", content="be terse"),
                AnthropicMessage(role="user", content="go"),
            ],
        )
        self.assertEqual([m.role for m in request.messages], ["user", "system", "user"])
        chat_request = serving._convert_to_chat_completion_request(request)
        self.assertEqual(
            [m.role for m in chat_request.messages], ["system", "user", "user"]
        )
        self.assertEqual(chat_request.messages[0].content, "be terse")

    def test_thinking_history_drop_on_missing_detector(self):
        """Replaying a thinking block on a non-reasoning model should not 400."""

        class _NoDetectorOpenAI(_FakeOpenAIServingChat):
            def wrap_reasoning_history(self, text):
                raise ValueError("no reasoning detector is configured")

        serving = AnthropicServing(_NoDetectorOpenAI())
        request = self._anthropic_request(
            stream=False,
            messages=[
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "I think..."}],
                },
                {"role": "user", "content": "follow-up"},
            ],
        )
        # Must convert successfully; the thinking block is silently dropped.
        chat_request = serving._convert_to_chat_completion_request(request)
        roles = [m.role for m in chat_request.messages]
        self.assertIn("user", roles)
        # The assistant turn was rendered (as empty placeholder) so
        # alternation is preserved.
        self.assertIn("assistant", roles)

    def test_stop_reason_content_filter_falls_back_with_warning(self):
        """Unmapped OpenAI finish_reasons default to 'end_turn' + log a warning.

        ``content_filter`` and ``abort`` have no entry in STOP_REASON_MAP
        because Anthropic's ``stop_reason`` Literal (end_turn/max_tokens/
        stop_sequence/tool_use) has no perfect target. The fallback must
        produce a spec-valid stop_reason and a WARNING so operators don't
        silently lose the safety/abort signal.
        """
        import logging

        response = ChatCompletionResponse.model_validate(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "content_filter",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )
        serving = self._serving()
        with self.assertLogs(
            "sglang.srt.entrypoints.anthropic.serving", level=logging.WARNING
        ) as log:
            anthropic_response = serving._convert_response(response)
        self.assertEqual(anthropic_response.stop_reason, "end_turn")
        self.assertTrue(
            any("content_filter" in rec for rec in log.output),
            f"expected a warning mentioning the unmapped finish_reason: {log.output}",
        )


class TestDetectInlineSystemSupport(unittest.TestCase):
    """Chat-template detection for mid-conversation system messages (#28883)."""

    def test_guarded_template_not_supported(self):
        guarded = (
            "{%- for message in messages %}"
            "{%- if message.role == 'system' and not loop.first %}"
            "{{- raise_exception('system must be first') }}"
            "{%- endif %}"
            "{%- endfor %}"
        )
        self.assertFalse(detect_inline_system_support(guarded))

    def test_inline_template_supported(self):
        inline = (
            "{%- for message in messages %}"
            "{{- message.role }}: {{ message.content }}\n"
            "{%- endfor %}"
        )
        self.assertTrue(detect_inline_system_support(inline))

    def test_silent_drop_template_not_supported(self):
        # Renders only the leading system; silently ignores later system turns.
        silent_drop = (
            "{%- if messages[0].role == 'system' %}"
            "{{ messages[0].content }}\n"
            "{%- endif %}"
            "{%- for message in messages %}"
            "{%- if message.role in ('user', 'assistant') %}"
            "{{ message.role }}: {{ message.content }}\n"
            "{%- endif %}"
            "{%- endfor %}"
        )
        self.assertFalse(detect_inline_system_support(silent_drop))

    def test_no_template_not_supported(self):
        self.assertFalse(detect_inline_system_support(None))
        self.assertFalse(detect_inline_system_support(""))


if __name__ == "__main__":
    unittest.main()
