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
    def _validate_request(self, chat_request):
        return None

    def _convert_to_internal_request(self, chat_request, raw_request):
        return SimpleNamespace(), chat_request

    async def _handle_non_streaming_request(
        self, adapted_request, processed_request, raw_request
    ):
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "type": "invalid_request_error",
                    "message": "context length exceeded",
                }
            },
        )


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

    def test_stream_reasoning_content_uses_thinking_block_with_signature_delta(self):
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
        self.assertEqual(content_events[0]["content_block"]["signature"], "")
        self.assertEqual(content_events[1]["delta"]["type"], "thinking_delta")
        self.assertEqual(content_events[1]["delta"]["thinking"], "think first")
        self.assertEqual(content_events[2]["delta"]["type"], "signature_delta")
        self.assertEqual(content_events[3]["type"], "content_block_stop")
        self.assertEqual(content_events[4]["content_block"]["type"], "text")

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
        request = AnthropicMessagesRequest.model_validate(
            {
                "model": "test-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "call a tool"}],
                "tools": [{"name": "custom_without_schema"}],
            }
        )

        with self.assertRaisesRegex(ValueError, "input_schema"):
            self._serving()._convert_to_chat_completion_request(request)

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


if __name__ == "__main__":
    unittest.main()
