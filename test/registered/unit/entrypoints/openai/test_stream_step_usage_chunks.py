"""Unit tests for `stream_options.step_usage_chunks` — no server, no model loading.

Streaming chat only emits an SSE chunk when a parser hands back visible text.
While the reasoning / tool-call parser buffers a partial marker it swallows the
step's text, so those tokens are already counted in usage but the client sees no
chunk and charges them to TTFT, which shrinks the decode window and inflates the
reported decode speed. `stream_options.step_usage_chunks` (off/first_token/all,
default off) makes such a step observable by emitting an empty-delta chunk that
carries usage.

The tests drive `OpenAIServingChat._generate_stream_content` directly with stubs
for the tokenizer manager and the parsers.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import asyncio
import json
import unittest

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, StreamOptions
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_CHUNK_ID = "chatcmpl-test"
_PROMPT_TOKENS = 7
_COMPLETION_TOKENS = 3
# tool_choice defaults are derived from tools: without tools it becomes "none",
# which would keep _generate_stream_content off the tool-call branch entirely.
_TOOLS = [
    {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {"type": "object"}},
    }
]


class _FakeServerArgs:
    incremental_streaming_output = False
    stream_response_default_include_usage = False
    enable_cache_report = False


class _FakeTokenizerManager:
    def __init__(self, contents=()):
        self.server_args = _FakeServerArgs()
        self._contents = contents

    async def generate_request(self, adapted_request, raw_request):
        for content in self._contents:
            yield content

    def create_abort_task(self, adapted_request):
        return None


class _StubServingChat:
    """Only the attributes and collaborators `_generate_stream_content` touches.

    `tool_emits=None` means the tool-call parser buffered the whole step and
    emitted nothing; otherwise it is the chunk the parser hands back this step.
    `reasoning_text` / `remaining_delta` stand in for the reasoning parser's two
    return values, and `unstreamed_args` for the tool-call arguments flushed when
    generation finishes.
    """

    def __init__(
        self,
        tool_emits=None,
        reasoning_parser=None,
        tool_call_parser="dsml",
        reasoning_text=None,
        remaining_delta="",
        unstreamed_args=None,
        contents=(),
    ):
        self.tokenizer_manager = _FakeTokenizerManager(contents)
        self.reasoning_parser = reasoning_parser
        self.tool_call_parser = tool_call_parser
        self._tool_emits = tool_emits
        self._reasoning_text = reasoning_text
        self._remaining_delta = remaining_delta
        self._unstreamed_args = unstreamed_args

    def _effective_tools(self, request):
        return [{"type": "function"}]

    def _tool_call_parsing_active(self, request):
        return OpenAIServingChat._tool_call_parsing_active(self, request)

    def _continuous_usage_cached_details(self, content):
        return None

    def _check_for_unstreamed_tool_args(self, parser, content, request, index):
        return self._unstreamed_args

    def _process_reasoning_stream(
        self, index, delta, reasoning_parser_dict, content, request, finish_reason_type
    ):
        return self._reasoning_text, self._remaining_delta

    async def _process_tool_call_stream(
        self,
        index,
        delta,
        parser_dict,
        content,
        request,
        has_tool_calls,
        continuous_usage_stats,
        flush=False,
    ):
        if self._tool_emits is not None:
            yield self._tool_emits


def _make_request(step_usage_chunks=None, continuous_usage_stats=True):
    stream_options = StreamOptions(
        include_usage=True,
        continuous_usage_stats=continuous_usage_stats,
    )
    if step_usage_chunks is not None:
        stream_options.step_usage_chunks = step_usage_chunks
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
        tools=_TOOLS,
        stream=True,
        stream_options=stream_options,
    )


async def _collect(
    stub,
    request,
    first_delta_seen,
    continuous_usage_stats=True,
    choice_logprobs=None,
    finish_reason_type=None,
    parser_dict=None,
):
    """Run one decode step and return the SSE chunks it produced."""
    content = {"text": "<|tool_call_start|>", "meta_info": {"id": _CHUNK_ID}}
    chunks = []
    async for chunk in OpenAIServingChat._generate_stream_content(
        stub,
        content=content,
        index=0,
        request=request,
        stream_offsets={},
        reasoning_parser_dict={},
        parser_dict={0: object()} if parser_dict is None else parser_dict,
        has_tool_calls={},
        choice_logprobs=choice_logprobs,
        finish_reason_type=finish_reason_type,
        continuous_usage_stats=continuous_usage_stats,
        prompt_tokens={0: _PROMPT_TOKENS},
        reasoning_tokens={0: 0},
        completion_tokens={0: _COMPLETION_TOKENS},
        first_delta_seen=first_delta_seen,
    ):
        chunks.append(chunk)
    return chunks


def _parse(chunk):
    prefix, _, payload = chunk.partition("data: ")
    assert prefix == "", f"unexpected SSE prefix: {chunk!r}"
    return json.loads(payload)


class TestStreamOptionsStepUsageChunks(CustomTestCase):
    """Protocol layer: the value contract of step_usage_chunks."""

    def test_default_is_off(self):
        self.assertEqual(StreamOptions().step_usage_chunks, "off")

    def test_accepts_documented_values(self):
        for value in ("off", "first_token", "all"):
            self.assertEqual(
                StreamOptions(step_usage_chunks=value).step_usage_chunks, value
            )

    def test_rejects_unknown_value(self):
        with self.assertRaises(ValueError):
            StreamOptions(step_usage_chunks="sometimes")

    def test_reachable_through_request(self):
        request = _make_request("first_token")
        self.assertEqual(request.stream_options.step_usage_chunks, "first_token")


class TestBufferedStepUsageChunk(CustomTestCase):
    """Steps whose text the parser buffered (no chunk emitted)."""

    def test_off_emits_nothing(self):
        chunks = asyncio.run(_collect(_StubServingChat(), _make_request("off"), {}))
        self.assertEqual(chunks, [])

    def test_missing_stream_options_emits_nothing(self):
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            tools=_TOOLS,
            stream=True,
        )
        chunks = asyncio.run(_collect(_StubServingChat(), request, {}))
        self.assertEqual(chunks, [])

    def test_first_token_emits_usage_only_chunk(self):
        chunks = asyncio.run(
            _collect(_StubServingChat(), _make_request("first_token"), {})
        )
        self.assertEqual(len(chunks), 1)

        data = _parse(chunks[0])
        self.assertEqual(data["id"], _CHUNK_ID)
        self.assertEqual(data["object"], "chat.completion.chunk")
        self.assertEqual(data["usage"]["prompt_tokens"], _PROMPT_TOKENS)
        self.assertEqual(data["usage"]["completion_tokens"], _COMPLETION_TOKENS)

        delta = data["choices"][0]["delta"]
        self.assertIsNone(delta.get("content"))
        self.assertIsNone(delta.get("reasoning_content"))
        self.assertIsNone(data["choices"][0]["finish_reason"])

    def test_first_token_stops_after_first_delta(self):
        chunks = asyncio.run(
            _collect(_StubServingChat(), _make_request("first_token"), {0: True})
        )
        self.assertEqual(chunks, [])

    def test_all_emits_after_first_delta(self):
        chunks = asyncio.run(
            _collect(_StubServingChat(), _make_request("all"), {0: True})
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            _parse(chunks[0])["usage"]["completion_tokens"], _COMPLETION_TOKENS
        )

    def test_requires_continuous_usage_stats(self):
        request = _make_request("all", continuous_usage_stats=False)
        chunks = asyncio.run(
            _collect(_StubServingChat(), request, {}, continuous_usage_stats=False)
        )
        self.assertEqual(chunks, [])


class TestVisibleStepNotAffected(CustomTestCase):
    """Steps with visible output must behave exactly as before."""

    def test_visible_chunk_passes_through_without_extra_chunk(self):
        stub = _StubServingChat(tool_emits='data: {"choices": []}\n\n')
        chunks = asyncio.run(_collect(stub, _make_request("all"), {}))
        self.assertEqual(chunks, ['data: {"choices": []}\n\n'])

    def test_visible_chunk_marks_first_delta_seen(self):
        stub = _StubServingChat(tool_emits='data: {"choices": []}\n\n')
        first_delta_seen = {}
        asyncio.run(_collect(stub, _make_request("first_token"), first_delta_seen))
        self.assertTrue(first_delta_seen[0])


class TestStepEmittedOnEveryEmitPath(CustomTestCase):
    """Every emit path must mark the step, otherwise a usage chunk is added."""

    def test_reasoning_chunk_counts_as_emitted(self):
        stub = _StubServingChat(
            reasoning_parser="deepseek-r1",
            tool_call_parser=None,
            reasoning_text="thinking...",
        )
        first_delta_seen = {}
        chunks = asyncio.run(
            _collect(stub, _make_request("first_token"), first_delta_seen)
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            _parse(chunks[0])["choices"][0]["delta"]["reasoning_content"],
            "thinking...",
        )
        self.assertTrue(first_delta_seen[0])

    def test_regular_content_chunk_counts_as_emitted(self):
        stub = _StubServingChat(tool_call_parser=None)
        first_delta_seen = {}
        chunks = asyncio.run(
            _collect(stub, _make_request("first_token"), first_delta_seen)
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            _parse(chunks[0])["choices"][0]["delta"]["content"], "<|tool_call_start|>"
        )
        self.assertTrue(first_delta_seen[0])

    def test_unstreamed_tool_args_counts_as_emitted(self):
        tail = 'data: {"choices": [{"index": 0}]}\n\n'
        stub = _StubServingChat(unstreamed_args=tail)
        first_delta_seen = {}
        chunks = asyncio.run(
            _collect(
                stub,
                _make_request("first_token"),
                first_delta_seen,
                finish_reason_type="stop",
            )
        )
        self.assertEqual(chunks, [tail])
        self.assertTrue(first_delta_seen[0])

    def test_logprobs_flush_counts_as_emitted(self):
        stub = _StubServingChat()
        first_delta_seen = {}
        chunks = asyncio.run(
            _collect(
                stub,
                _make_request("first_token"),
                first_delta_seen,
                choice_logprobs={"content": []},
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(_parse(chunks[0])["choices"][0]["logprobs"], {"content": []})
        self.assertTrue(first_delta_seen[0])


class TestStreamStateInit(CustomTestCase):
    """first_delta_seen initialization in _generate_chat_stream."""

    def test_first_delta_seen_initialized_per_stream(self):
        stub = _StubServingChat(contents=())
        request = ChatCompletionRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            tools=_TOOLS,
            stream=True,
            stream_options=StreamOptions(include_usage=False),
        )

        async def run():
            chunks = []
            async for chunk in OpenAIServingChat._generate_chat_stream(
                stub, adapted_request=None, request=request, raw_request=None
            ):
                chunks.append(chunk)
            return chunks

        self.assertEqual(asyncio.run(run()), ["data: [DONE]\n\n"])


if __name__ == "__main__":
    unittest.main()
