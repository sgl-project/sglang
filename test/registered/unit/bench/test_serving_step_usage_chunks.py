"""Unit tests for the usage-driven TTFT/ITL accounting in the chat bench client.

A decode step whose text the reasoning / tool-call parser buffered produces no
visible delta, and tokens turned into structured `tool_calls` never show up as
text either. Driving TTFT/ITL from `usage.completion_tokens` keeps those tokens
accounted for; without it the run reports ITL 0 and a token count that comes from
`max_tokens` rather than from the server.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede any import that pulls in sgl_kernel

import asyncio
import json
import unittest
from argparse import Namespace
from typing import List

import sglang.benchmark.serving as bench
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_API_URL = "http://127.0.0.1:30000/v1/chat/completions"


def _chunk(delta=None, usage_tokens=None, finish_reason=None):
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish_reason}],
    }
    if usage_tokens is not None:
        payload["usage"] = {"prompt_tokens": 5, "completion_tokens": usage_tokens}
    return f"data: {json.dumps(payload)}\n\n"


class _FakeContent:
    def __init__(self, lines: List[str]):
        self._lines = lines

    def __aiter__(self):
        async def gen():
            for line in self._lines:
                yield line.encode()

        return gen()


class _FakeResponse:
    def __init__(self, lines: List[str], status: int = 200):
        self.status = status
        self.reason = "OK"
        self.content = _FakeContent(lines)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """Captures the payload and replays a canned SSE stream."""

    def __init__(self, lines: List[str], captured: dict):
        self._lines = lines
        self._captured = captured

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url=None, json=None, headers=None):
        self._captured["url"] = url
        self._captured["payload"] = json
        return _FakeResponse(self._lines)


def _run(lines: List[str], step_usage_chunks: str = "all", output_len: int = 200):
    captured = {}
    saved_args = getattr(bench, "args", None)
    saved_session = bench._create_bench_client_session
    bench.args = Namespace(
        disable_stream=False,
        disable_ignore_eos=True,
        cache_report=False,
        print_requests=False,
        stream_step_usage_chunks=step_usage_chunks,
    )
    bench._create_bench_client_session = lambda: _FakeSession(lines, captured)
    try:
        request_input = bench.RequestFuncInput(
            prompt="hi",
            api_url=_API_URL,
            prompt_len=5,
            output_len=output_len,
            model="test-model",
            lora_name=None,
            image_data=None,
            extra_request_body={},
        )
        output = asyncio.run(bench.async_request_openai_chat_completions(request_input))
    finally:
        bench._create_bench_client_session = saved_session
        if saved_args is None:
            del bench.args
        else:
            bench.args = saved_args
    return output, captured


class TestStreamOptionsInjection(CustomTestCase):
    def test_off_does_not_touch_payload(self):
        _, captured = _run([_chunk(usage_tokens=1), "data: [DONE]\n\n"], "off")
        self.assertNotIn("stream_options", captured["payload"])

    def test_all_requests_usage_chunks(self):
        _, captured = _run([_chunk(usage_tokens=1), "data: [DONE]\n\n"], "all")
        self.assertEqual(
            captured["payload"]["stream_options"],
            {
                "include_usage": True,
                "continuous_usage_stats": True,
                "step_usage_chunks": "all",
            },
        )


class TestUsageDrivenAccounting(CustomTestCase):
    def test_buffered_steps_are_counted(self):
        """Empty-delta usage chunks contribute tokens, TTFT and ITL samples."""
        lines = [
            _chunk(delta={"role": "assistant", "content": ""}),
            _chunk(delta={"content": "\n"}, usage_tokens=1),
            # Parser buffered these steps: no visible delta, usage keeps moving.
            _chunk(usage_tokens=4),
            _chunk(usage_tokens=7),
            _chunk(delta={"tool_calls": [{"index": 0}]}, usage_tokens=10),
            _chunk(finish_reason="tool_calls", usage_tokens=10),
            "data: [DONE]\n\n",
        ]
        output, _ = _run(lines)
        self.assertTrue(output.success)
        self.assertEqual(output.output_len, 10)
        self.assertGreater(output.ttft, 0.0)
        # 10 tokens: the first sets TTFT, the other 9 become ITL samples.
        self.assertEqual(len(output.itl), 9)

    def test_tool_call_only_stream_is_not_empty(self):
        """A pure tool-call answer used to yield zero ITL samples."""
        lines = [
            _chunk(delta={"role": "assistant", "content": ""}),
            _chunk(delta={"tool_calls": [{"index": 0}]}, usage_tokens=1),
            _chunk(delta={"tool_calls": [{"index": 0}]}, usage_tokens=3),
            _chunk(finish_reason="tool_calls", usage_tokens=3),
            "data: [DONE]\n\n",
        ]
        output, _ = _run(lines)
        self.assertEqual(output.output_len, 3)
        self.assertEqual(len(output.itl), 2)

    def test_output_len_not_taken_from_max_tokens(self):
        """The reported token count comes from usage, not from the request."""
        lines = [
            _chunk(delta={"content": "a"}, usage_tokens=1),
            _chunk(delta={"content": "b"}, usage_tokens=2),
            "data: [DONE]\n\n",
        ]
        output, _ = _run(lines, output_len=999)
        self.assertEqual(output.output_len, 2)

    def test_falls_back_to_content_when_server_sends_no_usage(self):
        """Servers without per-chunk usage keep the previous behavior."""
        lines = [
            _chunk(delta={"role": "assistant", "content": ""}),
            _chunk(delta={"content": "a"}),
            _chunk(delta={"content": "b"}),
            _chunk(delta={"content": "c"}),
            "data: [DONE]\n\n",
        ]
        output, _ = _run(lines, "off", output_len=7)
        self.assertEqual(output.generated_text, "abc")
        self.assertEqual(len(output.itl), 2)
        # No usage anywhere: output_len stays at the requested value.
        self.assertEqual(output.output_len, 7)


if __name__ == "__main__":
    unittest.main()
