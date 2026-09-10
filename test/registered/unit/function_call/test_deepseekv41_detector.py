"""Unit tests for DeepSeekV41Detector (spaced DSML tags) -- no server, no model loading."""

import json

from sglang.srt.entrypoints.openai import encoding_dsv41
from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

CHUNK_SIZES = [1, 2, 3, 5, 7, 11, 23, 1000]


def _tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get weather information",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="lookup",
                description="Look up a value",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                        "flags": {"type": "array"},
                    },
                },
            ),
        ),
    ]


def _assemble(calls):
    """Streamed ToolCallItems -> [(name, parsed arguments)] per tool_index."""
    by_index = {}
    for call in calls:
        entry = by_index.setdefault(call.tool_index, {"name": None, "args": ""})
        if call.name:
            entry["name"] = call.name
        entry["args"] += call.parameters or ""
    return [
        (entry["name"], json.loads(entry["args"]))
        for _, entry in sorted(by_index.items())
    ]


class TestDeepSeekV41RoundTrip(CustomTestCase):
    """Encoder-rendered assistant tool calls parse back to the same arguments,
    in one shot and at every chunk size."""

    ARGUMENTS = {"query": '{"a": 1}', "limit": 2, "flags": [1, True, None]}

    def setUp(self):
        self.tools = _tools()
        self.completion = encoding_dsv41.render_message(
            1,
            [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "reasoning_content": "reason",
                    "content": "summary",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": json.dumps(self.ARGUMENTS),
                            },
                        }
                    ],
                },
            ],
            thinking_mode="thinking",
        )
        self.completion = "<think>" + self.completion
        self.expected = [("lookup", self.ARGUMENTS)]

    def test_one_shot(self):
        parser = FunctionCallParser(self.tools, "deepseekv41")
        reasoning, content = ReasoningParser("deepseek-v41").parse_non_stream(
            self.completion
        )
        self.assertEqual(reasoning, "reason")
        normal, calls = parser.parse_non_stream(content)
        self.assertEqual(normal, "summary")
        self.assertEqual(
            [(c.name, json.loads(c.parameters)) for c in calls], self.expected
        )

    def test_streaming_at_every_chunk_size(self):
        for chunk_size in CHUNK_SIZES:
            with self.subTest(chunk_size=chunk_size):
                reasoning_parser = ReasoningParser("deepseek-v41")
                tool_parser = FunctionCallParser(self.tools, "deepseekv41")
                reasoning, normal, calls = "", "", []
                for i in range(0, len(self.completion), chunk_size):
                    reason, content = reasoning_parser.parse_stream_chunk(
                        self.completion[i : i + chunk_size]
                    )
                    reasoning += reason or ""
                    text, delta = tool_parser.parse_stream_chunk(content or "")
                    normal += text
                    calls.extend(delta)
                reason, content = reasoning_parser.parse_stream_end()
                reasoning += reason or ""
                text, delta = tool_parser.parse_stream_chunk(content or "")
                normal += text
                calls.extend(delta)
                text, delta = tool_parser.parse_stream_end()
                normal += text
                calls.extend(delta)
                self.assertEqual(reasoning, "reason")
                # The blank line before the block is released or trimmed depending
                # on where the chunk boundary falls; the shared base behaves the
                # same for V4, so only the prose itself is pinned here.
                self.assertEqual(normal.strip(), "summary")
                self.assertEqual(_assemble(calls), self.expected)


if __name__ == "__main__":
    import unittest

    unittest.main()
