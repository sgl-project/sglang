"""Unit tests for DeepSeekV41Detector (spaced DSML tags) -- no server, no model loading."""

import json

from sglang.srt.entrypoints.openai import encoding_dsv41
from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.deepseekv41_detector import DeepSeekV41Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

CHUNK_SIZES = [1, 2, 3, 5, 7, 11, 23, 1000]
DSML = "｜DSML｜"


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


class TestDeepSeekV41ConstrainedDecoding(CustomTestCase):
    """A forced call must open the calls block before the first invoke; the
    per-tool legacy tag started the grammar at the invoke trigger, the model
    closed a block it had not opened, and the parser dropped the call."""

    def setUp(self):
        self.tools = _tools()
        self.detector = DeepSeekV41Detector()

    def test_no_builtin_structural_tag(self):
        """xgrammar's builtin deepseek_v4 tag is the unspaced grammar."""
        self.assertIsNone(self.detector.get_structural_tag_name())
        self.assertIsNone(self.detector.get_structural_tag([], "required"))

    def test_required_tag_wraps_invokes_in_the_calls_block(self):
        tag = self.detector.get_structural_tag(
            tools=self.tools, tool_choice="required"
        )
        opener, calls, closer = tag.format.elements
        self.assertEqual(opener.value, f"\n\n<{DSML} calls>\n")
        self.assertEqual(closer.value, f"</{DSML} calls>")
        self.assertTrue(calls.at_least_one)
        self.assertEqual(
            [t.begin for t in calls.tags],
            [f'<{DSML} invoke name="get_weather">', f'<{DSML} invoke name="lookup">'],
        )
        self.assertEqual({t.end for t in calls.tags}, {f"</{DSML} invoke>\n"})

    def test_named_tool_choice_keeps_only_that_tool(self):
        tag = self.detector.get_structural_tag(
            tools=self.tools,
            tool_choice=ToolChoice(function=ToolChoiceFuncName(name="lookup")),
        )
        _, call, _ = tag.format.elements
        self.assertEqual(call.begin, f'<{DSML} invoke name="lookup">')
        self.assertEqual(call.type, "tag")

    def test_parallel_off_allows_one_invoke(self):
        tag = self.detector.get_structural_tag(
            tools=self.tools, tool_choice="required", parallel_tool_calls=False
        )
        _, calls, _ = tag.format.elements
        self.assertEqual(calls.type, "or")
        self.assertEqual(len(calls.elements), 2)

    def test_auto_tag_triggers_on_the_calls_block(self):
        tag = self.detector.get_structural_tag(tools=self.tools, tool_choice="auto")
        self.assertEqual(tag.format.triggers, [f"<{DSML} calls>"])
        self.assertEqual(tag.format.tags[0].begin, f"<{DSML} calls>\n")
        self.assertEqual(tag.format.tags[0].end, f"</{DSML} calls>")

    def test_thinking_mode_prefixes_the_reasoning_span(self):
        tag = self.detector.get_structural_tag(
            tools=self.tools, tool_choice="required", thinking_mode=True
        )
        reasoning, body = tag.format.elements
        self.assertEqual(reasoning.end, "</think>")
        self.assertEqual(body.elements[0].value, f"\n\n<{DSML} calls>\n")

    def test_parser_uses_the_native_tag_for_required(self):
        parser = FunctionCallParser(self.tools, "deepseekv41")
        kind, tag = parser.get_structure_constraint("required")
        self.assertEqual(kind, "structural_tag")
        self.assertEqual(tag.format.elements[0].value, f"\n\n<{DSML} calls>\n")


if __name__ == "__main__":
    import unittest

    unittest.main()
