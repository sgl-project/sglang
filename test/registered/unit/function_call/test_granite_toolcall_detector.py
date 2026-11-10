"""Unit tests for GraniteToolCallDetector — no server, no model loading."""

import json
import random

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.granite_toolcall_detector import GraniteToolCallDetector
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def _call(name, params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params)
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def _assemble(detector, chunks, tools):
    text, calls = [], {}
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        text.append(result.normal_text)
        for item in result.calls:
            entry = calls.setdefault(item.tool_index, {"name": None, "args": ""})
            entry["name"] = item.name or entry["name"]
            entry["args"] += item.parameters or ""
    return "".join(text), [calls[i] for i in sorted(calls)]


class TestGraniteToolCallDetector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "days": {"type": "integer"},
                            "verbose": {"type": "boolean"},
                            "tags": {"type": "array"},
                        },
                    },
                ),
            ),
            Tool(function=Function(name="noargs", parameters={"type": "object"})),
        ]
        self.samples = [
            "plain text, no tool",
            _call("get_weather", [("city", "Paris"), ("days", "3")]),
            "Checking.\n"
            + _call("get_weather", [("city", "A"), ("verbose", "True")])
            + "\n"
            + _call("noargs", []),
            _call("get_weather", [("city", "multi\nline"), ("tags", '["a", 1]')]),
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nX\n<parameter=days>\n2\n</function>\n</tool_call>",
            "1 < 2 then " + _call("noargs", []),
        ]

    def test_non_streaming_matches_qwen3_coder(self):
        for text in self.samples:
            g = GraniteToolCallDetector().detect_and_parse(text, self.tools)
            q = Qwen3CoderDetector().detect_and_parse(text, self.tools)
            self.assertEqual(g.normal_text, q.normal_text)
            self.assertEqual(
                [(c.tool_index, c.name, c.parameters) for c in g.calls],
                [(c.tool_index, c.name, c.parameters) for c in q.calls],
            )

    def test_streaming_matches_qwen3_coder(self):
        rng = random.Random(0)
        for text in self.samples:
            splits = [list(text), [text]]
            splits.append([text[i : i + 5] for i in range(0, len(text), 5)])
            pieces, i = [], 0
            while i < len(text):
                n = rng.randint(1, 9)
                pieces.append(text[i : i + n])
                i += n
            splits.append(pieces)
            expected = _assemble(Qwen3CoderDetector(), [text], self.tools)
            for chunks in splits:
                self.assertEqual(
                    _assemble(GraniteToolCallDetector(), chunks, self.tools), expected
                )

    def test_typed_arguments(self):
        text = _call(
            "get_weather", [("city", "Paris"), ("days", "3"), ("verbose", "true")]
        )
        result = GraniteToolCallDetector().detect_and_parse(text, self.tools)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"city": "Paris", "days": 3, "verbose": True},
        )

    def test_content_is_not_stripped(self):
        # Newline handling belongs to the reasoning parser, not the tool parser.
        text = "\n\nanswer\n" + _call("noargs", [])
        result = GraniteToolCallDetector().detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "\n\nanswer\n")
        streamed, _ = _assemble(GraniteToolCallDetector(), list(text), self.tools)
        self.assertEqual(streamed, "\n\nanswer\n")

    def test_registered_as_granite_toolcall_parser(self):
        self.assertIs(
            FunctionCallParser.ToolCallParserEnum["granite_toolcall_parser"],
            GraniteToolCallDetector,
        )
        parser = FunctionCallParser(self.tools, "granite_toolcall_parser")
        text = _call("get_weather", [("city", "Oslo")])
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(normal, "")
        self.assertEqual(calls[0].name, "get_weather")


if __name__ == "__main__":
    import unittest

    unittest.main()
