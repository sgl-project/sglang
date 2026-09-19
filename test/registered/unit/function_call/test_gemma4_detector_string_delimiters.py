"""Unit tests for Gemma4Detector string arguments — no server, no model loading.

Gemma 4 wraps string arguments in ``<|"|>``, but sometimes omits the opening
delimiter and still emits the closing one, e.g. ``query:weather in Tokyo<|"|>``.
"""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.gemma4_detector import Gemma4Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

Q = '<|"|>'


class TestGemma4StringDelimiters(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="web_search",
                    description="Search the web",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                ),
            )
        ]

    def _parse(self, args: str) -> dict:
        text = f"<|tool_call>call:web_search{{{args}}}<tool_call|>"
        result = Gemma4Detector().detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "web_search")
        return json.loads(result.calls[0].parameters)

    def _stream(self, chunks: list) -> tuple:
        detector, name, params = Gemma4Detector(), None, ""
        for chunk in chunks:
            for call in detector.parse_streaming_increment(chunk, self.tools).calls:
                name = call.name or name
                params += call.parameters or ""
        return name, json.loads(params) if params else None

    def test_well_formed_string(self):
        self.assertEqual(
            self._parse(f"query:{Q}weather in Tokyo{Q}"), {"query": "weather in Tokyo"}
        )

    def test_omitted_opening_delimiter(self):
        self.assertEqual(
            self._parse(f"query:weather in Tokyo{Q}"), {"query": "weather in Tokyo"}
        )

    def test_omitted_opening_delimiter_with_comma(self):
        self.assertEqual(
            self._parse(f"query:Paris, France{Q},limit:5"),
            {"query": "Paris, France", "limit": 5},
        )

    def test_omitted_opening_delimiter_in_nested_object(self):
        self.assertEqual(
            self._parse(f"filters:{{city:Tokyo{Q}}},query:{Q}x{Q}"),
            {"filters": {"city": "Tokyo"}, "query": "x"},
        )

    def test_omitted_opening_delimiter_in_array(self):
        self.assertEqual(self._parse(f"query:[{Q}a{Q},b{Q}]"), {"query": ["a", "b"]})

    def test_other_values_unchanged(self):
        self.assertEqual(
            self._parse(f"limit:5,exact:true,query:{Q}Note: a, b{Q}"),
            {"limit": 5, "exact": True, "query": "Note: a, b"},
        )

    def test_malformed_array_terminates(self):
        text = f"<|tool_call>call:web_search{{query:[ k:{Q}1]]}}<tool_call|>"
        result = Gemma4Detector().detect_and_parse(text, self.tools)
        self.assertEqual([call.name for call in result.calls], ["web_search"])

    def test_streaming_omitted_opening_delimiter(self):
        chunks = [
            "<|tool_call>",
            "call",
            ":",
            "web",
            "_",
            "search",
            "{",
            "query",
            ":",
            "weather",
            " in",
            " Tokyo",
            Q,
            "}",
            "<tool_call|>",
        ]
        self.assertEqual(
            self._stream(chunks), ("web_search", {"query": "weather in Tokyo"})
        )

    def test_streaming_well_formed(self):
        chunks = [
            "<|tool_call>",
            "call",
            ":",
            "web",
            "_",
            "search",
            "{",
            "query",
            ":",
            Q,
            "weather",
            " in",
            " Tokyo",
            Q,
            "}",
            "<tool_call|>",
        ]
        self.assertEqual(
            self._stream(chunks), ("web_search", {"query": "weather in Tokyo"})
        )


if __name__ == "__main__":
    unittest.main()
