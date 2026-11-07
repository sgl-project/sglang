"""Unit tests for Gemma4Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.gemma4_detector import Gemma4Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

Q = '<|"|>'  # Gemma4 string delimiter


class TestGemma4Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search the web",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                ),
            ),
        ]
        self.detector = Gemma4Detector()

    @staticmethod
    def _call(name, args):
        body = ",".join(f"{k}:{Q}{v}{Q}" for k, v in args.items())
        return f"<|tool_call>call:{name}{{{body}}}<tool_call|>"

    @staticmethod
    def _collect(detector, text, tools, chunk=7):
        """Feed text in small chunks; return {tool_index: (name, arguments)}."""
        seen = {}
        for i in range(0, len(text), chunk):
            for call in detector.parse_streaming_increment(text[i : i + chunk], tools).calls:
                entry = seen.setdefault(call.tool_index, {"name": None, "args": ""})
                if call.name:
                    entry["name"] = call.name
                entry["args"] += call.parameters or ""
        return {k: (v["name"], json.loads(v["args"] or "{}")) for k, v in seen.items()}

    # ==================== Non-streaming ====================

    def test_detect_and_parse_single(self):
        text = self._call("get_weather", {"city": "Hà Nội", "unit": "celsius"})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"city": "Hà Nội", "unit": "celsius"}
        )

    def test_detect_and_parse_parallel(self):
        text = self._call("get_weather", {"city": "A"}) + self._call("get_weather", {"city": "B"})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual([c.name for c in result.calls], ["get_weather", "get_weather"])
        self.assertEqual([json.loads(c.parameters)["city"] for c in result.calls], ["A", "B"])

    # ==================== Streaming ====================

    def test_streaming_single_tool_call(self):
        text = self._call("search", {"query": "sglang"})
        calls = self._collect(Gemma4Detector(), text, self.tools)
        self.assertEqual(calls, {0: ("search", {"query": "sglang"})})

    def test_streaming_parallel_same_function_gets_distinct_indices(self):
        # Two calls of the same function must stream as index 0 and 1.
        # Before the fix both used the tool's position in the tools list
        # (index 0), so OpenAI clients merged their arguments into one call.
        text = self._call("get_weather", {"city": "Hà Nội"}) + self._call(
            "get_weather", {"city": "Đà Nẵng"}
        )
        calls = self._collect(Gemma4Detector(), text, self.tools)
        self.assertEqual(
            calls,
            {0: ("get_weather", {"city": "Hà Nội"}), 1: ("get_weather", {"city": "Đà Nẵng"})},
        )

    def test_streaming_parallel_different_functions_follow_call_order(self):
        # "search" is second in the tools list but first in the response:
        # streaming index must follow call order, not tools-list position.
        text = self._call("search", {"query": "x"}) + self._call("get_weather", {"city": "Y"})
        calls = self._collect(Gemma4Detector(), text, self.tools)
        self.assertEqual(
            calls, {0: ("search", {"query": "x"}), 1: ("get_weather", {"city": "Y"})}
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
