"""Unit tests for TrinityDetector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.trinity_detector import TrinityDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestTrinityDetector(CustomTestCase):
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
                            "city": {"type": "string", "description": "City name"},
                        },
                        "required": ["city"],
                    },
                ),
            ),
        ]

    def _stream(self, chunks):
        detector = TrinityDetector()
        normal = ""
        calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal += result.normal_text
            calls.extend(result.calls)
        return normal, calls

    def test_streaming_whole_think_tag(self):
        """A think tag arriving in one chunk is stripped, content kept."""
        normal, _ = self._stream(["<think>reasoning</think>hello"])
        self.assertEqual(normal, "reasoninghello")

    def test_streaming_split_think_tag(self):
        """A think tag split across chunk boundaries must still be stripped.

        Regression: stripping per-increment left `<think>`/`</think>` intact
        when the tag straddled two chunks (neither half held the full tag), so
        the raw tags leaked into normal_text.
        """
        normal, _ = self._stream(["<thi", "nk>reasoning</thi", "nk>hello"])
        self.assertEqual(normal, "reasoninghello")

    def test_streaming_char_by_char(self):
        """The tightest possible split (one char per chunk) is handled."""
        normal, _ = self._stream(list("<think>abc</think>hi"))
        self.assertEqual(normal, "abchi")

    def test_streaming_tag_lookalike_not_eaten(self):
        """Text that merely starts like `<think>` (e.g. `<thing>`) is preserved."""
        normal, _ = self._stream(["<thi", "ng>data"])
        self.assertEqual(normal, "<thing>data")

    def test_streaming_tool_call_inside_think_split(self):
        """A tool call inside a split think section still parses."""
        chunks = [
            "<thi",
            'nk><tool_call>\n{"name": "get_weather", ',
            '"arguments": {"city": "Paris"}}\n</tool_call></thi',
            "nk>",
        ]
        _, calls = self._stream(chunks)
        names = [c.name for c in calls if c.name]
        self.assertIn("get_weather", names)
        joined = "".join(c.parameters or "" for c in calls)
        self.assertIn("Paris", joined)
