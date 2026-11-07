"""Apertus tool-call streaming regressions, with no server or model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.apertus2509_detector import Apertus2509Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def _block(calls):
    return "<|tools_prefix|>" + json.dumps(calls) + "<|tools_suffix|>"


class TestApertus2509Streaming(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name=name, parameters={"type": "object", "properties": {}}
                ),
            )
            for name in ("get_weather", "get_time")
        ]

    def _feed(self, chunks):
        detector = Apertus2509Detector()
        normal, calls = "", []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal += result.normal_text
            calls.extend(result.calls)
        result = detector.finish(self.tools)
        return detector, normal + result.normal_text, calls + result.calls

    def _assembled(self, calls):
        result = {}
        for call in calls:
            entry = result.setdefault(call.tool_index, {"name": None, "arguments": ""})
            if call.name:
                entry["name"] = call.name
            entry["arguments"] += call.parameters
        return [
            (index, entry["name"], json.loads(entry["arguments"]))
            for index, entry in sorted(result.items())
        ]

    def test_chunk_invariance_for_multiple_blocks_and_trailing_text(self):
        text = (
            "before"
            + _block([{"get_weather": {"city": "Zürich"}}])
            + "between"
            + _block([{"get_time": {}}, {"get_weather": {"city": "Tokyo"}}])
            + "done"
        )
        one_shot = Apertus2509Detector().detect_and_parse(text, self.tools)
        expected = self._assembled(one_shot.calls)
        self.assertEqual(
            expected,
            [
                (0, "get_weather", {"city": "Zürich"}),
                (1, "get_time", {}),
                (2, "get_weather", {"city": "Tokyo"}),
            ],
        )
        for width in range(1, len(text) + 1):
            with self.subTest(width=width):
                detector, normal, calls = self._feed(
                    [text[i : i + width] for i in range(0, len(text), width)]
                )
                self.assertEqual(normal, one_shot.normal_text)
                self.assertEqual(self._assembled(calls), expected)
                self.assertEqual(detector._buffer, "")
                self.assertEqual(detector.current_tool_id, 3)

    def test_second_incomplete_block_stays_buffered(self):
        first = _block([{"get_weather": {}}])
        second = _block([{"get_time": {"zone": "UTC"}}])
        for split in range(1, len(second)):
            with self.subTest(split=split):
                detector = Apertus2509Detector()
                result = detector.parse_streaming_increment(
                    first + "between" + second[:split], self.tools
                )
                self.assertEqual(
                    self._assembled(result.calls), [(0, "get_weather", {})]
                )
                self.assertEqual(result.normal_text, "between")
                self.assertEqual(detector._buffer, second[:split])
                completion = detector.parse_streaming_increment(
                    second[split:] + "done", self.tools
                )
                self.assertEqual(
                    self._assembled(completion.calls),
                    [(1, "get_time", {"zone": "UTC"})],
                )
                self.assertEqual(completion.normal_text, "done")
                self.assertEqual(detector._buffer, "")

    def test_truncated_second_block_is_not_emitted_by_finish(self):
        partial = '<|tools_prefix|>[{"get_time": {"zone":'
        detector, normal, calls = self._feed([_block([{"get_weather": {}}]) + partial])
        self.assertEqual(self._assembled(calls), [(0, "get_weather", {})])
        self.assertEqual(normal, "")
        self.assertEqual(detector._buffer, partial)

    def test_plain_text_and_single_block_controls(self):
        for text in ("hello", _block([{"get_time": {}}]) + "done"):
            with self.subTest(text=text):
                expected = Apertus2509Detector().detect_and_parse(text, self.tools)
                detector, normal, calls = self._feed([text])
                self.assertEqual(normal, expected.normal_text)
                self.assertEqual(
                    self._assembled(calls), self._assembled(expected.calls)
                )
                self.assertEqual(detector._buffer, "")


if __name__ == "__main__":
    unittest.main()
