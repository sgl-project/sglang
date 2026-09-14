"""Unit tests for the DeepSeek V3 / V3.1 tool-call detectors — no server.

Regression: detect_and_parse wrapped the whole loop over tool calls in one
try/except, so a single malformed call (a non-matching detail regex, or invalid
JSON arguments) discarded every valid call parsed before it and returned the raw
markup as normal text. Each call must be parsed in isolation.
"""

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

BEGIN = "<｜tool▁calls▁begin｜>"
END = "<｜tool▁calls▁end｜>"
CB = "<｜tool▁call▁begin｜>"
CE = "<｜tool▁call▁end｜>"
SEP = "<｜tool▁sep｜>"


def _tools(*names):
    return [
        Tool(
            type="function",
            function=Function(
                name=n, description=n, parameters={"type": "object", "properties": {}}
            ),
        )
        for n in names
    ]


def _v3_call(name, args):
    return f"{CB}function{SEP}{name}\n```json\n{args}\n```{CE}"


def _v31_call(name, args):
    return f"{CB}{name}{SEP}{args}{CE}"


class TestDeepSeekV3Detector(CustomTestCase):
    def _parse(self, text, tools):
        return DeepSeekV3Detector().detect_and_parse(text, tools)

    def test_malformed_call_does_not_discard_valid_call(self):
        text = (
            "Sure. "
            + BEGIN
            + _v3_call("get_weather", '{"city": "Boston"}')
            + _v3_call("search", "{bad json")  # invalid JSON args
            + END
        )
        res = self._parse(text, _tools("get_weather", "search"))
        self.assertEqual([c.name for c in res.calls], ["get_weather"])
        self.assertNotIn(BEGIN, res.normal_text or "")

    def test_two_valid_calls_both_parse(self):
        text = (
            BEGIN
            + _v3_call("get_weather", '{"city": "Boston"}')
            + _v3_call("search", '{"q": "hi"}')
            + END
        )
        res = self._parse(text, _tools("get_weather", "search"))
        self.assertEqual([c.name for c in res.calls], ["get_weather", "search"])


class TestDeepSeekV31Detector(CustomTestCase):
    def _parse(self, text, tools):
        return DeepSeekV31Detector().detect_and_parse(text, tools)

    def test_malformed_call_does_not_discard_valid_call(self):
        text = (
            "Sure. "
            + BEGIN
            + _v31_call("get_weather", '{"city": "Boston"}')
            + _v31_call("search", "{bad json")
            + END
        )
        res = self._parse(text, _tools("get_weather", "search"))
        self.assertEqual([c.name for c in res.calls], ["get_weather"])
        self.assertNotIn(BEGIN, res.normal_text or "")


if __name__ == "__main__":
    import unittest

    unittest.main()
