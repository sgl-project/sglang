"""CPU-only regression tests for DeepSeek V3/V3.1 non-streaming calls."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.deepseekv3_detector import DeepSeekV3Detector
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, stage="base-a", runner_config="cpu")

SECTION_BEGIN = "<｜tool▁calls▁begin｜>"
SECTION_END = "<｜tool▁calls▁end｜>"
CALL_BEGIN = "<｜tool▁call▁begin｜>"
CALL_END = "<｜tool▁call▁end｜>"
SEPARATOR = "<｜tool▁sep｜>"


class _DeepSeekNonStreamingTests:
    def setUp(self):
        self.detector = self.detector_class()
        self.tools = [
            Tool(type="function", function=Function(name=name, parameters={}))
            for name in ("get_weather", "search")
        ]
        self.expected = [
            ("get_weather", {"city": "東京"}),
            ("search", {"query": "café"}),
        ]
        self.valid_calls = [
            self.call(name, json.dumps(params, ensure_ascii=False))
            for name, params in self.expected
        ]
        self.malformed_call = CALL_BEGIN + "missing function separator" + CALL_END

    def call(self, name, arguments):
        if self.detector_class is DeepSeekV3Detector:
            return (
                CALL_BEGIN
                + "function"
                + SEPARATOR
                + name
                + "\n```json\n"
                + arguments
                + "\n```"
                + CALL_END
            )
        return CALL_BEGIN + name + SEPARATOR + arguments + CALL_END

    def response(self, calls):
        return "Checking.\n" + SECTION_BEGIN + "\n".join(calls) + SECTION_END

    def assert_valid_calls(self, text):
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "Checking.")
        self.assertEqual(
            [(call.name, json.loads(call.parameters)) for call in result.calls],
            self.expected,
        )

    def test_valid_calls_preserve_order_and_preamble(self):
        self.assert_valid_calls(self.response(self.valid_calls))

    def test_invalid_json_does_not_discard_valid_siblings(self):
        for payload in ("{", '{"city":}', "not json", ""):
            for position in range(3):
                with self.subTest(payload=payload, position=position):
                    calls = self.valid_calls.copy()
                    calls.insert(position, self.call("get_weather", payload))
                    self.assert_valid_calls(self.response(calls))

    def test_malformed_call_does_not_discard_valid_siblings(self):
        for position in range(3):
            with self.subTest(position=position):
                calls = self.valid_calls.copy()
                calls.insert(position, self.malformed_call)
                self.assert_valid_calls(self.response(calls))

    def test_multiple_errors_do_not_discard_valid_siblings(self):
        self.assert_valid_calls(
            self.response(
                [
                    self.call("get_weather", "{"),
                    self.valid_calls[0],
                    self.malformed_call,
                    self.valid_calls[1],
                    self.call("search", "not json"),
                ]
            )
        )

    def test_all_malformed_calls_preserve_original_text(self):
        invalid_json = self.call("get_weather", "{")
        for calls in (
            [invalid_json],
            [self.malformed_call],
            [invalid_json, self.malformed_call],
        ):
            with self.subTest(calls=calls):
                text = "  " + self.response(calls) + "\nTrailing text.  "
                result = self.detector.detect_and_parse(text, self.tools)
                self.assertEqual(result.calls, [])
                self.assertEqual(result.normal_text, text)

    def test_unknown_tool_keeps_existing_filtering(self):
        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
            text = self.response([self.call("undefined_tool", "{}")])
            result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, "Checking.")

    def test_section_without_calls_keeps_existing_fallback(self):
        result = self.detector.detect_and_parse(self.response([]), self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, "Checking.")

    def test_plain_text_is_unchanged(self):
        text = "  No tool calls here.\n"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, text)


class TestDeepSeekV3NonStreaming(_DeepSeekNonStreamingTests, unittest.TestCase):
    detector_class = DeepSeekV3Detector


class TestDeepSeekV31NonStreaming(_DeepSeekNonStreamingTests, unittest.TestCase):
    detector_class = DeepSeekV31Detector


if __name__ == "__main__":
    unittest.main()
