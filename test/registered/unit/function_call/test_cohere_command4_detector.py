"""Unit tests for CohereCommand4Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.cohere_command4_detector import CohereCommand4Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

START = "<|START_ACTION|>"
END = "<|END_ACTION|>"


class TestCohereCommand4Detector(CustomTestCase):
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
        self.detector = CohereCommand4Detector()

    @staticmethod
    def _block(*actions: str) -> str:
        return START + "[" + ", ".join(actions) + "]" + END

    @staticmethod
    def _action(tool_name: str, params: dict) -> str:
        return json.dumps(
            {"tool_call_id": "0", "tool_name": tool_name, "parameters": params}
        )

    # ==================== has_tool_call ====================

    def test_has_tool_call_true(self):
        self.assertTrue(
            self.detector.has_tool_call(
                self._block(self._action("get_weather", {"city": "Beijing"}))
            )
        )

    def test_has_tool_call_false(self):
        self.assertFalse(self.detector.has_tool_call("The weather is nice today."))

    # ==================== _normalize_calls ====================

    def test_normalize_remaps_tool_name_and_drops_id(self):
        out = CohereCommand4Detector._normalize_calls(
            [
                {
                    "tool_call_id": "abc",
                    "tool_name": "get_weather",
                    "parameters": {"city": "X"},
                }
            ]
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "get_weather")
        self.assertNotIn("tool_call_id", out[0])
        self.assertNotIn("tool_name", out[0])

    def test_normalize_wraps_single_dict_and_skips_garbage(self):
        self.assertEqual(
            len(
                CohereCommand4Detector._normalize_calls(
                    {"tool_name": "search", "parameters": {}}
                )
            ),
            1,
        )
        self.assertEqual(CohereCommand4Detector._normalize_calls("not a list"), [])
        self.assertEqual(CohereCommand4Detector._normalize_calls([1, "x", None]), [])

    # ==================== detect_and_parse ====================

    def test_single_tool_call(self):
        text = self._block(self._action("get_weather", {"city": "Beijing"}))
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters)["city"], "Beijing")
        self.assertEqual(result.normal_text, "")

    def test_parallel_tool_calls_with_leading_text(self):
        text = "Sure! " + self._block(
            self._action("get_weather", {"city": "Tokyo"}),
            self._action("search", {"query": "ramen"}),
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "Sure! ")
        self.assertEqual([c.name for c in result.calls], ["get_weather", "search"])

    def test_truncated_body_uses_partial_fallback(self):
        # No closing <|END_ACTION|>; body is a valid-but-unterminated JSON array.
        text = (
            START
            + '[{"tool_call_id": "0", "tool_name": "get_weather", "parameters": {"city": "Paris"'
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

    def test_malformed_body_returns_normal_text(self):
        text = "prefix " + START + "[not valid json" + END
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "prefix ")

    def test_no_tool_call(self):
        result = self.detector.detect_and_parse("Just a plain answer.", self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "Just a plain answer.")

    # ==================== parse_streaming_increment ====================

    def test_streaming_text_then_block(self):
        detector = CohereCommand4Detector()
        block = self._block(self._action("get_weather", {"city": "Berlin"}))
        # split awkwardly: leading text, then the bot token, then the rest
        chunks = ["Let me check. ", block[: len(START)], block[len(START) :]]

        normal = ""
        calls = []
        for c in chunks:
            r = detector.parse_streaming_increment(c, self.tools)
            normal += r.normal_text
            calls.extend(r.calls)

        self.assertEqual(normal, "Let me check. ")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters)["city"], "Berlin")

    def test_streaming_waits_for_end_token(self):
        detector = CohereCommand4Detector()
        # Feed the start + body but not the closing token: no calls yet.
        r1 = detector.parse_streaming_increment(
            START
            + '[{"tool_call_id":"0","tool_name":"search","parameters":{"query":"x"}}]',
            self.tools,
        )
        self.assertEqual(len(r1.calls), 0)
        # Now the closing token arrives -> the call is emitted.
        r2 = detector.parse_streaming_increment(END, self.tools)
        self.assertEqual(len(r2.calls), 1)
        self.assertEqual(r2.calls[0].name, "search")


if __name__ == "__main__":
    import unittest

    unittest.main()
