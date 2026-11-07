"""Unit tests for the Harmony tool-call shapes GptOssDetector accepts, no server or model."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.gpt_oss_detector import GptOssDetector
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

ANALYSIS = "<|channel|>analysis<|message|>Need the weather.<|end|><|start|>assistant"
ARGS = '{"city": "Paris"}'
FULL_CALL = (
    "<|channel|>commentary to=functions.get_weather "
    f"<|constrain|>json<|message|>{ARGS}<|call|>"
)
NO_CONSTRAIN_CALL = (
    f"<|channel|>commentary to=functions.get_weather<|message|>{ARGS}<|call|>"
)
# Channel-first call with an adjacent <|constrain|> and a doubled <|call|>, as
# produced for a forced tool_choice.
FORCED_CALL = (
    "<|channel|>commentary to=functions.get_weather"
    '<|constrain|>json<|message|>{"city":"Paris"}<|call|><|call|>'
)
ROLE_RECIPIENT_CALL = (
    "<|start|>assistant to=functions.get_weather<|channel|>commentary "
    f"<|constrain|>json<|message|>{ARGS}<|call|>"
)


class TestGptOssCallShapes(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                ),
            )
        ]

    def _assert_single_call(self, text):
        result = GptOssDetector().detect_and_parse(text, self.tools)
        self.assertEqual([c.name for c in result.calls], ["get_weather"])
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Paris"})
        self.assertEqual(result.normal_text, "")

    def test_call_after_analysis(self):
        self._assert_single_call(ANALYSIS + FULL_CALL)

    def test_call_is_first_message(self):
        # The generation prompt already ends with <|start|>assistant.
        for text in (FULL_CALL, FORCED_CALL):
            with self.subTest(text=text):
                self.assertTrue(GptOssDetector().has_tool_call(text))
                self._assert_single_call(text)

    def test_call_without_constrain(self):
        for text in (NO_CONSTRAIN_CALL, ANALYSIS + NO_CONSTRAIN_CALL):
            with self.subTest(text=text):
                self._assert_single_call(text)

    def test_recipient_in_role_header(self):
        self.assertTrue(GptOssDetector().has_tool_call(ROLE_RECIPIENT_CALL))
        self._assert_single_call(ROLE_RECIPIENT_CALL)

    def test_call_after_reasoning_parser(self):
        # Same order as the chat endpoint: reasoning parser first, then the tool parser.
        for call in (FULL_CALL, NO_CONSTRAIN_CALL):
            with self.subTest(call=call):
                _, rest = ReasoningParser(model_type="gpt-oss").parse_non_stream(
                    ANALYSIS + call
                )
                self._assert_single_call(rest)

    def test_streaming_call_without_constrain(self):
        result = GptOssDetector().parse_streaming_increment(
            NO_CONSTRAIN_CALL, self.tools
        )
        self.assertEqual([c.name for c in result.calls], ["get_weather"])
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Paris"})

    def test_plain_text_is_not_a_call(self):
        for text in (
            "The weather in Paris is sunny.",
            "Pass to=functions.get_weather to the API.",
            "<|channel|>commentary<|message|>Checking the weather.<|end|>",
            "<|start|>assistant to=browser.search<|channel|>analysis<|message|>"
            "{}<|call|>",
        ):
            with self.subTest(text=text):
                detector = GptOssDetector()
                self.assertFalse(detector.has_tool_call(text))
                result = detector.detect_and_parse(text, self.tools)
                self.assertEqual(result.calls, [])
                self.assertEqual(result.normal_text, text)


if __name__ == "__main__":
    unittest.main()
