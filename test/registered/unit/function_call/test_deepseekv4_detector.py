"""Unit tests for DeepSeekV4Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestDeepSeekV4Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_favorite_tourist_spot",
                    description="Get the favorite tourist spot in a city",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = DeepSeekV4Detector()

    # ==================== has_tool_call Tests ====================

    def test_has_tool_call_true(self):
        text = "<｜DSML｜tool_calls>something</｜DSML｜tool_calls>"
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        text = "The weather is nice today."
        self.assertFalse(self.detector.has_tool_call(text))

    # ==================== detect_and_parse Tests ====================

    def test_single_invoke_direct_json(self):
        text = (
            "<｜DSML｜tool_calls>"
            '<｜DSML｜invoke name="get_favorite_tourist_spot">'
            '{ "city": "San Francisco" }'
            "</｜DSML｜invoke>"
            "</｜DSML｜tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_favorite_tourist_spot")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "San Francisco")

    def test_single_invoke_xml_parameter_tags(self):
        text = (
            "<｜DSML｜tool_calls>"
            '<｜DSML｜invoke name="get_favorite_tourist_spot">'
            '<｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>'
            "</｜DSML｜invoke>"
            "</｜DSML｜tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_favorite_tourist_spot")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "San Francisco")

    def test_tool_call_with_leading_text(self):
        text = (
            'I will find your spot. '
            "<｜DSML｜tool_calls>"
            '<｜DSML｜invoke name="get_favorite_tourist_spot">'
            '{ "city": "Tokyo" }'
            "</｜DSML｜invoke>"
            "</｜DSML｜tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_favorite_tourist_spot")
        self.assertEqual(result.normal_text, "I will find your spot. ")

    def test_no_tool_call(self):
        text = "The weather is nice today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "The weather is nice today.")

    def test_malformed_incomplete_doesnt_crash(self):
        text = "<｜DSML｜tool_calls><｜DSML｜invoke name="
        result = self.detector.detect_and_parse(text, self.tools)
        # Should not crash; may return 0 or partial calls depending on parser
        self.assertIsNotNone(result)


if __name__ == "__main__":
    import unittest

    unittest.main()
