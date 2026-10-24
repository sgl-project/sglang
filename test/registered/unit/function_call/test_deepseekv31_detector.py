import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepSeekV31PrefixText(CustomTestCase):
    def test_text_before_tool_call_is_emitted_once(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather", parameters={"type": "object", "properties": {}}
                ),
            )
        ]
        prefix = "Let me check the weather.\n"
        call = '<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city": "Tokyo"}<｜tool▁call▁end｜>'
        for outer_marker in ("<｜tool▁calls▁begin｜>", ""):
            for split in (0, len(prefix), len(prefix) + len(outer_marker)):
                with self.subTest(outer_marker=outer_marker, split=split):
                    detector = DeepSeekV31Detector()
                    text = prefix + outer_marker + call
                    chunks = [text[:split], text[split:], "", ""]
                    results = [
                        detector.parse_streaming_increment(chunk, tools)
                        for chunk in chunks
                    ]
                    self.assertEqual(
                        "".join(result.normal_text for result in results), prefix
                    )
                    calls = [call for result in results for call in result.calls]
                    self.assertEqual(
                        [call.name for call in calls if call.name], ["get_weather"]
                    )
                    self.assertEqual(
                        json.loads("".join(call.parameters for call in calls)),
                        {"city": "Tokyo"},
                    )

    def test_prefix_is_not_repeated_while_arguments_arrive(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather", parameters={"type": "object", "properties": {}}
                ),
            )
        ]
        detector = DeepSeekV31Detector()
        chunks = [
            "Checking... <｜tool▁calls▁begin｜>",
            '<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"city":',
            ' "Tokyo"}<｜tool▁call▁end｜>',
            "<｜tool▁calls▁end｜>",
        ]
        results = [detector.parse_streaming_increment(chunk, tools) for chunk in chunks]
        self.assertEqual(
            [result.normal_text for result in results], ["Checking... ", "", "", ""]
        )
        calls = [call for result in results for call in result.calls]
        self.assertEqual(
            json.loads("".join(call.parameters for call in calls)), {"city": "Tokyo"}
        )


if __name__ == "__main__":
    unittest.main()
