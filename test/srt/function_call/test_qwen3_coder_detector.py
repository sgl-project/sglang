import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector


class TestQwen3CoderDetector(unittest.TestCase):
    def setUp(self):
        self.detector = Qwen3CoderDetector()
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                    },
                ),
            )
        ]

    def test_full_parse(self):
        text = "<tool_call><function=get_weather><parameter=city>Shanghai</parameter><parameter=unit>celsius</parameter></tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            result.calls[0].parameters, '{"city": "Shanghai", "unit": "celsius"}'
        )

    def test_streaming_parse(self):
        # Initial text
        result = self.detector.parse_streaming_increment(
            "Thought: calling weather. ", self.tools
        )
        self.assertEqual(result.normal_text, "Thought: calling weather. ")
        self.assertEqual(len(result.calls), 0)

        # Tool call start
        result = self.detector.parse_streaming_increment("<tool_call>", self.tools)
        self.assertEqual(result.normal_text, "")

        # Function name
        result = self.detector.parse_streaming_increment(
            "<function=get_weather>", self.tools
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

        # Parameter
        result = self.detector.parse_streaming_increment(
            "<parameter=city>Shanghai</parameter>", self.tools
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, '{"city": "Shanghai"')

        # End
        result = self.detector.parse_streaming_increment("</tool_call>", self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].parameters, "}")


if __name__ == "__main__":
    unittest.main()
