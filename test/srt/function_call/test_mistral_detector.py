import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.mistral_detector import MistralDetector


class TestMistralDetector(unittest.TestCase):
    def setUp(self):
        self.detector = MistralDetector()
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

    def test_canonical_format(self):
        text = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[0].parameters, '{"city": "Paris"}')

    def test_compact_format(self):
        text = '[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}'
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[0].parameters, '{"city": "Paris"}')

    def test_streaming_canonical(self):
        result = self.detector.parse_streaming_increment("[TOOL_CALLS] ", self.tools)
        self.assertEqual(len(result.calls), 0)

        result = self.detector.parse_streaming_increment(
            '[{"name": "get_weather", ', self.tools
        )
        # Mistral original detector might buffer until complete for JSON array
        # or use BaseFormatDetector logic.

        result = self.detector.parse_streaming_increment(
            '"arguments": {"city": "Paris"}}] ', self.tools
        )
        self.assertEqual(len(result.calls), 1)

    def test_streaming_compact(self):
        result = self.detector.parse_streaming_increment(
            "[TOOL_CALLS]get_weather", self.tools
        )
        self.assertEqual(len(result.calls), 0)

        result = self.detector.parse_streaming_increment(
            '[ARGS]{"city": "Paris"}', self.tools
        )
        self.assertEqual(len(result.calls), 2)  # Name + Parameters
        self.assertEqual(result.calls[0].name, "get_weather")


if __name__ == "__main__":
    unittest.main()
