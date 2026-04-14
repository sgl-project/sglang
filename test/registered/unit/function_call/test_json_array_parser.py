import unittest
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool, Function
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.test.test_utils import CustomTestCase, register_cpu_ci

class TestJsonArrayParser(CustomTestCase):
    def setUp(self):
        self.parser = JsonArrayParser()
        self.tools = [
            Tool(type="function", function=Function(name="test_tool", description="", parameters={}))
        ]

    def test_initialization(self):
        """Test initialization sets correct tokens."""
        self.assertEqual(self.parser.bot_token, "[")
        self.assertEqual(self.parser.eot_token, "]")
        self.assertEqual(self.parser.tool_call_separator, ",")

    def test_has_tool_call(self):
        """Test has_tool_call pattern detection."""
        self.assertTrue(self.parser.has_tool_call("Here is the tool call: [{\n  \"name\": \"tool\""))
        self.assertTrue(self.parser.has_tool_call("{\"name\": \"tool\"}"))
        self.assertFalse(self.parser.has_tool_call("This is just regular text without square or curly braces."))

    def test_detect_and_parse_not_implemented(self):
        """Test detect_and_parse raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.parser.detect_and_parse('[{"name": "test"}]', self.tools)

    def test_structure_info_not_implemented(self):
        """Test structure_info raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.parser.structure_info()

    def test_parse_streaming_increment(self):
        """Test parse_streaming_increment handles partial texts inherited from BaseFormatDetector."""
        # Using base implementation via super()
        # Ensure it works with simple valid json
        text = "[\n  {\"name\": \"test_tool\", \"arguments\": {}}\n]"
        res = self.parser.parse_streaming_increment(text, self.tools)
        # Should return a StreamingParseResult where is_tool_call is typically evaluated
        self.assertIsNotNone(res)

register_cpu_ci(TestJsonArrayParser)

if __name__ == "__main__":
    unittest.main()
