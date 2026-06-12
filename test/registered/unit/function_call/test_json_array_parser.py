import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")


class TestJsonArrayParser(CustomTestCase):
    def setUp(self):
        self.parser = JsonArrayParser()
        self.tools = [
            Tool(
                type="function",
                function=Function(name="test_tool", description="", parameters={}),
            )
        ]

    def test_initialization(self):
        """Test initialization sets correct tokens."""
        self.assertEqual(self.parser.bot_token, "[")
        self.assertEqual(self.parser.eot_token, "]")
        self.assertEqual(self.parser.tool_call_separator, ",")

    def test_has_tool_call_with_array(self):
        """Test has_tool_call detects JSON arrays."""
        self.assertTrue(self.parser.has_tool_call('[{"name": "tool"}]'))

    def test_has_tool_call_with_object(self):
        """Test has_tool_call detects JSON objects."""
        self.assertTrue(self.parser.has_tool_call('{"name": "tool"}'))

    def test_has_tool_call_plain_text(self):
        """Test has_tool_call returns False for plain text."""
        self.assertFalse(
            self.parser.has_tool_call("This is regular text without brackets.")
        )

    def test_detect_and_parse_not_implemented(self):
        """Test detect_and_parse raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.parser.detect_and_parse('[{"name": "test"}]', self.tools)

    def test_structure_info_not_implemented(self):
        """Test structure_info raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.parser.structure_info()

    def test_parse_streaming_increment_complete(self):
        """Test streaming parse with a complete JSON array in one chunk."""
        text = '[{"name": "test_tool", "arguments": {}}]'
        res = self.parser.parse_streaming_increment(text, self.tools)
        self.assertIsNotNone(res)
        self.assertEqual(len(res.calls), 1)
        self.assertEqual(res.calls[0].name, "test_tool")

    def test_parse_streaming_increment_multi_chunk(self):
        """Test streaming parse across multiple incremental chunks."""
        # A fresh parser for this test to avoid state from other tests
        parser = JsonArrayParser()

        # Chunk 1: opening bracket and partial name
        res1 = parser.parse_streaming_increment('[{"name": "test_tool"', self.tools)
        # Parser is still buffering, no complete call yet is expected
        self.assertIsNotNone(res1)

        # Chunk 2: arguments and closing
        res2 = parser.parse_streaming_increment(', "arguments": {}}]', self.tools)
        self.assertIsNotNone(res2)
        # Verify that across both chunks, the tool was recognized
        all_calls = (res1.calls or []) + (res2.calls or [])
        tool_names = [c.name for c in all_calls if c.name]
        self.assertIn("test_tool", tool_names)


if __name__ == "__main__":
    unittest.main()
