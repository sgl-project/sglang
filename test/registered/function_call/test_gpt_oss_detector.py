import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.gpt_oss_detector import GptOssDetector


class TestGptOssDetector(unittest.TestCase):
    def setUp(self):
        self.detector = GptOssDetector()
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="read_file",
                    description="Read file content",
                    parameters={
                        "type": "object",
                        "properties": {"files": {"type": "array"}},
                    },
                ),
            )
        ]

    def test_detect_and_parse_truncate_normal(self):
        """Test that detect_and_parse truncates normal text after a tool call."""
        text = (
            "<|channel|>analysis<|message|>Reasoning...<|end|>"
            '<|start|>assistant<|channel|>commentary to=functions.read_file <|constrain|>json<|message|>{\n  "files": []\n}<|call|>'
            "<|start|>assistant<|channel|>final<|message|>This text should be truncated."
        )

        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "read_file")
        # The key assertion: normal_text should be empty because it appears AFTER the tool call
        self.assertEqual(result.normal_text, "")

    def test_parse_streaming_truncate_normal(self):
        """Test that parse_streaming_increment truncates normal text after a tool call."""
        # Use small chunk size to exercise streaming logic
        text = (
            "<|channel|>analysis<|message|>Reasoning...<|end|>"
            '<|start|>assistant<|channel|>commentary to=functions.read_file <|constrain|>json<|message|>{\n  "files": []\n}<|call|>'
            "<|start|>assistant<|channel|>final<|message|>This text should be truncated."
        )

        accumulated_normal = ""
        accumulated_calls = []

        # Reset detector state
        self.detector = GptOssDetector()

        chunk_size = 10
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            res = self.detector.parse_streaming_increment(chunk, self.tools)
            if res.normal_text:
                accumulated_normal += res.normal_text
            if res.calls:
                accumulated_calls.extend(res.calls)

        self.assertEqual(len(accumulated_calls), 1)
        self.assertEqual(accumulated_calls[0].name, "read_file")
        # The key assertion: normal_text should be empty
        self.assertEqual(accumulated_normal, "")

    def test_parse_streaming_fast_path_fix(self):
        """Test specifically the fast path logic fix in streaming."""
        # Chunk 1: Tool call completion
        chunk1 = "<|start|>assistant<|channel|>commentary to=functions.read_file <|constrain|>json<|message|>{}<|call|>"
        res1 = self.detector.parse_streaming_increment(chunk1, self.tools)

        self.assertEqual(len(res1.calls), 1)
        # current_tool_id should now be >= 0
        self.assertNotEqual(self.detector.current_tool_id, -1)

        # Chunk 2: Normal text without tool call markers
        # This triggers the "fast path" if not properly guarded
        chunk2 = "This is normal text that should be ignored."
        res2 = self.detector.parse_streaming_increment(chunk2, self.tools)

        # Should be empty if fix works
        self.assertEqual(res2.normal_text, "")


if __name__ == "__main__":
    unittest.main()
