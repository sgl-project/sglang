"""Unit tests for DeepSeekV32Detector streaming behavior.

These tests directly construct the mid-stream snapshot
"current_tool_id >= 0 + empty buffer" -- the moment just after the model
emits the closing </｜DSML｜function_calls> tag and is about to emit
boundary whitespace. Feeding the full tool_call section to
parse_streaming_increment in a single call leaves the closing tag in
_buffer, so subsequent input gets caught by the potentially_dsml branch
and bypasses the early-return path we want to exercise -- the bug would
not trigger.
"""

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestDeepSeekV32DetectorStreaming(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = DeepSeekV32Detector()

    def _enter_tool_call_mode(self):
        """Force detector state to simulate the moment when the previous
        chunk has fully processed a tool_call section: buffer is cleared
        and current_tool_id has advanced.
        """
        self.detector.current_tool_id = 0
        self.detector._buffer = ""

    # ============ Regression: #24293 ============

    def test_whitespace_after_tool_call_is_suppressed(self):
        """Once in tool_call mode, whitespace-only normal_text must be
        suppressed; otherwise it leaks out as delta.content and breaks
        strict Anthropic-spec clients.
        """
        for ws in ["\n", "\n\n", "   ", "\t\n "]:
            with self.subTest(whitespace=repr(ws)):
                self._enter_tool_call_mode()
                result = self.detector.parse_streaming_increment(ws, self.tools)
                self.assertEqual(
                    result.normal_text,
                    "",
                    f"expected empty normal_text for {ws!r}, "
                    f"got {result.normal_text!r}",
                )

    # ============ Behaviors the guard must not break ============

    def test_plain_dialog_whitespace_is_preserved(self):
        """Outside tool_call mode the guard never fires; legitimate
        whitespace is forwarded unchanged."""
        self.assertEqual(self.detector.current_tool_id, -1)
        result = self.detector.parse_streaming_increment("\n", self.tools)
        self.assertEqual(result.normal_text, "\n")
        result = self.detector.parse_streaming_increment("hello\n\nworld", self.tools)
        self.assertEqual(result.normal_text, "hello\n\nworld")

    def test_non_empty_text_after_tool_call_still_emitted(self):
        """In tool_call mode, real text content must still pass through."""
        self._enter_tool_call_mode()
        result = self.detector.parse_streaming_increment("All set!", self.tools)
        self.assertEqual(result.normal_text, "All set!")

    def test_whitespace_before_tool_call_is_unchanged(self):
        """Whitespace before any tool_call keeps existing behavior; the
        guard intentionally does not touch this case."""
        result = self.detector.parse_streaming_increment("\n\n", self.tools)
        self.assertEqual(self.detector.current_tool_id, -1)
        self.assertEqual(result.normal_text, "\n\n")


if __name__ == "__main__":
    import unittest

    unittest.main()
