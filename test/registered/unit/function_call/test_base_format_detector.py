"""Unit tests for BaseFormatDetector streaming: no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.llama32_detector import Llama32Detector
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.srt.function_call.trinity_detector import TrinityDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

PREFIX = "Let me check the weather for you. "
QWEN_CALL = (
    '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Tokyo"}}\n</tool_call>'
)
LLAMA_CALL = '<|python_tag|>{"name": "get_weather", "arguments": {"city": "Tokyo"}}'
# Sizes small enough that the start marker is still split across increments.
# A single increment carrying the whole answer drops the leading text through a
# different path, tracked in #31915, so it is deliberately not covered here.
CHUNK_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 17]


def _tools():
    return [
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
        )
    ]


class TestBaseFormatDetectorStreamingText(CustomTestCase):
    def setUp(self):
        self.tools = _tools()

    def _stream(self, detector, text, chunk_size):
        normal_text_parts = []
        calls = []
        for i in range(0, len(text), chunk_size):
            result = detector.parse_streaming_increment(
                text[i : i + chunk_size], self.tools
            )
            if result.normal_text:
                normal_text_parts.append(result.normal_text)
            calls.extend(result.calls)
        return "".join(normal_text_parts), calls

    def test_text_before_partial_bot_token_is_streamed(self):
        """An increment ending in a partial start marker still streams the text
        in front of it, instead of holding it back to be discarded."""
        detector = Qwen25Detector()
        result = detector.parse_streaming_increment("Let me check. <tool_", self.tools)
        self.assertEqual(result.normal_text, "Let me check. ")

    def test_streaming_is_chunk_size_invariant(self):
        """Text preceding a tool call survives verbatim and the call still
        parses, wherever the increment boundaries fall."""
        for detector_cls, call_text in (
            (Qwen25Detector, QWEN_CALL),
            (TrinityDetector, QWEN_CALL),
            (Llama32Detector, LLAMA_CALL),
        ):
            for chunk_size in CHUNK_SIZES:
                with self.subTest(detector=detector_cls.__name__, chunk=chunk_size):
                    normal_text, calls = self._stream(
                        detector_cls(), PREFIX + call_text, chunk_size
                    )
                    self.assertEqual(normal_text, PREFIX)
                    self.assertEqual([c.name for c in calls if c.name], ["get_weather"])
                    arguments = "".join(c.parameters for c in calls if c.parameters)
                    self.assertEqual(json.loads(arguments), {"city": "Tokyo"})


if __name__ == "__main__":
    unittest.main()
