"""CPU-only tests for the shared FunctionCallParser orchestration layer."""

from unittest.mock import Mock, patch

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _StubDetector(BaseFormatDetector):
    def has_tool_call(self, text: str) -> bool:
        return "<tool>" in text

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        return StreamingParseResult(normal_text=text)

    def structure_info(self):
        return lambda name: StructureInfo(
            begin=f'<tool name="{name}">',
            end="</tool>",
            trigger="<tool",
        )


class _TokenizerDetector(_StubDetector):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer


class TestFunctionCallParserCore(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                function=Function(
                    name="weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                )
            ),
            Tool(
                function=Function(
                    name="search",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                )
            ),
        ]

    def make_parser(self, tools=None, detector_class=_StubDetector):
        effective_tools = self.tools if tools is None else tools
        with patch.dict(
            FunctionCallParser.ToolCallParserEnum,
            {"test_stub": detector_class},
        ):
            return FunctionCallParser(effective_tools, "test_stub")

    def test_rejects_unsupported_parser(self):
        with self.assertRaisesRegex(ValueError, "Unsupported tool_call_parser"):
            FunctionCallParser(self.tools, "not-a-parser")

    def test_passes_tokenizer_only_to_supported_detector(self):
        tokenizer = object()
        with patch.dict(
            FunctionCallParser.ToolCallParserEnum,
            {
                "with_tokenizer": _TokenizerDetector,
                "without_tokenizer": _StubDetector,
            },
        ):
            parser = FunctionCallParser(
                self.tools, "with_tokenizer", tokenizer=tokenizer
            )
            parser_without_tokenizer = FunctionCallParser(
                self.tools, "without_tokenizer", tokenizer=tokenizer
            )

        self.assertIs(parser.detector.tokenizer, tokenizer)
        self.assertIsInstance(parser_without_tokenizer.detector, _StubDetector)

    def test_no_tools_short_circuits_every_parse_path(self):
        parser = self.make_parser(tools=[])
        parser.detector.has_tool_call = Mock()
        parser.detector.detect_and_parse = Mock()
        parser.detector.parse_streaming_increment = Mock()
        parser.detector.finish = Mock()

        self.assertFalse(parser.has_tool_call("<tool>"))
        self.assertEqual(parser.parse_non_stream("plain"), ("plain", []))
        self.assertEqual(parser.parse_stream_chunk("chunk"), ("chunk", []))
        self.assertEqual(parser.parse_stream_end(), ("", []))

        parser.detector.has_tool_call.assert_not_called()
        parser.detector.detect_and_parse.assert_not_called()
        parser.detector.parse_streaming_increment.assert_not_called()
        parser.detector.finish.assert_not_called()

    def test_non_stream_fallback_depends_on_marker_or_calls(self):
        parser = self.make_parser()
        parser.detector.has_tool_call = Mock(return_value=False)
        parser.detector.detect_and_parse = Mock(
            return_value=StreamingParseResult(normal_text="detector fallback")
        )

        self.assertEqual(parser.parse_non_stream("original"), ("original", []))

        parser.detector.has_tool_call.return_value = True
        self.assertEqual(
            parser.parse_non_stream("<tool>malformed"),
            ("detector fallback", []),
        )

        parsed_call = ToolCallItem(
            tool_index=0,
            name="weather",
            parameters='{"city":"Singapore"}',
        )
        parser.detector.has_tool_call.return_value = False
        parser.detector.detect_and_parse.return_value = StreamingParseResult(
            normal_text="", calls=[parsed_call]
        )
        self.assertEqual(parser.parse_non_stream("model output"), ("", [parsed_call]))

    def test_stream_chunk_and_end_delegate_to_detector(self):
        parser = self.make_parser()
        parsed_call = ToolCallItem(
            tool_index=0,
            name="weather",
            parameters="",
        )
        parser.detector.parse_streaming_increment = Mock(
            return_value=StreamingParseResult(normal_text="before", calls=[parsed_call])
        )
        parser.detector.finish = Mock(
            return_value=StreamingParseResult(normal_text="held-back")
        )

        self.assertEqual(parser.parse_stream_chunk("delta"), ("before", [parsed_call]))
        self.assertEqual(parser.parse_stream_end(), ("held-back", []))
        parser.detector.parse_streaming_increment.assert_called_once_with(
            "delta", self.tools
        )
        parser.detector.finish.assert_called_once_with(self.tools)


if __name__ == "__main__":
    import unittest

    unittest.main()
