import json
import logging

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "default")


class DummyDetector(BaseFormatDetector):
    def has_tool_call(self, text: str) -> bool:
        return True

    def detect_and_parse(self, text: str, tools):
        action = json.loads(text)
        return StreamingParseResult(
            normal_text="", calls=self.parse_base_json(action, tools)
        )

    def structure_info(self):
        pass


def test_unknown_tool_name_dropped_default(caplog):
    """Test that unknown tools are dropped by default (legacy behavior)."""
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
        tools = [
            Tool(
                function=Function(
                    name="get_weather", parameters={"type": "object", "properties": {}}
                )
            )
        ]
        detector = DummyDetector()
        with caplog.at_level(
            logging.WARNING, logger="sglang.srt.function_call.base_format_detector"
        ):
            result = detector.detect_and_parse(
                '{"name":"unknown_tool","parameters":{"city":"Paris"}}', tools
            )
        assert any(
            "Model attempted to call undefined function: unknown_tool" in m
            for m in caplog.messages
        )
        assert len(result.calls) == 0  # dropped in default mode


def test_unknown_tool_name_forwarded(caplog):
    """Test that unknown tools are forwarded when env var is True."""
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
        tools = [
            Tool(
                function=Function(
                    name="get_weather", parameters={"type": "object", "properties": {}}
                )
            )
        ]
        detector = DummyDetector()
        with caplog.at_level(
            logging.WARNING, logger="sglang.srt.function_call.base_format_detector"
        ):
            result = detector.detect_and_parse(
                '{"name":"unknown_tool","parameters":{"city":"Paris"}}', tools
            )
        assert any(
            "Model attempted to call undefined function: unknown_tool" in m
            for m in caplog.messages
        )
        assert len(result.calls) == 1
        assert result.calls[0].name == "unknown_tool"
        assert result.calls[0].tool_index == -1
        assert json.loads(result.calls[0].parameters)["city"] == "Paris"


def _make_tools():
    return [
        Tool(
            function=Function(
                name="get_weather",
                parameters={"type": "object", "properties": {}},
            )
        )
    ]


def test_streaming_unknown_tool_dropped_default():
    """Streaming: unknown tool call block is silently skipped, matching non-streaming.

    Uses Qwen25Detector which delegates to BaseFormatDetector.parse_streaming_increment.
    When the tool name is unknown and SGLANG_FORWARD_UNKNOWN_TOOLS is False,
    the streaming parser discards the entire tool call block (no calls, no normal text),
    consistent with the non-streaming path.
    """
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
        tools = _make_tools()
        detector = Qwen25Detector()

        # Simulate streaming tokens for: <tool_call>\n{"name":"unknown_tool","arguments":{"city":"Paris"}}\n</tool_call>
        chunks = [
            "<tool_call>\n",
            '{"name',
            '":"unknown_tool',
            '","arguments',
            '":{"city',
            '":"Paris"',
            "}}\n</tool_call>",
        ]

        all_calls = []
        all_normal = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, tools)
            all_calls.extend(result.calls)
            if result.normal_text:
                all_normal.append(result.normal_text)

        # No tool calls should have been emitted
        assert len(all_calls) == 0
        # No normal text either — entire block is discarded
        assert "".join(all_normal) == ""


def test_streaming_unknown_tool_forwarded():
    """Streaming: unknown tools are forwarded when env var is True.

    Uses Qwen25Detector which delegates to BaseFormatDetector.parse_streaming_increment.
    When SGLANG_FORWARD_UNKNOWN_TOOLS is True, the streaming parser should
    emit the unknown tool name and stream its arguments normally.
    """
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
        tools = _make_tools()
        detector = Qwen25Detector()

        # Simulate streaming tokens
        chunks = [
            "<tool_call>\n",
            '{"name',
            '":"unknown_tool',
            '","arguments',
            '":{"city',
            '":"Paris"',
            "}}\n</tool_call>",
        ]

        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, tools)
            all_calls.extend(result.calls)

        # First call should be the tool name
        assert len(all_calls) >= 1
        assert all_calls[0].name == "unknown_tool"
        assert all_calls[0].parameters == ""

        # Remaining calls should contain argument fragments
        arg_fragments = "".join(c.parameters for c in all_calls[1:])
        parsed_args = json.loads(arg_fragments)
        assert parsed_args["city"] == "Paris"
