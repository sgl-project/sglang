import json
import logging

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.mistral_detector import MistralDetector
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


def get_test_tools():
    """Helper to create test tools list."""
    return [
        Tool(
            function=Function(
                name="get_weather", parameters={"type": "object", "properties": {}}
            )
        )
    ]


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


def test_streaming_unknown_tool_dropped_default(caplog):
    """Test that unknown tools are dropped in streaming mode by default."""
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
        tools = get_test_tools()
        detector = MistralDetector()
        # Mistral format: [TOOL_CALLS] [{"name": "...", "arguments": {...}}]
        text = '[TOOL_CALLS] [{"name": "unknown_tool", "arguments": {"city": "Paris"}}]'

        with caplog.at_level(
            logging.WARNING, logger="sglang.srt.function_call.base_format_detector"
        ):
            result = detector.parse_streaming_increment(text, tools)

        # Should log warning about unknown tool
        assert any(
            "Model attempted to call undefined function: unknown_tool" in m
            for m in caplog.messages
        )
        # Should not return any tool calls (dropped)
        assert len(result.calls) == 0


def test_streaming_unknown_tool_forwarded(caplog):
    """Test that unknown tools are forwarded in streaming mode when env var is True."""
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
        tools = get_test_tools()
        detector = MistralDetector()
        # Mistral format: [TOOL_CALLS] [{"name": "...", "arguments": {...}}]
        text = '[TOOL_CALLS] [{"name": "unknown_tool", "arguments": {"city": "Paris"}}]'

        with caplog.at_level(
            logging.WARNING, logger="sglang.srt.function_call.base_format_detector"
        ):
            result = detector.parse_streaming_increment(text, tools)

        # Should log warning about unknown tool
        assert any(
            "Model attempted to call undefined function: unknown_tool" in m
            for m in caplog.messages
        )
        # Should return the tool call with tool_index=-1
        assert len(result.calls) == 1
        assert result.calls[0].name == "unknown_tool"
        assert result.calls[0].tool_index == -1


def test_streaming_unknown_tool_with_arguments(caplog):
    """Test that unknown tool arguments are correctly streamed when forwarding is enabled."""
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
        tools = get_test_tools()
        detector = MistralDetector()

        # Simulate streaming in chunks
        chunks = [
            '[TOOL_CALLS] [{"name": "unknown_tool"',
            ', "arguments": {"city": "Paris"}}]',
        ]

        all_calls = []
        with caplog.at_level(
            logging.WARNING, logger="sglang.srt.function_call.base_format_detector"
        ):
            for chunk in chunks:
                result = detector.parse_streaming_increment(chunk, tools)
                all_calls.extend(result.calls)

        # Should have received tool name and arguments
        assert len(all_calls) >= 1
        # First call should have the tool name
        assert all_calls[0].name == "unknown_tool"
        assert all_calls[0].tool_index == -1


def test_streaming_mixed_known_and_unknown_tools(caplog):
    """Test streaming with both known and unknown tools."""
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
        tools = get_test_tools()  # Only has "get_weather"
        detector = MistralDetector()

        # First call known tool
        text1 = '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "NYC"}}]'
        result1 = detector.parse_streaming_increment(text1, tools)

        # Known tool should have proper index (0)
        assert len(result1.calls) >= 1
        # Find the call with the name (first one)
        name_call = next((c for c in result1.calls if c.name == "get_weather"), None)
        assert name_call is not None
        assert name_call.tool_index == 0

        # Reset detector for second call
        detector = MistralDetector()

        # Now call unknown tool
        text2 = '[TOOL_CALLS] [{"name": "unknown_func", "arguments": {"x": 1}}]'
        with caplog.at_level(
            logging.WARNING, logger="sglang.srt.function_call.base_format_detector"
        ):
            result2 = detector.parse_streaming_increment(text2, tools)

        # Unknown tool should have index -1
        assert len(result2.calls) >= 1
        name_call = next((c for c in result2.calls if c.name == "unknown_func"), None)
        assert name_call is not None
        assert name_call.tool_index == -1


if __name__ == "__main__":
    pytest.main([__file__])
