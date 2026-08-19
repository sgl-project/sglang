import json
import logging

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "base-a-test-cpu")
register_cpu_ci(est_time=7, suite="base-c-test-cpu")


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


class StreamingDummyDetector(BaseFormatDetector):
    """Minimal detector exercising BaseFormatDetector.parse_streaming_increment.

    Uses `<tool>` / `</tool>` tokens so the base implementation's
    bot_token / JSON-after-bot flow applies directly.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool>"
        self.eot_token = "</tool>"

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools):
        return StreamingParseResult(normal_text=text)

    def structure_info(self):
        pass


def _run_streaming(detector, chunks, tools):
    calls = []
    normal_text = ""
    for chunk in chunks:
        res = detector.parse_streaming_increment(chunk, tools)
        calls.extend(res.calls)
        normal_text += res.normal_text
    return calls, normal_text


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


@pytest.fixture
def weather_tools():
    return [
        Tool(
            function=Function(
                name="get_weather", parameters={"type": "object", "properties": {}}
            )
        )
    ]


def test_streaming_unknown_tool_dropped_default(caplog, weather_tools):
    """Streaming path drops unknown tool by default, matching non-streaming."""
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
        detector = StreamingDummyDetector()
        chunks = ["<tool>", '{"name":"unknown_tool"', ',"arguments":{"x":1}}']
        with caplog.at_level(
            logging.WARNING, logger="sglang.srt.function_call.base_format_detector"
        ):
            calls, _ = _run_streaming(detector, chunks, weather_tools)
        assert any(
            "Model attempted to call undefined function: unknown_tool" in m
            for m in caplog.messages
        )
        assert calls == []


def test_streaming_unknown_tool_forwarded(caplog, weather_tools):
    """Streaming path forwards unknown tool (name + args) when env var is True.

    Previously the base parser reset state and dropped the tool call plus the
    following buffer, which broke agent frameworks that rely on their own
    unknown-tool error handling. With SGLANG_FORWARD_UNKNOWN_TOOLS=true the
    name and streamed arguments should reach the client verbatim.
    """
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
        detector = StreamingDummyDetector()
        chunks = ["<tool>", '{"name":"unknown_tool"', ',"arguments":{"x":1}}']
        with caplog.at_level(
            logging.WARNING, logger="sglang.srt.function_call.base_format_detector"
        ):
            calls, _ = _run_streaming(detector, chunks, weather_tools)
        assert any(
            "Model attempted to call undefined function: unknown_tool" in m
            for m in caplog.messages
        )
        assert calls, "expected at least one ToolCallItem"
        assert calls[0].name == "unknown_tool"
        merged_args = "".join(c.parameters for c in calls if c.parameters)
        assert '"x"' in merged_args and "1" in merged_args


def test_streaming_unknown_tool_forwarded_warns_once(caplog, weather_tools):
    """Forward mode must not repeat the undefined-function warning every chunk."""
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
        detector = StreamingDummyDetector()
        chunks = ['<tool>{"name":"unknown_tool"', ',"arguments":{"x":1', "}}"]
        with caplog.at_level(
            logging.WARNING, logger="sglang.srt.function_call.base_format_detector"
        ):
            _run_streaming(detector, chunks, weather_tools)
        warn_count = sum(
            1
            for m in caplog.messages
            if "Model attempted to call undefined function: unknown_tool" in m
        )
        assert warn_count == 1, f"expected one warning, got {warn_count}"


def test_streaming_unknown_tool_between_known_preserves_state(weather_tools):
    """Unknown tool mid-stream must not corrupt already-completed tool state."""
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
        detector = StreamingDummyDetector()
        chunks = [
            "<tool>",
            '{"name":"get_weather","arguments":{"x":1}}',
            ', {"name":"unknown","arguments":{}}',
            "",
            ', {"name":"get_weather","arguments":{"y":2}}',
            "",
        ]
        _run_streaming(detector, chunks, weather_tools)
        assert detector.streamed_args_for_tool == ['{"x": 1}', '{"y": 2}']
        assert [entry.get("arguments") for entry in detector.prev_tool_call_arr] == [
            {"x": 1},
            {"y": 2},
        ]


def test_streaming_unknown_tool_same_chunk_preserves_following_known(weather_tools):
    """Skipping a complete unknown tool must preserve later buffered calls."""
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
        detector = StreamingDummyDetector()
        chunks = [
            "<tool>",
            '{"name":"get_weather","arguments":{"x":1}}',
            (
                ', {"name":"unknown","arguments":{}}, '
                '{"name":"get_weather","arguments":{"y":2}}'
            ),
            "",
            "",
            "",
        ]
        calls, _ = _run_streaming(detector, chunks, weather_tools)
        assert [call.name for call in calls if call.name] == [
            "get_weather",
            "get_weather",
        ]
        assert detector.streamed_args_for_tool == ['{"x": 1}', '{"y": 2}']
        assert [entry.get("arguments") for entry in detector.prev_tool_call_arr] == [
            {"x": 1},
            {"y": 2},
        ]


def test_streaming_after_unknown_tool_continues(weather_tools):
    """Dropping an unknown tool must not poison subsequent normal text."""
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
        detector = StreamingDummyDetector()
        # Feed an unknown tool call, then follow with normal text. The parser
        # should drop the tool call cleanly and still emit later plain text
        # rather than staying stuck in an invalid state.
        calls, normal_text = _run_streaming(
            detector,
            [
                "<tool>",
                '{"name":"unknown_tool","arguments":{}}',
                "hello world",
            ],
            weather_tools,
        )
        assert calls == []
        # Once the parser resets, the trailing plain text should still flow out.
        assert "hello world" in normal_text


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
