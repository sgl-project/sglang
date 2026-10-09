import json
import logging

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "base-a-test-cpu")
register_cpu_ci(est_time=5, suite="stage-b-test-cpu-intel")


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


def _tools():
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
        tools = _tools()
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
        tools = _tools()
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


def test_non_object_entry_dropped_when_forwarding_unknown_tools(caplog):
    """A non-object entry has no name to forward, so it is skipped even with
    SGLANG_FORWARD_UNKNOWN_TOOLS=True; the valid call beside it survives."""
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
        detector = DummyDetector()
        with caplog.at_level(
            logging.WARNING, logger="sglang.srt.function_call.base_format_detector"
        ):
            result = detector.detect_and_parse(
                '[{"name":"get_weather","parameters":{"city":"Paris"}}, "junk"]',
                _tools(),
            )
        assert any(
            "Skipping non-object tool call entry of type str" in m
            for m in caplog.messages
        )
        assert [c.name for c in result.calls] == ["get_weather"]


VALID_CALL = '{"name":"get_weather","parameters":{"city":"Paris"}}'


@pytest.mark.parametrize(
    "junk", ['"junk"', "42", "null", "[]", '[{"name":"get_weather"}]', "true"]
)
@pytest.mark.parametrize("junk_first", [True, False])
def test_non_object_entry_skipped_keeping_valid_call(junk, junk_first):
    """A tool-call array mixing a well-formed call with a non-object entry must still
    return the well-formed call, whichever side the junk is on (sglang#37283)."""
    entries = [junk, VALID_CALL] if junk_first else [VALID_CALL, junk]
    result = DummyDetector().detect_and_parse(f"[{', '.join(entries)}]", _tools())
    assert [c.name for c in result.calls] == ["get_weather"]
    assert json.loads(result.calls[0].parameters) == {"city": "Paris"}


def test_all_entries_non_object_returns_no_calls():
    """An array with nothing but non-object entries yields no calls instead of raising."""
    result = DummyDetector().detect_and_parse('["junk", 42, null, []]', _tools())
    assert result.calls == []


def test_bare_non_object_action_returns_no_calls():
    """A scalar action (not wrapped in an array) is skipped, not crashed on."""
    assert DummyDetector().detect_and_parse('"junk"', _tools()).calls == []


def test_all_object_array_still_parses_every_call():
    """Control: the guard must not drop entries from an all-object array."""
    result = DummyDetector().detect_and_parse(f"[{VALID_CALL}, {VALID_CALL}]", _tools())
    assert [c.name for c in result.calls] == ["get_weather", "get_weather"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
