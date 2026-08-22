import json
import logging

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.glm47_moe_detector import Glm47MoeDetector
from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
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
def known_tool():
    return Tool(
        function=Function(name="known", parameters={"type": "object", "properties": {}})
    )


@pytest.mark.parametrize(
    ("detector_cls", "unknown_call", "known_call"),
    [
        (
            Glm4MoeDetector,
            "<tool_call>unknown\n</tool_call>",
            "<tool_call>known\n</tool_call>",
        ),
        (
            Glm47MoeDetector,
            "<tool_call>unknown</tool_call>",
            "<tool_call>known</tool_call>",
        ),
    ],
)
def test_glm_streaming_drops_unknown_then_parses_known(
    detector_cls, unknown_call, known_call, known_tool
):
    detector = detector_cls()
    calls = []

    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
        for character in unknown_call + known_call:
            calls.extend(
                detector.parse_streaming_increment(character, [known_tool]).calls
            )

    named_calls = [call for call in calls if call.name is not None]
    assert [(call.tool_index, call.name) for call in named_calls] == [(0, "known")]


@pytest.mark.parametrize(
    ("detector_cls", "unknown_call"),
    [
        (Glm4MoeDetector, "<tool_call>unknown\n</tool_call>"),
        (Glm47MoeDetector, "<tool_call>unknown</tool_call>"),
    ],
)
def test_glm_streaming_forwards_unknown_when_enabled(
    detector_cls, unknown_call, known_tool
):
    detector = detector_cls()
    calls = []

    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
        for character in unknown_call:
            calls.extend(
                detector.parse_streaming_increment(character, [known_tool]).calls
            )

    assert [call.name for call in calls if call.name is not None] == ["unknown"]


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
