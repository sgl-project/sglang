import json
import logging

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult
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


if __name__ == "__main__":
    pytest.main([__file__])
