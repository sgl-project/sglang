import json
import logging
import unittest

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.internlm_detector import InternlmDetector
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


class TestInternlmUnknownToolForwarding(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                function=Function(
                    name="known_tool",
                    parameters={"type": "object", "properties": {}},
                )
            )
        ]
        self.unknown_call = (
            '<|action_start|> <|plugin|>{"name":"unknown_tool",'
            '"parameters":{"value":1}}<|action_end|>'
        )

    def test_non_streaming_forwards_unknown_tool(self):
        """The forwarding opt-in must preserve an unknown InternLM tool call."""
        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            result = InternlmDetector().detect_and_parse(self.unknown_call, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].tool_index, -1)
        self.assertEqual(result.calls[0].name, "unknown_tool")
        self.assertEqual(json.loads(result.calls[0].parameters), {"value": 1})

    def test_streaming_forwards_unknown_tool(self):
        """Streaming InternLM parsing must not discard an opted-in unknown call."""
        with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
            result = InternlmDetector().parse_streaming_increment(
                self.unknown_call, self.tools
            )

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].tool_index, 0)
        self.assertEqual(result.calls[0].name, "unknown_tool")
        self.assertEqual(json.loads(result.calls[0].parameters), {"value": 1})


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
    import sys

    sys.exit(pytest.main([__file__]))
