import json
import logging

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.environ import envs
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.gemma4_detector import Gemma4Detector
from sglang.srt.function_call.kimik2_detector import KimiK2Detector
from sglang.srt.function_call.kimik3_detector import KimiK3Detector
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
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


# Detectors that override detect_and_parse and parse the model output themselves
# instead of delegating to parse_base_json, so they must apply the check on
# their own. Each sample declares a call to "unknown_tool" with {"city": "Paris"}.
ONE_SHOT_SAMPLES = [
    (
        Qwen3CoderDetector,
        "<tool_call><function=unknown_tool>"
        "<parameter=city>Paris</parameter></function></tool_call>",
    ),
    (
        Gemma4Detector,
        '<|tool_call>call:unknown_tool{city:<|"|>Paris<|"|>}<tool_call|>',
    ),
    (
        KimiK2Detector,
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.unknown_tool:0"
        '<|tool_call_argument_begin|>{"city": "Paris"}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>",
    ),
    (
        KimiK3Detector,
        "<|open|>tools<|sep|>"
        '<|open|>call tool="unknown_tool" index="0"<|sep|>'
        '<|open|>argument key="city" type="string"<|sep|>Paris'
        "<|close|>argument<|sep|>"
        "<|close|>call<|sep|>"
        "<|close|>tools<|sep|>",
    ),
]


@pytest.mark.parametrize(
    "detector_cls,text", ONE_SHOT_SAMPLES, ids=[c.__name__ for c, _ in ONE_SHOT_SAMPLES]
)
def test_one_shot_unknown_tool_name_dropped_default(detector_cls, text, caplog):
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(False):
        tools = [
            Tool(
                function=Function(
                    name="get_weather", parameters={"type": "object", "properties": {}}
                )
            )
        ]
        with caplog.at_level(logging.WARNING, logger="sglang.srt.function_call"):
            result = detector_cls().detect_and_parse(text, tools)
        assert any(
            "Model attempted to call undefined function: unknown_tool" in m
            for m in caplog.messages
        )
        assert len(result.calls) == 0


@pytest.mark.parametrize(
    "detector_cls,text", ONE_SHOT_SAMPLES, ids=[c.__name__ for c, _ in ONE_SHOT_SAMPLES]
)
def test_one_shot_unknown_tool_name_forwarded(detector_cls, text):
    with envs.SGLANG_FORWARD_UNKNOWN_TOOLS.override(True):
        tools = [
            Tool(
                function=Function(
                    name="get_weather", parameters={"type": "object", "properties": {}}
                )
            )
        ]
        result = detector_cls().detect_and_parse(text, tools)
        assert len(result.calls) == 1
        assert result.calls[0].name == "unknown_tool"
        assert json.loads(result.calls[0].parameters)["city"] == "Paris"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
