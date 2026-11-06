"""Unit tests for qwen3_coder xgrammar token_markers feature detection."""

import unittest
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call import base_format_detector
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")


def _tool() -> Tool:
    return Tool(
        type="function",
        function=Function(
            name="get_weather",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ),
    )


class _Recorder:
    def __init__(self):
        self.kwargs = None

    def record(self, kwargs):
        self.kwargs = kwargs
        return "tag"


def _new_api(recorder):
    def get_model_structural_tag(
        model,
        tools=None,
        tool_choice="auto",
        reasoning="enabled",
        parallel_tool_calls=True,
        token_markers=False,
    ):
        return recorder.record(
            dict(model=model, reasoning=reasoning, token_markers=token_markers)
        )

    return get_model_structural_tag


def _old_api(recorder):
    def get_model_structural_tag(
        model,
        tools=None,
        tool_choice="auto",
        reasoning="enabled",
        parallel_tool_calls=True,
    ):
        return recorder.record(dict(model=model, reasoning=reasoning))

    return get_model_structural_tag


class _OtherDetector(Qwen3CoderDetector):
    def use_structural_tag_token_markers(self) -> bool:
        return BaseFormatDetector.use_structural_tag_token_markers(self)


class TestQwen3CoderTokenMarkers(unittest.TestCase):
    def _call(self, api_factory, detector):
        recorder = _Recorder()
        with patch.object(
            base_format_detector, "get_model_structural_tag", api_factory(recorder)
        ):
            result = detector.get_structural_tag(tools=[_tool()], tool_choice="auto")
        self.assertEqual(result, "tag")
        return recorder.kwargs

    def test_token_markers_passed_when_supported(self):
        kwargs = self._call(_new_api, Qwen3CoderDetector())
        self.assertEqual(kwargs["model"], "qwen_3_coder")
        self.assertIs(kwargs["token_markers"], True)

    def test_falls_back_to_text_trigger_on_old_xgrammar(self):
        kwargs = self._call(_old_api, Qwen3CoderDetector())
        self.assertNotIn("token_markers", kwargs)

    def test_not_passed_for_detectors_without_opt_in(self):
        kwargs = self._call(_new_api, _OtherDetector())
        self.assertIs(kwargs["token_markers"], False)

    def test_no_xgrammar(self):
        with patch.object(base_format_detector, "get_model_structural_tag", None):
            self.assertIsNone(Qwen3CoderDetector().get_structural_tag(tools=[_tool()]))


if __name__ == "__main__":
    unittest.main()
