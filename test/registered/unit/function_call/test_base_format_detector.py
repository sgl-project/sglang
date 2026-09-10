"""CPU-only tests for shared BaseFormatDetector behavior."""

import json
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StructureInfo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _ConcreteDetector(BaseFormatDetector):
    def __init__(self):
        super().__init__()
        self.bot_token = "<tool>"
        self.eot_token = "</tool>"

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: list[Tool]):
        return super().detect_and_parse(text, tools)

    def structure_info(self):
        return lambda name: StructureInfo(
            begin=f'<tool name="{name}">',
            end="</tool>",
            trigger="<tool",
        )


class _NativeTagDetector(_ConcreteDetector):
    def get_structural_tag_name(self):
        return "test-native-format"


class TestBaseFormatDetectorCore(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                function=Function(
                    name="weather",
                    description="Get weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                )
            )
        ]

    def test_base_json_parsing_normalizes_object_list_and_argument_keys(self):
        detector = _ConcreteDetector()
        payload = [
            {"name": "weather", "arguments": {"city": "杭州"}},
            {"name": "weather", "parameters": {"city": "Singapore"}},
        ]

        list_result = detector.detect_and_parse(
            json.dumps(payload, ensure_ascii=False), self.tools
        )
        object_result = detector.detect_and_parse(
            json.dumps(payload[0], ensure_ascii=False), self.tools
        )

        self.assertEqual(len(list_result.calls), 2)
        self.assertEqual(
            [json.loads(call.parameters) for call in list_result.calls],
            [{"city": "杭州"}, {"city": "Singapore"}],
        )
        self.assertEqual(len(object_result.calls), 1)
        self.assertIn("杭州", object_result.calls[0].parameters)

    def test_partial_token_helper_finds_only_a_suffix_prefix(self):
        detector = _ConcreteDetector()

        self.assertEqual(detector._ends_with_partial_token("plain<too", "<tool>"), 4)
        self.assertEqual(detector._ends_with_partial_token("plain", "<tool>"), 0)
        self.assertEqual(detector._ends_with_partial_token("<tool>", "<tool>"), 0)

    def test_native_structural_tag_serializes_tools_and_string_choice(self):
        detector = _NativeTagDetector()
        native_tag = object()

        with patch(
            "sglang.srt.function_call.base_format_detector.get_model_structural_tag",
            return_value=native_tag,
        ) as get_model_structural_tag:
            result = detector.get_structural_tag(
                tools=self.tools,
                tool_choice="required",
                thinking_mode=True,
                parallel_tool_calls=False,
            )

        self.assertIs(result, native_tag)
        get_model_structural_tag.assert_called_once_with(
            model="test-native-format",
            tools=[self.tools[0].model_dump()],
            tool_choice="required",
            reasoning=True,
        )

    def test_native_structural_tag_serializes_named_tool_choice(self):
        detector = _NativeTagDetector()
        named_choice = ToolChoice(function=ToolChoiceFuncName(name="weather"))

        with patch(
            "sglang.srt.function_call.base_format_detector.get_model_structural_tag",
            return_value="native-tag",
        ) as get_model_structural_tag:
            result = detector.get_structural_tag(
                tools=self.tools,
                tool_choice=named_choice,
            )

        self.assertEqual(result, "native-tag")
        self.assertEqual(
            get_model_structural_tag.call_args.kwargs["tool_choice"],
            named_choice.model_dump(),
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
