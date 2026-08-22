"""Unit tests for GptOssDetector — no server, no model loading."""

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.gpt_oss_detector import GptOssDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


class TestGptOssDetector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Searches for information.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "topn": {"type": "integer"},
                        },
                        "required": ["query"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information for a city.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = GptOssDetector()

    def test_has_tool_call_detects_marker(self):
        # Guards the bot_token substring predicate, which detect_and_parse
        # does not exercise in isolation.
        self.assertTrue(
            self.detector.has_tool_call(
                "<|start|>assistant<|channel|>commentary to=get_weather<|return|>"
            )
        )
        self.assertFalse(self.detector.has_tool_call("no tool call here"))

    def test_get_model_structural_tag(self):
        import xgrammar as xgr

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True, tool_choice="required"
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False, tool_choice="required"
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        tool_choice_name = ToolChoiceFuncName(name="search")
        tool_choice = ToolChoice(function=tool_choice_name)
        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True, tool_choice=tool_choice
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False, tool_choice=tool_choice
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)


if __name__ == "__main__":
    import unittest

    unittest.main()
