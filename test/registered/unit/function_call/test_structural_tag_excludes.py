"""Unit tests for strip_structural_tag_excludes — no server, no model loading."""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.base_format_detector import get_model_structural_tag
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.utils import strip_structural_tag_excludes
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")

THINK_TOKENS = ("<think>", "</think>")


def _weather_tool() -> Tool:
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


def _collect_excludes(node, out):
    if isinstance(node, list):
        for item in node:
            _collect_excludes(item, out)
    elif isinstance(node, dict):
        if "excludes" in node:
            out.append(list(node["excludes"]))
        for key in ("format", "content", "tags", "elements"):
            if key in node:
                _collect_excludes(node[key], out)
    return out


class TestStripStructuralTagExcludes(unittest.TestCase):
    def test_strips_only_think_tokens_from_every_free_text_section(self):
        tag = {
            "type": "structural_tag",
            "format": {
                "type": "sequence",
                "elements": [
                    {
                        "type": "triggered_tags",
                        "triggers": ["<tool>"],
                        "tags": [
                            {
                                "type": "tag",
                                "begin": "<tool>",
                                "content": {
                                    "type": "any_text",
                                    "excludes": ["</think>", "</tool>"],
                                },
                                "end": "</tool>",
                            }
                        ],
                        "excludes": ["<think>", "</think>", "<tool_call>"],
                    },
                    {"type": "any_text", "excludes": ["<think>", "</think>"]},
                ],
            },
        }

        strip_structural_tag_excludes(tag, THINK_TOKENS)

        self.assertEqual(
            _collect_excludes(tag, []),
            [["<tool_call>"], ["</tool>"], []],
        )

    def test_no_tokens_is_a_no_op(self):
        tag = {"type": "any_text", "excludes": ["<think>"]}

        strip_structural_tag_excludes(tag, ())

        self.assertEqual(tag["excludes"], ["<think>"])

    @unittest.skipIf(get_model_structural_tag is None, "xgrammar builtin tags absent")
    def test_deepseek_v4_auto_tag_allows_think_tokens_after_strip(self):
        tag = DeepSeekV4Detector().get_structural_tag([_weather_tool()], "auto")
        before = _collect_excludes(tag.model_dump(by_alias=True), [])
        self.assertTrue(any("</think>" in ex for ex in before))

        strip_structural_tag_excludes(tag, THINK_TOKENS)

        after = _collect_excludes(tag.model_dump(by_alias=True), [])
        self.assertFalse(any(tok in ex for ex in after for tok in THINK_TOKENS))
        self.assertEqual(tag.format.triggers, ["<｜DSML｜tool_calls>"])


if __name__ == "__main__":
    unittest.main()
