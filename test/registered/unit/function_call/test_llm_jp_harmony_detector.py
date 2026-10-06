import json
import unittest

import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.llm_jp_harmony_detector import LlmJpHarmonyDetector
from sglang.srt.parser.reasoning_parser import (
    LlmJpHarmonyDetector as LlmJpHarmonyReasoningDetector,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _tool(name, properties):
    return Tool(
        type="function",
        function=Function(
            name=name,
            parameters={
                "type": "object",
                "properties": properties,
                "required": list(properties),
            },
        ),
    )


TOOLS = [
    _tool("get_time", {"tz": {"type": "string"}}),
    _tool("get_weather", {"city": {"type": "string"}}),
]


def _call(city, end, spaced=True):
    if spaced:
        header = (
            "<|start|> assistant to=functions.get_weather<|channel|> commentary "
            "<|constrain|>  json<|message|> "
        )
    else:
        header = (
            "<|start|>assistant to=functions.get_weather<|channel|>commentary "
            "<|constrain|> json<|message|>"
        )
    return header + json.dumps({"city": city}) + end


def _analysis(spaced=True):
    if spaced:
        return "<|channel|> analysis<|message|> Two cities.<|end|>"
    return "<|channel|>analysis<|message|>Two cities.<|end|>"


# Model output after the generation prompt "<|start|>assistant"; the final
# <|call|> is an EOS token that SGLang trims.
def _parallel_output(spaced=True):
    return (
        _analysis(spaced)
        + _call("Tokyo", "<|end|>", spaced)
        + _call("Osaka", "", spaced)
    )


class TestLlmJpHarmonyToolCalls(CustomTestCase):
    def _forwarded(self, text):
        # What the reasoning parser passes on to the function-call detector.
        return LlmJpHarmonyReasoningDetector().detect_and_parse(text)

    def test_parallel_calls_split_on_end_with_trimmed_final_call(self):
        for spaced in (True, False):
            with self.subTest(spaced=spaced):
                forwarded = self._forwarded(_parallel_output(spaced))
                self.assertEqual(forwarded.reasoning_text, "Two cities.")
                result = LlmJpHarmonyDetector().detect_and_parse(
                    forwarded.normal_text, TOOLS
                )
                self.assertEqual(result.normal_text, "")
                self.assertEqual(
                    [
                        (c.name, json.loads(c.parameters), c.tool_index)
                        for c in result.calls
                    ],
                    [
                        ("get_weather", {"city": "Tokyo"}, 1),
                        ("get_weather", {"city": "Osaka"}, 1),
                    ],
                )

    def test_streaming_emits_each_call_with_its_ordinal(self):
        forwarded = self._forwarded(_parallel_output()).normal_text
        detector = LlmJpHarmonyDetector()
        calls, normal = [], ""
        for ch in forwarded:
            result = detector.parse_streaming_increment(ch, TOOLS)
            calls += result.calls
            normal += result.normal_text
        # The last call has no terminator; it completes when the stream ends.
        self.assertEqual(len(calls), 1)
        result = detector.finish(TOOLS)
        calls += result.calls
        normal += result.normal_text
        self.assertEqual(normal, "")
        self.assertEqual(
            [(c.name, json.loads(c.parameters), c.tool_index) for c in calls],
            [
                ("get_weather", {"city": "Tokyo"}, 0),
                ("get_weather", {"city": "Osaka"}, 1),
            ],
        )

    def test_streaming_drops_an_unparseable_last_call(self):
        # The last call ends at EOS, so only finish() can see it; it must be
        # dropped as in non-streaming, not returned as content.
        cut = _call("Tokyo", "")[:-3]
        no_args = _call("Tokyo", "").split("{")[0]
        unknown = _call("Tokyo", "").replace("get_weather", "get_wether")
        for last in (cut, no_args, unknown):
            with self.subTest(last=last):
                forwarded = self._forwarded(_analysis() + last).normal_text
                detector = LlmJpHarmonyDetector()
                result = detector.parse_streaming_increment(forwarded, TOOLS)
                end = detector.finish(TOOLS)
                self.assertEqual(result.calls + end.calls, [])
                self.assertEqual(result.normal_text + end.normal_text, "")


class TestLlmJpHarmonyStructuralTag(CustomTestCase):
    """The grammar starts after the reasoning and must accept the model's own
    spelling: a space after every special token, <|end|> between calls, and the
    last call ended by EOS rather than by text."""

    def _grammar(self, tool_choice, parallel_tool_calls=True):
        tag = LlmJpHarmonyDetector().get_structural_tag(
            tools=TOOLS,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel_tool_calls,
        )
        return xgr.Grammar.from_structural_tag(tag)

    def test_required_accepts_parallel_calls(self):
        grammar = self._grammar("required")
        one = _call("Tokyo", "")
        two = _call("Tokyo", "<|end|>") + _call("Osaka", "")
        self.assertTrue(_is_grammar_accept_string(grammar, one))
        self.assertTrue(_is_grammar_accept_string(grammar, two))
        self.assertFalse(_is_grammar_accept_string(grammar, ""))

    def test_parallel_tool_calls_false_allows_one_call(self):
        grammar = self._grammar("required", parallel_tool_calls=False)
        two = _call("Tokyo", "<|end|>") + _call("Osaka", "")
        self.assertTrue(_is_grammar_accept_string(grammar, _call("Tokyo", "")))
        self.assertFalse(_is_grammar_accept_string(grammar, two))

    def test_named_choice_allows_only_that_tool(self):
        choice = ToolChoice(
            type="function", function=ToolChoiceFuncName(name="get_time")
        )
        grammar = self._grammar(choice)
        self.assertFalse(_is_grammar_accept_string(grammar, _call("Tokyo", "")))

    def test_auto_also_accepts_a_final_message(self):
        grammar = self._grammar("auto")
        final = "<|start|> assistant<|channel|> final<|message|> 東京です。"
        self.assertTrue(_is_grammar_accept_string(grammar, final))
        self.assertTrue(_is_grammar_accept_string(grammar, _call("Tokyo", "")))


if __name__ == "__main__":
    unittest.main()
