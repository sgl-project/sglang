"""Unit tests for DeepSeekV41Detector (spaced DSML tags) -- no server, no model loading."""

import json
import unittest
from typing import get_args

import xgrammar as xgr
from xgrammar.structural_tag import JSONSchemaFormat
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.entrypoints.openai import encoding_dsv41
from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.deepseekv41_detector import DeepSeekV41Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

CHUNK_SIZES = [1, 2, 3, 5, 7, 11, 23, 1000]
DSML = "｜DSML｜"


def _tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get weather information",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="lookup",
                description="Look up a value",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                        "flags": {"type": "array"},
                    },
                },
            ),
        ),
    ]


def _assemble(calls):
    """Streamed ToolCallItems -> [(name, parsed arguments)] per tool_index."""
    by_index = {}
    for call in calls:
        entry = by_index.setdefault(call.tool_index, {"name": None, "args": ""})
        if call.name:
            entry["name"] = call.name
        entry["args"] += call.parameters or ""
    return [
        (entry["name"], json.loads(entry["args"]))
        for _, entry in sorted(by_index.items())
    ]


class TestDeepSeekV41RoundTrip(CustomTestCase):
    """Encoder-rendered assistant tool calls parse back to the same arguments,
    in one shot and at every chunk size."""

    ARGUMENTS = {"query": '{"a": 1}', "limit": 2, "flags": [1, True, None]}

    def setUp(self):
        self.tools = _tools()
        self.completion = encoding_dsv41.render_message(
            1,
            [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "reasoning_content": "reason",
                    "content": "summary",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": json.dumps(self.ARGUMENTS),
                            },
                        }
                    ],
                },
            ],
            thinking_mode="thinking",
        )
        self.completion = "<think>" + self.completion
        self.expected = [("lookup", self.ARGUMENTS)]

    def test_one_shot(self):
        parser = FunctionCallParser(self.tools, "deepseekv41")
        reasoning, content = ReasoningParser("deepseek-v41").parse_non_stream(
            self.completion
        )
        self.assertEqual(reasoning, "reason")
        normal, calls = parser.parse_non_stream(content)
        self.assertEqual(normal, "summary")
        self.assertEqual(
            [(c.name, json.loads(c.parameters)) for c in calls], self.expected
        )

    def test_streaming_at_every_chunk_size(self):
        for chunk_size in CHUNK_SIZES:
            with self.subTest(chunk_size=chunk_size):
                reasoning_parser = ReasoningParser("deepseek-v41")
                tool_parser = FunctionCallParser(self.tools, "deepseekv41")
                reasoning, normal, calls = "", "", []
                for i in range(0, len(self.completion), chunk_size):
                    reason, content = reasoning_parser.parse_stream_chunk(
                        self.completion[i : i + chunk_size]
                    )
                    reasoning += reason or ""
                    text, delta = tool_parser.parse_stream_chunk(content or "")
                    normal += text
                    calls.extend(delta)
                reason, content = reasoning_parser.parse_stream_end()
                reasoning += reason or ""
                text, delta = tool_parser.parse_stream_chunk(content or "")
                normal += text
                calls.extend(delta)
                text, delta = tool_parser.parse_stream_end()
                normal += text
                calls.extend(delta)
                self.assertEqual(reasoning, "reason")
                # The blank line before the block is released or trimmed depending
                # on where the chunk boundary falls; the shared base behaves the
                # same for V4, so only the prose itself is pinned here.
                self.assertEqual(normal.strip(), "summary")
                self.assertEqual(_assemble(calls), self.expected)


class TestDeepSeekV41ConstrainedDecoding(CustomTestCase):
    """A forced call must open the calls block before the first invoke; the
    per-tool legacy tag started the grammar at the invoke trigger, the model
    closed a block it had not opened, and the parser dropped the call."""

    def setUp(self):
        self.tools = _tools()
        self.detector = DeepSeekV41Detector()

    def test_no_builtin_structural_tag(self):
        """xgrammar's builtin deepseek_v4 tag is the unspaced grammar."""
        self.assertIsNone(self.detector.get_structural_tag_name())
        self.assertIsNone(self.detector.get_structural_tag([], "required"))

    def test_required_tag_wraps_invokes_in_the_calls_block(self):
        tag = self.detector.get_structural_tag(tools=self.tools, tool_choice="required")
        opener, calls, closer = tag.format.elements
        self.assertEqual(opener.value, f"\n\n<{DSML} calls>\n")
        self.assertEqual(closer.value, f"</{DSML} calls>")
        self.assertTrue(calls.at_least_one)
        self.assertEqual(
            [t.begin for t in calls.tags],
            [
                f'<{DSML} invoke name="get_weather">\n',
                f'<{DSML} invoke name="lookup">\n',
            ],
        )
        self.assertEqual({t.end for t in calls.tags}, {f"</{DSML} invoke>\n"})

    def test_named_tool_choice_keeps_only_that_tool(self):
        tag = self.detector.get_structural_tag(
            tools=self.tools,
            tool_choice=ToolChoice(function=ToolChoiceFuncName(name="lookup")),
        )
        _, call, _ = tag.format.elements
        self.assertEqual(call.begin, f'<{DSML} invoke name="lookup">\n')
        self.assertEqual(call.type, "tag")

    def test_parallel_off_allows_one_invoke(self):
        tag = self.detector.get_structural_tag(
            tools=self.tools, tool_choice="required", parallel_tool_calls=False
        )
        _, calls, _ = tag.format.elements
        self.assertEqual(calls.type, "or")
        self.assertEqual(len(calls.elements), 2)

    def test_auto_tag_triggers_on_the_calls_block(self):
        tag = self.detector.get_structural_tag(tools=self.tools, tool_choice="auto")
        self.assertEqual(tag.format.triggers, [f"<{DSML} calls>"])
        self.assertEqual(tag.format.tags[0].begin, f"<{DSML} calls>\n")
        self.assertEqual(tag.format.tags[0].end, f"</{DSML} calls>")

    def test_thinking_mode_prefixes_the_reasoning_span(self):
        tag = self.detector.get_structural_tag(
            tools=self.tools, tool_choice="required", thinking_mode=True
        )
        reasoning, body = tag.format.elements
        self.assertEqual(reasoning.end, "</think>")
        self.assertEqual(body.elements[0].value, f"\n\n<{DSML} calls>\n")

    def test_parser_uses_the_native_tag_for_required(self):
        parser = FunctionCallParser(self.tools, "deepseekv41")
        kind, tag = parser.get_structure_constraint("required")
        self.assertEqual(kind, "structural_tag")
        self.assertEqual(tag.format.elements[0].value, f"\n\n<{DSML} calls>\n")

    def test_body_uses_available_xgrammar_style(self):
        """Older XGrammar must keep a compilable, schema-constrained JSON fallback."""
        self.tools[0].function.strict = True
        tag = self.detector.get_structural_tag(self.tools, "required")
        grammar = xgr.Grammar.from_structural_tag(tag)
        native_xml = tag.format.elements[1].tags[0].content.style == "deepseek_v4_1_xml"
        begin = f'\n\n<{DSML} calls>\n<{DSML} invoke name="get_weather">\n'
        end = f"</{DSML} invoke>\n</{DSML} calls>"
        xml = f'<{DSML} parameter name="city" string="true">Paris</{DSML} parameter>\n'
        self.assertEqual(
            _is_grammar_accept_string(grammar, begin + xml + end), native_xml
        )
        self.assertEqual(
            _is_grammar_accept_string(grammar, begin + '{"city":"Paris"}' + end),
            not native_xml,
        )
        self.assertFalse(_is_grammar_accept_string(grammar, begin + end))
        self.assertFalse(_is_grammar_accept_string(grammar, begin + "{}" + end))


@unittest.skipUnless(
    "deepseek_v4_1_xml" in get_args(JSONSchemaFormat.model_fields["style"].annotation),
    "Requires XGrammar's DeepSeek V4.1 XML style",
)
class TestDeepSeekV41ParameterGrammar(CustomTestCase):
    """The encoder emits DSML parameters; a JSON invoke body rejects valid output."""

    def setUp(self):
        self.tools = _tools()
        self.tools[1].function.parameters["properties"]["flags"]["items"] = True
        for tool in self.tools:
            tool.function.strict = True
            tool.function.parameters["additionalProperties"] = False

    @staticmethod
    def _render(arguments, *, thinking=False, count=1, name="get_weather"):
        return encoding_dsv41.render_message(
            1,
            [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "reason",
                    "wo_eos": True,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ]
                    * count,
                },
            ],
            thinking_mode="thinking" if thinking else "chat",
        )

    def _grammar(self, choice="required", thinking=False, parallel=True):
        constraint = FunctionCallParser(
            self.tools, "deepseekv41"
        ).get_structure_constraint(
            choice, thinking_mode=thinking, parallel_tool_calls=parallel
        )
        self.assertIsNotNone(constraint)
        self.assertEqual(constraint[0], "structural_tag")
        return xgr.Grammar.from_structural_tag(constraint[1])

    def test_encoder_output_matches_and_round_trips(self):
        arguments = {"query": '{"a": 1}', "limit": 2, "flags": [True, None, 1.5]}
        for thinking in (False, True):
            with self.subTest(thinking=thinking):
                output = self._render(arguments, thinking=thinking, name="lookup")
                grammar = self._grammar(thinking=thinking)
                self.assertTrue(_is_grammar_accept_string(grammar, output))
                if thinking:
                    output = output.split("</think>", 1)[1]
                parsed = DeepSeekV41Detector().detect_and_parse(output, self.tools)
                self.assertEqual(json.loads(parsed.calls[0].parameters), arguments)

    def test_strict_schema_rejects_missing_extra_and_wrong_type(self):
        for choice in (
            "auto",
            "required",
            ToolChoice(function=ToolChoiceFuncName(name="get_weather")),
        ):
            grammar = self._grammar(choice)
            self.assertTrue(
                _is_grammar_accept_string(grammar, self._render({"city": "杭州"}))
            )
            for arguments in ({}, {"city": 42}, {"city": "Paris", "extra": True}):
                with self.subTest(choice=choice, arguments=arguments):
                    self.assertFalse(
                        _is_grammar_accept_string(grammar, self._render(arguments))
                    )
            self.assertFalse(
                _is_grammar_accept_string(
                    grammar,
                    self._render({"city": "Paris"}).replace(
                        'string="true">Paris', 'string="false">42'
                    ),
                )
            )

    def test_parallel_and_named_choice_limit_calls(self):
        for parallel in (False, True):
            grammar = self._grammar(parallel=parallel)
            self.assertTrue(
                _is_grammar_accept_string(grammar, self._render({"city": "Paris"}))
            )
            self.assertEqual(
                _is_grammar_accept_string(
                    grammar, self._render({"city": "Paris"}, count=2)
                ),
                parallel,
            )
        grammar = self._grammar(
            ToolChoice(function=ToolChoiceFuncName(name="get_weather"))
        )
        self.assertFalse(
            _is_grammar_accept_string(grammar, self._render({"city": "Paris"}, count=2))
        )
        self.assertFalse(
            _is_grammar_accept_string(
                grammar, self._render({"query": "Paris"}, name="lookup")
            )
        )

    def test_non_strict_still_uses_native_parameters(self):
        self.tools[0].function.strict = False
        grammar = self._grammar()
        self.assertTrue(
            _is_grammar_accept_string(grammar, self._render({"extra": [True, None, 2]}))
        )
        self.assertFalse(
            _is_grammar_accept_string(
                grammar,
                self._render({"extra": [True, None, 2]}).replace(
                    "[true, null, 2]", "invalid"
                ),
            )
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
