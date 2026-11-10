"""Unit tests for DeepSeekV41Detector (spaced DSML tags) -- no server, no model loading."""

import json
import unittest
from typing import get_args

import xgrammar as xgr
from xgrammar.structural_tag import JSONSchemaFormat
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.entrypoints.openai.encoding_dsv41 import (
    dsml_token,
    encode_arguments_to_dsml,
    encode_messages,
    eos_token,
    thinking_end_token,
    tool_call_tag_name,
    tool_calls_block_name,
    tool_parameter_tag_name,
)
from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.deepseekv41_detector import DeepSeekV41Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.deepseek_recipe_renderer import load_deepseek_recipe
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

DSML = "｜DSML｜"
EOS = "<｜end▁of▁sentence｜>"


def _v41_wrapped(invoke: str) -> str:
    block = tool_calls_block_name
    return f"<{dsml_token}{block}>\n{invoke}\n</{dsml_token}{block}>"


def _v41_invoke(name: str, params: str = "") -> str:
    tag = tool_call_tag_name
    return f'<{dsml_token}{tag} name="{name}">\n{params}\n</{dsml_token}{tag}>'



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


def _render_assistant(message, *, thinking):
    """The assistant turn as deepseek-recipe renders it, minus the generation
    header (`<think>` / `</think>`) that ends the prompt before it."""
    recipe = load_deepseek_recipe()
    calls = [
        {**call, "id": f"call_{i}"} for i, call in enumerate(message["tool_calls"])
    ]
    results = [
        {"role": "tool", "tool_call_id": call["id"], "content": "ok"} for call in calls
    ]
    body = {
        "model": "DeepSeek-V4.1-Flash",
        "messages": [
            {"role": "user", "content": "question"},
            {**message, "tool_calls": calls},
            *results,
        ],
        "thinking": {"type": "enabled" if thinking else "disabled"},
        # Declared tools keep the turn's reasoning in the rendered history.
        "tools": [tool.model_dump(exclude_none=True) for tool in _tools()],
    }
    converted = recipe.ChatCompletionRequest(body).convert(recipe.ConversionOptions())
    prompt = recipe.DeepseekV41Encoding().render_conversation(converted.conversation)
    turn = prompt.prompt.split("<｜User｜>question<｜Assistant｜>", 1)[1]
    turn = turn[: turn.index(EOS) + len(EOS)]
    return turn.removeprefix("<think>" if thinking else "</think>")


class TestDeepSeekV41ConstrainedDecoding(CustomTestCase):
    """A forced call must open the calls block before the first invoke; the
    per-tool legacy tag started the grammar at the invoke trigger, the model
    closed a block it had not opened, and the parser dropped the call."""

    def setUp(self):
        self.tools = _tools()
        self.detector = DeepSeekV41Detector()

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
        return _render_assistant(
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "reason",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ]
                * count,
            },
            thinking=thinking,
        ).removesuffix(EOS)

    def _grammar(self, choice="required", thinking=False, parallel=True):
        constraint = FunctionCallParser(
            self.tools, "deepseekv41"
        ).get_structure_constraint(
            choice, thinking_mode=thinking, parallel_tool_calls=parallel
        )
        self.assertIsNotNone(constraint)
        self.assertEqual(constraint[0], "structural_tag")
        return xgr.Grammar.from_structural_tag(constraint[1])

    def test_encoder_output_matches_the_grammar(self):
        arguments = {"query": '{"a": 1}', "limit": 2, "flags": [True, None, 1.5]}
        for thinking in (False, True):
            with self.subTest(thinking=thinking):
                output = self._render(arguments, thinking=thinking, name="lookup")
                grammar = self._grammar(thinking=thinking)
                self.assertTrue(_is_grammar_accept_string(grammar, output))

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


class TestDeepSeekV41NameResolution(CustomTestCase):
    def test_cli_parser_names_resolve(self):
        self.assertIn("deepseekv41", FunctionCallParser.ToolCallParserEnum)
        self.assertIs(
            FunctionCallParser.ToolCallParserEnum["deepseekv41"], DeepSeekV41Detector
        )
        self.assertIn("deepseek-v41", ReasoningParser.DetectorMap)


class TestDeepSeekV41Detector(CustomTestCase):
    """SM70 fork: encoding-level checks of the spaced V4.1 DSML tags."""

    def setUp(self):
        self.tools = [
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
            )
        ]
        self.params = encode_arguments_to_dsml({"arguments": {"city": "SF"}})

    def test_tokens_use_spaced_v41_names(self):
        detector = DeepSeekV41Detector()
        self.assertEqual(detector.bot_token, f"<{dsml_token}{tool_calls_block_name}>")
        self.assertIn(tool_call_tag_name, detector.invoke_start_token)
        self.assertIn(tool_parameter_tag_name, detector.parameter_regex)
        self.assertNotEqual(detector.bot_token, "<｜DSML｜tool_calls>")
        self.assertNotIn("<｜DSML｜invoke ", detector.invoke_start_token)

    def test_non_streaming_parses_spaced_dsml(self):
        text = "Let me check.\n\n" + _v41_wrapped(
            _v41_invoke("get_weather", self.params)
        )
        result = DeepSeekV41Detector().detect_and_parse(text, self.tools)
        self.assertIn("Let me check.", result.normal_text)
        self.assertEqual([c.name for c in result.calls if c.name], ["get_weather"])
        self.assertIn("SF", result.calls[0].parameters)

    def test_streaming_agrees_with_one_shot(self):
        text = "Checking.\n\n" + _v41_wrapped(_v41_invoke("get_weather", self.params))
        detector = DeepSeekV41Detector()
        normal, calls = "", []
        for i in range(0, len(text), 7):
            result = detector.parse_streaming_increment(text[i : i + 7], self.tools)
            normal += result.normal_text
            calls.extend(result.calls)
        one_shot = DeepSeekV41Detector().detect_and_parse(text, self.tools)
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])
        self.assertEqual(normal, one_shot.normal_text)

    def test_unspaced_v4_tags_are_not_v41_calls(self):
        v4 = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_weather">\n'
            f'<{dsml_token}{tool_parameter_tag_name} name="city" string="true">'
            "SF"
            f"</{dsml_token}{tool_parameter_tag_name}>\n"
            "</｜DSML｜invoke>\n"
            "</｜DSML｜tool_calls>"
        )
        result = DeepSeekV41Detector().detect_and_parse(v4, self.tools)
        self.assertEqual(result.calls, [])

    def test_function_call_parser_name_parses(self):
        text = _v41_wrapped(_v41_invoke("get_weather", self.params))
        parser = FunctionCallParser(self.tools, "deepseekv41")
        result = parser.detector.detect_and_parse(text, self.tools)
        self.assertEqual([c.name for c in result.calls if c.name], ["get_weather"])


class TestDeepSeekV41ReasoningParser(CustomTestCase):
    def test_deepseek_v41_splits_think_tags(self):
        parser = ReasoningParser("deepseek-v41")
        reasoning, normal = parser.parse_non_stream(
            "<think>plan the call</think>\n\nanswer"
        )
        self.assertEqual(reasoning, "plan the call")
        self.assertEqual(normal, "\n\nanswer")

    def test_reasoning_then_spaced_tool_call_stays_in_normal(self):
        parser = ReasoningParser("deepseek-v41")
        call = _v41_wrapped(
            _v41_invoke(
                "get_weather",
                encode_arguments_to_dsml({"arguments": {"city": "SF"}}),
            )
        )
        reasoning, normal = parser.parse_non_stream(
            f"<think>need weather</think>\n\n{call}"
        )
        self.assertEqual(reasoning, "need weather")
        self.assertIn(tool_calls_block_name, normal)
        self.assertIn("get_weather", normal)


class TestDsv41StickyPrefix(CustomTestCase):
    def test_trailing_system_reminder_keeps_the_previous_prompt_as_prefix(self):
        """A follow-up that leaves the previous harness reminder where it was
        and appends a new one must keep the first turn's prompt plus the
        assistant text as a prefix. That is the sticky-cache contract: the
        next request's token ids have to start with the pinned sequence."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ]
        system = {
            "role": "system",
            "content": "You are a coding assistant.",
            "tools": tools,
        }
        user = {"role": "user", "content": "Read foo.py and tell me the first line."}
        reminder = {
            "role": "system",
            "content": (
                "<system-reminder>\nThe current date is 2026-09-22.\n</system-reminder>"
            ),
        }
        first = encode_messages(
            [system, user, reminder],
            thinking_mode="thinking",
            drop_thinking=False,
        )
        output = "need the file" + thinking_end_token + "\n\nReading it." + eos_token
        assistant = {
            "role": "assistant",
            "reasoning_content": "need the file",
            "content": "\n\nReading it.",
        }
        follow = encode_messages(
            [
                system,
                user,
                reminder,
                assistant,
                {"role": "user", "content": "also check bar.py"},
                {
                    "role": "system",
                    "content": "<system-reminder>\nfoo.py changed.\n</system-reminder>",
                },
            ],
            thinking_mode="thinking",
            drop_thinking=False,
        )
        self.assertTrue(follow.startswith(first + output))


if __name__ == "__main__":
    import unittest

    unittest.main()
