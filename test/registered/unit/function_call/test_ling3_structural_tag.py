"""Ling native tool constraints: real CPU grammar matching and parser round trips."""

import asyncio
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import xgrammar as xgr
from xgrammar.testing import _is_grammar_accept_string

from sglang.srt.constrained.reasoner_grammar_backend import ReasonerGrammarObject
from sglang.srt.constrained.xgrammar_backend import XGrammarGrammar
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    Function,
    Tool,
    ToolChoice,
)
from sglang.srt.environ import ToolStrictLevel
from sglang.srt.function_call.core_types import NativeToolConstraintError
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.parser.reasoning_parser import Ling3Detector as Ling3ReasoningDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _call(name="weather", city="Beijing", newline=""):
    return (
        f"<tool_call>{name}{newline}"
        f"<arg_key>city</arg_key><arg_value>{city}</arg_value>"
        "</tool_call>"
    )


class TestLing3StructuralTag(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="weather",
                    strict=True,
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_date",
                    strict=True,
                    parameters={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                ),
            ),
        ]

    def _parser(self, tools=None, strict_level=ToolStrictLevel.OFF):
        parser = FunctionCallParser(self.tools if tools is None else tools, "ling3")
        parser.tool_strict_level = strict_level
        return parser

    def _grammar(self, choice="required", tools=None, **kwargs):
        constraint = self._parser(tools).get_structure_constraint(choice, **kwargs)
        self.assertIsNotNone(constraint)
        self.assertEqual(constraint[0], "structural_tag")
        return xgr.Grammar.from_structural_tag(constraint[1])

    def test_required_accepts_native_layouts_not_json_array(self):
        grammar = self._grammar()
        for text in (_call(), _call(newline="\n"), "<tool_call>get_date</tool_call>"):
            with self.subTest(text=text):
                self.assertTrue(_is_grammar_accept_string(grammar, text))
        for text in (
            "",
            "Normal answer without a tool call",
            '[{"name":"weather","parameters":{"city":"Beijing"}}]',
            _call(name="unknown"),
            "<tool_call>weather</tool_call>",
        ):
            with self.subTest(text=text):
                self.assertFalse(_is_grammar_accept_string(grammar, text))

    def test_parallel_tool_calls_limit(self):
        text = _call() + "<tool_call>get_date</tool_call>"
        for choice in ("required", "auto"):
            for parallel in (True, False):
                with self.subTest(choice=choice, parallel=parallel):
                    grammar = self._grammar(choice, parallel_tool_calls=parallel)
                    self.assertTrue(_is_grammar_accept_string(grammar, _call()))
                    self.assertEqual(_is_grammar_accept_string(grammar, text), parallel)

    def test_reasoning_prefix_and_parallel_limit(self):
        grammar = self._grammar(thinking_mode=True, parallel_tool_calls=False)
        text = "Let me check the weather.</think>" + _call()
        self.assertTrue(_is_grammar_accept_string(grammar, text))
        self.assertFalse(_is_grammar_accept_string(grammar, text + _call()))

    def test_coalesced_stream_preserves_prose_and_multiple_calls(self):
        text = "Checking. " + _call() + "<tool_call>get_date</tool_call>" + " Done."
        normal, calls = self._parser().parse_stream_chunk(text)
        self.assertEqual(normal, "Checking.  Done.")
        self.assertEqual(
            [call.name for call in calls if call.name], ["weather", "get_date"]
        )

    def test_tool_start_without_think_end_activates_native_masks(self):
        """With thinking on and no </think>, the required native constraint must
        take effect at <tool_call> instead of leaving the call unconstrained."""
        text = (
            "<tool_call>weather\n"
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n"
            "</tool_call>"
        )
        chat = self._chat()
        chat.reasoning_parser = "ling3"
        chat._reasoning_detector = Ling3ReasoningDetector()
        request = ChatCompletionRequest(
            model="ling",
            messages=[{"role": "user", "content": "weather?"}],
            input_ids=[1],
            tools=self.tools,
            tool_choice="required",
            chat_template_kwargs={"enable_thinking": True},
        )
        processed = chat._process_messages(request, is_multimodal=False)
        self.assertTrue(processed.require_reasoning)
        self.assertEqual(processed.tool_call_constraint[0], "structural_tag")
        # Byte vocabulary drives the real matcher without model weights.
        tokenizer_info = xgr.TokenizerInfo(
            [bytes([i]) for i in range(256)] + [b"<eos>"],
            vocab_type=xgr.VocabType.RAW,
            stop_token_ids=[256],
        )
        compiled = xgr.GrammarCompiler(tokenizer_info).compile_structural_tag(
            processed.tool_call_constraint[1]
        )
        wrapper = ReasonerGrammarObject(
            XGrammarGrammar(
                xgr.GrammarMatcher(compiled, max_rollback_tokens=256),
                257,
                compiled,
                [256],
            ),
            think_end_ids=list(b"</think>"),
            tool_start_ids=list(b"<tool_call>"),
        )
        wrapper.maybe_init_reasoning(True)
        mask = torch.full((1, 9), -1, dtype=torch.int32)

        def allowed(token):
            mask.fill_(-1)
            wrapper.fill_vocab_mask(mask, 0)
            return bool(int(mask[0, token // 32]) & (1 << (token % 32)))

        prefix = "<tool_call>"
        for token in prefix.encode():
            wrapper.accept_token(token)
        self.assertTrue(wrapper._is_generation())
        self.assertFalse(allowed(ord("!")), "unknown tool name must be masked")
        self.assertFalse(allowed(256), "EOS must wait for a complete required call")
        for token in text[len(prefix) :].encode():
            self.assertTrue(allowed(token))
            wrapper.accept_token(token)
        self.assertTrue(allowed(256))

    def _decoded_arguments(self, text, tools):
        _, calls = self._parser(tools).parse_non_stream(text)
        decoded = [json.loads(calls[0].parameters)]
        for size in (1, len(text)):
            parser = self._parser(tools)
            arguments = ""
            for start in range(0, len(text), size):
                _, deltas = parser.parse_stream_chunk(text[start : start + size])
                arguments += "".join(call.parameters for call in deltas)
            decoded.append(json.loads(arguments))
        return decoded

    def test_xml_string_values_are_not_reinterpreted_as_json(self):
        """String values are raw text in grammar, streaming and non-streaming alike;
        only a bare null under a nullable schema decodes to null."""
        for city_schema in (
            {"type": "string"},
            {"type": ["string", "null"]},
            {"anyOf": [{"type": "string"}, {"type": "null"}]},
        ):
            tool = self.tools[0].model_copy(deep=True)
            tool.function.parameters["properties"]["city"] = city_schema
            grammar = self._grammar(tools=[tool])
            nullable = city_schema != {"type": "string"}
            for value in ("true", "null", '"quoted"', '{"x":1}', r"C:\tmp", " 北京 "):
                with self.subTest(schema=city_schema, value=value):
                    text = _call(city=value)
                    self.assertTrue(_is_grammar_accept_string(grammar, text))
                    expected = None if nullable and value == "null" else value
                    self.assertEqual(
                        self._decoded_arguments(text, [tool]), [{"city": expected}] * 3
                    )

    def test_tool_closer_inside_string_value_stays_in_the_value(self):
        """The grammar lets a string value contain </tool_call>; the parsers must
        not end the call there."""
        value = "a</tool_call>b"
        text = _call(city=value)
        self.assertTrue(_is_grammar_accept_string(self._grammar(), text))
        self.assertEqual(
            self._decoded_arguments(text, self.tools), [{"city": value}] * 3
        )
        calls, content, reason = self._chat()._process_tool_calls(
            text, self.tools, {"type": "stop"}, tool_choice="required"
        )
        self.assertEqual(json.loads(calls[0].function.arguments), {"city": value})
        self.assertEqual(content, "")
        self.assertEqual(reason["type"], "tool_calls")
        for size in (1, len(text)):
            with self.subTest(chunk_size=size):
                parser = self._parser()
                normal = ""
                for start in range(0, len(text), size):
                    chunk_normal, _ = parser.parse_stream_chunk(
                        text[start : start + size]
                    )
                    normal += chunk_normal
                self.assertEqual(normal, "")

    def _stream(self, text, size):
        parser = self._parser()
        normal, names, arguments = "", [], ""
        for start in range(0, len(text), size):
            chunk_normal, deltas = parser.parse_stream_chunk(text[start : start + size])
            normal += chunk_normal
            names += [call.name for call in deltas if call.name]
            arguments += "".join(call.parameters for call in deltas)
        return parser, normal, names, arguments

    def test_unfinished_value_is_not_reported_as_a_call(self):
        """A value cut off at the token limit stays open to the end of the output:
        no mode may close the call at a </tool_call> or open one inside it."""
        prefix = "<tool_call>weather<arg_key>city</arg_key><arg_value>a</tool_call>"
        for suffix in ("", "<tool_call>get_date</tool_call>"):
            text = prefix + suffix
            rest = "</arg_value></tool_call>"
            self.assertTrue(_is_grammar_accept_string(self._grammar(), text + rest))
            with self.subTest(suffix=suffix):
                _, calls = self._parser().parse_non_stream(text)
                self.assertEqual(calls, [])
                calls, content, reason = self._chat()._process_tool_calls(
                    text, self.tools, {"type": "length"}, tool_choice="required"
                )
                self.assertIsNone(calls)
                self.assertEqual((content, reason["type"]), ("", "length"))
                for size in (1, len(text)):
                    parser, normal, names, arguments = self._stream(text, size)
                    self.assertEqual((normal, names), ("", ["weather"]))
                    _, deltas = parser.parse_stream_chunk(rest)
                    arguments += "".join(call.parameters for call in deltas)
                    self.assertEqual(
                        json.loads(arguments), {"city": "a</tool_call>" + suffix}
                    )

    def test_malformed_calls_parse_in_linear_time(self):
        """Malformed or truncated output must not make the call regexes backtrack
        super-linearly, in either mode, for ling3 or the glm45 parent format."""
        many_values = "<tool_call>weather" + "".join(
            f"<arg_key>k{i}</arg_key><arg_value>x</arg_value>" for i in range(24)
        )
        cases = (
            ("ling3", "non_stream", many_values),
            (
                "ling3",
                "non_stream",
                "<tool_call>weather" + "\n" * 16000 + "x</tool_call>",
            ),
            ("ling3", "stream", "<tool_call>" + " " * 16000 + "x"),
            ("ling3", "stream", "<tool_call>weather" * 2048),
            ("glm45", "non_stream", "<tool_call>weather" * 4096),
            ("glm45", "non_stream", "<tool_call>weather" * 4096 + "</tool_call>"),
            ("glm45", "stream", "<tool_call>weather" * 4096),
        )
        for parser_name, mode, text in cases:
            with self.subTest(parser=parser_name, mode=mode, text=text[:40]):
                parser = FunctionCallParser(self.tools, parser_name)
                start = time.perf_counter()
                if mode == "stream":
                    parser.parse_stream_chunk(text)
                else:
                    _, calls = parser.parse_non_stream(text)
                    self.assertEqual(calls, [])
                self.assertLess(time.perf_counter() - start, 0.5)

    def test_missing_native_backend_rejects_constrained_requests(self):
        with patch(
            "sglang.srt.function_call.ling3_detector._ling3_native_structural_tag_available",
            return_value=False,
        ):
            parser = self._parser()
            self.assertFalse(parser.detector.supports_structural_tag())
            for choice in (
                "required",
                ToolChoice(type="function", function={"name": "get_date"}),
                "auto",  # strict=True also requires a working native grammar.
            ):
                with (
                    self.subTest(choice=choice),
                    self.assertRaisesRegex(NativeToolConstraintError, "glm_4_7"),
                ):
                    parser.get_structure_constraint(choice)
            tools = [tool.model_copy(deep=True) for tool in self.tools]
            for tool in tools:
                tool.function.strict = False
            self.assertIsNone(self._parser(tools).get_structure_constraint("auto"))
            self.assertIsNone(parser.get_structure_constraint("none"))

    def test_native_builder_failures_do_not_become_json_or_no_constraint(self):
        for outcome in (None, ValueError("unsupported schema"), RuntimeError("failed")):
            for choice in (
                "required",
                ToolChoice(type="function", function={"name": "get_date"}),
                "auto",
            ):
                with self.subTest(outcome=outcome, choice=choice):
                    kwargs = (
                        {"side_effect": outcome}
                        if isinstance(outcome, Exception)
                        else {"return_value": outcome}
                    )
                    with (
                        patch(
                            "sglang.srt.function_call.base_format_detector.get_model_structural_tag",
                            **kwargs,
                        ),
                        patch(
                            "sglang.srt.function_call.function_call_parser.get_json_schema_constraint"
                        ) as json_fallback,
                    ):
                        with self.assertRaisesRegex(
                            NativeToolConstraintError, "Refusing JSON fallback"
                        ):
                            self._parser().get_structure_constraint(choice)
                        json_fallback.assert_not_called()

    def _chat(self):
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

        chat = OpenAIServingChat.__new__(OpenAIServingChat)
        chat.tool_call_parser = "ling3"
        chat.reasoning_parser = None
        chat._reasoning_detector = None
        chat.template_manager = SimpleNamespace(
            reasoning_config=None, chat_template_name=None
        )
        chat.chat_encoding_spec = None
        chat.default_chat_template_kwargs = {}
        chat.is_gpt_oss = chat.is_gemma4 = False
        chat.tokenizer_manager = SimpleNamespace(
            tokenizer=None,
            model_config=SimpleNamespace(is_multimodal=False),
            request_logger=SimpleNamespace(log_requests=False),
        )
        return chat

    def test_serving_constraint_errors_return_400_before_generation(self):
        chat = self._chat()
        for stream in (True, False):
            for choice in (
                "required",
                ToolChoice(type="function", function={"name": "get_date"}),
            ):
                with self.subTest(stream=stream, choice=choice):
                    request = ChatCompletionRequest(
                        model="ling",
                        messages=[{"role": "user", "content": "date?"}],
                        tools=self.tools,
                        tool_choice=choice,
                        stream=stream,
                    )
                    with (
                        patch(
                            "sglang.srt.function_call.ling3_detector._ling3_native_structural_tag_available",
                            return_value=False,
                        ),
                        patch.object(chat, "_validate_request", return_value=None),
                        patch.object(chat, "_handle_streaming_request") as streaming,
                        patch.object(
                            chat, "_handle_non_streaming_request"
                        ) as non_streaming,
                        patch(
                            "sglang.srt.entrypoints.openai.serving_chat.get_json_schema_constraint"
                        ) as fallback,
                    ):
                        response = asyncio.run(chat.handle_request(request, None))
                    self.assertEqual(response.status_code, 400)
                    self.assertIn(
                        "refusing JSON fallback", json.loads(response.body)["message"]
                    )
                    fallback.assert_not_called()
                    streaming.assert_not_called()
                    non_streaming.assert_not_called()

    def test_serving_request_builds_native_constraint(self):
        for choice in (
            "required",
            ToolChoice(type="function", function={"name": "get_date"}),
        ):
            with self.subTest(choice=choice):
                request = ChatCompletionRequest(
                    model="ling",
                    messages=[{"role": "user", "content": "date?"}],
                    input_ids=[1],
                    tools=self.tools,
                    tool_choice=choice,
                )
                processed = self._chat()._process_messages(request, is_multimodal=False)
                self.assertEqual(processed.tool_call_constraint[0], "structural_tag")
                grammar = xgr.Grammar.from_structural_tag(
                    processed.tool_call_constraint[1]
                )
                self.assertTrue(
                    _is_grammar_accept_string(
                        grammar, "<tool_call>get_date</tool_call>"
                    )
                )

    def test_truncated_native_call_is_not_reported_as_complete(self):
        text = _call()
        for cutoff in (len("<tool_call>weather"), len(text) - 1):
            with self.subTest(cutoff=cutoff):
                truncated = text[:cutoff]
                self.assertFalse(_is_grammar_accept_string(self._grammar(), truncated))
                calls, _, reason = self._chat()._process_tool_calls(
                    truncated, self.tools, {"type": "length"}, tool_choice="required"
                )
                self.assertIsNone(calls)
                self.assertEqual(reason["type"], "length")

    def test_serving_required_and_named_use_native_parser(self):
        chat = self._chat()
        text = "<tool_call>get_date</tool_call>"
        for choice in (
            "required",
            ToolChoice(type="function", function={"name": "get_date"}),
        ):
            with self.subTest(choice=choice):
                result = chat._process_tool_calls(
                    text,
                    self.tools,
                    {"type": "stop"},
                    tool_choice=choice,
                )
                self.assertEqual(result[0][0].function.name, "get_date")
                self.assertEqual(result[0][0].function.arguments, "{}")
                self.assertEqual(result[2]["type"], "tool_calls")
                request = ChatCompletionRequest(
                    model="ling",
                    messages=[{"role": "user", "content": "date?"}],
                    tools=self.tools,
                    tool_choice=choice,
                    stream=True,
                )
                parsers, has_calls = {}, {}

                async def collect(parser_dict, request_obj, has_tool_calls):
                    return [
                        chunk
                        async for chunk in chat._process_tool_call_stream(
                            index=0,
                            delta=text,
                            parser_dict=parser_dict,
                            content={"meta_info": {"id": "test"}},
                            request=request_obj,
                            has_tool_calls=has_tool_calls,
                        )
                    ]

                chunks = asyncio.run(collect(parsers, request, has_calls))
                self.assertIsInstance(parsers[0], FunctionCallParser)
                self.assertTrue(has_calls[0])
                calls = [
                    json.loads(chunk.removeprefix("data: "))["choices"][0]["delta"][
                        "tool_calls"
                    ][0]
                    for chunk in chunks
                ]
                self.assertEqual(calls[0]["function"]["name"], "get_date")
                self.assertEqual(
                    "".join(call["function"]["arguments"] for call in calls), "{}"
                )


if __name__ == "__main__":
    unittest.main()
