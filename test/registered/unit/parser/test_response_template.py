"""Tests for checkpoint-driven response-template adapters."""

import json
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    Function,
    ResponsesRequest,
    Tool,
    ToolChoice,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.gemma4_detector import (
    Gemma4Detector as Gemma4ToolDetector,
)
from sglang.srt.parser.reasoning_parser import Gemma4Detector as Gemma4ReasoningDetector
from sglang.srt.parser.reasoning_parser import ReasoningParser
from sglang.srt.parser.response_template import (
    ResponseTemplateReasoningDetector,
    ResponseTemplateToolDetector,
)
from sglang.srt.parser.response_template_config import (
    resolve_detector_response_template,
    validate_response_template_for_serving,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Public checkpoint metadata shared by google/gemma-4-12B-it and
# google/gemma-4-26B-A4B-it.
GEMMA4_RESPONSE_TEMPLATE = {
    "defaults": {"role": "assistant"},
    "fields": {
        "content": {
            "close": ["<turn|>", "<|tool_response>", "<eos>"],
            "content": "text",
        },
        "thinking": {
            "close": "<channel|>",
            "content": "text",
            "open": "<|channel>thought\n",
        },
        "tool_calls": {
            "close": "<tool_call|>",
            "content": "json",
            "content_args": {
                "string_delims": [['<|"|>', '<|"|>']],
                "unquoted_keys": True,
            },
            "open_pattern": r"<\|tool_call>call:(?P<name>\w+)",
            "repeats": True,
            "transform": {
                "function": {
                    "arguments": "{content}",
                    "name": "{name}",
                },
                "type": "function",
            },
        },
    },
    "start_anchor": ["<|turn>model\n", "<tool_response|>"],
}

PREFIX = "<|turn>model\n"
THINKING = "<|channel>thought\nI should check the weather.<channel|>"
TOOL_CALL = (
    '<|tool_call>call:get_weather{location:<|"|>New York<|"|>,days:3,'
    "details:{metric:true},hours:[1,2]}<tool_call|>"
)


def _tool(name: str = "get_weather") -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            parameters={
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "days": {"type": "integer"},
                    "details": {"type": "object"},
                    "hours": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                },
            },
        ),
    )


def _call_values(calls):
    return [(call.name, json.loads(call.parameters)) for call in calls]


def _chunks(text, size):
    return [text[index : index + size] for index in range(0, len(text), size)]


def _collect_tool_stream(detector, chunks):
    normal_parts = []
    calls = {}
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, [_tool()])
        normal_parts.append(result.normal_text)
        for call in result.calls:
            entry = calls.setdefault(
                call.tool_index,
                {"name": "", "parameters": ""},
            )
            if call.name:
                entry["name"] = call.name
            if call.parameters:
                entry["parameters"] += call.parameters
    finished = detector.finish([_tool()])
    normal_parts.append(finished.normal_text)
    for call in finished.calls:
        entry = calls.setdefault(
            call.tool_index,
            {"name": "", "parameters": ""},
        )
        if call.name:
            entry["name"] = call.name
        if call.parameters:
            entry["parameters"] += call.parameters
    parsed_calls = [
        (call["name"], json.loads(call["parameters"]))
        for _, call in sorted(calls.items())
    ]
    return "".join(normal_parts), parsed_calls


class TestResponseTemplateLoading(unittest.TestCase):
    def test_loads_response_template_from_tokenizer_config(self):
        tokenizer = SimpleNamespace(
            init_kwargs={"response_template": GEMMA4_RESPONSE_TEMPLATE}
        )

        self.assertEqual(
            resolve_detector_response_template(tokenizer, None),
            GEMMA4_RESPONSE_TEMPLATE,
        )
        self.assertEqual(tokenizer.response_template, GEMMA4_RESPONSE_TEMPLATE)

    def test_rejects_unsupported_semantic_fields(self):
        template = {
            "start_anchor": "<assistant>",
            "fields": {
                "content": {"content": "text"},
                "citations": {
                    "open": "<citation>",
                    "close": "</citation>",
                    "content": "text",
                },
            },
        }

        with self.assertRaisesRegex(ValueError, "unsupported semantic fields"):
            validate_response_template_for_serving(template)
        with self.assertRaisesRegex(ValueError, "unsupported semantic fields"):
            ResponseTemplateReasoningDetector(response_template=template)

        template = {
            "defaults": {"role": "assistant", "metadata": {}},
            "start_anchor": "<assistant>",
            "fields": {"content": {"content": "text"}},
        }
        with self.assertRaisesRegex(ValueError, "unsupported semantic fields"):
            validate_response_template_for_serving(template)


class TestGemma4ResponseTemplateParity(unittest.TestCase):
    def test_non_streaming_without_reasoning_parity(self):
        text = "It will be sunny."
        existing = Gemma4ReasoningDetector().detect_and_parse(text)
        template = ResponseTemplateReasoningDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX,
        ).detect_and_parse(text)

        self.assertEqual(template.reasoning_text, existing.reasoning_text)
        self.assertEqual(template.normal_text, existing.normal_text)

    def test_non_streaming_reasoning_parity(self):
        text = THINKING + "It will be sunny."
        existing = Gemma4ReasoningDetector().detect_and_parse(text)
        template = ResponseTemplateReasoningDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX,
        ).detect_and_parse(text)

        self.assertEqual(template.reasoning_text, existing.reasoning_text)
        self.assertEqual(template.normal_text, existing.normal_text)

    def test_non_streaming_tool_call_parity(self):
        text = "Some text before " + TOOL_CALL
        existing = Gemma4ToolDetector().detect_and_parse(text, [_tool()])
        template = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX,
        ).detect_and_parse(text, [_tool()])

        self.assertEqual(template.normal_text, existing.normal_text)
        self.assertEqual(_call_values(template.calls), _call_values(existing.calls))

    def test_non_streaming_repeated_tool_call_parity(self):
        text = TOOL_CALL + TOOL_CALL
        existing = Gemma4ToolDetector().detect_and_parse(text, [_tool()])
        template = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX,
        ).detect_and_parse(text, [_tool()])

        self.assertEqual(template.normal_text, existing.normal_text)
        self.assertEqual(_call_values(template.calls), _call_values(existing.calls))

    def test_non_streaming_reasoning_and_tool_handoff_parity(self):
        text = THINKING + TOOL_CALL

        existing_reasoning = Gemma4ReasoningDetector().detect_and_parse(text)
        existing_tools = Gemma4ToolDetector().detect_and_parse(
            existing_reasoning.normal_text,
            [_tool()],
        )

        template_reasoning = ResponseTemplateReasoningDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX,
        ).detect_and_parse(text)
        template_tools = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX,
        ).detect_and_parse(template_reasoning.normal_text, [_tool()])

        self.assertEqual(
            template_reasoning.reasoning_text,
            existing_reasoning.reasoning_text,
        )
        self.assertEqual(template_tools.normal_text, existing_tools.normal_text)
        self.assertEqual(
            _call_values(template_tools.calls),
            _call_values(existing_tools.calls),
        )

    def test_malformed_tool_body_does_not_expose_reasoning(self):
        malformed = TOOL_CALL.replace("<tool_call|>", "unexpected<tool_call|>")
        parsed = ResponseTemplateReasoningDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX,
        ).detect_and_parse(THINKING + malformed)

        self.assertEqual(parsed.reasoning_text, "I should check the weather.")
        self.assertEqual(parsed.normal_text, malformed)

    def test_streaming_reasoning_parity(self):
        text = THINKING + "It will be sunny."
        for chunk_size in (1, 2, 3, 5, 7, 13, 29, len(text)):
            with self.subTest(chunk_size=chunk_size):
                existing = Gemma4ReasoningDetector()
                template = ResponseTemplateReasoningDetector(
                    response_template=GEMMA4_RESPONSE_TEMPLATE,
                    prefix=PREFIX,
                )
                existing_reasoning = []
                existing_normal = []
                template_reasoning = []
                template_normal = []
                for chunk in _chunks(text, chunk_size):
                    existing_result = existing.parse_streaming_increment(chunk)
                    existing_reasoning.append(existing_result.reasoning_text)
                    existing_normal.append(existing_result.normal_text)
                    template_result = template.parse_streaming_increment(chunk)
                    template_reasoning.append(template_result.reasoning_text)
                    template_normal.append(template_result.normal_text)

                existing_end = existing.finish()
                template_end = template.finish()
                existing_reasoning.append(existing_end.reasoning_text)
                existing_normal.append(existing_end.normal_text)
                template_reasoning.append(template_end.reasoning_text)
                template_normal.append(template_end.normal_text)

                self.assertEqual(
                    "".join(template_reasoning),
                    "".join(existing_reasoning),
                )
                self.assertEqual(
                    "".join(template_normal),
                    "".join(existing_normal),
                )

    def test_streaming_tool_call_parity(self):
        for chunk_size in (1, 2, 3, 5, 7, 13, 29, len(TOOL_CALL)):
            with self.subTest(chunk_size=chunk_size):
                existing = Gemma4ToolDetector()
                template = ResponseTemplateToolDetector(
                    response_template=GEMMA4_RESPONSE_TEMPLATE,
                    prefix=PREFIX,
                )
                chunks = _chunks(TOOL_CALL, chunk_size)

                self.assertEqual(
                    _collect_tool_stream(template, chunks),
                    _collect_tool_stream(existing, chunks),
                )

    def test_streaming_tool_name_timing_parity(self):
        opening = "<|tool_call>call:get_weather{"
        for detector in (
            Gemma4ToolDetector(),
            ResponseTemplateToolDetector(
                response_template=GEMMA4_RESPONSE_TEMPLATE,
                prefix=PREFIX,
            ),
        ):
            with self.subTest(detector=type(detector).__name__):
                parsed = detector.parse_streaming_increment(opening, [_tool()])

                self.assertEqual(parsed.normal_text, "")
                self.assertEqual(
                    [
                        (call.tool_index, call.name, call.parameters)
                        for call in parsed.calls
                    ],
                    [(0, "get_weather", "")],
                )


class TestResponseTemplateAdapters(unittest.TestCase):
    def test_parser_prefix_request_state_is_private(self):
        requests = (
            ChatCompletionRequest(messages=[]),
            ResponsesRequest(input="hello"),
        )

        for request in requests:
            with self.subTest(request=type(request).__name__):
                request._response_parser_prefix = PREFIX

                self.assertEqual(request._response_parser_prefix, PREFIX)
                self.assertNotIn("response_parser_prefix", request.model_dump())
                self.assertEqual(
                    request.model_copy()._response_parser_prefix,
                    PREFIX,
                )

    def test_response_template_backend_is_internal(self):
        self.assertNotIn("response_template", ReasoningParser.DetectorMap)
        self.assertNotIn(
            "response_template",
            FunctionCallParser.ToolCallParserEnum,
        )

    def test_internal_adapters_use_checkpoint_metadata(self):
        tokenizer = SimpleNamespace(response_template=GEMMA4_RESPONSE_TEMPLATE)
        reasoning = ReasoningParser(
            model_type="response_template",
            tokenizer=tokenizer,
            prefix=PREFIX,
        )
        tools = FunctionCallParser(
            tools=[_tool()],
            tool_call_parser="response_template",
            tokenizer=tokenizer,
            prefix=PREFIX,
        )

        reasoning_text, tool_wire = reasoning.parse_non_stream(THINKING + TOOL_CALL)
        normal_text, calls = tools.parse_non_stream(tool_wire)

        self.assertEqual(reasoning_text, "I should check the weather.")
        self.assertEqual(normal_text, "")
        self.assertEqual(
            _call_values(calls),
            [
                (
                    "get_weather",
                    {
                        "location": "New York",
                        "days": 3,
                        "details": {"metric": True},
                        "hours": [1, 2],
                    },
                )
            ],
        )

    def test_empty_thinking_prefill_routes_generated_content(self):
        detector = ResponseTemplateReasoningDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX + "<|channel>thought\n<channel|>",
        )

        parsed = detector.parse_streaming_increment("Hello<turn|>")
        finished = detector.finish()

        self.assertEqual(parsed.reasoning_text + finished.reasoning_text, "")
        self.assertEqual(parsed.normal_text + finished.normal_text, "Hello")

    def test_open_thinking_prefill_routes_reasoning_and_content(self):
        detector = ResponseTemplateReasoningDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX + "<|channel>thought\n",
        )

        parsed = detector.parse_streaming_increment(
            "Still thinking<channel|>Done<turn|>"
        )
        finished = detector.finish()

        self.assertEqual(
            parsed.reasoning_text + finished.reasoning_text,
            "Still thinking",
        )
        self.assertEqual(parsed.normal_text + finished.normal_text, "Done")

    def test_nonempty_prefix_content_is_not_replayed(self):
        reasoning = ResponseTemplateReasoningDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX + "<|channel>thought\nExisting reasoning",
        )
        content = ResponseTemplateReasoningDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX + "Existing answer",
        )
        non_streaming = ResponseTemplateReasoningDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX + "Existing answer",
        )

        reasoning_parsed = reasoning.parse_streaming_increment(
            " continued<channel|>Done<turn|>"
        )
        content_parsed = content.parse_streaming_increment(" continued<turn|>")
        non_streaming_parsed = non_streaming.detect_and_parse(" continued<turn|>")

        self.assertEqual(reasoning_parsed.reasoning_text, " continued")
        self.assertEqual(reasoning_parsed.normal_text, "Done")
        self.assertEqual(content_parsed.normal_text, " continued")
        self.assertEqual(non_streaming_parsed.normal_text, " continued")

    def test_checkpoint_json_null_is_parsed_as_null(self):
        detector = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX,
        )
        text = "<|tool_call>call:get_weather{value:null}<tool_call|>"

        parsed = detector.detect_and_parse(text, [_tool()])

        self.assertEqual(json.loads(parsed.calls[0].parameters), {"value": None})

    def test_content_between_tool_calls_is_preserved(self):
        detector = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX,
        )
        text = TOOL_CALL + " between calls " + TOOL_CALL

        parsed = detector.detect_and_parse(text, [_tool()])

        self.assertEqual(parsed.normal_text, " between calls ")
        self.assertEqual(len(parsed.calls), 2)

    def test_parser_preserves_special_tokens_without_changing_stop_trimming(self):
        request = SimpleNamespace(
            skip_special_tokens=True,
            no_stop_trim=False,
        )

        ResponseTemplateToolDetector.configure_request_for_parsing(request)

        self.assertFalse(request.skip_special_tokens)
        self.assertFalse(request.no_stop_trim)
        self.assertFalse(request.chat_template_kwargs["spaces_between_special_tokens"])

    def test_parser_config_disables_generated_special_token_spacing(self):
        request = ChatCompletionRequest(messages=[])

        ResponseTemplateToolDetector.configure_request_for_parsing(request)
        sampling_params = request.to_sampling_params([], {})

        self.assertFalse(sampling_params["skip_special_tokens"])
        self.assertFalse(sampling_params["spaces_between_special_tokens"])

    def test_reasoning_requires_explicit_enable_without_template_policy(self):
        detector = ResponseTemplateReasoningDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
        )

        self.assertEqual(detector.reasoning_default, "explicit_enable_thinking")

    def test_transform_each_tool_region_emits_all_calls(self):
        template = {
            "start_anchor": "<assistant>",
            "fields": {
                "content": {"content": "text"},
                "tool_calls": {
                    "open": "<calls>",
                    "close": "</calls>",
                    "content": "json",
                    "transform_each": True,
                    "transform": {
                        "type": "function",
                        "function": {
                            "name": "{name}",
                            "arguments": "{arguments}",
                        },
                    },
                },
            },
        }
        tools = [_tool(), _tool("get_forecast")]
        text = (
            '<calls>[{"name":"get_weather","arguments":{"location":"Paris"}},'
            '{"name":"get_forecast","arguments":{"days":3}}]</calls>'
        )
        detector = ResponseTemplateToolDetector(
            response_template=template,
            prefix="<assistant>",
        )

        parsed = detector.detect_and_parse(text, tools)

        self.assertEqual(parsed.normal_text, "")
        self.assertEqual(
            [(call.name, json.loads(call.parameters)) for call in parsed.calls],
            [
                ("get_weather", {"location": "Paris"}),
                ("get_forecast", {"days": 3}),
            ],
        )

        streaming = ResponseTemplateToolDetector(
            response_template=template,
            prefix="<assistant>",
        )
        opened = streaming.parse_streaming_increment("<calls>", tools)
        closed = streaming.parse_streaming_increment(
            text.removeprefix("<calls>"),
            tools,
        )
        self.assertEqual(opened.calls, [])
        self.assertEqual(
            [(call.name, json.loads(call.parameters)) for call in closed.calls],
            [
                ("get_weather", {"location": "Paris"}),
                ("get_forecast", {"days": 3}),
            ],
        )

    def test_malformed_call_is_preserved_as_content(self):
        detector = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
        )
        malformed = TOOL_CALL.replace("<tool_call|>", "unexpected<tool_call|>")

        result = detector.detect_and_parse(malformed, [_tool()])

        self.assertEqual(result.normal_text, malformed)
        self.assertEqual(result.calls, [])

    def test_streaming_malformed_call_restores_input_before_emission(self):
        detector = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
        )
        malformed = TOOL_CALL.replace("<tool_call|>", "unexpected<tool_call|>")

        parsed = detector.parse_streaming_increment(malformed, [_tool()])

        self.assertEqual(parsed.normal_text, malformed)
        self.assertEqual(parsed.calls, [])

    def test_streaming_malformed_call_does_not_roll_back_emitted_name(self):
        detector = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
        )
        opening, body = TOOL_CALL.split("{", 1)

        opened = detector.parse_streaming_increment(opening + "{", [_tool()])
        failed = detector.parse_streaming_increment(
            body.replace("<tool_call|>", "unexpected<tool_call|>"),
            [_tool()],
        )

        self.assertEqual(
            [(call.name, call.parameters) for call in opened.calls],
            [("get_weather", "")],
        )
        self.assertEqual(failed.normal_text, "")
        self.assertEqual(
            [(call.tool_index, call.name, call.parameters) for call in failed.calls],
            [
                (
                    0,
                    None,
                    '{location:<|"|>New York<|"|>,days:3,'
                    "details:{metric:true},hours:[1,2]}unexpected<tool_call|>",
                )
            ],
        )
        self.assertTrue(detector.has_incomplete_tool_call)
        self.assertEqual(
            detector.prev_tool_call_arr[0]["arguments"],
            failed.calls[0].parameters,
        )
        self.assertEqual(
            detector.streamed_args_for_tool[0],
            failed.calls[0].parameters,
        )

    def test_streaming_malformed_call_keeps_later_bytes_on_same_index(self):
        detector = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
        )
        opening, body = TOOL_CALL.split("{", 1)

        detector.parse_streaming_increment(opening + "{", [_tool()])
        failed = detector.parse_streaming_increment(
            body.replace("<tool_call|>", "unexpected<tool_call|>"),
            [_tool()],
        )
        later = detector.parse_streaming_increment(" trailing bytes", [_tool()])

        self.assertEqual(failed.calls[0].tool_index, 0)
        self.assertEqual(later.calls[0].tool_index, 0)
        self.assertIsNone(failed.calls[0].name)
        self.assertIsNone(later.calls[0].name)
        self.assertEqual(later.calls[0].parameters, " trailing bytes")

    def test_streaming_waits_when_tool_name_depends_on_content(self):
        template = {
            "start_anchor": "<assistant>",
            "fields": {
                "content": {"content": "text"},
                "tool_calls": {
                    "open": "<call>",
                    "close": "</call>",
                    "content": "json",
                    "repeats": True,
                    "transform": {
                        "type": "function",
                        "function": {
                            "name": "{content.name}",
                            "arguments": "{content.arguments}",
                        },
                    },
                },
            },
        }
        detector = ResponseTemplateToolDetector(
            response_template=template,
            prefix="<assistant>",
        )

        opened = detector.parse_streaming_increment("<call>", [_tool()])
        closed = detector.parse_streaming_increment(
            '{"name":"get_weather","arguments":{"location":"Paris"}}</call>',
            [_tool()],
        )

        self.assertEqual(opened.calls, [])
        self.assertEqual(
            [(call.name, json.loads(call.parameters)) for call in closed.calls],
            [("get_weather", {"location": "Paris"})],
        )

    def test_streaming_failure_does_not_repeat_emitted_content(self):
        detector = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
        )
        malformed = TOOL_CALL.replace("<tool_call|>", "unexpected<tool_call|>")

        content = detector.parse_streaming_increment("hello", [_tool()])
        failed = detector.parse_streaming_increment(malformed, [_tool()])

        self.assertEqual(content.normal_text, "hello")
        self.assertEqual(failed.normal_text, malformed)

    def test_prefilled_malformed_call_does_not_replay_prefix(self):
        opening, body = TOOL_CALL.split("{", 1)
        malformed_body = "{" + body.replace(
            "<tool_call|>",
            "unexpected<tool_call|>",
        )
        detector = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX + opening,
        )

        parsed = detector.parse_streaming_increment(malformed_body, [_tool()])

        self.assertEqual(parsed.normal_text, malformed_body)
        self.assertEqual(parsed.calls, [])

    def test_non_streaming_call_opened_in_prefix_is_parsed(self):
        opening, body = TOOL_CALL.split("{", 1)
        detector = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX + opening,
        )

        parsed = detector.detect_and_parse("{" + body, [_tool()])

        self.assertEqual(parsed.normal_text, "")
        self.assertEqual(
            _call_values(parsed.calls),
            [
                (
                    "get_weather",
                    {
                        "location": "New York",
                        "days": 3,
                        "details": {"metric": True},
                        "hours": [1, 2],
                    },
                )
            ],
        )

    def test_streaming_call_opened_in_prefix_emits_name_then_arguments(self):
        opening, body = TOOL_CALL.split("{", 1)
        detector = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
            prefix=PREFIX + opening,
        )

        parsed = detector.parse_streaming_increment("{" + body, [_tool()])

        self.assertEqual(
            [(call.tool_index, call.name) for call in parsed.calls],
            [(0, "get_weather"), (0, None)],
        )
        self.assertEqual(
            json.loads(parsed.calls[1].parameters),
            {
                "location": "New York",
                "days": 3,
                "details": {"metric": True},
                "hours": [1, 2],
            },
        )

    def test_complete_call_without_closing_delimiter_finalizes(self):
        without_close = TOOL_CALL.removesuffix("<tool_call|>")
        non_streaming = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
        ).detect_and_parse(without_close, [_tool()])
        detector = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
        )

        streamed = detector.parse_streaming_increment(without_close, [_tool()])
        finished = detector.finish([_tool()])

        self.assertEqual(
            _call_values(non_streaming.calls),
            [
                (
                    "get_weather",
                    {
                        "location": "New York",
                        "days": 3,
                        "details": {"metric": True},
                        "hours": [1, 2],
                    },
                )
            ],
        )
        self.assertEqual(
            [(call.name, call.parameters) for call in streamed.calls],
            [("get_weather", "")],
        )
        self.assertEqual(
            [(call.name, json.loads(call.parameters)) for call in finished.calls],
            [
                (
                    None,
                    {
                        "location": "New York",
                        "days": 3,
                        "details": {"metric": True},
                        "hours": [1, 2],
                    },
                )
            ],
        )

    def test_truncated_call_is_preserved_at_stream_end(self):
        detector = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
        )
        truncated = TOOL_CALL.removesuffix("<tool_call|>") + "<tool_"

        streamed = detector.parse_streaming_increment(truncated, [_tool()])
        finished = detector.finish([_tool()])

        self.assertEqual(
            [(call.name, call.parameters) for call in streamed.calls],
            [("get_weather", "")],
        )
        self.assertEqual(streamed.normal_text + finished.normal_text, "")
        self.assertEqual(
            [(call.tool_index, call.name, call.parameters) for call in finished.calls],
            [
                (
                    0,
                    None,
                    '{location:<|"|>New York<|"|>,days:3,'
                    "details:{metric:true},hours:[1,2]}<tool_",
                )
            ],
        )
        self.assertTrue(detector.has_incomplete_tool_call)

    def test_incomplete_state_tracks_only_truncated_call_index(self):
        detector = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
        )
        opening, body = TOOL_CALL.split("{", 1)

        detector.parse_streaming_increment(TOOL_CALL, [_tool()])
        detector.parse_streaming_increment(opening + "{" + body[:10], [_tool()])
        detector.finish([_tool()])

        self.assertEqual(detector.incomplete_tool_call_indices, {1})

    def test_unknown_tool_is_preserved_when_forwarding_is_disabled(self):
        detector = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
        )
        unknown = TOOL_CALL.replace("get_weather", "unknown")

        result = detector.detect_and_parse(unknown, [_tool()])

        self.assertEqual(result.normal_text, unknown)
        self.assertEqual(result.calls, [])

    def test_streaming_unknown_tool_is_not_emitted_early(self):
        detector = ResponseTemplateToolDetector(
            response_template=GEMMA4_RESPONSE_TEMPLATE,
        )
        unknown = TOOL_CALL.replace("get_weather", "unknown")
        opening, body = unknown.split("{", 1)

        opened = detector.parse_streaming_increment(opening + "{", [_tool()])
        closed = detector.parse_streaming_increment(body, [_tool()])

        self.assertEqual(opened.calls, [])
        self.assertEqual(closed.normal_text, unknown)
        self.assertEqual(closed.calls, [])

    def test_strict_tools_require_native_constraint_support(self):
        tokenizer = SimpleNamespace(response_template=GEMMA4_RESPONSE_TEMPLATE)
        tool = _tool()
        tool.function.strict = True
        parser = FunctionCallParser(
            tools=[tool],
            tool_call_parser="response_template",
            tokenizer=tokenizer,
        )

        with self.assertRaisesRegex(ValueError, "does not support strict"):
            parser.get_structure_constraint("auto")

    def test_required_and_named_tools_use_generic_json_schema(self):
        tokenizer = SimpleNamespace(response_template=GEMMA4_RESPONSE_TEMPLATE)
        parser = FunctionCallParser(
            tools=[_tool()],
            tool_call_parser="response_template",
            tokenizer=tokenizer,
        )

        for choice in (
            "required",
            ToolChoice(function={"name": "get_weather"}),
        ):
            with self.subTest(choice=choice):
                constraint = parser.get_structure_constraint(choice)
                self.assertEqual(constraint[0], "json_schema")

        strict_tool = _tool()
        strict_tool.function.strict = True
        strict_parser = FunctionCallParser(
            tools=[strict_tool],
            tool_call_parser="response_template",
            tokenizer=tokenizer,
        )
        self.assertEqual(
            strict_parser.get_structure_constraint("required")[0],
            "json_schema",
        )

    def test_parallel_false_auto_is_rejected_without_native_constraint(self):
        tokenizer = SimpleNamespace(response_template=GEMMA4_RESPONSE_TEMPLATE)
        parser = FunctionCallParser(
            tools=[_tool()],
            tool_call_parser="response_template",
            tokenizer=tokenizer,
        )

        with self.assertRaisesRegex(ValueError, "parallel_tool_calls=False"):
            parser.get_structure_constraint("auto", parallel_tool_calls=False)


if __name__ == "__main__":
    unittest.main()
