"""Schema-aware Kimi tool-name inference without model execution."""

import copy
import json
import unittest
from unittest.mock import patch

from jsonschema import Draft202012Validator

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.kimik2_detector import KimiK2Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=5, suite="stage-b-test-cpu-intel")


def _tool(name, schema):
    return Tool(function=Function(name=name, parameters=schema))


class TestKimiK2Inference(CustomTestCase):
    def setUp(self):
        self.detector = KimiK2Detector()
        self.path_schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        self.weather = _tool(
            "weather",
            {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )

    def _stream(self, chunks, tools=None):
        parser = FunctionCallParser(
            [self.weather, _tool("read", self.path_schema)] if tools is None else tools,
            tool_call_parser="kimi_k2",
        )
        results = [parser.parse_stream_chunk(chunk) for chunk in chunks]
        results.append(parser.parse_stream_end())
        self.assertEqual(parser.parse_stream_end(), ("", []))
        return "".join(text for text, _ in results), [
            call for _, calls in results for call in calls
        ]

    def test_fragmented_call_end_is_not_an_argument(self):
        end = "<|tool_call_end|>"
        for identifier, tools in (
            ("functions.weather:0", [self.weather, _tool("read", self.path_schema)]),
            ("call_9", [self.weather]),
        ):
            for ending in ([end[:-3], end[-3:]], list(end)):
                with self.subTest(identifier=identifier, ending=ending):
                    chunks = [
                        "<|tool_calls_section_begin|><|tool_call_begin|>"
                        + identifier
                        + '<|tool_call_argument_begin|>{"city":"Paris"}',
                        *ending,
                        "<|tool_calls_section_end|>",
                    ]
                    normal, calls = self._stream(chunks, tools)
                    self.assertEqual(normal, "")
                    self.assertEqual(
                        [item.name for item in calls if item.name], ["weather"]
                    )
                    self.assertEqual(
                        "".join(item.parameters for item in calls), '{"city":"Paris"}'
                    )

    def test_literal_angle_bracket_in_arguments_survives(self):
        normal, calls = self._stream(
            [
                "<|tool_calls_section_begin|><|tool_call_begin|>functions.weather:0"
                '<|tool_call_argument_begin|>{"city":"<',
                '"}',
                "<|tool_call_end|><|tool_calls_section_end|>",
            ]
        )
        self.assertEqual(normal, "")
        self.assertEqual(
            json.loads("".join(item.parameters for item in calls)), {"city": "<"}
        )

    def test_fragmented_section_end_is_not_normal_text(self):
        end = "<|tool_calls_section_end|>"
        call = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>550e8400-e29b-41d4-a716-446655440000"
            '<|tool_call_argument_begin|>{"city":"Paris"}<|tool_call_end|>'
        )
        for ending in ([end], [end[:-1], end[-1:]], list(end)):
            with self.subTest(ending=ending):
                normal, calls = self._stream([call, *ending])
                self.assertEqual(normal, "")
                self.assertEqual(
                    [item.name for item in calls if item.name], ["weather"]
                )
                self.assertEqual(
                    json.loads("".join(item.parameters for item in calls)),
                    {"city": "Paris"},
                )

    def test_fragmented_call_begin_after_rejected_call(self):
        chunks = [
            "<|tool_calls_section_begin|>",
            '<|tool_call_begin|>call_3<|tool_call_argument_begin|>{"unknown":"x"}<|tool_call_end|>',
            "<",
            '|tool_call_begin|>call_4<|tool_call_argument_begin|>{"city":"Paris"}<|tool_call_end|>',
            "<|tool_calls_section_end|>",
        ]
        normal, calls = self._stream(chunks)
        self.assertEqual(normal, "")
        self.assertEqual([item.name for item in calls if item.name], ["weather"])
        self.assertEqual({item.tool_index for item in calls}, {0})

    def test_finish_releases_unmatched_normal_text_prefix(self):
        for text in ("Compare 1 <", "Show <|tool_call_beg"):
            with self.subTest(text=text):
                normal, calls = self._stream([text])
                self.assertEqual(normal, text)
                self.assertEqual(calls, [])

    def test_finish_does_not_release_incomplete_call_markup(self):
        normal, calls = self._stream(
            [
                "Before the call. <|tool_calls_section_begin|>",
                '<|tool_call_begin|>call_3<|tool_call_argument_begin|>{"city":',
            ]
        )
        self.assertEqual(normal, "Before the call. ")
        self.assertEqual(calls, [])

    def test_evaluated_properties_preserve_unique_and_ambiguous_matches(self):
        schemas = {
            "local_ref": {"$defs": {"args": self.path_schema}, "$ref": "#/$defs/args"},
            "pattern": {
                "type": "object",
                "patternProperties": {"^path$": {"type": "string"}},
                "additionalProperties": False,
            },
            "composite": {
                "properties": {"tag": {"type": "string"}},
                "allOf": [self.path_schema],
            },
            "explicit_extra_branch": {
                "allOf": [
                    {"type": "object", "additionalProperties": {"type": "string"}}
                ]
            },
            "explicit_unevaluated": {"type": "object", "unevaluatedProperties": True},
        }
        for label, schema in schemas.items():
            with self.subTest(schema=label):
                read = _tool("read", schema)
                original = copy.deepcopy(read.function.parameters)
                for tools in ([read, self.weather], [self.weather, read]):
                    self.assertEqual(
                        self.detector._resolve_function_name(
                            "call_3", tools, '{"path":"/x"}'
                        ),
                        "read",
                    )
                delete = _tool("delete", self.path_schema)
                for tools in ([read, delete], [delete, read]):
                    self.assertIsNone(
                        self.detector._resolve_function_name(
                            "call_3", tools, '{"path":"/x"}'
                        )
                    )
                self.assertEqual(read.function.parameters, original)

    def test_validation_error_cannot_make_another_candidate_unique(self):
        broken = _tool("read", {"$ref": "#/$defs/missing", **self.path_schema})
        Draft202012Validator.check_schema(broken.function.parameters)
        delete = _tool("delete", self.path_schema)
        for tools in ([broken, delete], [delete, broken]):
            with self.subTest(first=tools[0].function.name):
                with self.assertLogs(
                    "sglang.srt.function_call.kimik2_detector", level="WARNING"
                ) as logs:
                    self.assertIsNone(
                        self.detector._resolve_function_name(
                            "call_3", tools, '{"path":"/x"}'
                        )
                    )
                self.assertIn("tool 'read'", "\n".join(logs.output))

    def test_external_reference_does_not_fetch(self):
        remote = _tool("read", {"$ref": "https://example.invalid/tool.json"})
        tools = [remote, _tool("delete", self.path_schema)]
        with patch(
            "urllib.request.urlopen", side_effect=AssertionError("network access")
        ) as opener:
            self.assertIsNone(
                self.detector._resolve_function_name("call_3", tools, '{"path":"/x"}')
            )
        opener.assert_not_called()

    def test_type_aliases_are_normalized_without_mutating_tools(self):
        schema = copy.deepcopy(self.path_schema)
        schema["properties"]["path"]["type"] = "varchar"
        read = _tool("read", schema)
        tools = [read, self.weather]
        for function_id in ("3", "call_3"):
            with self.subTest(function_id=function_id):
                self.assertEqual(
                    self.detector._resolve_function_name(
                        function_id, tools, '{"path":"/x"}'
                    ),
                    "read",
                )
        self.assertEqual(
            read.function.parameters["properties"]["path"]["type"], "varchar"
        )

    def test_invalid_values_are_not_identity_evidence_for_nameless_calls(self):
        tools = [_tool("read", self.path_schema), self.weather]
        for function_id in ("3", "call_3"):
            with self.subTest(function_id=function_id):
                self.assertIsNone(
                    self.detector._resolve_function_name(
                        function_id, tools, '{"path":5}'
                    )
                )
        # An explicit ID already establishes identity; argument checking stays
        # with the client on the existing standard-ID path.
        self.assertEqual(
            self.detector._resolve_function_name(
                "functions.read:0", tools, '{"path":5}'
            ),
            "read",
        )

    def test_unresolvable_call_is_skipped_without_losing_next_explicit_call(self):
        broken = _tool("delete", {"$ref": "#/$defs/missing", **self.path_schema})
        tools = [broken, self.weather]
        # Kimi special tokens are atomic; split only the ordinary JSON text.
        chunks = [
            "<|tool_calls_section_begin|>",
            '<|tool_call_begin|>call_3<|tool_call_argument_begin|>{"path":',
            '"/x"}',
            "<|tool_call_end|>",
            '<|tool_call_begin|>functions.weather:4<|tool_call_argument_begin|>{"city":',
            '"Paris"}',
            "<|tool_call_end|><|tool_calls_section_end|>",
        ]
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                detector = KimiK2Detector()
                if streaming:
                    results = [
                        detector.parse_streaming_increment(chunk, tools)
                        for chunk in chunks
                    ]
                else:
                    results = [detector.detect_and_parse("".join(chunks), tools)]
                calls = [call for result in results for call in result.calls]
                self.assertEqual(
                    [call.name for call in calls if call.name], ["weather"]
                )
                self.assertEqual(
                    json.loads("".join(call.parameters for call in calls)),
                    {"city": "Paris"},
                )
                self.assertEqual("".join(result.normal_text for result in results), "")
                self.assertEqual({call.tool_index for call in calls}, {0})


if __name__ == "__main__":
    unittest.main()
