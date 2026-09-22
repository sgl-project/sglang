import copy
import unittest
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.kimik2_detector import KimiK2Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _schema(**types):
    return {
        "type": "object",
        "properties": {key: {"type": value} for key, value in types.items()},
        "required": list(types),
    }


def _tool(name, schema):
    return Tool(function=Function(name=name, parameters=schema))


def _call(identifier, arguments):
    return (
        f"<|tool_call_begin|>{identifier}<|tool_call_argument_begin|>"
        f"{arguments}<|tool_call_end|>"
    )


def _section(calls):
    return f"<|tool_calls_section_begin|>{calls}<|tool_calls_section_end|>"


class TestKimiK2Inference(CustomTestCase):
    def setUp(self):
        self.path_schema = _schema(path="string")
        self.read = _tool(name="read", schema=self.path_schema)
        self.weather = _tool(name="weather", schema=_schema(city="string"))
        self.tools = [self.read, self.weather]

    def _parse(self, chunks, *, streaming, tools=None):
        parser = FunctionCallParser(
            tools=self.tools if tools is None else tools, tool_call_parser="kimi_k2"
        )
        if streaming:
            results = [parser.parse_stream_chunk(chunk) for chunk in chunks]
            results.append(parser.parse_stream_end())
            self.assertEqual(parser.parse_stream_end(), ("", []))
        else:
            results = [parser.parse_non_stream("".join(chunks))]
        calls = {}
        for _, deltas in results:
            for delta in deltas:
                call = calls.setdefault(delta.tool_index, ["", ""])
                call[0] += delta.name or ""
                call[1] += delta.parameters
        self.assertEqual(list(calls), list(range(len(calls))))
        return "".join(text for text, _ in results), list(calls.values())

    def test_client_ids_and_fragmented_markers(self):
        """Client IDs must survive parsing without leaking marker bytes into either channel."""
        for identifier in (
            "3",
            "call_3",
            "call_0123456789abcdef",
            "toolu_01Example",
            "550e8400-e29b-41d4-a716-446655440000",
            "functions.weather:7",
        ):
            text = "Before. " + _section(
                _call(identifier=identifier, arguments='{"city":"Paris"}')
            )
            for streaming, chunks in (
                (False, [text]),
                (True, [text]),
                (True, list(text)),
            ):
                with self.subTest(
                    identifier=identifier, chunks=len(chunks), streaming=streaming
                ):
                    self.assertEqual(
                        self._parse(chunks, streaming=streaming),
                        ("Before. ", [["weather", '{"city":"Paris"}']]),
                    )

    def test_inference_waits_for_complete_arguments(self):
        """A partial object cannot choose a tool before later keys disambiguate it."""
        tools = [
            self.read,
            _tool(name="write", schema=_schema(path="string", content="string")),
        ]
        detector = KimiK2Detector()
        for chunk in (
            '<|tool_calls_section_begin|><|tool_call_begin|>call_3<|tool_call_argument_begin|>{"path":"/x"',
            ',"content":"text"}',
        ):
            self.assertEqual(detector.parse_streaming_increment(chunk, tools).calls, [])
        result = detector.parse_streaming_increment("<|tool_call_end|>", tools)
        self.assertEqual(
            [(c.name, c.parameters) for c in result.calls],
            [("write", '{"path":"/x","content":"text"}')],
        )

    def test_single_tool_streams_arguments_without_validation(self):
        """Single-tool calls stream early, preserving literal '<' but consuming a split end marker."""
        parser = FunctionCallParser(tools=[self.read], tool_call_parser="kimi_k2")
        text, calls = parser.parse_stream_chunk(
            '<|tool_calls_section_begin|><|tool_call_begin|>call_9<|tool_call_argument_begin|>{"city":"<'
        )
        self.assertEqual(text, "")
        self.assertEqual(
            [(c.name, c.parameters) for c in calls], [("read", '{"city":"')]
        )
        for chunk in ['"}', *list("<|tool_call_end|><|tool_calls_section_end|>")]:
            text, delta = parser.parse_stream_chunk(chunk)
            self.assertEqual(text, "")
            calls.extend(delta)
        self.assertEqual("".join(c.parameters for c in calls), '{"city":"<"}')
        self.assertEqual(parser.parse_stream_end(), ("", []))

    def test_rejected_call_does_not_hide_the_next_call(self):
        """Ambiguous and unevaluable calls cannot consume the following call or its local index."""
        for schema in (self.path_schema, {"$ref": "#/$defs/missing"}):
            tools = self.tools + [_tool(name="delete", schema=schema)]
            text = _section(
                _call(identifier="call_3", arguments='{"path":"/x"}')
                + _call(identifier="functions.weather:9", arguments='{"city":"Paris"}')
            )
            for streaming in (False, True):
                with self.subTest(schema=schema, streaming=streaming):
                    self.assertEqual(
                        self._parse(list(text), streaming=streaming, tools=tools),
                        ("", [["weather", '{"city":"Paris"}']]),
                    )

    def test_finish_preserves_text_but_discards_incomplete_calls(self):
        """EOF must neither lose ordinary prefixes nor expose incomplete call markup."""
        for text, expected in (
            ("Compare 1 <", "Compare 1 <"),
            ("Show <|tool_call_beg", "Show <|tool_call_beg"),
            (
                'Before. <|tool_calls_section_begin|><|tool_call_begin|>call_3<|tool_call_argument_begin|>{"city":',
                "Before. ",
            ),
        ):
            with self.subTest(text=text):
                self.assertEqual(self._parse([text], streaming=True), (expected, []))

    def test_inference_rejects_invalid_and_ambiguous_arguments(self):
        """Key overlap must not select a wrong tool, regardless of tool order."""
        tools = self.tools + [
            _tool(name="write", schema=_schema(path="string", content="string"))
        ]
        for args, expected in (
            ('{"path":"/x"}', "read"),
            ('{"path":"/x","content":"text"}', "write"),
            ('{"path":"/x","unknown":1}', None),
            ('{"content":"text"}', None),
            ('{"path":5}', None),
            ("[]", None),
        ):
            for ordered in (tools, tools[::-1]):
                with self.subTest(args=args, first=ordered[0].function.name):
                    self.assertEqual(
                        KimiK2Detector()._infer_tool_name(ordered, args), expected
                    )

    def test_schema_semantics_and_ambiguity(self):
        """References and composed schemas must not disappear as potential competing tools."""
        schemas = (
            {"$defs": {"args": self.path_schema}, "$ref": "#/$defs/args"},
            {
                "patternProperties": {"^path$": {"type": "string"}},
                "additionalProperties": False,
            },
            {"anyOf": [self.path_schema, _schema(city="string")]},
            {"allOf": [{"additionalProperties": {"type": "string"}}]},
            {"unevaluatedProperties": True},
            _schema(path="varchar"),
        )
        for schema in schemas:
            candidate = _tool(name="candidate", schema=schema)
            original = copy.deepcopy(candidate.function.parameters)
            for other, expected in ((self.weather, "candidate"), (self.read, None)):
                for tools in ([candidate, other], [other, candidate]):
                    with self.subTest(
                        schema=schema,
                        other=other.function.name,
                        first=tools[0].function.name,
                    ):
                        self.assertEqual(
                            KimiK2Detector()._infer_tool_name(tools, '{"path":"/x"}'),
                            expected,
                        )
            self.assertEqual(candidate.function.parameters, original)

    def test_unevaluable_schema_cannot_establish_uniqueness(self):
        """A failed reference must not select another tool or retrieve a remote schema."""
        for ref in ("#/$defs/missing", "https://example.invalid/tool.json"):
            broken = _tool(name="broken", schema={"$ref": ref})
            for tools in ([broken, self.read], [self.read, broken]):
                with self.subTest(ref=ref, first=tools[0].function.name):
                    with patch(
                        "urllib.request.urlopen", side_effect=AssertionError("network")
                    ) as opener:
                        with self.assertLogs(
                            "sglang.srt.function_call.kimik2_detector", level="WARNING"
                        ) as logs:
                            self.assertIsNone(
                                KimiK2Detector()._infer_tool_name(
                                    tools, '{"path":"/x"}'
                                )
                            )
                    opener.assert_not_called()
                    self.assertIn("tool 'broken'", "\n".join(logs.output))

    def test_schema_values_and_empty_arguments(self):
        """Inference must honor schema constraints rather than treating every declared key as a match."""
        for schema, args, expected in (
            (False, "{}", None),
            (True, "{}", "candidate"),
            ({}, '{"path":"/x"}', "read"),
            ({"anyOf": [self.path_schema, _schema(city="string")]}, "{}", None),
            (
                {"properties": {"action": {"const": "delete"}}, "required": ["action"]},
                '{"action":"read"}',
                None,
            ),
            ({"additionalProperties": {"type": "string"}}, '{"extra":5}', None),
        ):
            with self.subTest(schema=schema, args=args):
                tools = [_tool(name="candidate", schema=schema), self.read]
                self.assertEqual(
                    KimiK2Detector()._infer_tool_name(tools, args), expected
                )


if __name__ == "__main__":
    unittest.main()
