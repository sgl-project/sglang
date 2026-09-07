"""Unit tests for DeepSeekV41Detector (spaced DSML tags) -- no server, no model loading."""

import json
from unittest.mock import patch

from sglang.srt.entrypoints.openai import encoding_dsv41
from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.deepseekv41_detector import DeepSeekV41Detector
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

DSML = "｜DSML｜"
CHUNK_SIZES = [1, 2, 3, 5, 7, 11, 23, 1000]


def _wrapped(invoke: str) -> str:
    return f"<{DSML} calls>\n{invoke}\n</{DSML} calls>"


def _invoke(name: str, params: str = "") -> str:
    return f'<{DSML} invoke name="{name}">\n{params}\n</{DSML} invoke>'


def _param(name: str, is_string: str, value: str) -> str:
    return f'<{DSML} parameter name="{name}" string="{is_string}">{value}</{DSML} parameter>'


def _weather_call(city: str = "SF") -> str:
    return _wrapped(_invoke("get_weather", _param("city", "true", city)))


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


def _feed(detector, text, tools, chunk_size):
    """Returns (normal_text, calls) accumulated over fixed-size chunks."""
    normal, calls = "", []
    for i in range(0, len(text), chunk_size):
        result = detector.parse_streaming_increment(text[i : i + chunk_size], tools)
        normal += result.normal_text
        calls.extend(result.calls)
    return normal, calls


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


class TestDeepSeekV41Streaming(CustomTestCase):
    def setUp(self):
        self.tools = _tools()

    def test_registered_parser_name(self):
        parser = FunctionCallParser(self.tools, "deepseekv41")
        self.assertIsInstance(parser.detector, DeepSeekV41Detector)

    def test_preamble_in_same_delta_as_tool_call(self):
        text = "Let me check.\n" + _weather_call()
        normal, calls = _feed(DeepSeekV41Detector(), text, self.tools, len(text))

        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])
        self.assertEqual(
            normal, DeepSeekV41Detector().detect_and_parse(text, self.tools).normal_text
        )

    def test_preamble_before_bare_invoke_without_wrapper(self):
        text = "Checking.\n" + _invoke("get_weather", _param("city", "true", "SF"))
        normal, calls = _feed(DeepSeekV41Detector(), text, self.tools, len(text))

        self.assertIn("Checking.", normal)
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])

    def test_no_dsml_markers_leak_into_normal_text(self):
        text = "Prose.\n" + _weather_call()
        normal, _ = _feed(DeepSeekV41Detector(), text, self.tools, 4)

        self.assertNotIn(DSML, normal)

    def test_malformed_partial_json_falls_back_to_raw_value(self):
        result = DeepSeekV41Detector().parse_streaming_increment(
            f'<{DSML} calls>\n<{DSML} invoke name="get_weather">\n'
            f'<{DSML} parameter name="city" string="false">{{"a"',
            self.tools,
        )

        self.assertEqual([c.name for c in result.calls if c.name], ["get_weather"])

    def test_non_streaming_parses_every_tool_calls_section(self):
        result = DeepSeekV41Detector().detect_and_parse(
            f"{_weather_call('SF')}\n{_weather_call('NY')}", self.tools
        )

        self.assertEqual(len(result.calls), 2)

    def test_parse_error_neither_swallows_nor_duplicates(self):
        detector = DeepSeekV41Detector()

        with patch.object(
            DeepSeekV41Detector,
            "_parse_parameters_from_xml",
            side_effect=RuntimeError("boom"),
        ):
            first = detector.parse_streaming_increment(_weather_call(), self.tools)
            self.assertEqual(detector._buffer, "")
            second = detector.parse_streaming_increment(" tail", self.tools)

        self.assertIn("get_weather", first.normal_text)
        self.assertNotIn("get_weather", second.normal_text)
        self.assertEqual(first.calls, [])

    def test_unspaced_v4_tags_are_not_tool_calls(self):
        """The V4 grammar must fall through as plain text for a V4.1 detector."""
        v4_text = (
            _weather_call().replace(f"{DSML} ", DSML).replace(" calls", "tool_calls")
        )
        detector = DeepSeekV41Detector()
        self.assertFalse(detector.has_tool_call(v4_text))
        result = detector.detect_and_parse(v4_text, self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, v4_text)


class TestDeepSeekV41RoundTrip(CustomTestCase):
    """Encoder-rendered assistant tool calls parse back to the same arguments,
    in one shot and at every chunk size."""

    ARGUMENTS = {"query": "value", "limit": 2, "flags": [1, True, None]}

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
        # What the tool detector sees after the reasoning parser strips
        # `reason</think>`.
        self.after_think = self.completion.split("</think>", 1)[1]
        self.expected = [("lookup", self.ARGUMENTS)]

    def test_matches_reference_style_completion_parser(self):
        parsed = encoding_dsv41.parse_message_from_completion_text(
            self.completion, thinking_mode="thinking"
        )
        self.assertEqual(
            [
                (tc["function"]["name"], json.loads(tc["function"]["arguments"]))
                for tc in parsed["tool_calls"]
            ],
            self.expected,
        )

    def test_one_shot(self):
        result = DeepSeekV41Detector().detect_and_parse(self.after_think, self.tools)
        self.assertEqual(result.normal_text, "summary")
        self.assertEqual(
            [(c.name, json.loads(c.parameters)) for c in result.calls], self.expected
        )

    def test_streaming_at_every_chunk_size(self):
        for chunk_size in CHUNK_SIZES:
            with self.subTest(chunk_size=chunk_size):
                normal, calls = _feed(
                    DeepSeekV41Detector(), self.after_think, self.tools, chunk_size
                )
                # The blank line before the block is released or trimmed depending
                # on where the chunk boundary falls; the shared base behaves the
                # same for V4, so only the prose itself is pinned here.
                self.assertEqual(normal.strip(), "summary")
                self.assertEqual(_assemble(calls), self.expected)

    def test_string_parameter_shaped_like_json_stays_a_string(self):
        text = _wrapped(_invoke("lookup", _param("query", "true", '{"a": 1}')))
        for chunk_size in CHUNK_SIZES:
            with self.subTest(chunk_size=chunk_size):
                _, calls = _feed(DeepSeekV41Detector(), text, self.tools, chunk_size)
                self.assertEqual(_assemble(calls), [("lookup", {"query": '{"a": 1}'})])


class TestDeepSeekV41ConstrainedDecoding(CustomTestCase):
    def test_structure_info_uses_spaced_tags(self):
        info = DeepSeekV41Detector().structure_info()("get_weather")
        self.assertEqual(info.begin, f'<{DSML} invoke name="get_weather">')
        self.assertEqual(info.end, f"</{DSML} invoke>")
        self.assertEqual(info.trigger, f"<{DSML} invoke")

    def test_no_builtin_structural_tag(self):
        """xgrammar's builtin deepseek_v4 tag is the unspaced grammar; using it
        would force V4.1 into the wrong tool-call format."""
        detector = DeepSeekV41Detector()
        self.assertIsNone(detector.get_structural_tag_name())
        self.assertIsNone(detector.get_structural_tag(_tools(), "required"))


class TestDsmlTagLiterals(CustomTestCase):
    """The model formats dictate these strings; the shared base assembles them
    from tag names, so pin each family's literals."""

    def test_tokens_per_family(self):
        for detector, block, invoke, parameter in (
            (DeepSeekV32Detector(), "function_calls", "invoke", "parameter"),
            (DeepSeekV4Detector(), "tool_calls", "invoke", "parameter"),
            (DeepSeekV41Detector(), " calls", " invoke", " parameter"),
        ):
            with self.subTest(detector=type(detector).__name__):
                self.assertEqual(detector.bot_token, f"<{DSML}{block}>")
                self.assertEqual(detector.eot_token, f"</{DSML}{block}>")
                self.assertEqual(detector.invoke_start_token, f"<{DSML}{invoke}")
                self.assertEqual(detector.invoke_end_token, f"</{DSML}{invoke}>")
                self.assertEqual(
                    detector.function_calls_regex,
                    f"<{DSML}{block}>(.*?)</{DSML}{block}>",
                )
                self.assertEqual(
                    detector.parameter_regex,
                    f'<{DSML}{parameter}\\s+name="([^"]+)"\\s+string="([^"]+)"\\s*>'
                    f"(.*?)</{DSML}{parameter}>",
                )
                self.assertEqual(
                    detector.prefix_parameter_end_call, ["</", DSML, parameter]
                )
                self.assertEqual(
                    detector.prefix_invoke_end_call,
                    ["</", DSML, invoke[:-3], invoke[-3:]],
                )

    def test_v4_literals_unchanged(self):
        detector = DeepSeekV4Detector()
        self.assertEqual(detector.bot_token, "<｜DSML｜tool_calls>")
        self.assertEqual(detector.eot_token, "</｜DSML｜tool_calls>")
        self.assertEqual(detector.invoke_end_token, "</｜DSML｜invoke>")
        self.assertEqual(
            detector.invoke_regex,
            r'<｜DSML｜invoke\s+name="(?P<name>[^"]+)"\s*'
            r"(?:(?P<self_close>/>)"
            r"|>(?P<body>.*?)(?P<end>(?:</｜DSML｜invoke>|$)))",
        )
        self.assertEqual(
            detector.partial_parameter_regex,
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*)$',
        )
        self.assertEqual(
            detector.prefix_invoke_end_call, ["</", "｜DSML｜", "inv", "oke"]
        )
        self.assertEqual(detector.get_structural_tag_name(), "deepseek_v4")
        info = detector.structure_info()("x")
        self.assertEqual(
            (info.begin, info.end, info.trigger),
            ('<｜DSML｜invoke name="x">', "</｜DSML｜invoke>", "<｜DSML｜invoke"),
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
