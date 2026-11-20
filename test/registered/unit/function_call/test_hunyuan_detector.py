"""Unit tests for HunyuanDetector - no server, no model loading."""

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
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.hunyuan_detector import HunyuanDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


def _make_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="get_current_date",
                description="Get the current date",
                parameters={},
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get weather information",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "date": {"type": "string", "description": "Date"},
                    },
                    "required": ["city"],
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="search",
                description="Search the web",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        },
                        "count": {
                            "type": "integer",
                            "description": "Number of results",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="calculate",
                description="Calculate expression",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                        "precision": {"type": "number"},
                        "verbose": {"type": "boolean"},
                    },
                },
            ),
        ),
    ]


class _Hy4Tokenizer:
    def get_vocab(self):
        return {
            "<tool_calls:opensource>": 1,
            "<tool_call:opensource>": 2,
            "<arg_key:opensource>": 3,
            "<arg_value:opensource>": 4,
            "<think:opensource>": 5,
        }


class TestHunyuanDetectorHasToolCall(CustomTestCase):
    def setUp(self):
        self.detector = HunyuanDetector()

    def test_has_tool_call_true(self):
        text = (
            "<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>"
        )
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        self.assertFalse(
            self.detector.has_tool_call("The weather in Beijing is sunny today.")
        )

    def test_has_tool_call_partial_tag(self):
        self.assertTrue(self.detector.has_tool_call("<tool_call>"))
        self.assertFalse(self.detector.has_tool_call("<tool_call"))

    def test_has_tool_call_with_surrounding_text(self):
        self.assertTrue(
            self.detector.has_tool_call("text before <tool_calls> text after")
        )


class TestHunyuanDetectorDetectAndParse(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()
        self.detector = HunyuanDetector()

    def test_no_tool_call(self):
        text = "This is a plain response."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_hy4_format_without_tool_separator(self):
        detector = HunyuanDetector(_Hy4Tokenizer())
        text = (
            "<tool_calls:opensource>"
            "<tool_call:opensource>get_weather"
            "<arg_key:opensource>city</arg_key:opensource>"
            "<arg_value:opensource>Beijing</arg_value:opensource>"
            "</tool_call:opensource></tool_calls:opensource>"
        )

        result = detector.detect_and_parse(text, self.tools)

        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Beijing"})

    def test_zero_arg_inline(self):
        text = (
            "<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_current_date")
        self.assertEqual(json.loads(result.calls[0].parameters), {})

    def test_zero_arg_newline(self):
        text = (
            "<tool_calls>\n"
            "<tool_call>get_current_date<tool_sep>\n"
            "</tool_call>\n"
            "</tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_current_date")

    def test_single_string_arg(self):
        text = (
            "<tool_calls><tool_call>get_weather<tool_sep>"
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
            "</tool_call></tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args, {"city": "Beijing"})

    def test_multiple_args_same_line(self):
        text = (
            "<tool_calls><tool_call>get_weather<tool_sep>"
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
            "<arg_key>date</arg_key><arg_value>2026-03-30</arg_value>"
            "</tool_call></tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        self.assertEqual(args["date"], "2026-03-30")

    def test_args_with_newlines(self):
        text = (
            "<tool_calls>\n"
            "<tool_call>get_weather<tool_sep>\n"
            "<arg_key>city</arg_key>\n"
            "<arg_value>Beijing</arg_value>\n"
            "<arg_key>date</arg_key>\n"
            "<arg_value>2026-03-30</arg_value>\n"
            "</tool_call>\n"
            "</tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Beijing")
        self.assertEqual(args["date"], "2026-03-30")

    def test_content_before_tool_call(self):
        text = (
            "Checking."
            "<tool_calls>\n"
            "<tool_call>get_current_date<tool_sep>\n"
            "</tool_call>\n"
            "</tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.normal_text, "Checking.")

    def test_multiple_tool_calls(self):
        text = (
            "<tool_calls>\n"
            "<tool_call>get_weather<tool_sep>\n"
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n"
            "</tool_call>\n"
            "<tool_call>get_weather<tool_sep>\n"
            "<arg_key>city</arg_key>\n<arg_value>Hangzhou</arg_value>\n"
            "</tool_call>\n"
            "</tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(json.loads(result.calls[0].parameters)["city"], "Beijing")
        self.assertEqual(json.loads(result.calls[1].parameters)["city"], "Hangzhou")

    def test_empty_content_returns_empty_normal_text(self):
        text = "<tool_calls>\n<tool_call>get_current_date<tool_sep>\n</tool_call>\n</tool_calls>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "")

    def test_unknown_tool_skipped(self):
        text = (
            "<tool_calls><tool_call>nonexistent_func<tool_sep>"
            "<arg_key>x</arg_key><arg_value>1</arg_value>"
            "</tool_call></tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_mixed_known_and_unknown_tools(self):
        """Known tools should be parsed, unknown ones skipped."""
        text = (
            "<tool_calls>"
            "<tool_call>get_current_date<tool_sep></tool_call>"
            "<tool_call>nonexistent<tool_sep></tool_call>"
            "<tool_call>search<tool_sep>"
            "<arg_key>query</arg_key><arg_value>test</arg_value>"
            "</tool_call>"
            "</tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_current_date")
        self.assertEqual(result.calls[1].name, "search")

    def test_three_parallel_tool_calls(self):
        text = (
            "<tool_calls>"
            "<tool_call>get_weather<tool_sep>"
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
            "</tool_call>"
            "<tool_call>get_weather<tool_sep>"
            "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>"
            "</tool_call>"
            "<tool_call>get_current_date<tool_sep></tool_call>"
            "</tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 3)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[1].name, "get_weather")
        self.assertEqual(result.calls[2].name, "get_current_date")
        # tool_index maps to position in tools list
        self.assertEqual(result.calls[0].tool_index, 1)  # get_weather is index 1
        self.assertEqual(result.calls[2].tool_index, 0)  # get_current_date is index 0


class TestHunyuanDetectorArgDeserialization(CustomTestCase):
    """Test type-aware argument deserialization."""

    def setUp(self):
        self.tools = _make_tools()
        self.detector = HunyuanDetector()

    def test_integer_arg(self):
        text = (
            "<tool_calls><tool_call>search<tool_sep>"
            "<arg_key>query</arg_key><arg_value>restaurants</arg_value>"
            "<arg_key>count</arg_key><arg_value>5</arg_value>"
            "</tool_call></tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["query"], "restaurants")
        self.assertEqual(args["count"], 5)
        self.assertIsInstance(args["count"], int)

    def test_float_arg(self):
        text = (
            "<tool_calls><tool_call>calculate<tool_sep>"
            "<arg_key>expression</arg_key><arg_value>1+1</arg_value>"
            "<arg_key>precision</arg_key><arg_value>0.01</arg_value>"
            "</tool_call></tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["expression"], "1+1")
        self.assertAlmostEqual(args["precision"], 0.01)

    def test_boolean_arg(self):
        text = (
            "<tool_calls><tool_call>calculate<tool_sep>"
            "<arg_key>expression</arg_key><arg_value>2+2</arg_value>"
            "<arg_key>verbose</arg_key><arg_value>true</arg_value>"
            "</tool_call></tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        args = json.loads(result.calls[0].parameters)
        self.assertIs(args["verbose"], True)

    def test_top_level_composed_schema_args(self):
        cases = (
            ("anyOf", {"type": "integer"}, "7", 7),
            ("oneOf", {"type": "boolean"}, "true", True),
            ("allOf", {"type": "array", "items": {"type": "integer"}}, "[1,2]", [1, 2]),
        )
        for keyword, arg_schema, raw_value, expected in cases:
            with self.subTest(keyword=keyword):
                function_name = f"composed_{keyword}"
                tools = [
                    Tool(
                        type="function",
                        function=Function(
                            name=function_name,
                            description="Composed schema",
                            parameters={
                                keyword: [
                                    {
                                        "type": "object",
                                        "properties": {"value": arg_schema},
                                    }
                                ]
                            },
                        ),
                    )
                ]
                text = (
                    f"<tool_calls><tool_call>{function_name}<tool_sep>"
                    f"<arg_key>value</arg_key><arg_value>{raw_value}</arg_value>"
                    "</tool_call></tool_calls>"
                )

                result = self.detector.detect_and_parse(text, tools)
                args = json.loads(result.calls[0].parameters)

                self.assertEqual(args, {"value": expected})

    def test_string_arg_not_deserialized(self):
        """String-typed args should stay as strings even if they look like JSON."""
        text = (
            "<tool_calls><tool_call>search<tool_sep>"
            '<arg_key>query</arg_key><arg_value>{"key": "value"}</arg_value>'
            "</tool_call></tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["query"], '{"key": "value"}')
        self.assertIsInstance(args["query"], str)

    def test_non_json_value_stays_string(self):
        """Non-JSON-parseable values for non-string types should fall back to string."""
        text = (
            "<tool_calls><tool_call>search<tool_sep>"
            "<arg_key>query</arg_key><arg_value>hello world</arg_value>"
            "<arg_key>count</arg_key><arg_value>not a number</arg_value>"
            "</tool_call></tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["count"], "not a number")


def _collect_streamed_tool_calls(all_calls):
    """Accumulate streaming ToolCallItems (name + arg-JSON fragments) by tool_index."""
    tools = {}
    for c in all_calls:
        idx = c.tool_index
        if idx not in tools:
            tools[idx] = {"name": c.name or "", "parameters": c.parameters or ""}
        else:
            if c.name:
                tools[idx]["name"] += c.name
            if c.parameters:
                tools[idx]["parameters"] += c.parameters
    return [tools[i] for i in sorted(tools.keys())]


class TestHunyuanDetectorStreaming(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()

    def _new_detector(self):
        return HunyuanDetector()

    def test_normal_text_only(self):
        detector = self._new_detector()
        result = detector.parse_streaming_increment(
            "Hello, I can help you with that.", self.tools
        )
        self.assertEqual(result.normal_text, "Hello, I can help you with that.")
        self.assertEqual(len(result.calls), 0)

    def test_complete_tool_call_single_chunk(self):
        detector = self._new_detector()
        text = (
            "<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>"
        )
        result = detector.parse_streaming_increment(text, self.tools)
        collected = _collect_streamed_tool_calls(result.calls)
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0]["name"], "get_current_date")
        self.assertEqual(json.loads(collected[0]["parameters"]), {})

    def test_hy4_format_without_tool_separator_char_by_char(self):
        detector = HunyuanDetector(_Hy4Tokenizer())
        text = (
            "<tool_calls:opensource>"
            "<tool_call:opensource>get_weather"
            "<arg_key:opensource>city</arg_key:opensource>"
            "<arg_value:opensource>Tokyo</arg_value:opensource>"
            "</tool_call:opensource></tool_calls:opensource>"
        )
        all_calls = []
        for char in text:
            all_calls.extend(detector.parse_streaming_increment(char, self.tools).calls)

        collected = _collect_streamed_tool_calls(all_calls)
        self.assertEqual(collected[0]["name"], "get_weather")
        self.assertEqual(json.loads(collected[0]["parameters"]), {"city": "Tokyo"})

    def test_chunked_tool_call(self):
        detector = self._new_detector()
        chunks = [
            "<tool_calls>",
            "<tool_call>get_weather<tool_sep>",
            "<arg_key>city</arg_key>",
            "<arg_value>Tokyo</arg_value>",
            "</tool_call>",
            "</tool_calls>",
        ]
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        collected = _collect_streamed_tool_calls(all_calls)
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0]["name"], "get_weather")
        args = json.loads(collected[0]["parameters"])
        self.assertEqual(args["city"], "Tokyo")

    def test_normal_text_before_tool(self):
        detector = self._new_detector()
        r1 = detector.parse_streaming_increment("Let me check. ", self.tools)
        self.assertIn("Let me check.", r1.normal_text)

        r2 = detector.parse_streaming_increment(
            "<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>",
            self.tools,
        )
        collected = _collect_streamed_tool_calls(r2.calls)
        self.assertEqual([c["name"] for c in collected], ["get_current_date"])

    def test_multiple_tool_calls_chunked(self):
        detector = self._new_detector()
        chunks = [
            "<tool_calls>\n",
            "<tool_call>get_weather<tool_sep>\n",
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>\n",
            "</tool_call>\n",
            "<tool_call>get_weather<tool_sep>\n",
            "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>\n",
            "</tool_call>\n",
            "</tool_calls>",
        ]
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        collected = _collect_streamed_tool_calls(all_calls)
        self.assertEqual(len(collected), 2)
        self.assertEqual(json.loads(collected[0]["parameters"])["city"], "Beijing")
        self.assertEqual(json.loads(collected[1]["parameters"])["city"], "Tokyo")

    def test_partial_bot_token_buffered(self):
        """Partial <tool_calls> at end of chunk should be buffered, not emitted."""
        detector = self._new_detector()
        r1 = detector.parse_streaming_increment("Hello <tool_", self.tools)
        # "Hello " should be emitted, "<tool_" buffered
        self.assertIn("Hello", r1.normal_text)
        self.assertNotIn("<tool_", r1.normal_text)

    def test_char_by_char_streaming(self):
        """Simulate extreme character-by-character streaming."""
        detector = self._new_detector()
        full = (
            "<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>"
        )
        all_calls = []
        for ch in full:
            result = detector.parse_streaming_increment(ch, self.tools)
            all_calls.extend(result.calls)

        collected = _collect_streamed_tool_calls(all_calls)
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0]["name"], "get_current_date")
        self.assertEqual(json.loads(collected[0]["parameters"]), {})

    def test_streaming_three_tools_sequential(self):
        """Three different tool calls arriving sequentially."""
        detector = self._new_detector()
        chunks = [
            "<tool_calls>",
            "<tool_call>get_current_date<tool_sep></tool_call>",
            "<tool_call>get_weather<tool_sep><arg_key>city</arg_key><arg_value>SF</arg_value></tool_call>",
            "<tool_call>search<tool_sep><arg_key>query</arg_key><arg_value>test</arg_value></tool_call>",
            "</tool_calls>",
        ]
        all_calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(result.calls)

        collected = _collect_streamed_tool_calls(all_calls)
        self.assertEqual(len(collected), 3)
        self.assertEqual(collected[0]["name"], "get_current_date")
        self.assertEqual(collected[1]["name"], "get_weather")
        self.assertEqual(collected[2]["name"], "search")
        # Streaming uses sequential tool_index (0, 1, 2)
        self.assertEqual(sorted({c.tool_index for c in all_calls}), [0, 1, 2])

    def test_streaming_normal_text_not_lost(self):
        """All normal text before tool_calls should be fully emitted."""
        detector = self._new_detector()
        all_normal = ""
        for chunk in ["I will ", "check the ", "date now. "]:
            result = detector.parse_streaming_increment(chunk, self.tools)
            all_normal += result.normal_text

        result = detector.parse_streaming_increment(
            "<tool_calls><tool_call>get_current_date<tool_sep></tool_call></tool_calls>",
            self.tools,
        )
        all_normal += result.normal_text
        self.assertIn("I will check the date now.", all_normal)

    def test_streaming_name_comes_before_args(self):
        """The name delta must arrive before any arg deltas (two-phase contract)."""
        detector = self._new_detector()
        text = (
            "<tool_calls><tool_call>get_weather<tool_sep>"
            "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
            "</tool_call></tool_calls>"
        )
        all_calls = []
        for ch in text:
            all_calls.extend(detector.parse_streaming_increment(ch, self.tools).calls)

        name_indices = [i for i, c in enumerate(all_calls) if c.name]
        param_indices = [i for i, c in enumerate(all_calls) if c.parameters]
        self.assertTrue(name_indices, "expected at least one name delta")
        self.assertTrue(param_indices, "expected at least one arg delta")
        self.assertLess(min(name_indices), min(param_indices))

    def test_streaming_typed_args_coerced(self):
        """Streaming must apply schema-aware type coercion (int/float/bool)."""
        detector = self._new_detector()
        chunks = [
            "<tool_calls>",
            "<tool_call>search<tool_sep>",
            "<arg_key>query</arg_key><arg_value>pizza</arg_value>",
            "<arg_key>count</arg_key><arg_value>7</arg_value>",
            "</tool_call></tool_calls>",
        ]
        all_calls = []
        for chunk in chunks:
            all_calls.extend(
                detector.parse_streaming_increment(chunk, self.tools).calls
            )
        collected = _collect_streamed_tool_calls(all_calls)
        args = json.loads(collected[0]["parameters"])
        self.assertEqual(args["query"], "pizza")
        self.assertEqual(args["count"], 7)
        self.assertIsInstance(args["count"], int)

    def test_streaming_string_arg_holds_back_partial_end_tag(self):
        """Char-by-char string streaming must not leak `</arg_value>` into the value."""
        detector = self._new_detector()
        full = (
            "<tool_calls><tool_call>get_weather<tool_sep>"
            "<arg_key>city</arg_key><arg_value>San Francisco</arg_value>"
            "</tool_call></tool_calls>"
        )
        all_calls = []
        for ch in full:
            all_calls.extend(detector.parse_streaming_increment(ch, self.tools).calls)

        collected = _collect_streamed_tool_calls(all_calls)
        args = json.loads(collected[0]["parameters"])
        self.assertEqual(args["city"], "San Francisco")

    def test_streaming_all_in_one_delta(self):
        """Entire tool call arriving in a single delta."""
        detector = self._new_detector()
        text = (
            "<tool_calls>\n<tool_call>get_current_date<tool_sep>\n"
            "</tool_call>\n</tool_calls>"
        )
        result = detector.parse_streaming_increment(text, self.tools)
        collected = _collect_streamed_tool_calls(result.calls)
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0]["name"], "get_current_date")
        self.assertEqual(json.loads(collected[0]["parameters"]), {})

    def test_streaming_content_before(self):
        """Normal text preceding a tool call must be surfaced."""
        detector = self._new_detector()
        deltas = [
            "Checking.",
            "<tool_calls>",
            "\n<tool_call>",
            "get_current_date",
            "<tool_sep>",
            "\n</tool_call>",
            "\n</tool_calls>",
        ]
        all_calls = []
        all_normal = ""
        for d in deltas:
            r = detector.parse_streaming_increment(d, self.tools)
            all_calls.extend(r.calls)
            all_normal += r.normal_text
        self.assertIn("Checking.", all_normal)
        collected = _collect_streamed_tool_calls(all_calls)
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0]["name"], "get_current_date")


class TestHunyuanDetectorStructureInfo(CustomTestCase):
    def setUp(self):
        self.detector = HunyuanDetector()

    def test_structure_info_content(self):
        info_fn = self.detector.structure_info()
        info = info_fn("get_weather")
        self.assertIn("get_weather", info.begin)
        self.assertIn("<tool_call>", info.begin)
        self.assertIn("<tool_sep>", info.begin)
        self.assertIn("</tool_call>", info.end)
        self.assertEqual(info.trigger, "<tool_calls>")

    def test_supports_structural_tag(self):
        self.assertFalse(self.detector.supports_structural_tag())


_S = ":opensource"


def _hy4_call(name, args=()):
    pairs = "".join(
        f"<arg_key{_S}>{k}</arg_key{_S}><arg_value{_S}>{v}</arg_value{_S}>"
        for k, v in args
    )
    return f"<tool_call{_S}>{name}{pairs}</tool_call{_S}>"


def _hy4_calls(*calls):
    return f"<tool_calls{_S}>" + "".join(calls) + f"</tool_calls{_S}>"


class TestHunyuanForcedToolGrammar(CustomTestCase):
    """Required/named tool_choice must constrain output to a native tool call;
    an unconstrained request lets the model answer in plain text."""

    def setUp(self):
        self.tools = _make_tools()
        self.weather = _hy4_call("get_weather", [("city", "Paris")])
        self.date = _hy4_call("get_current_date")

    def _grammar(self, tool_choice, tokenizer=_Hy4Tokenizer(), **kwargs):
        parser = FunctionCallParser(self.tools, "hunyuan", tokenizer=tokenizer)
        constraint = parser.get_structure_constraint(tool_choice, **kwargs)
        self.assertIsNotNone(constraint)
        kind, tag = constraint
        self.assertEqual(kind, "structural_tag")
        return xgr.Grammar.from_structural_tag(tag)

    def _assert_accepts(self, grammar, accepted, rejected):
        for text in accepted:
            self.assertTrue(_is_grammar_accept_string(grammar, text), text)
        for text in rejected:
            self.assertFalse(_is_grammar_accept_string(grammar, text), text)

    def test_required_forces_native_tool_calls(self):
        self._assert_accepts(
            self._grammar("required"),
            accepted=[_hy4_calls(self.weather), _hy4_calls(self.weather, self.date)],
            rejected=[
                "The weather in Paris is sunny.",
                _hy4_calls(_hy4_call("undefined_tool")),
                '[{"name": "get_weather", "parameters": {"city": "Paris"}}]',
            ],
        )

    def test_required_constrains_argument_keys(self):
        # Required keys follow the properties order, not the required list's.
        self.tools.append(
            Tool(
                type="function",
                function=Function(
                    name="route",
                    parameters={
                        "type": "object",
                        "properties": {"src": {}, "dst": {}},
                        "required": ["dst", "src"],
                    },
                ),
            )
        )
        self._assert_accepts(
            self._grammar("required"),
            accepted=[
                _hy4_calls(_hy4_call("get_weather", [("city", "P"), ("date", "d")])),
                _hy4_calls(_hy4_call("calculate")),
                _hy4_calls(_hy4_call("route", [("src", "a"), ("dst", "b")])),
            ],
            rejected=[
                _hy4_calls(_hy4_call("get_weather", [("date", "d")])),
                _hy4_calls(_hy4_call("get_weather", [("city", "P"), ("zone", "z")])),
                _hy4_calls(_hy4_call("route", [("dst", "b"), ("src", "a")])),
            ],
        )

    def test_named_choice_forces_one_call_of_that_tool(self):
        choice = ToolChoice(function=ToolChoiceFuncName(name="get_weather"))
        self._assert_accepts(
            self._grammar(choice),
            accepted=[_hy4_calls(self.weather)],
            rejected=[
                _hy4_calls(self.date),
                _hy4_calls(self.weather, self.weather),
                "Paris is sunny.",
            ],
        )

    def test_parallel_tool_calls_false_allows_one_call(self):
        self._assert_accepts(
            self._grammar("required", parallel_tool_calls=False),
            accepted=[_hy4_calls(self.date)],
            rejected=[_hy4_calls(self.weather, self.date)],
        )

    def test_thinking_mode_skips_reasoning_before_the_call(self):
        reasoning = f"Need the weather.</think{_S}>"
        self._assert_accepts(
            self._grammar("required", thinking_mode=True),
            accepted=[reasoning + _hy4_calls(self.weather)],
            rejected=[reasoning + "Paris is sunny."],
        )

    def test_hy3_dialect_accepts_tool_sep_and_line_breaks(self):
        text = (
            "<tool_calls>\n<tool_call>get_weather<tool_sep>\n"
            "<arg_key>city</arg_key>\n<arg_value>Beijing</arg_value>\n"
            "</tool_call>\n<tool_call>get_current_date<tool_sep></tool_call>\n"
            "</tool_calls>"
        )
        grammar = self._grammar("required", tokenizer=None)
        self._assert_accepts(grammar, accepted=[text], rejected=["Beijing."])


class TestHunyuanDetectorV46Contracts(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()

    def test_local_ref_and_anchor_nullable_args(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    parameters={
                        "$defs": {
                            "NullableText": {
                                "$anchor": "nullable_text",
                                "anyOf": [{"type": "string"}, {"type": "null"}],
                            },
                            "Count": {"type": "integer"},
                        },
                        "type": "object",
                        "properties": {
                            "query": {"$ref": "#nullable_text"},
                            "note": {"type": ["string", "null"]},
                            "count": {
                                "anyOf": [
                                    {"$ref": "#/$defs/Count"},
                                    {"type": "null"},
                                ]
                            },
                        },
                    },
                ),
            )
        ]
        text = (
            "<tool_calls><tool_call>search<tool_sep>"
            "<arg_key>query</arg_key><arg_value>null</arg_value>"
            "<arg_key>note</arg_key><arg_value>null</arg_value>"
            "<arg_key>count</arg_key><arg_value>7</arg_value>"
            "</tool_call></tool_calls>"
        )
        result = HunyuanDetector().detect_and_parse(text, tools)
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"query": None, "note": None, "count": 7},
        )

    def test_bare_tool_call_non_streaming(self):
        text = "Checking.<tool_call>get_current_date<tool_sep></tool_call>"
        detector = HunyuanDetector()
        self.assertTrue(detector.has_tool_call(text))
        result = detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "Checking.")
        self.assertEqual(result.calls[0].name, "get_current_date")
        self.assertEqual(json.loads(result.calls[0].parameters), {})

    def test_bare_tool_call_streaming(self):
        detector = HunyuanDetector()
        text = "<tool_call>get_current_date<tool_sep></tool_call>"
        calls = []
        for char in text:
            calls.extend(detector.parse_streaming_increment(char, self.tools).calls)
        collected = _collect_streamed_tool_calls(calls)
        self.assertEqual(collected[0]["name"], "get_current_date")
        self.assertEqual(json.loads(collected[0]["parameters"]), {})

    def test_duplicate_args_keep_streamed_json_prefix_stable(self):
        detector = HunyuanDetector()
        chunks = [
            "<tool_calls><tool_call>search<tool_sep>"
            "<arg_key>query</arg_key><arg_value>first</arg_value>"
            "<arg_key>query</arg_key><arg_value>sec",
            "ond</arg_value></tool_call></tool_calls>",
        ]
        emissions = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            if detector.streamed_args_for_tool:
                emissions.append(detector.streamed_args_for_tool[0])
        final_json = emissions[-1]
        self.assertEqual(final_json.count('"query"'), 2)
        self.assertEqual(json.loads(final_json), {"query": "second"})
        for emission in emissions:
            self.assertTrue(final_json.startswith(emission), emission)

    def test_adjacent_bare_calls_across_chunks(self):
        detector = HunyuanDetector()
        chunks = [
            "<tool_call>get_current_date<tool_sep></tool_call>"
            "<tool_call>search<tool_sep>"
            "<arg_key>query</arg_key><arg_value>hel",
            "lo</arg_value></tool_call>",
        ]
        normal_parts = []
        calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal_parts.append(result.normal_text)
            calls.extend(result.calls)
        collected = _collect_streamed_tool_calls(calls)
        self.assertEqual(
            [call["name"] for call in collected], ["get_current_date", "search"]
        )
        self.assertEqual(json.loads(collected[1]["parameters"]), {"query": "hello"})
        self.assertNotIn("<arg_key", "".join(normal_parts))

    def test_adjacent_bare_calls_across_chunks_hy4_suffixed(self):
        suffix = ":opensource"
        detector = HunyuanDetector(_Hy4Tokenizer())
        chunks = [
            f"<tool_call{suffix}>get_current_date</tool_call{suffix}>"
            f"<tool_call{suffix}>search"
            f"<arg_key{suffix}>query</arg_key{suffix}>"
            f"<arg_value{suffix}>hel",
            f"lo</arg_value{suffix}></tool_call{suffix}>",
        ]
        normal_parts = []
        calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal_parts.append(result.normal_text)
            calls.extend(result.calls)
        collected = _collect_streamed_tool_calls(calls)
        self.assertEqual(
            [call["name"] for call in collected], ["get_current_date", "search"]
        )
        self.assertEqual(json.loads(collected[1]["parameters"]), {"query": "hello"})
        self.assertNotIn("<arg_key", "".join(normal_parts))

    def test_bare_call_outer_close_and_trailing_text(self):
        detector = HunyuanDetector()
        normal_parts = []
        for chunk in (
            "<tool_call>get_current_date<tool_sep></tool_call>",
            "</tool",
            "_calls>Done.",
        ):
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal_parts.append(result.normal_text)
        normal_parts.append(detector.finish(self.tools).normal_text)
        self.assertEqual("".join(normal_parts), "Done.")

    def test_unfinished_string_closes_with_valid_json(self):
        detector = HunyuanDetector()
        chunks = [
            "<tool_calls><tool_call>search<tool_sep>"
            "<arg_key>count</arg_key><arg_value>7</arg_value>"
            "<arg_key>query</arg_key><arg_value>hel",
            "lo</tool_call></tool_calls>",
        ]
        emissions = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            if detector.streamed_args_for_tool:
                emissions.append(detector.streamed_args_for_tool[0])
        final_json = emissions[-1]
        self.assertEqual(json.loads(final_json), {"count": 7, "query": "hello"})
        for emission in emissions:
            self.assertTrue(final_json.startswith(emission), emission)

    def test_unfinished_string_split_tool_close(self):
        detector = HunyuanDetector()
        chunks = [
            "<tool_calls><tool_call>search<tool_sep>"
            "<arg_key>count</arg_key><arg_value>7</arg_value>"
            "<arg_key>query</arg_key><arg_value>hello</t",
            "ool_call></tool_calls>",
        ]
        emissions = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            if detector.streamed_args_for_tool:
                emissions.append(detector.streamed_args_for_tool[0])
        final_json = emissions[-1]
        self.assertEqual(json.loads(final_json), {"count": 7, "query": "hello"})
        for emission in emissions:
            self.assertTrue(final_json.startswith(emission), emission)

    def test_bare_call_newline_before_outer_close(self):
        detector = HunyuanDetector()
        normal_parts = []
        calls = []
        for chunk in (
            "<tool_call>get_current_date<tool_sep></tool_call>",
            "\n",
            "</tool_calls>",
        ):
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal_parts.append(result.normal_text)
            calls.extend(result.calls)
        self.assertEqual(
            _collect_streamed_tool_calls(calls)[0]["name"], "get_current_date"
        )
        self.assertEqual("".join(normal_parts), "\n")

    def test_outer_close_preserves_partial_next_call_marker(self):
        detector = HunyuanDetector()
        chunks = [
            "<tool_call>get_current_date<tool_sep></tool_call></tool_calls><tool_",
            "call>search<tool_sep><arg_key>query</arg_key><arg_value>hello</arg_value></tool_call>",
        ]
        normal_parts = []
        calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal_parts.append(result.normal_text)
            calls.extend(result.calls)
        collected = _collect_streamed_tool_calls(calls)
        self.assertEqual(
            [call["name"] for call in collected], ["get_current_date", "search"]
        )
        self.assertEqual(json.loads(collected[1]["parameters"]), {"query": "hello"})
        self.assertNotIn("<tool_", "".join(normal_parts))

    def test_bare_close_then_wrapped_open_ordering(self):
        for suffix in ("", ":opensource"):
            for char_mode in (False, True):
                with self.subTest(suffix=suffix, char_mode=char_mode):
                    detector = HunyuanDetector(_Hy4Tokenizer() if suffix else None)
                    bare = f"<tool_call{suffix}>get_current_date"
                    if not suffix:
                        bare += "<tool_sep>"
                    bare += f"</tool_call{suffix}>"
                    wrapped = (
                        f"Between.<tool_calls{suffix}><tool_call{suffix}>get_current_date"
                        + ("" if suffix else "<tool_sep>")
                        + f"</tool_call{suffix}></tool_calls{suffix}>"
                    )
                    text = bare + f"</tool_calls{suffix}>" + wrapped
                    chunks = list(text) if char_mode else [text]
                    normal_parts = []
                    calls = []
                    for chunk in chunks:
                        result = detector.parse_streaming_increment(chunk, self.tools)
                        normal_parts.append(result.normal_text)
                        calls.extend(result.calls)
                    normal_parts.append(detector.finish(self.tools).normal_text)
                    self.assertEqual("".join(normal_parts), "Between.")
                    self.assertEqual(len(_collect_streamed_tool_calls(calls)), 2)

    def test_outer_close_next_bare_call_finishes_later(self):
        detector = HunyuanDetector()
        chunks = [
            "<tool_call>get_current_date<tool_sep></tool_call></tool_calls>"
            "<tool_call>search<tool_sep><arg_key>query</arg_key><arg_value>hel",
            "lo</arg_value></tool_call>",
        ]
        normal_parts = []
        calls = []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal_parts.append(result.normal_text)
            calls.extend(result.calls)
        collected = _collect_streamed_tool_calls(calls)
        self.assertEqual(
            [call["name"] for call in collected], ["get_current_date", "search"]
        )
        self.assertEqual(json.loads(collected[1]["parameters"]), {"query": "hello"})
        self.assertNotIn("<tool_", "".join(normal_parts))

    def test_bare_call_then_wrapped_call(self):
        for suffix in ("", ":opensource"):
            for char_mode in (False, True):
                with self.subTest(suffix=suffix, char_mode=char_mode):
                    detector = HunyuanDetector(_Hy4Tokenizer() if suffix else None)
                    bare = f"<tool_call{suffix}>get_current_date"
                    if not suffix:
                        bare += "<tool_sep>"
                    bare += f"</tool_call{suffix}>Done."
                    wrapped = (
                        f"<tool_calls{suffix}><tool_call{suffix}>get_current_date"
                        + ("" if suffix else "<tool_sep>")
                        + f"</tool_call{suffix}></tool_calls{suffix}>"
                    )
                    chunks = list(bare + wrapped) if char_mode else [bare, wrapped]
                    normal_parts = []
                    calls = []
                    for chunk in chunks:
                        result = detector.parse_streaming_increment(chunk, self.tools)
                        normal_parts.append(result.normal_text)
                        calls.extend(result.calls)
                    normal_parts.append(detector.finish(self.tools).normal_text)
                    self.assertEqual("".join(normal_parts), "Done.")
                    self.assertEqual(len(_collect_streamed_tool_calls(calls)), 2)

    def test_nested_union_ref_and_nullable_string_streaming(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    parameters={
                        "type": "object",
                        "$defs": {
                            "Count": {"type": "integer"},
                            "NullableText": {
                                "anyOf": [{"type": "string"}, {"type": "null"}]
                            },
                        },
                        "properties": {
                            "query": {"$ref": "#/$defs/NullableText"},
                            "count": {
                                "anyOf": [
                                    {"oneOf": [{"$ref": "#/$defs/Count"}]},
                                    {"type": "null"},
                                ]
                            },
                        },
                    },
                ),
            )
        ]
        text = (
            "<tool_calls><tool_call>search<tool_sep>"
            "<arg_key>query</arg_key><arg_value>null</arg_value>"
            "<arg_key>count</arg_key><arg_value>7</arg_value>"
            "</tool_call></tool_calls>"
        )
        expected = {"query": None, "count": 7}
        result = HunyuanDetector().detect_and_parse(text, tools)
        self.assertEqual(json.loads(result.calls[0].parameters), expected)

        detector = HunyuanDetector()
        calls = []
        for char in text:
            calls.extend(detector.parse_streaming_increment(char, tools).calls)
        collected = _collect_streamed_tool_calls(calls)
        self.assertEqual(json.loads(collected[0]["parameters"]), expected)


class TestHunyuanDetectorFunctionCallParser(CustomTestCase):
    """Test through the FunctionCallParser interface."""

    def setUp(self):
        self.tools = _make_tools()

    def test_parser_registry(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        parser = FunctionCallParser(self.tools, "hunyuan")
        self.assertIsInstance(parser.detector, HunyuanDetector)

    def test_parse_non_stream(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        parser = FunctionCallParser(self.tools, "hunyuan")
        text = (
            "Checking.<tool_calls><tool_call>get_weather<tool_sep>"
            "<arg_key>city</arg_key><arg_value>Tokyo</arg_value>"
            "</tool_call></tool_calls>"
        )
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(normal, "Checking.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters)["city"], "Tokyo")

    def test_parse_stream_chunks(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        parser = FunctionCallParser(self.tools, "hunyuan")
        chunks = [
            "<tool_calls>",
            "<tool_call>get_current_date<tool_sep></tool_call>",
            "</tool_calls>",
        ]
        all_calls = []
        for chunk in chunks:
            normal, calls = parser.parse_stream_chunk(chunk)
            all_calls.extend(calls)

        collected = _collect_streamed_tool_calls(all_calls)
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0]["name"], "get_current_date")
        self.assertEqual(json.loads(collected[0]["parameters"]), {})

    def test_has_tool_call_through_parser(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        parser = FunctionCallParser(self.tools, "hunyuan")
        self.assertTrue(parser.has_tool_call("<tool_calls>foo</tool_calls>"))
        self.assertFalse(parser.has_tool_call("no tools here"))


if __name__ == "__main__":
    unittest.main()
