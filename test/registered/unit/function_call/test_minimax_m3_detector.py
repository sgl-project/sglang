import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.minimax_m3 import MINIMAX_NS_TOKEN, MinimaxM3Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

NS = MINIMAX_NS_TOKEN


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
                        "query": {"type": "string"},
                        "count": {"type": "integer"},
                        "ratio": {"type": "number"},
                        "verbose": {"type": "boolean"},
                    },
                    "required": ["query"],
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="create_event",
                description="Create a calendar event",
                parameters={
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "location": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "zip": {"type": "integer"},
                            },
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="add_note",
                description="Add a free-form note",
                parameters={
                    "type": "object",
                    "properties": {"note": {"type": "string"}},
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="configure",
                description="Configure runtime options",
                parameters={
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["none", "low", "high"],
                        },
                        "optional": {"type": ["string", "null"]},
                    },
                },
            ),
        ),
    ]


def _wire(*lines):
    return "".join(NS + line for line in lines)


def _segments(*lines):
    return [NS + line for line in lines]


def _collect_streamed_tool_calls(all_calls):
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


def _stream_segments(segments, tools):
    detector = MinimaxM3Detector()
    all_calls = []
    for seg in _segments(*segments):
        all_calls.extend(detector.parse_streaming_increment(seg, tools).calls)
    collected = _collect_streamed_tool_calls(all_calls)
    return [{"name": c["name"], "args": json.loads(c["parameters"])} for c in collected]


def _parse_segments(segments, tools):
    detector = MinimaxM3Detector()
    result = detector.detect_and_parse(_wire(*segments), tools)
    return [
        {"name": c.name, "args": json.loads(c.parameters)} for c in result.calls
    ], result.normal_text


class TestMinimaxM3HasToolCall(CustomTestCase):
    def setUp(self):
        self.detector = MinimaxM3Detector()

    def test_has_tool_call_true(self):
        text = _wire(
            "<tool_call>",
            '<invoke name="get_current_date">',
            "</invoke>",
            "</tool_call>",
        )
        self.assertTrue(self.detector.has_tool_call(text))

    def test_has_tool_call_false(self):
        self.assertFalse(
            self.detector.has_tool_call("The weather in Beijing is sunny.")
        )


class TestMinimaxM3DetectAndParse(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()

    def test_no_tool_call(self):
        text = "This is a plain response with no tool call."
        calls, normal = _parse_segments_text(text, self.tools)
        self.assertEqual(len(calls), 0)
        self.assertEqual(normal, text)

    def test_single_tool_call(self):
        segments = (
            "<tool_call>",
            '<invoke name="get_weather">',
            "<city>Beijing",
            "</city>",
            "</invoke>",
            "</tool_call>",
        )
        calls, _ = _parse_segments(segments, self.tools)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertEqual(calls[0]["args"], {"city": "Beijing"})

    def test_zero_arg_tool_call(self):
        segments = (
            "<tool_call>",
            '<invoke name="get_current_date">',
            "</invoke>",
            "</tool_call>",
        )
        calls, _ = _parse_segments(segments, self.tools)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_current_date")
        self.assertEqual(calls[0]["args"], {})

    def test_multiple_tool_calls_separate_blocks(self):
        segments = (
            "<tool_call>",
            '<invoke name="get_weather">',
            "<city>Beijing",
            "</city>",
            "</invoke>",
            "</tool_call>",
            "<tool_call>",
            '<invoke name="get_weather">',
            "<city>Tokyo",
            "</city>",
            "</invoke>",
            "</tool_call>",
        )
        calls, _ = _parse_segments(segments, self.tools)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["args"]["city"], "Beijing")
        self.assertEqual(calls[1]["args"]["city"], "Tokyo")

    def test_multiple_invokes_one_block(self):
        segments = (
            "<tool_call>",
            '<invoke name="get_weather">',
            "<city>Beijing",
            "</city>",
            "</invoke>",
            '<invoke name="get_weather">',
            "<city>Tokyo",
            "</city>",
            "</invoke>",
            "</tool_call>",
        )
        calls, _ = _parse_segments(segments, self.tools)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["args"]["city"], "Beijing")
        self.assertEqual(calls[1]["args"]["city"], "Tokyo")

    def test_content_before_tool_call(self):
        segments = (
            "<tool_call>",
            '<invoke name="get_weather">',
            "<city>Paris",
            "</city>",
            "</invoke>",
            "</tool_call>",
        )
        detector = MinimaxM3Detector()
        text = "Let me check the weather." + _wire(*segments)
        result = detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.normal_text, "Let me check the weather.")

    def test_typed_scalars(self):
        segments = (
            "<tool_call>",
            '<invoke name="search">',
            "<query>pizza",
            "</query>",
            "<count>7",
            "</count>",
            "<ratio>0.5",
            "</ratio>",
            "<verbose>true",
            "</verbose>",
            "</invoke>",
            "</tool_call>",
        )
        calls, _ = _parse_segments(segments, self.tools)
        args = calls[0]["args"]
        self.assertEqual(args["query"], "pizza")
        self.assertEqual(args["count"], 7)
        self.assertIsInstance(args["count"], int)
        self.assertAlmostEqual(args["ratio"], 0.5)
        self.assertIs(args["verbose"], True)

    def test_nested_object_and_array(self):
        segments = (
            "<tool_call>",
            '<invoke name="create_event">',
            "<title>Standup",
            "</title>",
            "<location>",
            "<city>NYC",
            "</city>",
            "<zip>10001",
            "</zip>",
            "</location>",
            "<tags>",
            "<item>red",
            "</item>",
            "<item>blue",
            "</item>",
            "</tags>",
            "</invoke>",
            "</tool_call>",
        )
        calls, _ = _parse_segments(segments, self.tools)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            calls[0]["args"],
            {
                "title": "Standup",
                "location": {"city": "NYC", "zip": 10001},
                "tags": ["red", "blue"],
            },
        )

    def test_string_with_special_characters(self):
        value = 'a "quoted" word with \\ backslash and\nnewline'
        segments = (
            "<tool_call>",
            '<invoke name="add_note">',
            "<note>" + value,
            "</note>",
            "</invoke>",
            "</tool_call>",
        )
        calls, _ = _parse_segments(segments, self.tools)
        self.assertEqual(calls[0]["args"], {"note": value})


class TestMinimaxM3NoneNullRegression(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()

    def _configure_segments(self, param, value):
        return (
            "<tool_call>",
            '<invoke name="configure">',
            "<{}>{}".format(param, value),
            "</{}>".format(param),
            "</invoke>",
            "</tool_call>",
        )

    def test_plain_string_none_not_coerced(self):
        for value in ("none", "nil", "null"):
            with self.subTest(value=value):
                segments = self._configure_segments("mode", value)
                calls, _ = _parse_segments(segments, self.tools)
                self.assertEqual(calls[0]["args"], {"mode": value})
                self.assertIsInstance(calls[0]["args"]["mode"], str)

    def test_plain_string_none_not_coerced_streaming(self):
        for value in ("none", "nil", "null"):
            with self.subTest(value=value):
                segments = self._configure_segments("mode", value)
                calls = _stream_segments(segments, self.tools)
                self.assertEqual(calls[0]["args"], {"mode": value})
                self.assertIsInstance(calls[0]["args"]["mode"], str)

    def test_streaming_and_non_streaming_agree_for_string(self):
        for value in ("none", "nil", "null"):
            with self.subTest(value=value):
                segments = self._configure_segments("mode", value)
                non_stream, _ = _parse_segments(segments, self.tools)
                stream = _stream_segments(segments, self.tools)
                self.assertEqual(non_stream, stream)

    def test_nullable_param_null_becomes_none(self):
        segments = self._configure_segments("optional", "null")
        calls, _ = _parse_segments(segments, self.tools)
        self.assertEqual(calls[0]["args"], {"optional": None})

    def test_nullable_param_none_stays_string(self):
        segments = self._configure_segments("optional", "none")
        calls, _ = _parse_segments(segments, self.tools)
        self.assertEqual(calls[0]["args"], {"optional": "none"})


class TestMinimaxM3Streaming(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()

    def test_normal_text_only(self):
        detector = MinimaxM3Detector()
        result = detector.parse_streaming_increment("Hello there.", self.tools)
        self.assertEqual(result.normal_text, "Hello there.")
        self.assertEqual(len(result.calls), 0)

    def test_single_tool_call_chunked(self):
        segments = (
            "<tool_call>",
            '<invoke name="get_weather">',
            "<city>Beijing",
            "</city>",
            "</invoke>",
            "</tool_call>",
        )
        calls = _stream_segments(segments, self.tools)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertEqual(calls[0]["args"], {"city": "Beijing"})

    def test_streaming_matches_non_streaming(self):
        cases = {
            "weather": (
                "<tool_call>",
                '<invoke name="get_weather">',
                "<city>Beijing",
                "</city>",
                "</invoke>",
                "</tool_call>",
            ),
            "typed": (
                "<tool_call>",
                '<invoke name="search">',
                "<query>pizza",
                "</query>",
                "<count>7",
                "</count>",
                "<ratio>0.5",
                "</ratio>",
                "<verbose>true",
                "</verbose>",
                "</invoke>",
                "</tool_call>",
            ),
            "nested": (
                "<tool_call>",
                '<invoke name="create_event">',
                "<title>Standup",
                "</title>",
                "<location>",
                "<city>NYC",
                "</city>",
                "<zip>10001",
                "</zip>",
                "</location>",
                "<tags>",
                "<item>red",
                "</item>",
                "<item>blue",
                "</item>",
                "</tags>",
                "</invoke>",
                "</tool_call>",
            ),
            "special": (
                "<tool_call>",
                '<invoke name="add_note">',
                "<note>" + 'a "q" and \\ back\nslash',
                "</note>",
                "</invoke>",
                "</tool_call>",
            ),
            "multi_invoke": (
                "<tool_call>",
                '<invoke name="get_weather">',
                "<city>Beijing",
                "</city>",
                "</invoke>",
                '<invoke name="get_weather">',
                "<city>Tokyo",
                "</city>",
                "</invoke>",
                "</tool_call>",
            ),
        }
        for name, segments in cases.items():
            with self.subTest(case=name):
                non_stream, _ = _parse_segments(segments, self.tools)
                stream = _stream_segments(segments, self.tools)
                self.assertEqual(non_stream, stream)

    def test_streaming_sequential_tool_index(self):
        segments = (
            "<tool_call>",
            '<invoke name="get_weather">',
            "<city>Beijing",
            "</city>",
            "</invoke>",
            '<invoke name="get_weather">',
            "<city>Tokyo",
            "</city>",
            "</invoke>",
            "</tool_call>",
        )
        detector = MinimaxM3Detector()
        all_calls = []
        for seg in _segments(*segments):
            all_calls.extend(detector.parse_streaming_increment(seg, self.tools).calls)
        self.assertEqual(sorted({c.tool_index for c in all_calls}), [0, 1])


class TestMinimaxM3Malformed(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()

    def test_truncated_no_closing_tags(self):
        text = _wire(
            "<tool_call>",
            '<invoke name="get_weather">',
            "<city>Beijing",
        )
        detector = MinimaxM3Detector()
        result = detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_mismatched_closing_tag(self):
        text = _wire(
            "<tool_call>",
            '<invoke name="get_weather">',
            "<city>Beijing",
            "</wrong>",
            "</invoke>",
            "</tool_call>",
        )
        detector = MinimaxM3Detector()
        result = detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, text)

    def test_truncated_streaming_does_not_crash(self):
        segments = (
            "<tool_call>",
            '<invoke name="get_weather">',
            "<city>Beijing",
        )
        detector = MinimaxM3Detector()
        for seg in _segments(*segments):
            detector.parse_streaming_increment(seg, self.tools)


def _parse_segments_text(text, tools):
    detector = MinimaxM3Detector()
    result = detector.detect_and_parse(text, tools)
    return [
        {"name": c.name, "args": json.loads(c.parameters)} for c in result.calls
    ], result.normal_text


class TestMinimaxM3TopLevelOneOf(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="acme",
                    description="Send a value to Acme.",
                    parameters={
                        "type": "object",
                        "oneOf": [
                            {
                                "type": "object",
                                "properties": {
                                    "count": {"type": "integer"},
                                    "verbose": {"type": "boolean"},
                                },
                                "required": ["count", "verbose"],
                            },
                            {
                                "type": "object",
                                "properties": {"kind": {"const": "other"}},
                                "required": ["kind"],
                            },
                        ],
                    },
                ),
            ),
        ]
        self.segments = (
            "<tool_call>",
            '<invoke name="acme">',
            "<count>7",
            "</count>",
            "<verbose>true",
            "</verbose>",
            "</invoke>",
            "</tool_call>",
        )
        self.expected = {"count": 7, "verbose": True}

    def test_detect_and_parse(self):
        calls, _ = _parse_segments(self.segments, self.tools)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["args"], self.expected)


class TestMinimaxM3SchemaLessContainers(CustomTestCase):
    """Members of a bare {"type": "object"} have no schema to type them by."""

    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="api_fetch",
                    description="Call an API endpoint",
                    parameters={
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "body": {"type": "object"},
                        },
                        "required": ["url", "body"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="deep_think",
                    description="Manage the todo list",
                    parameters={
                        "type": "object",
                        "properties": {
                            "todo_list": {
                                "type": "array",
                                "items": {"type": "object"},
                            },
                        },
                    },
                ),
            ),
        ]

    def _fetch(self, *body):
        return (
            "<tool_call>",
            '<invoke name="api_fetch">',
            "<url>http://127.0.0.1:1000",
            "</url>",
            "<body>",
            *body,
            "</body>",
            "</invoke>",
            "</tool_call>",
        )

    def _assert_body(self, segments, expected):
        calls, _ = _parse_segments(segments, self.tools)
        self.assertEqual(calls[0]["args"]["body"], expected)
        self.assertEqual(_stream_segments(segments, self.tools), calls)

    def test_two_dimensional_array_and_bool(self):
        segments = self._fetch(
            "<range>Q!M5:O5",
            "</range>",
            "<values>",
            "<item>",
            "<item>Allocated",
            "</item>",
            "<item>521",
            "</item>",
            "<item>Emerging",
            "</item>",
            "</item>",
            "</values>",
            "<flag>true",
            "</flag>",
        )
        self._assert_body(
            segments,
            {
                "range": "Q!M5:O5",
                "values": [["Allocated", "521", "Emerging"]],
                "flag": True,
            },
        )

    def test_single_item_array(self):
        segments = self._fetch("<tags>", "<item>red", "</item>", "</tags>")
        self._assert_body(segments, {"tags": ["red"]})

    def test_literals_convert_other_leaves_stay_verbatim(self):
        segments = self._fetch(
            "<off>false",
            "</off>",
            "<missing>null",
            "</missing>",
            "<count>521",
            "</count>",
            "<shout>True",
            "</shout>",
        )
        self._assert_body(
            segments,
            {"off": False, "missing": None, "count": "521", "shout": "True"},
        )

    def test_item_beside_named_children_stays_object(self):
        segments = self._fetch(
            "<meta>",
            "<item>a",
            "</item>",
            "<name>b",
            "</name>",
            "</meta>",
        )
        self._assert_body(segments, {"meta": {"item": "a", "name": "b"}})

    def test_array_of_bare_objects(self):
        segments = (
            "<tool_call>",
            '<invoke name="deep_think">',
            "<todo_list>",
            "<item>",
            "<title>Outline",
            "</title>",
            "<done>true",
            "</done>",
            "</item>",
            "<item>",
            "<title>Draft",
            "</title>",
            "<done>false",
            "</done>",
            "</item>",
            "</todo_list>",
            "</invoke>",
            "</tool_call>",
        )
        calls, _ = _parse_segments(segments, self.tools)
        self.assertEqual(
            calls[0]["args"],
            {
                "todo_list": [
                    {"title": "Outline", "done": True},
                    {"title": "Draft", "done": False},
                ]
            },
        )
        self.assertEqual(_stream_segments(segments, self.tools), calls)

    _UNTYPED_PARAMS = (
        "<tool_call>",
        '<invoke name="{name}">',
        "<flag>true",
        "</flag>",
        "<tags>",
        "<item>a",
        "</item>",
        "<item>b",
        "</item>",
        "</tags>",
        "<count>521",
        "</count>",
        "</invoke>",
        "</tool_call>",
    )

    def _untyped_params(self, name):
        return tuple(s.format(name=name) for s in self._UNTYPED_PARAMS)

    def test_top_level_param_without_type(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="loose",
                    description="Parameters declared without a type",
                    parameters={
                        "type": "object",
                        "properties": {
                            "flag": {"description": "no type"},
                            "tags": {"description": "no type"},
                        },
                    },
                ),
            )
        ]
        segments = self._untyped_params("loose")
        expected = {"flag": True, "tags": ["a", "b"], "count": "521"}
        calls, _ = _parse_segments(segments, tools)
        self.assertEqual(calls[0]["args"], expected)
        self.assertEqual(_stream_segments(segments, tools), calls)

    def test_untyped_string_keeps_its_whitespace_when_streamed(self):
        """A schema-less value that turns out to be a string keeps its leading
        whitespace when streamed, as when parsed whole; only spacing before a
        nested tag is markup."""
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="loose",
                    description="A parameter declared without a type",
                    parameters={"type": "object", "properties": {"value": {}}},
                ),
            )
        ]
        wire = _wire(
            "<tool_call>",
            '<invoke name="loose">',
            "<value>  hello\n  world",
            "</value>",
            "</invoke>",
            "</tool_call>",
        )
        result = MinimaxM3Detector().detect_and_parse(wire, tools)
        whole = json.loads(result.calls[0].parameters)
        self.assertTrue(whole["value"].startswith("  hello"))
        for size in (1, 7, len(wire)):
            detector = MinimaxM3Detector()
            calls = []
            for i in range(0, len(wire), size):
                calls += detector.parse_streaming_increment(
                    wire[i : i + size], tools
                ).calls
            with self.subTest(chunk_size=size):
                streamed = _collect_streamed_tool_calls(calls)[0]["parameters"]
                self.assertEqual(json.loads(streamed), whole)

    def test_undeclared_tool_streams_like_non_stream(self):
        parser = FunctionCallParser(tools=[], tool_call_parser="minimax-m3")
        wire = _wire(*self._untyped_params("undeclared"))
        _, non_stream = parser.parse_non_stream(wire)

        parser = FunctionCallParser(tools=[], tool_call_parser="minimax-m3")
        streamed = []
        for chunk in _segments(*self._untyped_params("undeclared")):
            streamed += parser.parse_stream_chunk(chunk)[1]
        streamed += parser.parse_stream_end()[1]
        stream_args = _collect_streamed_tool_calls(streamed)[0]["parameters"]

        expected = {"flag": True, "tags": ["a", "b"], "count": "521"}
        self.assertEqual(json.loads(non_stream[0].parameters), expected)
        self.assertEqual(json.loads(stream_args), expected)

    def test_small_stream_chunks_match_non_stream(self):
        segments = self._fetch(
            "<values>",
            "<item>",
            "<item>Allocated",
            "</item>",
            "<item>521",
            "</item>",
            "</item>",
            "</values>",
            "<flag>true",
            "</flag>",
        )
        expected, _ = _parse_segments(segments, self.tools)
        wire = _wire(*segments)
        for size in (1, 2, 3, 5, 7):
            with self.subTest(chunk_size=size):
                detector = MinimaxM3Detector()
                streamed = []
                for i in range(0, len(wire), size):
                    chunk = wire[i : i + size]
                    streamed += detector.parse_streaming_increment(
                        chunk, self.tools
                    ).calls
                collected = _collect_streamed_tool_calls(streamed)
                self.assertEqual(
                    json.loads(collected[0]["parameters"]), expected[0]["args"]
                )


class TestMinimaxM3WithoutTools(CustomTestCase):
    """A request that declares no tools still gets the model's tool call parsed."""

    _CALL = ("<tool_call>", '<invoke name="list_skills">', "</invoke>", "</tool_call>")

    def setUp(self):
        self.parser = FunctionCallParser(tools=[], tool_call_parser="minimax-m3")

    def test_non_stream_keeps_undeclared_call(self):
        normal, calls = self.parser.parse_non_stream("Retrying." + _wire(*self._CALL))
        self.assertEqual(normal, "Retrying.")
        self.assertEqual(
            [(c.name, json.loads(c.parameters)) for c in calls], [("list_skills", {})]
        )

    def test_stream_keeps_undeclared_call(self):
        normal, calls = "", []
        for chunk in ["Retrying."] + _segments(*self._CALL):
            chunk_text, chunk_calls = self.parser.parse_stream_chunk(chunk)
            normal += chunk_text
            calls += chunk_calls
        end_text, end_calls = self.parser.parse_stream_end()
        self.assertEqual(normal + end_text, "Retrying.")
        self.assertEqual(
            [c["name"] for c in _collect_streamed_tool_calls(calls + end_calls)],
            ["list_skills"],
        )

    def test_stream_end_releases_held_back_text(self):
        """A trailing "]" held back as a possible marker start is released at stream end."""
        parser = FunctionCallParser(tools=_make_tools(), tool_call_parser="minimax-m3")
        normal = "".join(parser.parse_stream_chunk(c)[0] for c in ("[SILENCE", "]"))
        self.assertEqual(normal + parser.parse_stream_end()[0], "[SILENCE]")

    def test_other_parsers_skip_requests_without_tools(self):
        text = '<tool_call>{"name": "lookup", "arguments": {}}</tool_call>'
        parser = FunctionCallParser(tools=[], tool_call_parser="hermes")
        self.assertEqual(parser.parse_non_stream(text), (text, []))


if __name__ == "__main__":
    unittest.main()
