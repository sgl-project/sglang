import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
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
                        "fallback": {"type": ["string", "integer", "null"]},
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

    def test_nullable_param_null_becomes_none_streaming(self):
        """A string parameter that allows null, written as null or left empty, streams as JSON null."""
        for param in ("optional", "fallback"):
            for value in ("null", "Null", "NULL", ""):
                with self.subTest(param=param, value=value):
                    segments = self._configure_segments(param, value)
                    calls = _stream_segments(segments, self.tools)
                    self.assertEqual(calls[0]["args"], {param: None})

    def test_nullable_param_agrees_at_every_split_point(self):
        """Streamed arguments equal detect_and_parse wherever the chunk boundary falls."""
        for value in ("null", "", "nu", "nullable", "hello"):
            segments = self._configure_segments("optional", value)
            expected, _ = _parse_segments(segments, self.tools)
            wire = _wire(*segments)
            for split in range(len(wire) + 1):
                with self.subTest(value=value, split=split):
                    detector = MinimaxM3Detector()
                    all_calls = []
                    for text in (wire[:split], wire[split:]):
                        result = detector.parse_streaming_increment(text, self.tools)
                        all_calls.extend(result.calls)
                    (call,) = _collect_streamed_tool_calls(all_calls)
                    self.assertEqual(
                        json.loads(call["parameters"]), expected[0]["args"]
                    )

    def test_empty_value_in_one_chunk_matches_non_streaming(self):
        """An empty value whose closing tag shares its chunk keeps the type detect_and_parse gives."""
        for param in ("note", "undeclared"):
            with self.subTest(param=param):
                segments = (
                    "<tool_call>",
                    '<invoke name="add_note">',
                    "<{}>".format(param),
                    "</{}>".format(param),
                    "</invoke>",
                    "</tool_call>",
                )
                expected, _ = _parse_segments(segments, self.tools)
                result = MinimaxM3Detector().parse_streaming_increment(
                    _wire(*segments), self.tools
                )
                (call,) = _collect_streamed_tool_calls(result.calls)
                self.assertEqual(json.loads(call["parameters"]), expected[0]["args"])

    def test_nullable_param_text_streams_before_closing_tag(self):
        """A non-null value of a nullable string still streams before its closing tag."""
        detector = MinimaxM3Detector()
        for text in _segments(
            "<tool_call>", '<invoke name="configure">', "<optional>hello"
        ):
            result = detector.parse_streaming_increment(text, self.tools)
        self.assertIn('"hello', "".join(c.parameters for c in result.calls))


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

    def test_mismatched_closing_tag_after_opening_tag_in_one_chunk(self):
        """A wrong closing tag right after a scalar's opening tag is not streamed as its value."""
        text = _wire(
            "<tool_call>",
            '<invoke name="add_note">',
            "<note>",
            "</wrong>",
            "</note>",
            "</invoke>",
            "</tool_call>",
        )
        try:
            calls = (
                MinimaxM3Detector().parse_streaming_increment(text, self.tools).calls
            )
        except ValueError:
            calls = []
        self.assertNotIn("wrong", "".join(c.parameters or "" for c in calls))

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


if __name__ == "__main__":
    unittest.main()
