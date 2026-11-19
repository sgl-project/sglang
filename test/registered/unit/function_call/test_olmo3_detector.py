"""Unit tests for Olmo3Detector - no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.olmo3_detector import Olmo3Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _tool(name: str) -> Tool:
    return Tool(
        type="function",
        function=Function(name=name, parameters={"type": "object", "properties": {}}),
    )


TOOLS = [_tool(name) for name in ("get_weather", "register_user", "write_file")]


def _parse(text: str):
    result = Olmo3Detector().detect_and_parse(text=text, tools=TOOLS)
    return result.normal_text, [
        (c.name, json.loads(c.parameters)) for c in result.calls
    ]


def _stream(text: str, chunk_size: int):
    detector = Olmo3Detector()
    normal, calls, indices = "", [], []
    results = [
        detector.parse_streaming_increment(
            new_text=text[i : i + chunk_size], tools=TOOLS
        )
        for i in range(0, len(text), chunk_size)
    ]
    results.append(detector.finish(TOOLS))
    for result in results:
        normal += result.normal_text
        for call in result.calls:
            calls.append((call.name, json.loads(call.parameters)))
            indices.append(call.tool_index)
    return normal, calls, indices, detector


class TestOlmo3Detector(CustomTestCase):
    def assertParses(self, text, expected_calls, expected_normal=""):
        """Check non-streaming, then streaming at several chunk sizes."""
        self.assertEqual(_parse(text), (expected_normal, expected_calls))
        for chunk_size in (1, 3, 7, len(text) or 1):
            with self.subTest(chunk_size=chunk_size):
                normal, calls, indices, detector = _stream(
                    text=text, chunk_size=chunk_size
                )
                self.assertEqual((normal, calls), (expected_normal, expected_calls))
                self.assertEqual(indices, list(range(len(expected_calls))))
                self.assertEqual(
                    [json.loads(args) for args in detector.streamed_args_for_tool],
                    [args for _, args in expected_calls],
                )
                self.assertEqual(
                    detector.prev_tool_call_arr,
                    [{"name": n, "arguments": a} for n, a in expected_calls],
                )

    def test_registered(self):
        self.assertIs(FunctionCallParser.ToolCallParserEnum["olmo3"], Olmo3Detector)

    def test_text_without_a_block_is_unchanged(self):
        for text in (
            "How can I help you today?",
            "<p>hello",
            "<b>bold</b>",
            "<function> is an HTML-like tag, not a tool call.",
            "ends with a partial tag <function_ca",
        ):
            with self.subTest(text=text):
                self.assertFalse(Olmo3Detector().has_tool_call(text))
                self.assertParses(text, [], text)

    def test_simple_call(self):
        self.assertParses(
            "<function_calls>get_weather(city='San Francisco', metric='celsius')"
            "</function_calls>",
            [("get_weather", {"city": "San Francisco", "metric": "celsius"})],
        )

    def test_python_and_json_literals(self):
        expected = {
            "name": "John Doe",
            "age": 37,
            "address": {"city": "San Francisco", "state": "CA"},
            "role": None,
            "passed_test": True,
            "failed": False,
            "aliases": ["John", "Johnny"],
            "offset": -2.5,
        }
        for role, passed, failed in (
            ("None", "True", "False"),
            ("null", "true", "false"),
        ):
            with self.subTest(literals=role):
                self.assertParses(
                    "<function_calls>register_user(name='John Doe', age=37, "
                    "address={'city': 'San Francisco', 'state': 'CA'}, "
                    f"role={role}, passed_test={passed}, failed={failed}, "
                    "aliases=['John', 'Johnny'], offset=-2.5)</function_calls>",
                    [("register_user", expected)],
                )

    def test_parameterless_and_empty_containers(self):
        self.assertParses(
            "<function_calls>get_weather()\nregister_user(data={}, steps=[])"
            "</function_calls>",
            [("get_weather", {}), ("register_user", {"data": {}, "steps": []})],
        )

    def test_escaped_strings(self):
        self.assertParses(
            r"""<function_calls>get_weather(city='Martha\'s Vineyard', metric='"cool units"')</function_calls>""",
            [("get_weather", {"city": "Martha's Vineyard", "metric": '"cool units"'})],
        )

    def test_parallel_calls_with_text_around(self):
        self.assertParses(
            "Let me check.\n<function_calls>\n"
            'get_weather(city="Paris")\n'
            'get_weather(city="Tokyo")\n'
            "</function_calls>Done.",
            [("get_weather", {"city": "Paris"}), ("get_weather", {"city": "Tokyo"})],
            "Let me check.\nDone.",
        )

    def test_multiline_string_and_call_spanning_lines(self):
        self.assertParses(
            "<function_calls>write_file(\n"
            '    path="a.py",\n'
            '    content="def f():\\n    return 1\\n",\n'
            '    notes="""line one\nline two""",\n'
            ")\nget_weather(city='Paris')</function_calls>",
            [
                (
                    "write_file",
                    {
                        "path": "a.py",
                        "content": "def f():\n    return 1\n",
                        "notes": "line one\nline two",
                    },
                ),
                ("get_weather", {"city": "Paris"}),
            ],
        )

    def test_closing_tag_inside_a_string_argument(self):
        self.assertParses(
            "<function_calls>write_file(path='x.txt', "
            "content='<function_calls>a()</function_calls>')</function_calls>after",
            [
                (
                    "write_file",
                    {
                        "path": "x.txt",
                        "content": "<function_calls>a()</function_calls>",
                    },
                )
            ],
            "after",
        )

    def test_indented_calls(self):
        self.assertParses(
            "<function_calls>\n  get_weather(city='Paris')\n  get_weather(city='Rome')\n"
            "</function_calls>",
            [("get_weather", {"city": "Paris"}), ("get_weather", {"city": "Rome"})],
        )

    def test_unclosed_block_at_end_of_output(self):
        self.assertParses(
            "<function_calls>get_weather(city='Paris')",
            [("get_weather", {"city": "Paris"})],
        )

    def test_truncated_or_invalid_block_stays_text(self):
        for text in (
            "<function_calls>get_weather(city='Par",
            "<function_calls>print('hi'); x = 1</function_calls>tail",
            "<function_calls>just some prose</function_calls>",
            "<function_calls>get_weather('positional')</function_calls>",
        ):
            with self.subTest(text=text):
                self.assertParses(text, [], text)

    def test_unknown_tool_and_unconvertible_arguments_skip_only_that_call(self):
        self.assertParses(
            "<function_calls>delete_everything(force=true)\n"
            "get_weather(city=some_variable)\n"
            "get_weather(city=1e999)\n"
            "get_weather(city='Paris')</function_calls>",
            [("get_weather", {"city": "Paris"})],
        )

    def test_two_blocks(self):
        self.assertParses(
            "<function_calls>get_weather(city='Paris')</function_calls>"
            " and "
            "<function_calls>get_weather(city='Rome')</function_calls>",
            [("get_weather", {"city": "Paris"}), ("get_weather", {"city": "Rome"})],
            " and ",
        )

    def test_parser_end_to_end(self):
        parser = FunctionCallParser(tools=TOOLS, tool_call_parser="olmo3")
        text = "Sure.<function_calls>get_weather(city='Paris', days=3)</function_calls>"
        self.assertTrue(parser.has_tool_call(text))
        normal, calls = parser.parse_non_stream(text)
        self.assertEqual(normal, "Sure.")
        self.assertEqual(
            [(c.name, json.loads(c.parameters)) for c in calls],
            [("get_weather", {"city": "Paris", "days": 3})],
        )


if __name__ == "__main__":
    unittest.main()
