"""Unit tests for MultiFormatDetector."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.hermes_detector import HermesDetector
from sglang.srt.function_call.multi_format_detector import MultiFormatDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "stage-a-test-cpu")


def _make_tool(name, parameters=None):
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters=parameters or {
                "type": "object",
                "properties": {"x": {"type": "string"}},
            },
        ),
    )


class TestMultiFormatDispatch(unittest.TestCase):
    """MultiFormatDetector picks the right inner parser based on tool_format."""

    def test_default_dialect_delegates_to_hermes(self):
        det = MultiFormatDetector(tool_format="default")
        self.assertIsInstance(det._delegate, HermesDetector)

    def test_missing_tool_format_defaults_to_default(self):
        det = MultiFormatDetector()
        self.assertIsInstance(det._delegate, HermesDetector)

    def test_unknown_dialect_errors(self):
        with self.assertRaisesRegex(ValueError, "Unsupported tool_format"):
            MultiFormatDetector(tool_format="not-a-dialect")

    def test_embedded_dialects_set_delegate_to_none(self):
        for dialect in ("minimax", "dsv32", "glm", "gptoss", "python"):
            det = MultiFormatDetector(tool_format=dialect)
            self.assertIsNone(det._delegate, f"{dialect} should have no delegate")
            self.assertEqual(det.tool_format, dialect)

    def test_qwen3_dialect_delegates_to_qwen3coder(self):
        from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector

        det = MultiFormatDetector(tool_format="qwen3")
        self.assertIsInstance(det._delegate, Qwen3CoderDetector)


class TestMinimaxDialect(unittest.TestCase):
    def setUp(self):
        self.tools = [_make_tool("get_weather")]
        self.det = MultiFormatDetector(tool_format="minimax")

    def test_single_call(self):
        text = (
            'prefix<tool_calls>'
            '<invoke name="get_weather">'
            '<parameter name="x">NYC</parameter>'
            '</invoke></tool_calls>'
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"x": "NYC"})
        self.assertEqual(result.normal_text, "prefix")

    def test_numeric_value_parses_as_int_via_json_fallback(self):
        text = (
            '<tool_calls><invoke name="get_weather">'
            '<parameter name="x">42</parameter>'
            '</invoke></tool_calls>'
        )
        result = self.det.detect_and_parse(text, self.tools)
        # Minimax dialect: JSON parse first, fall back to string.
        self.assertEqual(json.loads(result.calls[0].parameters), {"x": 42})

    def test_no_tool_call_returns_full_text(self):
        result = self.det.detect_and_parse("plain text", self.tools)
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, "plain text")


class TestQwen3Dialect(unittest.TestCase):
    def setUp(self):
        self.tools = [
            _make_tool(
                "get_weather",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                },
            ),
            _make_tool(
                "get_time",
                {
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string"},
                    },
                },
            ),
        ]
        self.det = MultiFormatDetector(tool_format="qwen3")

    def test_bare_function_without_tool_call_wrapper(self):
        text = (
            "<function=get_weather>\n"
            "<parameter=city>SF</parameter>\n"
            "<parameter=days>3</parameter>\n"
            "</function>"
        )

        self.assertTrue(self.det.has_tool_call(text))
        result = self.det.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"city": "SF", "days": 3},
        )

    def test_preserves_text_between_multiple_tool_blocks(self):
        text = (
            "pre"
            "<tool_call><function=get_weather>"
            "<parameter=city>SF</parameter>"
            "</function></tool_call>"
            "middle"
            "<tool_call><function=get_time>"
            "<parameter=timezone>UTC</parameter>"
            "</function></tool_call>"
            "post"
        )

        result = self.det.detect_and_parse(text, self.tools)

        self.assertEqual(result.normal_text, "premiddle")
        self.assertEqual(
            [call.name for call in result.calls], ["get_weather", "get_time"]
        )
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "SF"})
        self.assertEqual(json.loads(result.calls[1].parameters), {"timezone": "UTC"})


class TestDsv32Dialect(unittest.TestCase):
    def setUp(self):
        self.tools = [_make_tool("get_weather")]
        self.det = MultiFormatDetector(tool_format="dsv32")

    def test_string_flag_forces_string(self):
        text = (
            '<tool_calls><invoke name="get_weather">'
            '<parameter name="x" string="true">42</parameter>'
            '</invoke></tool_calls>'
        )
        result = self.det.detect_and_parse(text, self.tools)
        # With string="true", the value 42 must remain "42" (a string), not become int 42.
        self.assertEqual(json.loads(result.calls[0].parameters), {"x": "42"})

    def test_no_string_flag_parses_as_json(self):
        text = (
            '<tool_calls><invoke name="get_weather">'
            '<parameter name="x">42</parameter>'
            '</invoke></tool_calls>'
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(json.loads(result.calls[0].parameters), {"x": 42})

    def test_string_false_is_equivalent_to_absent_flag(self):
        """string="false" must take the same branch as absent — JSON-parse."""
        text = (
            '<tool_calls><invoke name="get_weather">'
            '<parameter name="x" string="false">42</parameter>'
            '</invoke></tool_calls>'
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(json.loads(result.calls[0].parameters), {"x": 42})

    def test_unknown_tool_is_forwarded(self):
        text = (
            '<tool_calls><invoke name="not_a_registered_tool">'
            '<parameter name="x">NYC</parameter>'
            '</invoke></tool_calls>'
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].tool_index, -1)
        self.assertEqual(result.calls[0].name, "not_a_registered_tool")
        self.assertEqual(json.loads(result.calls[0].parameters), {"x": "NYC"})


class TestDsv32VsDeepSeekV32Disjoint(unittest.TestCase):
    """Confirms that the ported 'dsv32' dialect does not collide with the
    existing DeepSeekV32Detector. Both must parse only their own wire format."""

    def test_ported_dsv32_ignores_dsml_markers(self):
        det = MultiFormatDetector(tool_format="dsv32")
        dsml_text = (
            "<｜DSML｜function_calls>"
            '<｜DSML｜invoke name="x"><｜DSML｜parameter name="a" string="true">1</｜DSML｜parameter>'
            "</｜DSML｜invoke></｜DSML｜function_calls>"
        )
        result = det.detect_and_parse(dsml_text, [_make_tool("x")])
        self.assertEqual(result.calls, [])

    def test_existing_dsv32_ignores_plain_tool_calls(self):
        from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector

        det = DeepSeekV32Detector()
        plain_text = (
            '<tool_calls><invoke name="x">'
            '<parameter name="a" string="true">1</parameter></invoke></tool_calls>'
        )
        result = det.detect_and_parse(plain_text, [_make_tool("x")])
        self.assertEqual(result.calls, [])


class TestGlmDialect(unittest.TestCase):
    def setUp(self):
        self.tools = [
            _make_tool(
                "get_weather",
                {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "days": {"type": "integer"},
                    },
                },
            )
        ]
        self.det = MultiFormatDetector(tool_format="glm")

    def test_basic_call(self):
        text = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>NYC</arg_value>"
            "<arg_key>days</arg_key><arg_value>3</arg_value>"
            "</tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        # city is declared "string", so "NYC" stays a string.
        # days is not "string", so "3" is deserialized to int 3.
        self.assertEqual(args["city"], "NYC")
        self.assertEqual(args["days"], 3)

    def test_no_arg_keys(self):
        text = "<tool_call>get_weather</tool_call>"
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {}.copy())

    def test_python_tuple_literal_value(self):
        text = (
            "<tool_call>get_weather"
            "<arg_key>days</arg_key><arg_value>(1, 2)</arg_value>"
            "</tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        args = json.loads(result.calls[0].parameters)
        # ast.literal handles tuple syntax; serialized as a JSON array.
        self.assertEqual(args["days"], [1, 2])


class TestGptOssDialect(unittest.TestCase):
    def setUp(self):
        self.tools = [_make_tool("get_weather")]
        self.det = MultiFormatDetector(tool_format="gptoss")

    def test_basic_call(self):
        text = (
            "Some preamble.\n"
            '<tool_call>assistant to=functions.get_weather json\n'
            '{"city": "NYC"}\n'
            "</tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "NYC"})
        self.assertEqual(result.normal_text, "Some preamble.\n")

    def test_call_without_assistant_prefix(self):
        text = (
            '<tool_call>to=functions.get_weather\n'
            '{"city": "NYC"}\n'
            "</tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls[0].name, "get_weather")

    def test_malformed_json_is_rejected(self):
        text = (
            '<tool_call>assistant to=functions.get_weather json\n'
            'not json\n'
            "</tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls, [])

    def test_unknown_tool_is_forwarded(self):
        text = (
            '<tool_call>assistant to=functions.unregistered_fn json\n'
            '{"x": 1}\n'
            "</tool_call>"
        )
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].tool_index, -1)
        self.assertEqual(result.calls[0].name, "unregistered_fn")
        self.assertEqual(json.loads(result.calls[0].parameters), {"x": 1})


class TestPythonDialect(unittest.TestCase):
    def setUp(self):
        self.tools = [_make_tool("get_weather")]
        self.det = MultiFormatDetector(tool_format="python")

    def test_basic_call(self):
        text = '<tool_call>get_weather(city="NYC", days=3)</tool_call>'
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"city": "NYC", "days": 3}
        )

    def test_python_literal_dict_value(self):
        text = '<tool_call>get_weather(x={"a": 1, "b": [2, 3]})</tool_call>'
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"x": {"a": 1, "b": [2, 3]}}
        )

    def test_python_lowercase_true_false_none(self):
        text = '<tool_call>get_weather(a=true, b=false, c=null)</tool_call>'
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"a": True, "b": False, "c": None},
        )

    def test_rejects_arbitrary_expression(self):
        # 2+2 is an ast.BinOp, not a literal; AST walker must reject.
        text = '<tool_call>get_weather(x=2+2)</tool_call>'
        result = self.det.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls, [])


class TestUnknownToolForwarding(unittest.TestCase):
    def setUp(self):
        self.tools = [_make_tool("get_weather")]

    def test_embedded_dialects_forward_unknown_tool_names(self):
        cases = {
            "minimax": (
                '<tool_calls><invoke name="not_registered">'
                '<parameter name="x">NYC</parameter>'
                "</invoke></tool_calls>"
            ),
            "dsv32": (
                '<tool_calls><invoke name="not_registered">'
                '<parameter name="x" string="true">NYC</parameter>'
                "</invoke></tool_calls>"
            ),
            "glm": (
                "<tool_call>not_registered"
                "<arg_key>x</arg_key><arg_value>NYC</arg_value>"
                "</tool_call>"
            ),
            "gptoss": (
                '<tool_call>assistant to=functions.not_registered json\n'
                '{"x": "NYC"}\n'
                "</tool_call>"
            ),
            "python": '<tool_call>not_registered(x="NYC")</tool_call>',
        }

        for dialect, text in cases.items():
            with self.subTest(dialect=dialect):
                result = MultiFormatDetector(tool_format=dialect).detect_and_parse(
                    text, self.tools
                )
                self.assertEqual(len(result.calls), 1)
                self.assertEqual(result.calls[0].tool_index, -1)
                self.assertEqual(result.calls[0].name, "not_registered")
                self.assertEqual(json.loads(result.calls[0].parameters), {"x": "NYC"})


class TestFunctionCallParserPlumbing(unittest.TestCase):
    """FunctionCallParser must forward chat_template_kwargs to MultiFormatDetector
    and ignore the kwarg for detectors that don't accept it."""

    def test_multi_format_receives_tool_format_from_kwargs(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        tools = [_make_tool("get_weather")]
        parser = FunctionCallParser(
            tools=tools,
            tool_call_parser="multi_format",
            chat_template_kwargs={"tool_format": "minimax"},
        )
        self.assertEqual(parser.detector.tool_format, "minimax")

    def test_other_detectors_unaffected(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        from sglang.srt.function_call.kimik2_detector import KimiK2Detector

        tools = [_make_tool("get_weather")]
        # Existing kimi_k2 detector takes no args; passing kwargs must not crash.
        parser = FunctionCallParser(
            tools=tools,
            tool_call_parser="kimi_k2",
            chat_template_kwargs={"tool_format": "minimax"},
        )
        self.assertIsInstance(parser.detector, KimiK2Detector)


class TestMultiFormatEndToEnd(unittest.TestCase):
    def test_full_pipeline_minimax(self):
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        tools = [_make_tool("get_weather")]
        req = ChatCompletionRequest(
            model="ifm-xllm",
            messages=[{"role": "user", "content": "hi"}],
            tools=[t.model_dump() for t in tools],
            chat_template_kwargs={"tool_format": "minimax"},
        )

        parser = FunctionCallParser(
            tools=tools,
            tool_call_parser="multi_format",
            chat_template_kwargs=req.chat_template_kwargs,
        )

        wire_output = (
            "Some preamble. "
            '<tool_calls><invoke name="get_weather">'
            '<parameter name="x">NYC</parameter>'
            "</invoke></tool_calls>"
        )

        normal_text, calls = parser.parse_non_stream(wire_output)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "get_weather")
        self.assertEqual(json.loads(calls[0].parameters), {"x": "NYC"})
        self.assertIn("Some preamble.", normal_text)


if __name__ == "__main__":
    unittest.main()
