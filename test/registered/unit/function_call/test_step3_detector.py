"""Unit tests for Step3Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.step3_detector import Step3Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")


def _tool_call(name, arguments):
    parameters = "".join(
        f'<steptml:parameter name="{key}">{value}</steptml:parameter>'
        for key, value in arguments.items()
    )
    return (
        "<｜tool_call_begin｜>function<｜tool_sep｜>"
        f'<steptml:invoke name="{name}">{parameters}</steptml:invoke>'
        "<｜tool_call_end｜>"
    )


FIRST = _tool_call("get_weather", {"city": "Beijing"})
SECOND = _tool_call("search", {"query": "news"})


class TestStep3Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
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
                    name="search",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                ),
            ),
        ]

    def test_detect_and_parse_multiple_tool_calls(self):
        text = f"<｜tool_calls_begin｜>\n{FIRST}\n{SECOND}\n<｜tool_calls_end｜>"
        result = Step3Detector().detect_and_parse(text, self.tools)
        self.assertEqual(
            [(c.name, json.loads(c.parameters)) for c in result.calls],
            [("get_weather", {"city": "Beijing"}), ("search", {"query": "news"})],
        )

    def test_streaming_block_end_with_unparsed_tool_calls(self):
        # The end of the tool block arrives while whole tool calls are still
        # in the buffer.
        detector = Step3Detector()
        chunks = [
            f"<｜tool_calls_begin｜>\n{FIRST}",
            f"\n{SECOND}\n<｜tool_calls_end｜>",
        ]
        normal_text, names, params = "", [], {}
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal_text += result.normal_text
            for call in result.calls:
                if call.name:
                    names.append(call.name)
                params[call.tool_index] = params.get(call.tool_index, "") + (
                    call.parameters or ""
                )

        self.assertEqual(normal_text, "")
        self.assertEqual(names, ["get_weather", "search"])
        self.assertEqual(
            [json.loads(params[i]) for i in sorted(params)],
            [{"city": "Beijing"}, {"query": "news"}],
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
