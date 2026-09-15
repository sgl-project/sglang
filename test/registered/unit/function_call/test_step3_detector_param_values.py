"""Unit tests for Step3Detector parameter-value parsing - no server, no model loading.

Regression tests for parameter values that contain '<' (code snippets,
comparisons, HTML/XML fragments), which the previous ``[^<]*`` value regex
silently dropped in both one-shot and streaming parsing.
"""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.step3_detector import Step3Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _make_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="run_code",
                description="Run a code snippet",
                parameters={
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Code to run"},
                        "lang": {"type": "string", "description": "Language"},
                    },
                    "required": ["code"],
                },
            ),
        ),
    ]


def _wrap_tool_call(detector, invoke_body):
    return (
        f"{detector.bot_token}{detector.tool_call_begin}"
        f"function{detector.tool_sep}{invoke_body}"
        f"{detector.tool_call_end}{detector.eot_token}"
    )


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


class TestStep3DetectorParamValuesWithLt(CustomTestCase):
    def setUp(self):
        self.detector = Step3Detector()
        self.tools = _make_tools()

    def test_detect_and_parse_value_containing_lt(self):
        invoke_body = (
            '<steptml:invoke name="run_code">'
            '<steptml:parameter name="code">if a < b: print(a)</steptml:parameter>'
            '<steptml:parameter name="lang">python</steptml:parameter>'
            "</steptml:invoke>"
        )
        text = _wrap_tool_call(self.detector, invoke_body)
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "run_code")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["code"], "if a < b: print(a)")
        self.assertEqual(args["lang"], "python")

    def test_detect_and_parse_multiline_value_with_lt(self):
        invoke_body = (
            '<steptml:invoke name="run_code">'
            '<steptml:parameter name="code">x = 1\nif x < 2:\n    print(x)</steptml:parameter>'
            "</steptml:invoke>"
        )
        text = _wrap_tool_call(self.detector, invoke_body)
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["code"], "x = 1\nif x < 2:\n    print(x)")

    def test_detect_and_parse_multi_param_pairing_unchanged(self):
        # Non-greedy matching must not swallow the boundary between
        # adjacent parameters.
        invoke_body = (
            '<steptml:invoke name="run_code">'
            '<steptml:parameter name="code">print(1)</steptml:parameter>'
            '<steptml:parameter name="lang">python</steptml:parameter>'
            "</steptml:invoke>"
        )
        text = _wrap_tool_call(self.detector, invoke_body)
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertEqual(len(result.calls), 1)
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args, {"code": "print(1)", "lang": "python"})

    def test_streaming_value_containing_lt(self):
        invoke_body = (
            '<steptml:invoke name="run_code">'
            '<steptml:parameter name="code">if a < b: print(a)</steptml:parameter>'
            '<steptml:parameter name="lang">python</steptml:parameter>'
            "</steptml:invoke>"
        )
        text = _wrap_tool_call(self.detector, invoke_body)

        all_calls = []
        chunk_size = 8
        for i in range(0, len(text), chunk_size):
            result = self.detector.parse_streaming_increment(
                text[i : i + chunk_size], self.tools
            )
            all_calls.extend(result.calls)

        collected = _collect_streamed_tool_calls(all_calls)
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0]["name"], "run_code")
        args = json.loads(collected[0]["parameters"])
        self.assertEqual(args["code"], "if a < b: print(a)")
        self.assertEqual(args["lang"], "python")


if __name__ == "__main__":
    unittest.main()
