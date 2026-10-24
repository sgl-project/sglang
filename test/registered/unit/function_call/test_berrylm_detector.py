"""Unit tests of the BerryLM parsers: the XML tool-call detector
(``--tool-call-parser berrylm``) and the ``<think>`` reasoning detector
(``--reasoning-parser berrylm``). CPU only, no model."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.berrylm_detector import BerryLMFunctionCallDetector
from sglang.srt.parser.reasoning_parser import BerryLMDetector, ReasoningParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "base-a-test-cpu")

TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"

WEATHER_CALL = (
    f"{TOOL_CALL_START}\n<function=get_weather>\n"
    "<parameter=city>Tokyo</parameter>\n"
    "<parameter=days>3</parameter>\n"
    f"</function>\n{TOOL_CALL_END}"
)


def _make_tool(name="get_weather"):
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City"},
                    "days": {"type": "integer", "description": "Forecast days"},
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["city"],
            },
        ),
    )


def _collect_streaming_tool_calls(detector, chunks, tools):
    """Run streaming chunks through a detector and collect assembled tool calls."""
    tool_calls = []
    all_normal_text = ""
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        all_normal_text += result.normal_text
        for tc_chunk in result.calls:
            if tc_chunk.tool_index is not None:
                while len(tool_calls) <= tc_chunk.tool_index:
                    tool_calls.append({"name": "", "parameters": ""})
                tc = tool_calls[tc_chunk.tool_index]
                if tc_chunk.name:
                    tc["name"] = tc_chunk.name
                if tc_chunk.parameters:
                    tc["parameters"] += tc_chunk.parameters
    return tool_calls, all_normal_text


class TestBerryLMFunctionCallDetector(unittest.TestCase):
    def setUp(self):
        self.detector = BerryLMFunctionCallDetector()
        self.tools = [_make_tool(), _make_tool("write_file")]

    def test_has_tool_call(self):
        self.assertTrue(self.detector.has_tool_call(WEATHER_CALL))
        self.assertFalse(self.detector.has_tool_call("Just an answer."))

    def test_single_tool_call_typed_arguments(self):
        result = self.detector.detect_and_parse(WEATHER_CALL, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        # ``days`` is an integer in the tool schema: converted from the XML text
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"city": "Tokyo", "days": 3}
        )
        self.assertEqual(result.normal_text.strip(), "")

    def test_multiple_tool_calls(self):
        text = WEATHER_CALL + (
            f"\n{TOOL_CALL_START}\n<function=get_weather>\n"
            "<parameter=city>Osaka</parameter>\n"
            f"</function>\n{TOOL_CALL_END}"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual([c.name for c in result.calls], ["get_weather", "get_weather"])
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"city": "Tokyo", "days": 3}
        )
        self.assertEqual(json.loads(result.calls[1].parameters), {"city": "Osaka"})

    def test_normal_text_before_tool_call(self):
        result = self.detector.detect_and_parse(
            "Checking the forecast.\n" + WEATHER_CALL, self.tools
        )
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.normal_text.strip(), "Checking the forecast.")

    def test_no_tool_call(self):
        result = self.detector.detect_and_parse("Just an answer.", self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "Just an answer.")

    def test_multiline_value(self):
        text = (
            f"{TOOL_CALL_START}\n<function=write_file>\n"
            "<parameter=path>/tmp/x.py</parameter>\n"
            "<parameter=content>\nline 1\nline 2\n</parameter>\n"
            f"</function>\n{TOOL_CALL_END}"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls[0].name, "write_file")
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"path": "/tmp/x.py", "content": "line 1\nline 2"},
        )

    def test_streaming_single_tool_call(self):
        chunks = [
            f"{TOOL_CALL_START}\n",
            "<function=get_weather>\n",
            "<parameter=city>Tok",
            "yo</parameter>\n",
            "<parameter=days>3</parameter>\n",
            "</function>\n",
            f"{TOOL_CALL_END}",
        ]
        calls, normal = _collect_streaming_tool_calls(
            BerryLMFunctionCallDetector(), chunks, self.tools
        )
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "get_weather")
        self.assertEqual(
            json.loads(calls[0]["parameters"]), {"city": "Tokyo", "days": 3}
        )
        self.assertEqual(normal.strip(), "")

    def test_streaming_multiple_tool_calls(self):
        chunks = [
            f"{TOOL_CALL_START}\n<function=get_weather>\n",
            "<parameter=city>Tokyo</parameter>\n</function>\n",
            f"{TOOL_CALL_END}\n{TOOL_CALL_START}\n",
            "<function=get_weather>\n<parameter=city>Osaka</parameter>\n",
            f"</function>\n{TOOL_CALL_END}",
        ]
        calls, _ = _collect_streaming_tool_calls(
            BerryLMFunctionCallDetector(), chunks, self.tools
        )
        self.assertEqual([c["name"] for c in calls], ["get_weather", "get_weather"])
        self.assertEqual(json.loads(calls[0]["parameters"]), {"city": "Tokyo"})
        self.assertEqual(json.loads(calls[1]["parameters"]), {"city": "Osaka"})

    def test_streaming_plain_text_passes_through(self):
        det = BerryLMFunctionCallDetector()
        normal = ""
        for chunk in ["Just ", "an answer", " with a < sign."]:
            normal += det.parse_streaming_increment(chunk, self.tools).normal_text
        self.assertEqual(normal, "Just an answer with a < sign.")

    def test_registered_in_function_call_parser(self):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        self.assertIs(
            FunctionCallParser.ToolCallParserEnum["berrylm"],
            BerryLMFunctionCallDetector,
        )


class TestBerryLMReasoningDetector(unittest.TestCase):
    def test_registered(self):
        parser = ReasoningParser(model_type="berrylm")
        self.assertIsInstance(parser.detector, BerryLMDetector)

    def test_think_then_content(self):
        det = BerryLMDetector()
        result = det.detect_and_parse(
            "<think>Plan the answer.</think>The answer is 42."
        )
        self.assertEqual(result.reasoning_text, "Plan the answer.")
        self.assertEqual(result.normal_text, "The answer is 42.")

    def test_template_opened_think(self):
        """The chat template ends the prompt with ``<think>``, so the generated
        text starts inside the reasoning block; the server passes
        ``force_reasoning=True`` for such templates (template manager)."""
        det = BerryLMDetector(force_reasoning=True)
        result = det.detect_and_parse("Plan the answer.</think>The answer is 42.")
        self.assertEqual(result.reasoning_text, "Plan the answer.")
        self.assertEqual(result.normal_text, "The answer is 42.")

    def test_no_forced_reasoning_is_plain_content(self):
        """Without ``force_reasoning`` (thinking off: the template closes an
        empty think block in the prompt) text without ``<think>`` is content."""
        det = BerryLMDetector()
        result = det.detect_and_parse("The answer is 42.")
        self.assertEqual(result.reasoning_text, "")
        self.assertEqual(result.normal_text, "The answer is 42.")

    def test_tool_call_without_think_end(self):
        """The model may open ``<tool_call>`` without closing ``</think>``: the
        tool call must reach the tool parser, never the reasoning text."""
        det = BerryLMDetector()
        result = det.detect_and_parse("<think>I need the forecast." + WEATHER_CALL)
        self.assertNotIn(TOOL_CALL_START, result.reasoning_text)
        self.assertIn("I need the forecast.", result.reasoning_text)
        self.assertNotIn("<think>", result.reasoning_text)
        self.assertTrue(result.normal_text.startswith(TOOL_CALL_START))

    def _run_streaming(self, chunks, **kwargs):
        det = BerryLMDetector(**kwargs)
        reasoning, normal = "", ""
        for chunk in chunks:
            r = det.parse_streaming_increment(chunk)
            reasoning += r.reasoning_text
            normal += r.normal_text
        return reasoning, normal

    def test_streaming_think_then_content(self):
        reasoning, normal = self._run_streaming(
            ["<think>", "Plan the ", "answer.", "</think>", "The answer", " is 42."]
        )
        self.assertEqual(reasoning, "Plan the answer.")
        self.assertEqual(normal, "The answer is 42.")

    def test_streaming_template_opened_think(self):
        reasoning, normal = self._run_streaming(
            ["Plan the ", "answer.", "</think>", "The answer", " is 42."],
            force_reasoning=True,
        )
        self.assertEqual(reasoning, "Plan the answer.")
        self.assertEqual(normal, "The answer is 42.")

    def test_streaming_tool_call_without_think_end(self):
        reasoning, normal = self._run_streaming(
            [
                "<think>",
                "I need the forecast.",
                TOOL_CALL_START,
                "\n<function=get_weather>",
            ]
        )
        self.assertNotIn(TOOL_CALL_START, reasoning)
        self.assertIn("I need the forecast.", reasoning)
        self.assertIn(TOOL_CALL_START, normal)


if __name__ == "__main__":
    unittest.main()
