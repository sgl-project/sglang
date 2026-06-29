"""Unit tests for GigaChat3Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.gigachat3_detector import GigaChat3Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")

ROLE_SEP = "<|role_sep|>"
MSG_SEP = "<|message_sep|>"
FUNC_CALL_TOKEN = "<|function_call|>"


def _role_sep_call(name: str, args: dict) -> str:
    return f"function call{ROLE_SEP}\n" + json.dumps({"name": name, "arguments": args})


def _func_call_token_call(name: str, args: dict) -> str:
    return FUNC_CALL_TOKEN + json.dumps({"name": name, "arguments": args})


class TestGigaChat3Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
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
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                ),
            ),
        ]
        self.detector = GigaChat3Detector()

    def test_has_tool_call_role_sep(self):
        self.assertTrue(
            self.detector.has_tool_call(
                _role_sep_call("get_weather", {"city": "Tokyo"})
            )
        )

    def test_has_tool_call_function_call_token(self):
        self.assertTrue(
            self.detector.has_tool_call(
                _func_call_token_call("get_weather", {"city": "Tokyo"})
            )
        )

    def test_has_tool_call_false(self):
        self.assertFalse(self.detector.has_tool_call("The weather is fine today."))

    def test_single_call_role_sep(self):
        text = _role_sep_call("get_weather", {"city": "Tokyo"})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["city"], "Tokyo")

    def test_single_call_function_call_token(self):
        text = _func_call_token_call("search", {"query": "python"})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args["query"], "python")

    def test_no_tool_call(self):
        text = "The weather is fine today."
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertIn("weather", result.normal_text)

    def test_content_before_call(self):
        content = "Sure, let me check."
        text = content + MSG_SEP + _role_sep_call("get_weather", {"city": "Paris"})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.normal_text, content)

    def test_strips_eos_token(self):
        text = _role_sep_call("get_weather", {"city": "Berlin"}) + "</s>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")

    def test_malformed_json_returns_text(self):
        text = f"function call{ROLE_SEP}\nnot valid json"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertIn("not valid json", result.normal_text)

    def test_missing_name_field_returns_empty(self):
        text = f"function call{ROLE_SEP}\n" + json.dumps(
            {"arguments": {"city": "Tokyo"}}
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_streaming_plain_text(self):
        detector = GigaChat3Detector()
        result = detector.parse_streaming_increment("Hello world. ", self.tools)
        self.assertEqual(result.normal_text, "Hello world. ")
        self.assertEqual(len(result.calls), 0)

    def test_streaming_tool_name_sent(self):
        detector = GigaChat3Detector()
        payload = json.dumps({"name": "get_weather", "arguments": {"city": "Oslo"}})
        chunks = [
            f"function call{ROLE_SEP}\n",
            payload[:10],
            payload[10:],
        ]
        all_calls = []
        for chunk in chunks:
            r = detector.parse_streaming_increment(chunk, self.tools)
            all_calls.extend(r.calls)

        named = [c for c in all_calls if c.name]
        self.assertEqual(len(named), 1)
        self.assertEqual(named[0].name, "get_weather")

    def test_streaming_args_delta(self):
        detector = GigaChat3Detector()
        payload = json.dumps({"name": "search", "arguments": {"query": "sglang"}})
        full_text = f"function call{ROLE_SEP}\n{payload}"
        all_calls = []
        for char in full_text:
            r = detector.parse_streaming_increment(char, self.tools)
            all_calls.extend(r.calls)

        named = [c for c in all_calls if c.name]
        self.assertEqual(len(named), 1)
        self.assertEqual(named[0].name, "search")

    def test_supports_structural_tag_false(self):
        self.assertFalse(self.detector.supports_structural_tag())

    def test_structure_info_raises(self):
        with self.assertRaises(NotImplementedError):
            self.detector.structure_info()


if __name__ == "__main__":
    import unittest

    unittest.main()
