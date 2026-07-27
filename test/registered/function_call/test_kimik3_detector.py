import json
import sys
import unittest

import pytest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.kimik3_detector import (
    KimiK3Detector as KimiK3FuncDetector,
)
from sglang.srt.function_call.kimik3_format import (
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    TOOLS_CLOSE,
    TOOLS_OPEN,
)
from sglang.srt.parser.reasoning_parser import KimiK3Detector as KimiK3ReasoningDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_tool(name):
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string"}},
            },
        ),
    )


def _call_block(tool, index, args):
    parts = [f'<|open|>call tool="{tool}" index="{index}"<|sep|>']
    for key, (arg_type, value) in args.items():
        parts.append(
            f'<|open|>argument key="{key}" type="{arg_type}"<|sep|>'
            f"{value}<|close|>argument<|sep|>"
        )
    parts.append("<|close|>call<|sep|>")
    return "".join(parts)


def _stream_chunks(text, size):
    return [text[i : i + size] for i in range(0, len(text), size)]


class TestKimiK3FuncDetector(unittest.TestCase):
    def setUp(self):
        self.detector = KimiK3FuncDetector()
        self.tools = [_make_tool("python")]

    def _stream(self, chunks):
        text = ""
        calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            text += result.normal_text
            calls.extend(result.calls)
        return text, calls

    def test_detect_and_parse_single_call(self):
        text = (
            f"{RESPONSE_OPEN}Let me run it.{RESPONSE_CLOSE}"
            f"{TOOLS_OPEN}"
            + _call_block(
                "python",
                1,
                {"code": ("string", "print(1)"), "opts": ("object", '{"a": 1}')},
            )
            + f"{TOOLS_CLOSE}"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "Let me run it.")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "python")
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"code": "print(1)", "opts": {"a": 1}},
        )

    def test_detect_and_parse_no_tools_channel(self):
        text = f"{RESPONSE_OPEN}hi there{RESPONSE_CLOSE}{MESSAGE_CLOSE}"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "hi there")
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_multiple_calls(self):
        text = (
            f"{TOOLS_OPEN}"
            + _call_block("python", 1, {"code": ("string", "a")})
            + _call_block("python", 2, {"code": ("string", "b")})
            + f"{TOOLS_CLOSE}"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].tool_index, 0)
        self.assertEqual(result.calls[1].tool_index, 1)
        self.assertEqual(json.loads(result.calls[1].parameters), {"code": "b"})

    def test_detect_and_parse_unclosed_tools_section(self):
        text = f"{TOOLS_OPEN}" + _call_block("python", 1, {"code": ("string", "x")})
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters), {"code": "x"})

    def test_attr_unescaping_and_raw_string_args(self):
        text = (
            f"{TOOLS_OPEN}"
            '<|open|>call tool="a&amp;b" index="1"<|sep|>'
            '<|open|>argument key="q" type="string"<|sep|>'
            "say &quot;hi&quot;<|close|>argument<|sep|>"
            "<|close|>call<|sep|>"
            f"{TOOLS_CLOSE}"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(result.calls[0].name, "a&b")
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"q": "say &quot;hi&quot;"}
        )

    def test_non_string_arg_json_decoding(self):
        text = (
            f"{TOOLS_OPEN}"
            + _call_block(
                "python",
                1,
                {
                    "n": ("number", "42"),
                    "flag": ("boolean", "true"),
                    "bad": ("object", "{not json"),
                },
            )
            + f"{TOOLS_CLOSE}"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"n": 42, "flag": True, "bad": "{not json"},
        )

    def test_streaming_split_markers(self):
        chunks = [
            "<|open|>",
            "response",
            "<|sep|>Hel",
            "lo!",
            "<|close|>respo",
            "nse<|sep|>",
            "<|open|>tools",
            "<|sep|>",
            '<|open|>call tool="py',
            'thon" index="1"<|sep|>',
            '<|open|>argument key="code" type="string"<|sep|>',
            "print(2)",
            "<|close|>argument<|sep|>",
            "<|close|>call<|sep|>",
            "<|close|>tools<|sep|>",
        ]
        text, calls = self._stream(chunks)
        self.assertEqual(text, "Hello!")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "python")
        self.assertEqual(json.loads(calls[0].parameters), {"code": "print(2)"})

    def test_streaming_two_calls_small_chunks(self):
        text = (
            f"{TOOLS_OPEN}"
            + _call_block("python", 1, {"code": ("string", "a")})
            + _call_block("python", 2, {"code": ("string", "b")})
            + f"{TOOLS_CLOSE}"
        )
        _, calls = self._stream(_stream_chunks(text, 7))
        self.assertEqual(len(calls), 2)
        self.assertEqual([c.tool_index for c in calls], [0, 1])
        self.assertEqual(json.loads(calls[0].parameters), {"code": "a"})
        self.assertEqual(json.loads(calls[1].parameters), {"code": "b"})

    def test_streaming_plain_text_only(self):
        text, calls = self._stream(["just a ", "plain ", "reply"])
        self.assertEqual(text, "just a plain reply")
        self.assertEqual(calls, [])

    def test_streaming_bookkeeping_for_serving_layer(self):
        text = (
            f"{TOOLS_OPEN}"
            + _call_block("python", 1, {"code": ("string", "a")})
            + f"{TOOLS_CLOSE}"
        )
        self._stream(_stream_chunks(text, 9))
        self.assertEqual(self.detector.current_tool_id, 0)
        self.assertEqual(
            self.detector.prev_tool_call_arr[0],
            {"name": "python", "arguments": {"code": "a"}},
        )
        self.assertEqual(
            json.loads(self.detector.streamed_args_for_tool[0]), {"code": "a"}
        )

    def test_has_tool_call(self):
        self.assertTrue(self.detector.has_tool_call(f"x{TOOLS_OPEN}y"))
        self.assertFalse(self.detector.has_tool_call(f"{RESPONSE_OPEN}x"))

    def test_constraint_capabilities(self):
        self.assertTrue(self.detector.supports_structural_tag())
        self.assertFalse(self.detector.parses_required_natively())


class TestKimiK3ReasoningDetector(unittest.TestCase):
    def _stream(self, detector, chunks):
        reasoning = ""
        content = ""
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk)
            reasoning += result.reasoning_text or ""
            content += result.normal_text or ""
        return reasoning, content

    def test_non_stream_full_markers(self):
        detector = KimiK3ReasoningDetector(force_reasoning=True)
        result = detector.detect_and_parse(
            f"{THINK_OPEN}deep thought{THINK_CLOSE}"
            f"{RESPONSE_OPEN}the answer{RESPONSE_CLOSE}{MESSAGE_CLOSE}"
        )
        self.assertEqual(result.reasoning_text, "deep thought")
        self.assertEqual(result.normal_text, "the answer")

    def test_non_stream_consumed_prefix(self):
        detector = KimiK3ReasoningDetector(force_reasoning=True)
        result = detector.detect_and_parse(
            f"thinking...{THINK_CLOSE}{RESPONSE_OPEN}done{RESPONSE_CLOSE}"
        )
        self.assertEqual(result.reasoning_text, "thinking...")
        self.assertEqual(result.normal_text, "done")

    def test_non_stream_thinking_disabled(self):
        detector = KimiK3ReasoningDetector(force_reasoning=False)
        result = detector.detect_and_parse(
            f"{RESPONSE_OPEN}plain reply{RESPONSE_CLOSE}{MESSAGE_CLOSE}"
        )
        self.assertFalse(result.reasoning_text)
        self.assertEqual(result.normal_text, "plain reply")

    def test_non_stream_truncated_reasoning(self):
        detector = KimiK3ReasoningDetector(force_reasoning=True)
        result = detector.detect_and_parse("still going")
        self.assertEqual(result.reasoning_text, "still going")
        self.assertFalse(result.normal_text)

    def test_non_stream_tools_channel_passthrough(self):
        detector = KimiK3ReasoningDetector(force_reasoning=True)
        tools_channel = (
            f"{TOOLS_OPEN}"
            + _call_block("python", 1, {"code": ("string", "1")})
            + f"{TOOLS_CLOSE}"
        )
        result = detector.detect_and_parse(
            f"t{THINK_CLOSE}{RESPONSE_OPEN}r{RESPONSE_CLOSE}{tools_channel}"
        )
        self.assertEqual(result.reasoning_text, "t")
        self.assertEqual(result.normal_text, f"r{tools_channel}")

    def test_streaming_split_markers(self):
        detector = KimiK3ReasoningDetector(force_reasoning=True)
        chunks = [
            "I am thinking<|close|>thi",
            "nk<|sep|>",
            "<|open|>respon",
            "se<|sep|>Ans",
            "wer here",
            "<|close|>response<|sep|>",
            "<|close|>message<|sep|>",
        ]
        reasoning, content = self._stream(detector, chunks)
        self.assertEqual(reasoning, "I am thinking")
        self.assertEqual(content, "Answer here")

    def test_streaming_char_by_char(self):
        detector = KimiK3ReasoningDetector(force_reasoning=True)
        full = (
            f"{THINK_OPEN}abc{THINK_CLOSE}"
            f"{RESPONSE_OPEN}xyz{RESPONSE_CLOSE}{MESSAGE_CLOSE}"
        )
        reasoning, content = self._stream(detector, list(full))
        self.assertEqual(reasoning, "abc")
        self.assertEqual(content, "xyz")

    def test_streaming_chains_into_tool_detector(self):
        detector = KimiK3ReasoningDetector(force_reasoning=True)
        full = (
            f"hmm{THINK_CLOSE}{RESPONSE_OPEN}ok{RESPONSE_CLOSE}"
            f"{TOOLS_OPEN}"
            + _call_block("python", 1, {"code": ("string", "1")})
            + f"{TOOLS_CLOSE}"
        )
        reasoning, content = self._stream(detector, _stream_chunks(full, 5))
        self.assertEqual(reasoning, "hmm")
        self.assertTrue(content.startswith(f"ok{TOOLS_OPEN}"))

        func_detector = KimiK3FuncDetector()
        result = func_detector.parse_streaming_increment(
            content, [_make_tool("python")]
        )
        self.assertEqual(result.normal_text, "ok")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters), {"code": "1"})

    def test_streaming_thinking_disabled(self):
        detector = KimiK3ReasoningDetector(force_reasoning=False)
        full = f"{RESPONSE_OPEN}plain{RESPONSE_CLOSE}{MESSAGE_CLOSE}"
        reasoning, content = self._stream(detector, _stream_chunks(full, 6))
        self.assertEqual(reasoning, "")
        self.assertEqual(content, "plain")

    def test_streaming_in_band_open_marker(self):
        detector = KimiK3ReasoningDetector(force_reasoning=False)
        full = (
            f"{THINK_OPEN}abc{THINK_CLOSE}"
            f"{RESPONSE_OPEN}xyz{RESPONSE_CLOSE}{MESSAGE_CLOSE}"
        )
        reasoning, content = self._stream(detector, _stream_chunks(full, 4))
        self.assertEqual(reasoning, "abc")
        self.assertEqual(content, "xyz")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
