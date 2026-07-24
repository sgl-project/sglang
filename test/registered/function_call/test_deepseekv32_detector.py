"""Tests for the DeepSeek V3.2 / V4 DSML streaming detector.

Regression tests for parse_streaming_increment when a non-string parameter
(string="false") streams a value that is not (yet) valid JSON — e.g. free-form
prose, Cyrillic text. Before the fix, partial_json_parser raised MalformedJSON,
which escaped the `except json.JSONDecodeError` in _parse_parameters_from_xml,
hit the outer catch-all, and the detector then:
  1. leaked the raw DSML buffer to the client as normal_text, and
  2. never cleared self._buffer, so every subsequent chunk re-parsed and
     re-emitted the whole growing buffer (clients saw the same text repeated
     on every token; proxies with repetition detectors killed the stream).
"""

import json
import logging
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(5, "base-a-test-cpu")

DETECTOR_LOGGER = "sglang.srt.function_call.deepseekv32_detector"


def _make_tool(name):
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters={
                "type": "object",
                "properties": {"payload": {"type": "object"}},
            },
        ),
    )


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _stream(detector, chunks, tools):
    """Feed chunks through the detector; return (normal_texts, calls, errors)."""
    capture = _LogCapture()
    logger = logging.getLogger(DETECTOR_LOGGER)
    logger.addHandler(capture)
    try:
        normal_texts, calls = [], []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, tools)
            normal_texts.append(result.normal_text)
            calls.extend(result.calls)
    finally:
        logger.removeHandler(capture)
    errors = [m for m in capture.messages if "parse_streaming_increment" in m]
    return normal_texts, calls, errors


class TestNonJsonObjectParamStreaming(unittest.TestCase):
    """string="false" parameter whose streamed value is not valid JSON."""

    CHUNKS = [
        '<｜DSML｜function_calls>\n<｜DSML｜invoke name="write_note">\n',
        '<｜DSML｜parameter name="payload" string="false">Здра',
        "вствуйте, мир",
        "! Это тест",
        "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>",
    ]

    def test_no_parser_error_and_no_dsml_leak(self):
        tools = [_make_tool("write_note")]
        normal_texts, calls, errors = _stream(DeepSeekV32Detector(), self.CHUNKS, tools)

        self.assertEqual(errors, [], "parser must not log parse errors")
        for text in normal_texts:
            self.assertNotIn("｜DSML｜", text, "raw DSML markup leaked to client")

        self.assertIn("write_note", [c.name for c in calls if c.name])
        streamed_args = "".join(c.parameters or "" for c in calls)
        self.assertEqual(
            json.loads(streamed_args),
            {"payload": "Здравствуйте, мир! Это тест"},
        )

    def test_no_repeated_buffer_reemission(self):
        tools = [_make_tool("write_note")]
        normal_texts, _, _ = _stream(DeepSeekV32Detector(), self.CHUNKS, tools)

        non_empty = [t for t in normal_texts if t]
        regrowth = sum(
            1 for prev, cur in zip(non_empty, non_empty[1:]) if cur.startswith(prev)
        )
        self.assertEqual(regrowth, 0, "buffer re-emitted on subsequent chunks")

    def test_v4_tool_calls_tokens(self):
        """DeepSeekV4Detector inherits the fix (tool_calls wrapper tokens)."""
        chunks = [c.replace("function_calls", "tool_calls") for c in self.CHUNKS]
        tools = [_make_tool("write_note")]
        normal_texts, calls, errors = _stream(DeepSeekV4Detector(), chunks, tools)

        self.assertEqual(errors, [])
        for text in normal_texts:
            self.assertNotIn("｜DSML｜", text)
        streamed_args = "".join(c.parameters or "" for c in calls)
        self.assertEqual(
            json.loads(streamed_args),
            {"payload": "Здравствуйте, мир! Это тест"},
        )


class TestBufferResetOnError(unittest.TestCase):
    def test_buffer_cleared_after_unexpected_error(self):
        """A one-off internal error must not poison every later increment."""
        tools = [_make_tool("write_note")]
        detector = DeepSeekV32Detector()

        original = detector._parse_parameters_from_xml
        detector._parse_parameters_from_xml = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("injected")
        )
        first = detector.parse_streaming_increment(
            '<｜DSML｜function_calls>\n<｜DSML｜invoke name="write_note">\nboom',
            tools,
        )
        detector._parse_parameters_from_xml = original

        follow_up = detector.parse_streaming_increment("plain text", tools)
        self.assertNotIn(first.normal_text[:20] or "<never>", follow_up.normal_text)
        self.assertNotIn("｜DSML｜", follow_up.normal_text)


class TestJsonParamStreamingUnchanged(unittest.TestCase):
    def test_json_object_param_still_streams(self):
        """Happy path: a valid (partial) JSON value keeps working as before."""
        chunks = [
            '<｜DSML｜function_calls>\n<｜DSML｜invoke name="write_note">\n',
            '<｜DSML｜parameter name="payload" string="false">{"ci',
            'ty": "San',
            ' Francisco"}',
            "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜function_calls>",
        ]
        tools = [_make_tool("write_note")]
        normal_texts, calls, errors = _stream(DeepSeekV32Detector(), chunks, tools)

        self.assertEqual(errors, [])
        for text in normal_texts:
            self.assertNotIn("｜DSML｜", text)
        streamed_args = "".join(c.parameters or "" for c in calls)
        self.assertEqual(
            json.loads(streamed_args),
            {"payload": {"city": "San Francisco"}},
        )


if __name__ == "__main__":
    unittest.main()
