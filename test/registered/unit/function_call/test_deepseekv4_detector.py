"""Unit tests for DeepSeekV4Detector DSML streaming — no server, no model loading."""

import json
from unittest.mock import patch

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

DSML = "｜DSML｜"


def _wrapped(invoke: str) -> str:
    return f"<{DSML}tool_calls>\n{invoke}\n</{DSML}tool_calls>"


def _invoke(name: str, params: str = "") -> str:
    return f'<{DSML}invoke name="{name}">\n{params}\n</{DSML}invoke>'


def _param(name: str, is_string: str, value: str) -> str:
    return (
        f'<{DSML}parameter name="{name}" string="{is_string}">{value}</{DSML}parameter>'
    )


def _weather_call(city: str = "SF") -> str:
    return _wrapped(_invoke("get_weather", _param("city", "true", city)))


class TestDeepSeekV4Streaming(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            )
        ]

    def _feed(self, chunks):
        """Returns (normal_text, calls) accumulated over the chunks."""
        detector = DeepSeekV4Detector()
        normal, calls = "", []
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal += result.normal_text
            calls.extend(result.calls)
        return normal, calls

    def test_preamble_in_same_delta_as_tool_call(self):
        """Prose sharing a delta with the tool call must not be dropped, and the
        streaming and one-shot paths must agree on it."""
        text = "Let me check.\n" + _weather_call()
        normal, calls = self._feed([text])

        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])
        self.assertEqual(
            normal, DeepSeekV4Detector().detect_and_parse(text, self.tools).normal_text
        )

    def test_preamble_before_bare_invoke_without_wrapper(self):
        """The bare `<｜DSML｜invoke …>` form has no tool_calls wrapper to walk
        back to, so the preamble is computed from the invoke itself."""
        text = "Checking.\n" + _invoke("get_weather", _param("city", "true", "SF"))
        normal, calls = self._feed([text])

        self.assertIn("Checking.", normal)
        self.assertEqual([c.name for c in calls if c.name], ["get_weather"])

    def test_no_dsml_markers_leak_into_normal_text(self):
        text = "Prose.\n" + _weather_call()
        normal, _ = self._feed([text[i : i + 4] for i in range(0, len(text), 4)])

        self.assertNotIn(DSML, normal)

    def test_malformed_partial_json_falls_back_to_raw_value(self):
        """A partial non-string parameter must not escape as MalformedJSON."""
        detector = DeepSeekV4Detector()
        result = detector.parse_streaming_increment(
            f'<{DSML}tool_calls>\n<{DSML}invoke name="get_weather">\n'
            f'<{DSML}parameter name="city" string="false">{{"a"',
            self.tools,
        )

        self.assertEqual([c.name for c in result.calls if c.name], ["get_weather"])

    def test_non_streaming_parses_every_tool_calls_section(self):
        """A turn with two tool_calls sections must yield both calls."""
        result = DeepSeekV4Detector().detect_and_parse(
            f"{_weather_call('SF')}\n{_weather_call('NY')}", self.tools
        )

        self.assertEqual(len(result.calls), 2)

    def test_parse_error_neither_swallows_nor_duplicates(self):
        """An unexpected parse error must not empty the turn, and the dropped
        buffer must not come back on the next delta."""
        detector = DeepSeekV4Detector()

        with patch.object(
            DeepSeekV4Detector,
            "_parse_parameters_from_xml",
            side_effect=RuntimeError("boom"),
        ):
            first = detector.parse_streaming_increment(_weather_call(), self.tools)
            self.assertEqual(detector._buffer, "")
            second = detector.parse_streaming_increment(" tail", self.tools)

        self.assertIn("get_weather", first.normal_text)
        self.assertNotIn("get_weather", second.normal_text)
        # No half-formed call: the failure can land between a tool's name and its
        # arguments, so an argument-less named call must not reach the client.
        self.assertEqual(first.calls, [])


class TestDeepSeekV4UnwrapWrapper(CustomTestCase):
    """DeepSeek V4 occasionally wraps real arguments under an `arguments`/`input`
    key (object, JSON-string, or bare-scalar inner). The parser must repair such
    payloads against the declared tool schema instead of forwarding them nested.
    """

    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="bash",
                    parameters={
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="read",
                    parameters={
                        "type": "object",
                        "properties": {
                            "filePath": {"type": "string"},
                            "offset": {"type": "integer"},
                            "limit": {"type": "integer"},
                        },
                        "required": ["filePath"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="weird",
                    parameters={
                        "type": "object",
                        "properties": {
                            "arguments": {"type": "string"},
                            "input": {"type": "object"},
                        },
                    },
                ),
            ),
        ]
        self.detector_cls = DeepSeekV4Detector

    def _args_of(self, text, chunks=None):
        """Return the (name, arguments_json) for the first tool call.

        ``chunks=None`` exercises the non-streaming path; otherwise the text is
        streamed in increments of the given size (including char-by-char).
        """
        if chunks is None:
            calls = self.detector_cls().detect_and_parse(text, self.tools).calls
            assert calls, "expected at least one parsed call"
            c = calls[0]
            return c.name, c.parameters
        detector = self.detector_cls()
        detector._buffer = ""
        name, args = None, ""
        for i in range(0, len(text), chunks):
            result = detector.parse_streaming_increment(
                text[i : i + chunks], self.tools
            )
            for c in result.calls:
                if c.name:
                    name = c.name
                if c.parameters:
                    args += c.parameters
        for c in detector.finish(self.tools).calls:
            if c.name:
                name = c.name
            if c.parameters:
                args += c.parameters
        return name, args

    def _assert_repaired(self, text, expected, chunks=(None, 1, 3, 100)):
        for size in chunks:
            name, args = self._args_of(text, size)
            self.assertEqual(name, "bash", f"chunk={size}")
            self.assertEqual(
                json.loads(args),
                expected,
                f"chunk={size} produced nested/unrepaired arguments {args!r}",
            )

    def test_direct_json_object_wrapper(self):
        body = '{"arguments": {"command": "ls -la /x"}}'
        self._assert_repaired(_wrapped(_invoke("bash", body)), {"command": "ls -la /x"})

    def test_direct_json_input_wrapper(self):
        body = '{"input": {"command": "pwd"}}'
        self._assert_repaired(_wrapped(_invoke("bash", body)), {"command": "pwd"})

    def test_xml_param_object_wrapper(self):
        body = _param("arguments", "false", '{"command":"cd /tmp"}')
        self._assert_repaired(_wrapped(_invoke("bash", body)), {"command": "cd /tmp"})

    def test_xml_param_scalar_maps_to_single_field(self):
        body = _param("arguments", "true", "ls -la")
        self._assert_repaired(_wrapped(_invoke("bash", body)), {"command": "ls -la"})

    def test_json_string_inner_unwrapped_on_multi_field_tool(self):
        inner = '{"filePath": "/x", "offset": 300, "limit": 350}'
        body = '{"arguments": "%s"}' % inner.replace('"', '\\"')
        for size in (None, 1, 100):
            name, args = self._args_of(_wrapped(_invoke("read", body)), size)
            self.assertEqual(name, "read", f"chunk={size}")
            self.assertEqual(
                json.loads(args),
                {"filePath": "/x", "offset": 300, "limit": 350},
                f"chunk={size} produced {args!r}",
            )

    def test_declared_arguments_param_not_unwrapped(self):
        # `weird` legitimately declares `arguments`; it must be preserved.
        body = (
            _param("arguments", "true", "x")
            + "\n"
            + _param("input", "false", '{"a": 1}')
        )
        name, args = self._args_of(_wrapped(_invoke("weird", body)))
        self.assertEqual(name, "weird")
        self.assertEqual(json.loads(args), {"arguments": "x", "input": {"a": 1}})

    def test_unknown_inner_key_not_unwrapped(self):
        body = '{"arguments": {"command": "x", "nope": 1}}'
        name, args = self._args_of(_wrapped(_invoke("bash", body)))
        self.assertEqual(name, "bash")
        self.assertEqual(json.loads(args), {"arguments": {"command": "x", "nope": 1}})

    def test_multi_field_scalar_wrapper_not_unwrapped(self):
        # `read` declares three fields; a bare-scalar wrapper has no safe mapping.
        body = _param("arguments", "true", "/x")
        name, args = self._args_of(_wrapped(_invoke("read", body)))
        self.assertEqual(name, "read")
        self.assertEqual(json.loads(args), {"arguments": "/x"})

    def test_normal_flat_call_untouched_and_incremental(self):
        body = '{"command": "echo hi"}'
        self._assert_repaired(_wrapped(_invoke("bash", body)), {"command": "echo hi"})

    def test_zero_arg_self_close_kept(self):
        text = f'<{DSML}tool_calls>\n<{DSML}invoke name="read"/>\n</{DSML}tool_calls>'
        for size in (None, 1, 100):
            name, args = self._args_of(text, size)
            self.assertEqual(name, "read", f"chunk={size}")
            self.assertEqual(args, "{}", f"chunk={size}")


if __name__ == "__main__":
    import unittest

    unittest.main()
