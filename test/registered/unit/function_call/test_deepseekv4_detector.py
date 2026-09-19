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


def _json_tool(name: str, properties: dict, required=None) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            parameters={
                "type": "object",
                "properties": properties,
                "required": list(properties) if required is None else required,
            },
        ),
    )


class TestDeepSeekV4WrappedArguments(CustomTestCase):
    """Regression tests for #38924: the model sometimes wraps the real
    parameters in a spurious top-level ``arguments``/``input`` key, e.g.
    ``{"arguments": {"path": "/x"}}`` or ``{"arguments": "cmd"}``, which reaches
    the client verbatim and fails schema validation. The detector must unwrap it
    back to the tool's declared parameters in both delivery modes, and the
    streaming and one-shot paths must agree."""

    def setUp(self):
        self.tools = [
            _json_tool("bash", {"command": {"type": "string"}}),
            _json_tool("read", {"path": {"type": "string"}}),
            _json_tool(
                "move",
                {"src": {"type": "string"}, "dst": {"type": "string"}},
                required=["src", "dst"],
            ),
            # A tool whose real parameter is literally named "arguments".
            _json_tool("argtool", {"arguments": {"type": "string"}}),
        ]

    def _non_streaming(self, invoke: str):
        result = DeepSeekV4Detector().detect_and_parse(_wrapped(invoke), self.tools)
        return [(c.name, json.loads(c.parameters)) for c in result.calls]

    def _streaming(self, invoke: str, chunk: int):
        detector = DeepSeekV4Detector()
        text = _wrapped(invoke)
        name, args = None, ""
        for i in range(0, len(text), chunk):
            result = detector.parse_streaming_increment(text[i : i + chunk], self.tools)
            for c in result.calls:
                name = c.name or name
                args += c.parameters or ""
        return name, json.loads(args)

    def _assert_both(self, invoke: str, expected_name: str, expected_args: dict):
        self.assertEqual(self._non_streaming(invoke), [(expected_name, expected_args)])
        for chunk in (1, 4, len(_wrapped(invoke))):
            self.assertEqual(
                self._streaming(invoke, chunk), (expected_name, expected_args)
            )

    def test_dict_wrapper_is_unwrapped(self):
        """``{"arguments": {...}}`` -> the inner object."""
        self._assert_both(
            _invoke("read", '{"arguments": {"path": "/x"}}'), "read", {"path": "/x"}
        )

    def test_input_wrapper_is_unwrapped(self):
        """``input`` is a wrapper key too."""
        self._assert_both(
            _invoke("read", '{"input": {"path": "/x"}}'), "read", {"path": "/x"}
        )

    def test_scalar_wrapper_is_remapped_to_sole_property(self):
        """A bare scalar payload is remapped onto the tool's only property."""
        self._assert_both(
            _invoke("bash", '{"arguments": "cd /x && ls"}'),
            "bash",
            {"command": "cd /x && ls"},
        )

    def test_xml_parameter_tag_wrapper_is_remapped(self):
        """The XML ``<parameter name="arguments" ...>`` shape hits the same bug."""
        self._assert_both(
            _invoke("bash", _param("arguments", "true", "cd /x")),
            "bash",
            {"command": "cd /x"},
        )

    def test_double_json_encoded_wrapper_is_decoded(self):
        """A wrapper whose value is itself a JSON-object string is decoded once."""
        self._assert_both(
            _invoke("read", '{"arguments": "{\\"path\\": \\"/x\\"}"}'),
            "read",
            {"path": "/x"},
        )

    def test_declared_arguments_property_is_not_unwrapped(self):
        """When ``arguments`` is a genuine declared property, it must be left as
        is -- unwrapping it would drop the caller's real parameter."""
        self._assert_both(
            _invoke("argtool", '{"arguments": "hi"}'), "argtool", {"arguments": "hi"}
        )

    def test_ambiguous_scalar_wrapper_is_left_untouched(self):
        """A scalar wrapper cannot be safely remapped when the tool has no single
        target property, so it is left untouched rather than guessed."""
        self._assert_both(
            _invoke("move", '{"arguments": "x"}'), "move", {"arguments": "x"}
        )

    def test_wrapped_and_good_calls_in_one_turn(self):
        """A wrapped call and a well-formed call in the same turn are both
        correct; the fix must not disturb the non-buggy call."""
        invoke = (
            _invoke("read", '{"arguments": {"path": "/a"}}')
            + "\n"
            + _invoke("bash", '{"command": "ls"}')
        )
        expected = [("read", {"path": "/a"}), ("bash", {"command": "ls"})]
        self.assertEqual(self._non_streaming(invoke), expected)


if __name__ == "__main__":
    import unittest

    unittest.main()
