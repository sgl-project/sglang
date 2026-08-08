"""Unit tests for MinimaxM2Detector."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")


def _make_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="bash",
                description="Run a shell command",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["command"],
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="get_weather",
                description="Get weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string"},
                    },
                    "required": ["city"],
                },
            ),
        ),
    ]


def _stream(detector, tools, chunks):
    all_calls = []
    all_normal = ""
    for ch in chunks:
        result = detector.parse_streaming_increment(ch, tools)
        all_calls.extend(result.calls)
        all_normal += result.normal_text or ""
    return all_normal, all_calls


def _merge_by_index(calls):
    # Merge deltas the way an OpenAI-compatible client does: same tool_index
    # accumulates name + parameters into one tool_call.
    merged = {}
    for c in calls:
        slot = merged.setdefault(c.tool_index, {"name": None, "args": ""})
        if c.name is not None:
            slot["name"] = c.name
        if c.parameters:
            slot["args"] += c.parameters
    return merged


class TestMinimaxM2Detector(CustomTestCase):
    def setUp(self):
        self.tools = _make_tools()

    # ==================== has_tool_call ====================

    def test_has_tool_call_true(self):
        text = '<minimax:tool_call><invoke name="bash"><parameter name="command">ls</parameter></invoke></minimax:tool_call>'
        self.assertTrue(MinimaxM2Detector().has_tool_call(text))

    def test_has_tool_call_false_plain_text(self):
        self.assertFalse(MinimaxM2Detector().has_tool_call("Just a normal reply."))

    # ==================== detect_and_parse (non-streaming) ====================

    def test_non_streaming_single_call(self):
        det = MinimaxM2Detector()
        text = '<minimax:tool_call><invoke name="bash"><parameter name="command">ls -la</parameter></invoke></minimax:tool_call>'
        result = det.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "bash")
        self.assertEqual(json.loads(result.calls[0].parameters), {"command": "ls -la"})

    def test_non_streaming_multiple_params(self):
        det = MinimaxM2Detector()
        text = (
            '<minimax:tool_call><invoke name="bash">'
            '<parameter name="command">cat foo</parameter>'
            '<parameter name="description">read file</parameter>'
            "</invoke></minimax:tool_call>"
        )
        result = det.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(
            json.loads(result.calls[0].parameters),
            {"command": "cat foo", "description": "read file"},
        )

    # ==================== Streaming ====================

    def test_streaming_one_chunk(self):
        det = MinimaxM2Detector()
        text = (
            '<minimax:tool_call><invoke name="bash">'
            '<parameter name="command">cat | head -100</parameter>'
            '<parameter name="description">Run tests</parameter>'
            "</invoke></minimax:tool_call>"
        )
        normal, calls = _stream(det, self.tools, [text])
        merged = _merge_by_index(calls)
        self.assertEqual(normal, "")
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["name"], "bash")
        self.assertEqual(
            json.loads(merged[0]["args"]),
            {"command": "cat | head -100", "description": "Run tests"},
        )

    def test_streaming_split_at_param_boundaries(self):
        det = MinimaxM2Detector()
        chunks = [
            '<minimax:tool_call><invoke name="bash">',
            '<parameter name="command">cat | head -100</parameter>',
            '<parameter name="description">Run tests</parameter>',
            "</invoke></minimax:tool_call>",
        ]
        _, calls = _stream(det, self.tools, chunks)
        merged = _merge_by_index(calls)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["name"], "bash")
        self.assertEqual(
            json.loads(merged[0]["args"]),
            {"command": "cat | head -100", "description": "Run tests"},
        )

    def test_streaming_split_inside_param_value(self):
        det = MinimaxM2Detector()
        chunks = [
            '<minimax:tool_call><invoke name="bash"><parameter name="command">cat',
            ' | head -100</parameter><parameter name="description">Run',
            " tests</parameter></invoke></minimax:tool_call>",
        ]
        _, calls = _stream(det, self.tools, chunks)
        merged = _merge_by_index(calls)
        self.assertEqual(len(merged), 1)
        self.assertEqual(
            json.loads(merged[0]["args"]),
            {"command": "cat | head -100", "description": "Run tests"},
        )

    def test_streaming_char_by_char(self):
        # Chunks shorter than "<minimax:tool_call>" must be buffered, not
        # flushed as normal text.
        det = MinimaxM2Detector()
        text = (
            '<minimax:tool_call><invoke name="bash">'
            '<parameter name="command">ls</parameter>'
            "</invoke></minimax:tool_call>"
        )
        normal, calls = _stream(det, self.tools, list(text))
        merged = _merge_by_index(calls)
        self.assertEqual(normal, "")
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["name"], "bash")
        self.assertEqual(json.loads(merged[0]["args"]), {"command": "ls"})

    def test_streaming_two_parallel_tool_calls(self):
        # #23071: two invokes in the same chunk must not share parameter
        # values via re.finditer scanning across boundaries.
        det = MinimaxM2Detector()
        text = (
            '<minimax:tool_call><invoke name="bash">'
            '<parameter name="command">A</parameter>'
            "</invoke></minimax:tool_call>"
            '<minimax:tool_call><invoke name="bash">'
            '<parameter name="command">B</parameter>'
            "</invoke></minimax:tool_call>"
        )
        _, calls = _stream(det, self.tools, [text])
        merged = _merge_by_index(calls)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["name"], "bash")
        self.assertEqual(json.loads(merged[0]["args"]), {"command": "A"})
        self.assertEqual(merged[1]["name"], "bash")
        self.assertEqual(json.loads(merged[1]["args"]), {"command": "B"})

    def test_streaming_no_double_emission_for_same_tool(self):
        # #23071: one invoke must stay on one tool_index, and the merged
        # args must be a single well-formed JSON object.
        det = MinimaxM2Detector()
        chunks = [
            '<minimax:tool_call><invoke name="bash"><parameter name="command">cat | head -100</parameter><',
            'parameter name="description">Run tests</parameter></invoke></minimax:tool_call>',
        ]
        _, calls = _stream(det, self.tools, chunks)
        for c in calls:
            self.assertEqual(c.tool_index, 0)
        merged = _merge_by_index(calls)
        self.assertEqual(len(merged), 1)
        self.assertEqual(
            json.loads(merged[0]["args"]),
            {"command": "cat | head -100", "description": "Run tests"},
        )

    def test_streaming_normal_text_then_tool(self):
        det = MinimaxM2Detector()
        chunks = [
            "Sure, running it now. ",
            '<minimax:tool_call><invoke name="bash">',
            '<parameter name="command">ls</parameter>',
            "</invoke></minimax:tool_call>",
        ]
        normal, calls = _stream(det, self.tools, chunks)
        merged = _merge_by_index(calls)
        self.assertEqual(normal, "Sure, running it now. ")
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["name"], "bash")
        self.assertEqual(json.loads(merged[0]["args"]), {"command": "ls"})

    def test_streaming_invalid_tool_name_dropped(self):
        det = MinimaxM2Detector()
        text = (
            '<minimax:tool_call><invoke name="not_a_real_tool">'
            '<parameter name="x">y</parameter>'
            "</invoke></minimax:tool_call>"
        )
        _, calls = _stream(det, self.tools, [text])
        self.assertEqual(len(calls), 0)

    def test_streaming_partial_start_token_held(self):
        # A chunk ending with a strict prefix of the start token must be
        # held in the buffer, not flushed.
        det = MinimaxM2Detector()
        normal1, calls1 = _stream(det, self.tools, ["<minimax:tool_"])
        self.assertEqual(calls1, [])
        self.assertNotIn("<minimax:tool_", normal1)
        normal2, calls2 = _stream(
            det,
            self.tools,
            [
                'call><invoke name="bash"><parameter name="command">ls</parameter></invoke></minimax:tool_call>'
            ],
        )
        merged = _merge_by_index(calls2)
        self.assertEqual(normal2, "")
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["name"], "bash")
        self.assertEqual(json.loads(merged[0]["args"]), {"command": "ls"})

    def test_streaming_two_invokes_in_one_wrapper(self):
        # #23071: distinct param values across two invokes inside a single
        # <minimax:tool_call>. Old code's re.finditer scanned across both
        # invokes and overwrote the first invoke's parameters with the
        # second's. The first invoke must keep its own values.
        det = MinimaxM2Detector()
        text = (
            '<minimax:tool_call>'
            '<invoke name="bash"><parameter name="command">FIRST</parameter></invoke>'
            '<invoke name="bash">'
            '<parameter name="command">SECOND</parameter>'
            '<parameter name="description">D</parameter>'
            '</invoke>'
            '</minimax:tool_call>'
        )
        _, calls = _stream(det, self.tools, [text])
        merged = _merge_by_index(calls)
        self.assertEqual(len(merged), 2)
        self.assertEqual(json.loads(merged[0]["args"]), {"command": "FIRST"})
        self.assertEqual(
            json.loads(merged[1]["args"]),
            {"command": "SECOND", "description": "D"},
        )

    def test_streaming_two_separate_wrappers_distinct_params(self):
        # Same data-corruption shape as the previous test, but with each
        # invoke in its own wrapper.
        det = MinimaxM2Detector()
        text = (
            '<minimax:tool_call><invoke name="bash">'
            '<parameter name="command">FIRST</parameter>'
            '</invoke></minimax:tool_call>'
            '<minimax:tool_call><invoke name="bash">'
            '<parameter name="command">SECOND</parameter>'
            '<parameter name="description">D</parameter>'
            '</invoke></minimax:tool_call>'
        )
        _, calls = _stream(det, self.tools, [text])
        merged = _merge_by_index(calls)
        self.assertEqual(len(merged), 2)
        self.assertEqual(json.loads(merged[0]["args"]), {"command": "FIRST"})
        self.assertEqual(
            json.loads(merged[1]["args"]),
            {"command": "SECOND", "description": "D"},
        )

    def test_streaming_invalid_invoke_does_not_drop_following_valid_invoke(
        self,
    ):
        # Old code dumped the entire buffer into normal text on an invalid
        # tool name, silently swallowing any valid invoke that came after.
        det = MinimaxM2Detector()
        text = (
            '<minimax:tool_call><invoke name="bogus">'
            '<parameter name="x">y</parameter>'
            '</invoke></minimax:tool_call>'
            '<minimax:tool_call><invoke name="bash">'
            '<parameter name="command">VALID</parameter>'
            '</invoke></minimax:tool_call>'
        )
        _, calls = _stream(det, self.tools, [text])
        merged = _merge_by_index(calls)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["name"], "bash")
        self.assertEqual(json.loads(merged[0]["args"]), {"command": "VALID"})

    def test_streaming_invoke_with_no_parameters(self):
        # Empty <invoke></invoke> must yield args = "{}", not "" — clients
        # call json.loads() on the merged args and "" is not valid JSON.
        det = MinimaxM2Detector()
        text = (
            '<minimax:tool_call><invoke name="bash"></invoke></minimax:tool_call>'
        )
        _, calls = _stream(det, self.tools, [text])
        merged = _merge_by_index(calls)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["name"], "bash")
        self.assertEqual(json.loads(merged[0]["args"]), {})

    def test_streaming_duplicate_parameter_keeps_first(self):
        # Model glitch: same parameter appears twice. Take the first
        # occurrence and ignore the duplicate (rather than re-emitting).
        det = MinimaxM2Detector()
        text = (
            '<minimax:tool_call><invoke name="bash">'
            '<parameter name="command">FIRST</parameter>'
            '<parameter name="command">SECOND</parameter>'
            '</invoke></minimax:tool_call>'
        )
        _, calls = _stream(det, self.tools, [text])
        merged = _merge_by_index(calls)
        self.assertEqual(len(merged), 1)
        self.assertEqual(json.loads(merged[0]["args"]), {"command": "FIRST"})

    def test_streaming_multi_param_types(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="set_config",
                    description="Configure",
                    parameters={
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer"},
                            "ratio": {"type": "number"},
                            "enabled": {"type": "boolean"},
                        },
                    },
                ),
            )
        ]
        det = MinimaxM2Detector()
        text = (
            '<minimax:tool_call><invoke name="set_config">'
            '<parameter name="count">3</parameter>'
            '<parameter name="ratio">0.5</parameter>'
            '<parameter name="enabled">true</parameter>'
            "</invoke></minimax:tool_call>"
        )
        _, calls = _stream(det, tools, [text])
        merged = _merge_by_index(calls)
        self.assertEqual(len(merged), 1)
        args = json.loads(merged[0]["args"])
        self.assertEqual(args["count"], 3)
        self.assertEqual(args["ratio"], 0.5)
        self.assertIs(args["enabled"], True)


if __name__ == "__main__":
    import unittest

    unittest.main()
