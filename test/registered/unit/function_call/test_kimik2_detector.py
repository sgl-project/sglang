"""Unit tests for KimiK2Detector — no server, no model loading."""

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.kimik2_detector import KimiK2Detector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

SB = "<|tool_calls_section_begin|>"
SE = "<|tool_calls_section_end|>"
CB = "<|tool_call_begin|>"
CE = "<|tool_call_end|>"
AB = "<|tool_call_argument_begin|>"


def _call(name: str, idx: int, args_json: str) -> str:
    return f"{CB}functions.{name}:{idx}{AB}{args_json}{CE}"


def _section(*calls: str) -> str:
    return SB + "".join(calls) + SE


class TestKimiK2Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="write_to_file",
                    description="Write content to a file",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="execute_command",
                    description="Run a shell command",
                    parameters={
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                ),
            ),
        ]
        self.detector = KimiK2Detector()

    def _run_stream(self, deltas):
        """Feed ``deltas`` in order; reconstruct per-index {name, args} and the
        concatenated normal text. A fresh detector isolates request state."""
        detector = KimiK2Detector()
        acc = {}
        normal = []
        for delta in deltas:
            res = detector.parse_streaming_increment(delta, self.tools)
            if res.normal_text:
                normal.append(res.normal_text)
            for item in res.calls:
                entry = acc.setdefault(item.tool_index, {"name": None, "args": ""})
                if item.name:
                    entry["name"] = item.name
                if item.parameters:
                    entry["args"] += item.parameters
        return acc, "".join(normal)

    # ==================== has_tool_call ====================

    def test_has_tool_call_true(self):
        self.assertTrue(
            self.detector.has_tool_call(_section(_call("execute_command", 0, "{}")))
        )

    def test_has_tool_call_false(self):
        self.assertFalse(self.detector.has_tool_call("just some normal assistant text"))

    # ==================== detect_and_parse (non-streaming) ====================

    def test_detect_and_parse_single(self):
        text = "Let me help. " + _section(
            _call("execute_command", 0, '{"command": "ls"}')
        )
        res = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(res.normal_text, "Let me help. ")
        self.assertEqual(len(res.calls), 1)
        self.assertEqual(res.calls[0].name, "execute_command")
        self.assertEqual(json.loads(res.calls[0].parameters), {"command": "ls"})

    def test_detect_and_parse_multiple(self):
        text = _section(
            _call("write_to_file", 0, '{"path": "a.txt", "content": "hi"}'),
            _call("execute_command", 1, '{"command": "ls"}'),
        )
        res = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(
            [c.name for c in res.calls], ["write_to_file", "execute_command"]
        )
        self.assertEqual([c.tool_index for c in res.calls], [0, 1])

    def test_detect_and_parse_nested_braces_not_truncated(self):
        # Arguments containing literal braces / markdown must survive parsing.
        args = {"path": "x.md", "content": "# T\n```py\nd={'k': 1}\n```\ndone } { more"}
        text = _section(_call("write_to_file", 0, json.dumps(args)))
        res = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(res.calls), 1)
        self.assertEqual(json.loads(res.calls[0].parameters), args)

    def test_detect_and_parse_no_tool_call(self):
        res = self.detector.detect_and_parse("no tools here", self.tools)
        self.assertEqual(res.normal_text, "no tools here")
        self.assertEqual(res.calls, [])

    # ==================== streaming (marker-aligned increments) ====================

    def test_stream_single_call(self):
        acc, normal = self._run_stream(
            [SB, CB, "functions.execute_command:0", AB, '{"command": "ls"}', CE, SE]
        )
        self.assertEqual(normal, "")
        self.assertEqual(acc[0]["name"], "execute_command")
        self.assertEqual(json.loads(acc[0]["args"]), {"command": "ls"})

    def test_stream_multiple_calls(self):
        text = _section(
            _call("write_to_file", 0, '{"path": "a.txt", "content": "hi"}'),
            _call("execute_command", 1, '{"command": "ls"}'),
        )
        acc, _ = self._run_stream([text])
        self.assertEqual(acc[0]["name"], "write_to_file")
        self.assertEqual(json.loads(acc[0]["args"]), {"path": "a.txt", "content": "hi"})
        self.assertEqual(acc[1]["name"], "execute_command")
        self.assertEqual(json.loads(acc[1]["args"]), {"command": "ls"})

    def test_stream_empty_args(self):
        acc, _ = self._run_stream([_section(_call("execute_command", 0, "{}"))])
        self.assertEqual(acc[0]["name"], "execute_command")
        self.assertEqual(json.loads(acc[0]["args"]), {})

    def test_stream_prefix_text_preserved(self):
        acc, normal = self._run_stream(
            [
                "Sure, doing it now. ",
                _section(_call("execute_command", 0, '{"command": "ls"}')),
            ]
        )
        self.assertEqual(normal, "Sure, doing it now. ")
        self.assertEqual(json.loads(acc[0]["args"]), {"command": "ls"})

    # ============ regression: <|tool_call_end|> split across increments ============
    # A special-token marker is not guaranteed to arrive whole (cf. #25071, which
    # added the begin-marker holdback). Before the fix, the trailing fragment of a
    # split <|tool_call_end|> leaked into the streamed arguments as broken JSON.

    def test_stream_end_marker_split_two_pieces(self):
        acc, normal = self._run_stream(
            [
                SB,
                CB,
                "functions.execute_command:0",
                AB,
                '{"command":',
                ' "ls"}',
                "<|tool_call_",  # end marker, piece 1
                "end|>",  # end marker, piece 2
                SE,
            ]
        )
        self.assertEqual(acc[0]["name"], "execute_command")
        # Arguments must be exactly the JSON, with no marker fragment leaked in.
        self.assertEqual(acc[0]["args"], '{"command": "ls"}')
        self.assertEqual(json.loads(acc[0]["args"]), {"command": "ls"})
        self.assertNotIn("tool_call", acc[0]["args"])
        self.assertEqual(normal, "")

    def test_stream_end_marker_split_many_pieces(self):
        acc, normal = self._run_stream(
            [
                SB + CB + "functions.write_to_file:0" + AB,
                '{"path": "a.txt", ',
                '"content": "hello"}',
                "<|tool",
                "_call",
                "_end|>",
                SE,
            ]
        )
        self.assertEqual(acc[0]["name"], "write_to_file")
        self.assertEqual(
            json.loads(acc[0]["args"]), {"path": "a.txt", "content": "hello"}
        )
        self.assertNotIn("<|", acc[0]["args"])
        self.assertEqual(normal, "")


if __name__ == "__main__":
    import unittest

    unittest.main()
