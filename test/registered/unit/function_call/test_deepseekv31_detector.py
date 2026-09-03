"""Unit tests for DeepSeekV31Detector streaming tool calls — no server, no model loading."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.deepseekv31_detector import DeepSeekV31Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")

CALLS_BEGIN = "<｜tool▁calls▁begin｜>"
CALLS_END = "<｜tool▁calls▁end｜>"
CALL_BEGIN = "<｜tool▁call▁begin｜>"
CALL_SEP = "<｜tool▁sep｜>"
CALL_END = "<｜tool▁call▁end｜>"


def _call(name: str, args: str) -> str:
    return f"{CALL_BEGIN}{name}{CALL_SEP}{args}{CALL_END}"


class TestDeepSeekV31Streaming(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="f1",
                    description="first tool",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="f2",
                    description="second tool",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

    def _feed(self, chunks):
        """Feeds chunks to a fresh detector, returns calls reconstructed per tool_index."""
        detector = DeepSeekV31Detector()
        emitted = []
        for chunk in chunks:
            emitted.extend(detector.parse_streaming_increment(chunk, self.tools).calls)

        reconstructed = {}
        for item in emitted:
            entry = reconstructed.setdefault(
                item.tool_index, {"name": None, "arguments": ""}
            )
            if item.name:
                entry["name"] = item.name
            entry["arguments"] += item.parameters
        return [reconstructed[i] for i in sorted(reconstructed)]

    def test_both_calls_in_single_increment(self):
        """One chunk carrying two complete calls (batched detokenization,
        stream_interval > 1, speculative decode) must emit both calls, not one
        call whose name swallowed the first tool call."""
        text = (
            CALLS_BEGIN + _call("f1", '{"a": 1}') + _call("f2", '{"b": 2}') + CALLS_END
        )
        calls = self._feed([text])

        self.assertEqual([c["name"] for c in calls], ["f1", "f2"])
        self.assertEqual(json.loads(calls[0]["arguments"]), {"a": 1})
        self.assertEqual(json.loads(calls[1]["arguments"]), {"b": 2})

        # Raw-event pin for the fused end-token -> next-separator boundary:
        # _feed() aggregates by tool_index, which would mask duplicate or
        # out-of-order emissions; the raw stream must carry each name exactly
        # once, in order, with no swallowed markup.
        detector = DeepSeekV31Detector()
        raw = detector.parse_streaming_increment(text, self.tools).calls
        self.assertEqual([c.name for c in raw if c.name], ["f1", "f2"])
        self.assertEqual([c.tool_index for c in raw if c.name], [0, 1])

    def test_one_call_per_increment(self):
        """Chunks each carrying one complete call must keep the two calls
        separate even though the first call is never trimmed mid-stream."""
        calls = self._feed(
            [CALLS_BEGIN + _call("f1", '{"a": 1}'), _call("f2", '{"b": 2}')]
        )

        self.assertEqual([c["name"] for c in calls], ["f1", "f2"])
        self.assertEqual(json.loads(calls[0]["arguments"]), {"a": 1})
        self.assertEqual(json.loads(calls[1]["arguments"]), {"b": 2})

    def test_token_by_token(self):
        """Fine-grained streaming (one lattice token per chunk) must keep working."""
        lattice = [
            CALLS_BEGIN,
            CALL_BEGIN,
            "f",
            "1",
            CALL_SEP,
            '{"',
            "a",
            '":',
            " ",
            "1",
            "}",
            CALL_END,
            CALL_BEGIN,
            "f",
            "2",
            CALL_SEP,
            '{"',
            "b",
            '":',
            " ",
            "2",
            "}",
            CALL_END,
            CALLS_END,
        ]
        calls = self._feed(lattice)

        self.assertEqual([c["name"] for c in calls], ["f1", "f2"])
        self.assertEqual(json.loads(calls[0]["arguments"]), {"a": 1})
        self.assertEqual(json.loads(calls[1]["arguments"]), {"b": 2})

    def test_single_call_across_increments(self):
        """A call whose name and arguments arrive in separate chunks must still
        be streamed as one complete call."""
        chunks = [
            CALLS_BEGIN + CALL_BEGIN + "f",
            "1" + CALL_SEP + "{",
            '"a": 1}',
            CALL_END + CALLS_END,
        ]
        calls = self._feed(chunks)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "f1")
        self.assertEqual(json.loads(calls[0]["arguments"]), {"a": 1})

    def test_streaming_matches_detect_and_parse(self):
        """Streaming and one-shot parsing must agree on the parsed calls."""
        text = (
            CALLS_BEGIN + _call("f1", '{"a": 1}') + _call("f2", '{"b": 2}') + CALLS_END
        )
        streaming_calls = self._feed([text])
        one_shot_calls = DeepSeekV31Detector().detect_and_parse(text, self.tools).calls

        self.assertEqual(
            [(c["name"], json.loads(c["arguments"])) for c in streaming_calls],
            [(c.name, json.loads(c.parameters)) for c in one_shot_calls],
        )


if __name__ == "__main__":
    unittest.main()
