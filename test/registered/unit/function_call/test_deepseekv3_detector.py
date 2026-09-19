"""Unit tests for DeepSeekV3Detector — no server, no model loading.

Covers the streaming fence-leak regression: when the DeepSeek tool-call fence
tokens (``<｜tool▁calls▁begin｜>`` etc.) are split across streaming chunks,
the detector must hold back the partial fence prefix instead of flushing it
into the content stream.
"""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "base-a-test-cpu")

# DeepSeek-V3 special tokens (▁ = U+2581, ｜ = U+FF5C).
B = "<｜tool▁calls▁begin｜>"
CB = "<｜tool▁call▁begin｜>"
SEP = "<｜tool▁sep｜>"
CE = "<｜tool▁call▁end｜>"
E = "<｜tool▁calls▁end｜>"

ALL_FENCES = (B, CB, SEP, CE, E)


def _one_call(name: str, args: str) -> str:
    return CB + "function" + SEP + name + "\n```json\n" + args + "\n```" + CE


class TestDeepSeekV3Detector(CustomTestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get current weather for a city.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["city", "unit"],
                    },
                ),
            ),
        ]
        self.single = (
            B + _one_call("get_weather", '{"city": "Chengdu", "unit": "celsius"}') + E
        )
        self.parallel = (
            B
            + _one_call("get_weather", '{"city": "Beijing", "unit": "celsius"}')
            + _one_call("get_weather", '{"city": "Shanghai", "unit": "celsius"}')
            + E
        )

    def _feed(self, pieces):
        """Feed text pieces to a fresh parser; return (normal_text, calls)."""
        parser = FunctionCallParser(tools=self.tools, tool_call_parser="deepseekv3")
        out, calls = [], {}
        for piece in pieces:
            if not piece:
                continue
            normal, items = parser.parse_stream_chunk(piece)
            if normal:
                out.append(normal)
            for it in items or []:
                d = calls.setdefault(it.tool_index, {"name": None, "params": ""})
                if it.name:
                    d["name"] = it.name
                if it.parameters:
                    d["params"] += it.parameters
        return "".join(out), calls

    @staticmethod
    def _splits(text, n):
        k = (len(text) + n - 1) // n
        return [text[i : i + k] for i in range(0, len(text), k)]

    def _assert_no_fence_leak(self, normal, label):
        for tok in ALL_FENCES:
            self.assertNotIn(tok, normal, f"{label}: fence {tok!r} leaked into content")

    # ==================== fence-leak regression ====================

    def test_single_call_no_leak_char_by_char(self):
        normal, calls = self._feed(list(self.single))
        self._assert_no_fence_leak(normal, "single/char")
        self.assertIn(0, calls)
        self.assertEqual(calls[0]["name"], "get_weather")

    def test_single_call_no_leak_all_two_piece_splits(self):
        for i in range(1, len(self.single)):
            normal, _ = self._feed([self.single[:i], self.single[i:]])
            self._assert_no_fence_leak(normal, f"single/split@{i}")

    def test_single_call_no_leak_chunked_granularities(self):
        for n in (2, 3, 5, 7, 13, 23):
            normal, _ = self._feed(self._splits(self.single, n))
            self._assert_no_fence_leak(normal, f"single/chunked-{n}")

    def test_parallel_calls_no_leak_char_by_char(self):
        normal, calls = self._feed(list(self.parallel))
        self._assert_no_fence_leak(normal, "parallel/char")
        self.assertEqual(len(calls), 2)

    def test_parallel_calls_no_leak_chunked(self):
        for n in (2, 5, 11):
            normal, _ = self._feed(self._splits(self.parallel, n))
            self._assert_no_fence_leak(normal, f"parallel/chunked-{n}")

    # ==================== content preservation ====================

    def test_leading_normal_text_survives(self):
        text = "Sure, let me check.\n" + self.single
        normal, calls = self._feed(list(text))
        self._assert_no_fence_leak(normal, "lead/char")
        self.assertTrue(normal.startswith("Sure, let me check.\n"))
        self.assertIn(0, calls)

    def test_leading_text_survives_when_begin_fence_is_split(self):
        # Regression for the case where leading normal text and a partial
        # begin fence arrive in the same chunk and the fence is completed in
        # the next chunk. The leading text must be flushed immediately and
        # not trapped in the buffer (where it would be discarded once the
        # parser switches to tool-call mode).
        lead = "Sure, let me check.\n"
        text = lead + self.single
        for i in range(1, len(B)):
            split = len(lead) + i
            normal, calls = self._feed([text[:split], text[split:]])
            self._assert_no_fence_leak(normal, f"lead/split-fence@{i}")
            self.assertTrue(
                normal.startswith(lead),
                f"lead/split-fence@{i}: leading text lost, got {normal!r}",
            )
            self.assertIn(0, calls)

    def test_trailing_normal_text_survives(self):
        text = self.single + "All done."
        normal, calls = self._feed(list(text))
        self._assert_no_fence_leak(normal, "trail/char")
        self.assertTrue(normal.rstrip().endswith("All done."))
        self.assertIn(0, calls)


if __name__ == "__main__":
    unittest.main()
