"""Unit tests for Apertus2509Detector streaming multi-block flush.

When one streaming chunk holds a tool block followed by another tool block
(and possibly text), parse_streaming_increment must yield every complete
block, not just the first one, so streaming matches detect_and_parse.

No server, no model loading.
"""

import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.apertus2509_detector import Apertus2509Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "base-a-test-cpu")


def _make_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name=n,
                parameters={"type": "object", "properties": {}},
            ),
        )
        for n in ("get_weather", "get_time")
    ]


TEXT = (
    '<|tools_prefix|>[{"get_weather": {}}]<|tools_suffix|>'
    '<|tools_prefix|>[{"get_time": {}}]<|tools_suffix|>done'
)


class TestApertus2509DetectorFinish(unittest.TestCase):
    def setUp(self):
        self.tools = _make_tools()

    def _names(self, *results):
        return [c.name for c in sum((r.calls for r in results), []) if c.name]

    def _text(self, *results):
        return "".join(r.normal_text for r in results)

    def test_same_chunk_two_blocks_matches_one_shot(self):
        """Regression: a chunk holding two tool blocks plus trailing text
        used to drop the second block and the text (issue #41821)."""
        one = Apertus2509Detector().detect_and_parse(TEXT, self.tools)
        d = Apertus2509Detector()
        r = d.parse_streaming_increment(TEXT, self.tools)
        f = d.finish(self.tools)
        self.assertEqual(self._names(r, f), self._names(one))
        self.assertEqual(self._text(r, f), one.normal_text)
        self.assertEqual(d._buffer, "")

    def test_second_block_split_across_chunks(self):
        """A complete block split over two chunks is not lost."""
        d = Apertus2509Detector()
        chunk1 = (
            '<|tools_prefix|>[{"get_weather": {}}]<|tools_suffix|>'
            '<|tools_prefix|>[{"get_t'
        )
        chunk2 = 'ime": {}}]<|tools_suffix|>done'
        r1 = d.parse_streaming_increment(chunk1, self.tools)
        r2 = d.parse_streaming_increment(chunk2, self.tools)
        f = d.finish(self.tools)
        self.assertEqual(self._names(r1, r2, f), ["get_weather", "get_time"])
        self.assertEqual(self._text(r1, r2, f), "done")

    def test_single_block_with_trailing_text(self):
        d = Apertus2509Detector()
        text = '<|tools_prefix|>[{"get_weather": {}}]<|tools_suffix|>hi'
        one = Apertus2509Detector().detect_and_parse(text, self.tools)
        r = d.parse_streaming_increment(text, self.tools)
        f = d.finish(self.tools)
        self.assertEqual(self._names(r, f), self._names(one))
        self.assertEqual(self._text(r, f), one.normal_text)

    def test_plain_text_passes_through(self):
        d = Apertus2509Detector()
        r = d.parse_streaming_increment("hello world", self.tools)
        f = d.finish(self.tools)
        self.assertEqual(self._text(r, f), "hello world")
        self.assertEqual(self._names(r, f), [])

    def test_partial_marker_waits_for_next_chunk(self):
        """A partial marker prefix is buffered, not emitted as text."""
        d = Apertus2509Detector()
        r1 = d.parse_streaming_increment(
            '<|tools_prefix|>[{"get_weather": {}}]<|tools_suffix|><|tools_p',
            self.tools,
        )
        self.assertEqual(self._names(r1), ["get_weather"])
        r2 = d.parse_streaming_increment(
            'refix|>[{"get_time": {}}]<|tools_suffix|>done', self.tools
        )
        f = d.finish(self.tools)
        self.assertEqual(self._names(r1, r2, f), ["get_weather", "get_time"])
        self.assertEqual(self._text(r1, r2, f), "done")


if __name__ == "__main__":
    unittest.main()
