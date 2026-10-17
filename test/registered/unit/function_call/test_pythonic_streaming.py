import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.pythonic_detector import PythonicDetector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPythonicStreaming(CustomTestCase):
    def setUp(self):
        super().setUp()
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                ),
            )
        ]

    def assert_stream(self, chunks, expected_text, expected_arguments):
        detector = PythonicDetector()
        text = []
        calls = {}
        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            text.append(result.normal_text)
            for call in result.calls:
                entry = calls.setdefault(call.tool_index, [None, ""])
                if call.name is not None:
                    self.assertIsNone(entry[0], "tool index was reused")
                    entry[0] = call.name
                entry[1] += call.parameters

        self.assertEqual("".join(text), expected_text)
        self.assertEqual(list(calls), list(range(len(expected_arguments))))
        self.assertEqual(
            [(name, json.loads(arguments)) for name, arguments in calls.values()],
            [("search", arguments) for arguments in expected_arguments],
        )
        # The final input chunk must emit everything complete, without relying
        # on a later empty increment to flush buffered calls or trailing prose.
        self.assertEqual(detector._buffer, "")

    def assert_all_chunkings(self, text, expected_text, expected_arguments):
        chunkings = [[text], list(text)]
        chunkings.extend([text[:split], text[split:]] for split in range(1, len(text)))
        for chunks in chunkings:
            with self.subTest(chunks=chunks):
                self.assert_stream(chunks, expected_text, expected_arguments)

    def test_quoted_brackets_match_non_streaming_for_every_split(self):
        literals = [
            '"hello]world"',
            "'an unmatched [ in a query'",
            r'"escaped quote: \" ] and backslash: \\"',
            r"'escaped quote: \' ] and backslash: \\'",
            "'''a single ' and ] inside a multiline\nquery'''",
            '"""a double " and [ inside a multiline\nquery"""',
        ]
        for literal in literals:
            text = f"[search(query={literal})]"
            with self.subTest(literal=literal):
                result = PythonicDetector().detect_and_parse(text, self.tools)
                self.assertEqual(len(result.calls), 1)
                self.assert_all_chunkings(
                    text, "", [json.loads(result.calls[0].parameters)]
                )

    def test_nested_containers_preserve_string_brackets(self):
        text = "[search(query={'terms': [']', {'value': '['}]})]"
        self.assert_all_chunkings(
            text, "", [{"query": {"terms": ["]", {"value": "["}]}}]
        )

    def test_trailing_text_is_emitted_with_the_final_tool_call(self):
        text = "Searching. [search(query='hello')] Done."
        self.assert_all_chunkings(text, "Searching.  Done.", [{"query": "hello"}])

    def test_repeated_blocks_have_distinct_indexes_and_drain_final_chunk(self):
        text = "[search(query='one')][search(query='two')] Done."
        self.assert_all_chunkings(text, " Done.", [{"query": "one"}, {"query": "two"}])

    def test_parallel_calls_and_later_block_keep_order(self):
        text = "[search(query='one'), search(query='two')][search(query='three')]"
        self.assert_all_chunkings(
            text, "", [{"query": value} for value in ("one", "two", "three")]
        )

    def test_incomplete_call_stays_buffered_until_outer_bracket(self):
        detector = PythonicDetector()
        result = detector.parse_streaming_increment(
            "[search(query='unfinished]')", self.tools
        )
        self.assertEqual(result.calls, [])
        self.assertEqual(result.normal_text, "")
        result = detector.parse_streaming_increment("] Done.", self.tools)
        self.assertEqual(result.normal_text, " Done.")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"query": "unfinished]"}
        )

    def test_special_markers_can_split_after_a_complete_call(self):
        text = "<|python_start|>[search(query='hello')]<|python_end|> Done."
        self.assert_all_chunkings(text, " Done.", [{"query": "hello"}])

    def test_bracketed_prose_with_apostrophes_is_not_a_call(self):
        for text in ("[O'Reilly] docs", "[don't panic] done", "[1] and []"):
            self.assert_all_chunkings(text, text, [])
        self.assert_all_chunkings(
            '[O\'Reilly] docs [search(query="]")] Done.',
            "[O'Reilly] docs  Done.",
            [{"query": "]"}],
        )


if __name__ == "__main__":
    unittest.main()
