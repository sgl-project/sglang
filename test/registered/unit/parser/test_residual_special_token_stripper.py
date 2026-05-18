"""Unit tests for srt/parser/residual_special_token_stripper.py"""

import unittest
from types import SimpleNamespace

from sglang.srt.parser.residual_special_token_stripper import (
    StreamingResidualStringStripper,
    get_residual_special_token_strings,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _added_token(content: str, special: bool = True) -> SimpleNamespace:
    return SimpleNamespace(content=content, special=special)


class _FakeTokenizer:
    def __init__(self, added_tokens_decoder):
        self.added_tokens_decoder = added_tokens_decoder


class TestGetResidualSpecialTokenStrings(CustomTestCase):
    def test_qwen_style_im_start_extracted(self):
        tok = _FakeTokenizer(
            {
                151644: _added_token("<|im_start|>"),
                151645: _added_token("<|im_end|>"),
                151643: _added_token("<|endoftext|>"),
            }
        )
        markers = get_residual_special_token_strings(tok)
        self.assertIn("<|im_start|>", markers)
        self.assertIn("<|im_end|>", markers)
        self.assertIn("<|endoftext|>", markers)

    def test_parser_consumed_markers_excluded_by_default(self):
        tok = _FakeTokenizer(
            {
                1: _added_token("<think>"),
                2: _added_token("</think>"),
                3: _added_token("<tool_call>"),
                4: _added_token("</tool_call>"),
                5: _added_token("<|im_start|>"),
            }
        )
        markers = get_residual_special_token_strings(tok)
        self.assertEqual(markers, ["<|im_start|>"])

    def test_non_special_tokens_ignored(self):
        tok = _FakeTokenizer(
            {
                1: _added_token("<|im_start|>", special=True),
                2: _added_token("regular_word", special=False),
            }
        )
        markers = get_residual_special_token_strings(tok)
        self.assertEqual(markers, ["<|im_start|>"])

    def test_extra_exclude_honored(self):
        tok = _FakeTokenizer(
            {
                1: _added_token("<|im_start|>"),
                2: _added_token("<|custom_marker|>"),
            }
        )
        markers = get_residual_special_token_strings(
            tok, extra_exclude={"<|custom_marker|>"}
        )
        self.assertEqual(markers, ["<|im_start|>"])

    def test_sorted_longest_first(self):
        tok = _FakeTokenizer(
            {
                1: _added_token("<|a|>"),
                2: _added_token("<|abc|>"),
                3: _added_token("<|ab|>"),
            }
        )
        markers = get_residual_special_token_strings(tok)
        self.assertEqual([len(m) for m in markers], sorted([7, 6, 5], reverse=True))

    def test_fallback_to_all_special_tokens(self):
        class _OldStyleTokenizer:
            added_tokens_decoder = None
            all_special_tokens = ["<|im_start|>", "<think>", "<|im_end|>"]

            def get_added_vocab(self):  # pragma: no cover — unused branch
                return {}

        markers = get_residual_special_token_strings(_OldStyleTokenizer())
        self.assertIn("<|im_start|>", markers)
        self.assertIn("<|im_end|>", markers)
        self.assertNotIn("<think>", markers)

    def test_empty_tokenizer(self):
        tok = _FakeTokenizer({})
        self.assertEqual(get_residual_special_token_strings(tok), [])


class TestStreamingResidualStringStripper(CustomTestCase):
    MARKER = "<|im_start|>"

    def _stream(self, stripper, chunks):
        out = "".join(stripper.feed(c) for c in chunks)
        return out + stripper.flush()

    def test_passthrough_when_no_markers(self):
        stripper = StreamingResidualStringStripper([])
        self.assertFalse(stripper.active)
        self.assertEqual(
            self._stream(stripper, ["hello ", "world"]), "hello world"
        )

    def test_marker_fully_in_one_chunk(self):
        stripper = StreamingResidualStringStripper([self.MARKER])
        self.assertTrue(stripper.active)
        result = self._stream(
            stripper, ["thinking... ", f"{self.MARKER}user", " continues"]
        )
        self.assertEqual(result, "thinking... user continues")

    def test_marker_split_across_chunks(self):
        stripper = StreamingResidualStringStripper([self.MARKER])
        chunks = ["before <|im_", "start|>after"]
        self.assertEqual(self._stream(stripper, chunks), "before after")

    def test_marker_split_byte_by_byte(self):
        stripper = StreamingResidualStringStripper([self.MARKER])
        chunks = ["x"] + list(self.MARKER) + ["y"]
        self.assertEqual(self._stream(stripper, chunks), "xy")

    def test_multiple_markers_in_one_chunk(self):
        stripper = StreamingResidualStringStripper([self.MARKER])
        result = self._stream(
            stripper, [f"a{self.MARKER}b{self.MARKER}c"]
        )
        self.assertEqual(result, "abc")

    def test_marker_at_boundary_then_flush(self):
        stripper = StreamingResidualStringStripper([self.MARKER])
        # First chunk ends with a prefix of the marker; second chunk
        # never completes it. Expect the partial to be flushed at end.
        chunks = ["okay <|im_", "start"]
        self.assertEqual(self._stream(stripper, chunks), "okay <|im_start")

    def test_partial_marker_only_flushed_on_close(self):
        stripper = StreamingResidualStringStripper([self.MARKER])
        # The model emits text that happens to be a prefix of a marker
        # but is never completed. We must NOT swallow it.
        early = stripper.feed("looks like <|im_")
        self.assertEqual(early, "looks like ")
        late = stripper.flush()
        self.assertEqual(late, "<|im_")

    def test_overlapping_markers_longest_match_wins(self):
        # If two markers share a prefix, longest-first sorting in
        # get_residual_special_token_strings ensures the longer is
        # matched. Here we pass both directly to verify the stripper.
        stripper = StreamingResidualStringStripper(
            ["<|im_start|>user", "<|im_start|>"]
        )
        self.assertEqual(
            self._stream(stripper, ["pre <|im_start|>user post"]),
            "pre  post",
        )

    def test_excluded_markers_preserved(self):
        # The stripper itself does not know about exclusions — that's the
        # caller's job. Verify that markers NOT passed in are preserved.
        stripper = StreamingResidualStringStripper([self.MARKER])
        result = self._stream(stripper, ["<think>reasoning</think>actual"])
        self.assertEqual(result, "<think>reasoning</think>actual")

    def test_empty_chunks_are_safe(self):
        stripper = StreamingResidualStringStripper([self.MARKER])
        self.assertEqual(stripper.feed(""), "")
        self.assertEqual(stripper.flush(), "")

    def test_flush_idempotent(self):
        stripper = StreamingResidualStringStripper([self.MARKER])
        # End with '<', which is a one-char prefix of the marker, so it
        # gets buffered until flush.
        early = stripper.feed("hello<")
        self.assertEqual(early, "hello")
        first = stripper.flush()
        second = stripper.flush()
        self.assertEqual(first, "<")
        self.assertEqual(second, "")


if __name__ == "__main__":
    unittest.main()
