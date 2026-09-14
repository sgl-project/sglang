"""Unit tests for srt/utils/token_sequence_matcher.py -- CPU, no server."""

import unittest
from typing import Optional, Sequence

from sglang.srt.utils.token_sequence_matcher import TokenSequenceMatcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _first_match_end(
    matcher: TokenSequenceMatcher, tokens: Sequence[int]
) -> Optional[int]:
    """Exclusive end index of the first full match, or None."""
    matched = 0
    for index, token in enumerate(tokens):
        matched = matcher.advance(matched, token)
        if matched == len(matcher):
            return index + 1
    return None


class TestTokenSequenceMatcher(CustomTestCase):
    def test_empty_pattern_raises(self):
        for pattern in ([], (), range(0)):
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(ValueError, "at least one token"):
                    TokenSequenceMatcher(pattern)

    def test_stores_an_independent_tuple_copy(self):
        src = [7, 8, 9]
        matcher = TokenSequenceMatcher(src)
        self.assertEqual(matcher.pattern, (7, 8, 9))
        self.assertEqual(len(matcher), 3)
        src[0] = 0
        self.assertEqual(matcher.pattern, (7, 8, 9))

    def test_prefix_lengths_no_self_overlap(self):
        matcher = TokenSequenceMatcher([1, 2, 3])
        self.assertEqual(matcher.prefix_lengths, (0, 0, 0))

    def test_prefix_lengths_repeated_token(self):
        # AAA -> LPS [0, 1, 2]
        matcher = TokenSequenceMatcher([1, 1, 1])
        self.assertEqual(matcher.prefix_lengths, (0, 1, 2))

    def test_prefix_lengths_periodic_prefix(self):
        # ABABC -> LPS [0, 0, 1, 2, 0]
        matcher = TokenSequenceMatcher([1, 2, 1, 2, 3])
        self.assertEqual(matcher.prefix_lengths, (0, 0, 1, 2, 0))

    def test_prefix_lengths_classic_kmp(self):
        # AABAACAABAA -> LPS [0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5]
        matcher = TokenSequenceMatcher([1, 1, 2, 1, 1, 3, 1, 1, 2, 1, 1])
        self.assertEqual(matcher.prefix_lengths, (0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5))

    def test_single_token_match_and_miss(self):
        matcher = TokenSequenceMatcher([42])
        self.assertEqual(matcher.prefix_lengths, (0,))
        self.assertEqual(matcher.advance(0, 41), 0)
        self.assertEqual(matcher.advance(0, 42), 1)
        self.assertEqual(_first_match_end(matcher, [41, 41, 42]), 3)

    def test_contiguous_match(self):
        matcher = TokenSequenceMatcher([10, 20, 30])
        self.assertEqual(_first_match_end(matcher, [1, 10, 20, 30, 99]), 4)
        self.assertIsNone(_first_match_end(matcher, [10, 20, 31]))

    def test_mismatch_falls_back_to_prefix(self):
        # ABABC in ABABABC: after ABAB the next A continues as prefix AB, not 0.
        matcher = TokenSequenceMatcher([1, 2, 1, 2, 3])
        self.assertEqual(_first_match_end(matcher, [1, 2, 1, 2, 1, 2, 3]), 7)

    def test_mismatch_walks_prefix_chain(self):
        # ABAC in ABAABAC: mismatch at C walks LPS twice (3 -> 1 -> 0+1).
        matcher = TokenSequenceMatcher([1, 2, 1, 3])
        self.assertEqual(matcher.prefix_lengths, (0, 0, 1, 0))
        self.assertEqual(_first_match_end(matcher, [1, 2, 1, 1, 2, 1, 3]), 7)

    def test_self_overlapping_run_reports_first_full_match(self):
        matcher = TokenSequenceMatcher([1, 1, 1])
        self.assertEqual(_first_match_end(matcher, [1, 1, 1, 1]), 3)

    def test_false_start_then_real_match(self):
        matcher = TokenSequenceMatcher([1, 2, 3])
        self.assertEqual(_first_match_end(matcher, [1, 2, 1, 2, 3]), 5)

    def test_partial_state_survives_across_chunks(self):
        # Callers keep matched between separately arriving token chunks.
        matcher = TokenSequenceMatcher([8, 9])
        matched = matcher.advance(0, 8)
        self.assertEqual(matched, 1)
        matched = matcher.advance(matched, 7)
        self.assertEqual(matched, 0)
        matched = matcher.advance(matched, 8)
        self.assertEqual(matched, 1)
        matched = matcher.advance(matched, 9)
        self.assertEqual(matched, 2)
        self.assertEqual(matched, len(matcher))

    def test_mismatch_at_first_token_stays_zero(self):
        matcher = TokenSequenceMatcher([5, 6])
        self.assertEqual(matcher.advance(0, 6), 0)


if __name__ == "__main__":
    unittest.main()
