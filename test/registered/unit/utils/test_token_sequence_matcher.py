"""Unit tests for the token-sequence matcher used by reasoning parsers."""

import unittest

from sglang.srt.utils.token_sequence_matcher import TokenSequenceMatcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestTokenSequenceMatcher(CustomTestCase):
    def test_rejects_empty_pattern(self):
        with self.assertRaisesRegex(
            ValueError, "pattern must contain at least one token"
        ):
            TokenSequenceMatcher([])

    def test_builds_prefix_lengths_for_overlapping_pattern(self):
        matcher = TokenSequenceMatcher([1, 2, 1, 2, 1])

        self.assertEqual(len(matcher), 5)
        self.assertEqual(matcher.pattern, (1, 2, 1, 2, 1))
        self.assertEqual(matcher.prefix_lengths, (0, 0, 1, 2, 3))

    def test_advance_matches_tokens(self):
        matcher = TokenSequenceMatcher([1, 2, 1, 2])
        matched = 0

        for token in [1, 2, 1, 2]:
            matched = matcher.advance(matched, token)

        self.assertEqual(matched, len(matcher))

        # A completed match can be restarted by the caller without creating a
        # new matcher.
        matched = 0
        for token in [1, 2, 1, 2]:
            matched = matcher.advance(matched, token)
        self.assertEqual(matched, len(matcher))

    def test_advance_falls_back_after_mismatch(self):
        matcher = TokenSequenceMatcher([1, 2, 1, 3])
        matched = 0

        for token in [1, 2, 1, 2, 1, 3]:
            matched = matcher.advance(matched, token)

        self.assertEqual(matched, len(matcher))

    def test_advance_ignores_unmatched_tokens(self):
        matcher = TokenSequenceMatcher([4, 5])

        self.assertEqual(matcher.advance(0, 9), 0)
        self.assertEqual(matcher.advance(0, 4), 1)
        self.assertEqual(matcher.advance(1, 9), 0)


if __name__ == "__main__":
    unittest.main()
