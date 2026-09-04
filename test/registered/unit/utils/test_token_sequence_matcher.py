"""CPU contract tests for srt/utils/token_sequence_matcher.py."""

import unittest

from sglang.srt.utils.token_sequence_matcher import TokenSequenceMatcher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _advance_stream(matcher, matched, tokens):
    states = []
    for token in tokens:
        matched = matcher.advance(matched, token)
        states.append(matched)
    return matched, states


class TestTokenSequenceMatcher(CustomTestCase):
    def test_rejects_empty_patterns(self):
        for pattern in ([], ()):
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(
                    ValueError, "pattern must contain at least one token"
                ):
                    TokenSequenceMatcher(pattern)

    def test_copies_mutable_pattern_input(self):
        pattern = [11, 22]
        matcher = TokenSequenceMatcher(pattern)

        pattern[1] = 99

        matched, states = _advance_stream(matcher, 0, [11, 22])
        self.assertEqual(len(matcher), 2)
        self.assertEqual(states, [1, 2])
        self.assertEqual(matched, len(matcher))

    def test_falls_back_through_overlapping_prefix(self):
        matcher = TokenSequenceMatcher([1, 2, 1, 3])

        matched, states = _advance_stream(matcher, 0, [1, 2, 1, 2, 1, 3])

        self.assertEqual(states, [1, 2, 3, 2, 3, 4])
        self.assertEqual(matched, len(matcher))

    def test_recovers_after_mismatch(self):
        matcher = TokenSequenceMatcher([4, 5, 4])

        matched, states = _advance_stream(matcher, 0, [4, 5, 9, 4, 5, 4])

        self.assertEqual(states, [1, 2, 0, 1, 2, 3])
        self.assertEqual(matched, len(matcher))

    def test_preserves_partial_match_across_chunks(self):
        matcher = TokenSequenceMatcher([101, 102, 103])

        matched, first_states = _advance_stream(matcher, 0, [7, 101])
        matched, second_states = _advance_stream(matcher, matched, [102])
        matched, third_states = _advance_stream(matcher, matched, [103])

        self.assertEqual(first_states, [0, 1])
        self.assertEqual(second_states, [2])
        self.assertEqual(third_states, [3])
        self.assertEqual(matched, len(matcher))

    def test_matches_single_token_pattern(self):
        matcher = TokenSequenceMatcher([42])

        self.assertEqual(matcher.advance(0, 7), 0)
        self.assertEqual(matcher.advance(0, 42), 1)


if __name__ == "__main__":
    unittest.main()
