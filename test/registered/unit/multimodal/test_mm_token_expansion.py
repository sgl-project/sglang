"""CPU contracts for expanding media placeholders in token space.

partly expanded IDs + boundary -> suffix matcher -> IDs

The matcher preserves history and never scans inserted tokens.
"""

import unittest

from sglang.srt.multimodal.mm_token_expansion import expand_token_placeholders
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMMTokenExpansion(unittest.TestCase):
    def test_preserves_non_media_ids_and_does_not_expand_insertions_again(self):
        original_input_ids = [71, 72, 99, 80, 98, 99, 81]
        expanded_input_ids = expand_token_placeholders(
            original_input_ids,
            [([99], [[99, 99], [99]]), ([98], [[98, 99, 98]])],
        )
        self.assertEqual(expanded_input_ids, [71, 72, 99, 99, 80, 98, 99, 98, 99, 81])
        self.assertEqual(original_input_ids, [71, 72, 99, 80, 98, 99, 81])
        self.assertEqual(
            expand_token_placeholders(original_input_ids, []), original_input_ids
        )

    def test_matches_whole_patterns_and_preserves_untouched_prefix(self):
        # [history | START PAD PAD END text] -> [history | fragment text insertion]
        # The text anchor is retained in its replacement; callers provide no positions.
        for history in ([], [70, 99, 99, 71]):
            with self.subTest(history=history):
                original = history + [80, 99, 99, 81, 72]
                media_fragments = ([[99, 99]] if history else []) + [[88, 99, 77]]
                self.assertEqual(
                    expand_token_placeholders(
                        original,
                        [
                            ([80, 99, 99, 81], media_fragments),
                            ([72], [[72, 98, 99, 98]]),
                        ],
                        mm_token_expansion_start_len=len(history),
                    ),
                    history + [88, 99, 77, 72, 98, 99, 98],
                )
                self.assertEqual(original, history + [80, 99, 99, 81, 72])

    def test_adjacent_patterns_follow_rule_order_without_overlapping(self):
        # [START PAD][START PAD][PAD] -> [fragment 0][fragment 1][bare PAD fragment]
        # Inner PADs and lower-priority START rules must not consume fragments.
        self.assertEqual(
            expand_token_placeholders(
                [80, 99, 80, 99, 99],
                [([80, 99], [[88], [89]]), ([80], []), ([99], [[77]])],
            ),
            [88, 89, 77],
        )
        # A bare START after a complete match can still use the lower-priority rule.
        self.assertEqual(
            expand_token_placeholders(
                [80, 99, 80],
                [([80, 99], [[88]]), ([80], [[77]])],
            ),
            [88, 77],
        )
        with self.assertRaisesRegex(ValueError, "patterns must not be empty"):
            expand_token_placeholders([80], [([], [[99]])])

    def test_validates_expansion_boundary_before_matching(self):
        input_ids = [10, 99]
        mm_token_expansion_spec = [([99], [[99, 99]])]
        for start in (-1, 3, 0.5):
            with (
                self.subTest(start=start),
                self.assertRaisesRegex(ValueError, "mm_token_expansion_start_len"),
            ):
                expand_token_placeholders(input_ids, mm_token_expansion_spec, start)
        self.assertEqual(
            expand_token_placeholders(
                input_ids, mm_token_expansion_spec, len(input_ids)
            ),
            input_ids,
        )

    def test_rejects_missing_or_surplus_media(self):
        for input_ids, mm_token_expansion_spec in [
            ([], [([99], [[99]])]),
            ([99], [([99], [])]),
            ([80, 99, 81], [([80, 99, 81], [])]),
            ([80, 99], [([80, 99, 81], [[99]])]),
        ]:
            with self.subTest(input_ids=input_ids), self.assertRaises(ValueError):
                expand_token_placeholders(input_ids, mm_token_expansion_spec)

    def test_expansion_suffix_uses_trailing_media_and_preserves_prefix(self):
        # One modality's expanded fragment can contain another modality's marker.
        history = [71, 99, 99, 72, 98, 99, 98, 73]
        historical_expansion_spec = [([99], [[99, 99]]), ([98], [[98, 99, 98]])]
        mm_token_expansion_spec = [
            ([99], [[99, 99], [99, 99, 99]]),
            ([98], [[98, 99, 98]]),
        ]
        for suffix, expected_suffix in [([99, 74], [99, 99, 99, 74]), ([74], [74])]:
            with self.subTest(suffix=suffix):
                original = history + suffix
                self.assertEqual(
                    expand_token_placeholders(
                        original,
                        mm_token_expansion_spec
                        if 99 in suffix
                        else historical_expansion_spec,
                        mm_token_expansion_start_len=len(history),
                    ),
                    history + expected_suffix,
                )
        with self.assertRaises(ValueError):
            expand_token_placeholders(
                history + [98, 98], mm_token_expansion_spec, len(history)
            )


if __name__ == "__main__":
    unittest.main()
