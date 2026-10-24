"""CPU contracts for expanding media placeholders in token space.

expand_token_placeholders(IDs, spec, start_len)
  [P 80 Q P], spec for P, Q -> each replaced once, insertions not rescanned
  [history | P], start_len = len(history) -> only the last expansion used
  start_len < 0, > len(IDs) or not int -> ValueError before matching
  placeholder count != media count -> ValueError
"""

import unittest

from sglang.srt.multimodal.token_space_mm_process.expansion import (
    expand_token_placeholders,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMMTokenExpansion(unittest.TestCase):
    def test_preserves_non_media_ids_and_does_not_expand_insertions_again(self):
        original_input_ids = [71, 72, 99, 80, 98, 99, 81]
        expanded_input_ids = expand_token_placeholders(
            original_input_ids,
            {99: [[99, 99], [99]], 98: [[98, 99, 98]]},
        )
        self.assertEqual(expanded_input_ids, [71, 72, 99, 99, 80, 98, 99, 98, 99, 81])
        self.assertEqual(original_input_ids, [71, 72, 99, 80, 98, 99, 81])
        self.assertEqual(
            expand_token_placeholders(original_input_ids, {}), original_input_ids
        )

    def test_suffix_takes_trailing_expansions_and_keeps_the_prefix(self):
        # [history, already expanded | PAD text] -> [history | last expansion text]
        history = [70, 99, 99, 71]
        original = history + [99, 72]
        self.assertEqual(
            expand_token_placeholders(
                original,
                {99: [[99, 99], [88, 99, 77]]},
                mm_token_expansion_start_len=len(history),
            ),
            history + [88, 99, 77, 72],
        )
        self.assertEqual(original, history + [99, 72])

    def test_validates_expansion_boundary_before_matching(self):
        input_ids = [10, 99]
        mm_token_expansion_spec = {99: [[99, 99]]}
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
            ([], {99: [[99]]}),
            ([99], {99: []}),
            ([99, 99], {99: [[99]]}),
        ]:
            with self.subTest(input_ids=input_ids), self.assertRaises(ValueError):
                expand_token_placeholders(input_ids, mm_token_expansion_spec)


if __name__ == "__main__":
    unittest.main()
