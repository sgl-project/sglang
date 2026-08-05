"""CPU contract tests for srt/utils/flatten.py."""

import unittest

from sglang.srt.utils.flatten import flatten_hidden, flatten_ragged
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFlattenHelpers(CustomTestCase):
    def test_flatten_ragged_preserves_positions_and_pairs(self):
        values = [[0.25, -1.5], None, [], [2.0]]
        token_ids = [[7, 8], None, [], [9]]

        flat_values, flat_ids, position_lengths = flatten_ragged(
            values, token_ids
        )

        self.assertEqual(flat_values, [0.25, -1.5, 2.0])
        self.assertEqual(flat_ids, [7, 8, 9])
        self.assertEqual(position_lengths, [2, 0, 0, 1])

    def test_flatten_ragged_accepts_empty_columns(self):
        for values, token_ids in ((None, None), ([], [])):
            with self.subTest(values=values, token_ids=token_ids):
                self.assertEqual(
                    flatten_ragged(values, token_ids),
                    ([], [], []),
                )

    def test_flatten_ragged_rejects_malformed_pairs(self):
        cases = (
            ([[0.5]], [], "positions"),
            ([[0.5, 0.75]], [[3]], "position 0"),
            ([None], [[3]], "val is empty"),
        )

        for values, token_ids, message in cases:
            with self.subTest(values=values, token_ids=token_ids):
                with self.assertRaisesRegex(AssertionError, message):
                    flatten_ragged(values, token_ids)

    def test_flatten_hidden_preserves_recursive_row_boundaries(self):
        values, row_lengths = flatten_hidden(
            [[1, 2.5], [], 3, [[4], [5.5, 6]]]
        )

        self.assertEqual(values, [1.0, 2.5, 3.0, 4.0, 5.5, 6.0])
        self.assertEqual(row_lengths, [2, 0, 1, 3])

    def test_flatten_hidden_accepts_empty_input(self):
        for hidden_states in (None, []):
            with self.subTest(hidden_states=hidden_states):
                self.assertEqual(flatten_hidden(hidden_states), ([], []))


if __name__ == "__main__":
    unittest.main()
