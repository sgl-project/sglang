"""Unit tests for `flatten_hidden` in srt/utils/flatten.py — no server, no model loading."""

import unittest

from sglang.srt.utils.flatten import (
    HIDDEN_ROW_MATRIX_FIRST,
    HIDDEN_ROW_MATRIX_NEXT,
    HIDDEN_ROW_VECTOR,
    HIDDEN_ROW_WHOLE,
    NestedRowColumns,
    flatten_hidden,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestFlattenHidden(CustomTestCase):
    def test_prefill_matrix_sends_one_row_per_token(self):
        # `return_hidden_states=True`: two prefill tokens, then one decode step.
        vals, lens, kinds = flatten_hidden([[[0.5, 0.25], [-1.0, 2.0]], [4.0, 8.0]])
        self.assertEqual(vals, [0.5, 0.25, -1.0, 2.0, 4.0, 8.0])
        self.assertEqual(lens, [2, 2, 2])
        self.assertEqual(
            kinds,
            [HIDDEN_ROW_MATRIX_FIRST, HIDDEN_ROW_MATRIX_NEXT, HIDDEN_ROW_VECTOR],
        )

    def test_one_token_prefill_is_still_a_matrix(self):
        # The rest of the prompt came from the prefix cache.
        _, lens, kinds = flatten_hidden([[[0.5, 0.25]], [4.0, 8.0]])
        self.assertEqual(lens, [2, 2])
        self.assertEqual(kinds, [HIDDEN_ROW_MATRIX_FIRST, HIDDEN_ROW_VECTOR])

    def test_second_prefill_starts_a_new_matrix(self):
        _, _, kinds = flatten_hidden([[[0.5]], [1.0], [[2.0], [3.0]]])
        self.assertEqual(
            kinds,
            [
                HIDDEN_ROW_MATRIX_FIRST,
                HIDDEN_ROW_VECTOR,
                HIDDEN_ROW_MATRIX_FIRST,
                HIDDEN_ROW_MATRIX_NEXT,
            ],
        )

    def test_last_vector_is_one_row(self):
        # `return_hidden_states="last"`: the value is the vector itself.
        vals, lens, kinds = flatten_hidden([0.5, 0.25, -1.0, 2.0])
        self.assertEqual(vals, [0.5, 0.25, -1.0, 2.0])
        self.assertEqual(lens, [4])
        self.assertEqual(kinds, [HIDDEN_ROW_WHOLE])

    def test_no_hidden_states_is_empty(self):
        self.assertEqual(flatten_hidden(None), ([], [], []))
        self.assertEqual(flatten_hidden([]), ([], [], []))

    def test_columns_carry_a_kind_per_row(self):
        columns = NestedRowColumns("h", [[[[0.5], [1.5]], [2.5]], None, [3.5]])
        columns.fill(3)
        req, pos, kind = columns.header_cols()
        self.assertEqual(req, [3, 0, 1])
        self.assertEqual(pos, [1, 1, 1, 1])
        self.assertEqual(
            kind,
            [
                HIDDEN_ROW_MATRIX_FIRST,
                HIDDEN_ROW_MATRIX_NEXT,
                HIDDEN_ROW_VECTOR,
                HIDDEN_ROW_WHOLE,
            ],
        )


if __name__ == "__main__":
    unittest.main()
