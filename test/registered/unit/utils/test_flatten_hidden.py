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

    def test_kinds_are_in_an_order_the_rust_side_accepts(self):
        # A copy of `hidden_rowkinds_nest` in the Rust `message/response.rs`,
        # which rejects the frame otherwise. Change the two together.
        def nests(kinds):
            if kinds == [HIDDEN_ROW_WHOLE]:
                return True
            in_matrix = False
            for kind in kinds:
                if kind == HIDDEN_ROW_MATRIX_NEXT:
                    if not in_matrix:
                        return False
                elif kind not in (HIDDEN_ROW_VECTOR, HIDDEN_ROW_MATRIX_FIRST):
                    return False
                in_matrix = kind != HIDDEN_ROW_VECTOR
            return True

        for hidden_states in (
            None,
            [],
            [0.5, 0.25],
            [[0.5, 0.25]],
            [[[0.5, 0.25]]],
            [[[0.5], [1.5]], [2.5], [3.5]],
            # Two prefill steps, as a chunked prefill gives.
            [[[0.5], [1.5]], [[2.5]], [3.5]],
            [[[0.5]], [1.5], [[2.5], [3.5]], [4.5]],
            # An item with no rows.
            [[], [0.5]],
        ):
            with self.subTest(hidden_states=hidden_states):
                self.assertTrue(nests(flatten_hidden(hidden_states)[2]))

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
