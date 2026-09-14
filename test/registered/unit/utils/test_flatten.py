"""CPU contract tests for srt/utils/flatten.py."""

import math
import unittest
from array import array

from sglang.srt.utils.flatten import (
    FlatPairColumns,
    NestedRowColumns,
    RaggedPairColumns,
    flatten_hidden,
    flatten_ragged,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _decode_array(typecode: str, payload: bytes) -> list:
    decoded = array(typecode)
    decoded.frombytes(payload)
    return decoded.tolist()


class TestFlattenHelpers(CustomTestCase):
    def test_flatten_ragged_preserves_positions_and_pairs(self):
        values = [[0.25, -1.5], None, [], [2.0]]
        token_ids = [[7, 8], None, [], [9]]

        flat_values, flat_ids, position_lengths = flatten_ragged(values, token_ids)

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
        values, row_lengths = flatten_hidden([[1, 2.5], [], 3, [[4], [5.5, 6]]])

        self.assertEqual(values, [1.0, 2.5, 3.0, 4.0, 5.5, 6.0])
        self.assertEqual(row_lengths, [2, 0, 1, 3])

    def test_flatten_hidden_accepts_empty_input(self):
        for hidden_states in (None, []):
            with self.subTest(hidden_states=hidden_states):
                self.assertEqual(flatten_hidden(hidden_states), ([], []))


class TestFlatPairColumns(CustomTestCase):
    def test_columns_expose_original_sources(self):
        values = [[0.25]]
        token_ids = [[7]]
        columns = FlatPairColumns("scores", values, token_ids)

        self.assertEqual(
            columns.columns(),
            (("scores_val", values), ("scores_idx", token_ids)),
        )

    def test_accumulates_requests_and_encodes_32_bit_buffers(self):
        values = [[None, 0.5], [1.25], None]
        token_ids = [[101, 102], [103], None]
        columns = FlatPairColumns(
            "scores",
            values,
            token_ids,
            first_none_to_nan=True,
        )

        for request_index in range(3):
            columns.accept(request_index)

        self.assertEqual(array("f").itemsize, 4)
        self.assertEqual(array("i").itemsize, 4)
        self.assertEqual(columns.header_cols(), [[2, 1, 0]])

        value_bytes, index_bytes = columns.data_cols()
        self.assertEqual(len(value_bytes), 3 * 4)
        self.assertEqual(len(index_bytes), 3 * 4)

        decoded_values = _decode_array("f", value_bytes)
        self.assertTrue(math.isnan(decoded_values[0]))
        self.assertAlmostEqual(decoded_values[1], 0.5)
        self.assertAlmostEqual(decoded_values[2], 1.25)
        self.assertEqual(_decode_array("i", index_bytes), [101, 102, 103])

    def test_rejects_mismatched_value_and_index_counts(self):
        columns = FlatPairColumns(
            "scores",
            [[0.25, 0.5]],
            [[7]],
        )

        with self.assertRaisesRegex(
            AssertionError,
            "scores: request 0 has 1 idx entries but 2 vals",
        ):
            columns.accept(0)


class TestNestedColumns(CustomTestCase):
    def test_ragged_pairs_accumulate_request_and_position_lengths(self):
        values = [
            [[0.1, 0.2], None, [0.3]],
            None,
            [[], [1.5]],
        ]
        token_ids = [
            [[11, 12], None, [13]],
            None,
            [[], [14]],
        ]
        columns = RaggedPairColumns("top", values, token_ids)

        for request_index in range(3):
            columns.accept(request_index)

        self.assertEqual(
            columns.columns(),
            (("top_val", values), ("top_idx", token_ids)),
        )
        self.assertEqual(columns.header_cols(), [[3, 0, 2], [2, 0, 1, 0, 1]])

        value_bytes, index_bytes = columns.data_cols()
        self.assertEqual(len(value_bytes), 4 * 4)
        self.assertEqual(len(index_bytes), 4 * 4)

        decoded_values = _decode_array("f", value_bytes)
        for actual, expected in zip(decoded_values, [0.1, 0.2, 0.3, 1.5]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(_decode_array("i", index_bytes), [11, 12, 13, 14])

    def test_hidden_rows_accumulate_request_and_row_lengths(self):
        rows = [
            [[1, 2], [[3], [4.5]]],
            None,
            [[], 6],
        ]
        columns = NestedRowColumns("hidden_states", rows)

        for request_index in range(3):
            columns.accept(request_index)

        self.assertEqual(columns.columns(), (("hidden_states", rows),))
        self.assertEqual(columns.header_cols(), [[2, 0, 2], [2, 2, 0, 1]])

        (value_bytes,) = columns.data_cols()
        self.assertEqual(len(value_bytes), 5 * 4)
        decoded_values = _decode_array("f", value_bytes)
        for actual, expected in zip(decoded_values, [1, 2, 3, 4.5, 6]):
            self.assertAlmostEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
