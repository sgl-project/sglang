"""Unit tests for observability/utils.py - no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.observability.utils import (
    exponential_buckets,
    generate_buckets,
    two_sides_exponential_buckets,
)
from sglang.test.test_utils import CustomTestCase


class TestTwoSidesExponentialBuckets(CustomTestCase):
    def test_middle_value_always_present(self):
        """Middle value is always included in the output."""
        result = two_sides_exponential_buckets(5.0, 3.0, 6)
        self.assertIn(5.0, result)

    def test_symmetric_expansion(self):
        """Symmetric expansion around middle produces the correct sorted values."""
        # half_count = ceil(4/2) = 2
        # i=0: distance=2 -> 12.0, 8.0; i=1: distance=4 -> 14.0, 6.0
        result = two_sides_exponential_buckets(10.0, 2.0, 4)
        self.assertEqual(result, [6.0, 8.0, 10.0, 12.0, 14.0])

    def test_odd_count_applies_ceil(self):
        """Odd count: math.ceil determines half_count, producing an extra pair."""
        # count=3 -> half_count=ceil(3/2)=2
        # i=0: distance=2 -> 7.0, 3.0; i=1: distance=4 -> 9.0, 1.0
        result = two_sides_exponential_buckets(5.0, 2.0, 3)
        self.assertEqual(result, [1.0, 3.0, 5.0, 7.0, 9.0])

    def test_left_side_clamped_to_zero(self):
        """Negative left-side values are clamped to 0 and not emitted negative."""
        # half_count=1; i=0: distance=2 -> 3.0, max(0, 1.0-2)=0.0
        result = two_sides_exponential_buckets(1.0, 2.0, 2)
        self.assertNotIn(-1.0, result)
        self.assertIn(0.0, result)

    def test_repeated_zero_clamps_deduplicated(self):
        """Multiple left-side values clamped to 0 appear only once in the output."""
        # middle=0.0: all left distances clamp to 0.0; set removes duplicates
        result = two_sides_exponential_buckets(0.0, 2.0, 4)
        self.assertEqual(result.count(0.0), 1)

    def test_count_one(self):
        """count=1 produces middle plus exactly one expansion pair."""
        # half_count=ceil(1/2)=1; i=0: distance=2 -> 12.0, 8.0
        result = two_sides_exponential_buckets(10.0, 2.0, 1)
        self.assertIn(10.0, result)
        self.assertIn(12.0, result)
        self.assertIn(8.0, result)

    def test_zero_clamp_dedup_with_positive_middle(self):
        """Zero-clamped duplicates from a positive middle appear only once."""
        # middle=4.0, base=2.0, count=6 -> half_count=3
        # i=1: distance=4 -> max(0, 0.0)=0.0; i=2: distance=8 -> max(0,-4.0)=0.0
        # raw list has two 0.0s; deduplication must produce exactly one
        result = two_sides_exponential_buckets(4.0, 2.0, 6)
        self.assertEqual(result.count(0.0), 1)
        self.assertEqual(len(result), len(set(result)))


class TestGenerateBuckets(CustomTestCase):
    # unsorted and contains a duplicate to exercise both sort and dedup
    _DEFAULT = [5.0, 1.0, 3.0, 1.0]

    def test_empty_rule_falls_back_to_default(self):
        """Empty buckets_rule list returns sorted, deduplicated default_buckets."""
        result = generate_buckets([], self._DEFAULT)
        self.assertEqual(result, sorted(set(self._DEFAULT)))

    def test_default_rule_deduplicates_and_sorts(self):
        """Rule ['default'] returns sorted, deduplicated default_buckets."""
        result = generate_buckets(["default"], self._DEFAULT)
        self.assertEqual(result, sorted(set(self._DEFAULT)))

    def test_tse_rule_delegates_to_two_sides(self):
        """Rule ['tse', ...] output matches two_sides_exponential_buckets directly."""
        result = generate_buckets(["tse", "10.0", "2.0", "4"], [])
        expected = two_sides_exponential_buckets(10.0, 2.0, 4)
        self.assertEqual(result, expected)

    def test_tse_base_equal_one_raises(self):
        """Rule ['tse', ...] with base==1.0 raises AssertionError."""
        with self.assertRaises(AssertionError):
            generate_buckets(["tse", "5.0", "1.0", "4"], [])

    def test_tse_base_below_one_raises(self):
        """Rule ['tse', ...] with base<1.0 raises AssertionError."""
        with self.assertRaises(AssertionError):
            generate_buckets(["tse", "5.0", "0.5", "4"], [])

    def test_custom_rule_converts_sorts_and_deduplicates(self):
        """Rule ['custom', ...] converts remainder to float, sorts, and deduplicates."""
        result = generate_buckets(["custom", "3.0", "1.0", "2.0", "2.0"], [])
        self.assertEqual(result, [1.0, 2.0, 3.0])

    def test_unknown_rule_raises_assertion_error(self):
        """Unrecognized rule triggers AssertionError via the assert rule == 'custom' guard."""
        with self.assertRaises(AssertionError):
            generate_buckets(["unknown_rule"], [1.0])


class TestExponentialBuckets(CustomTestCase):
    def test_geometric_sequence(self):
        """Output is exactly start * (width ** i) for i in range(length)."""
        result = exponential_buckets(1.0, 2.0, 4)
        self.assertEqual(result, [1.0, 2.0, 4.0, 8.0])

    def test_length_of_output(self):
        """Output contains exactly `length` elements."""
        result = exponential_buckets(2.0, 3.0, 5)
        self.assertEqual(len(result), 5)

    def test_zero_length_returns_empty(self):
        """length=0 returns an empty list."""
        result = exponential_buckets(1.0, 2.0, 0)
        self.assertEqual(result, [])

    def test_width_one_produces_constant_sequence(self):
        """width=1.0 yields all values equal to start."""
        result = exponential_buckets(5.0, 1.0, 3)
        self.assertEqual(result, [5.0, 5.0, 5.0])

    def test_non_unit_start_scales_all_values(self):
        """Non-unit start scales every element proportionally."""
        result = exponential_buckets(2.0, 3.0, 3)
        self.assertAlmostEqual(result[0], 2.0)
        self.assertAlmostEqual(result[1], 6.0)
        self.assertAlmostEqual(result[2], 18.0)


if __name__ == "__main__":
    unittest.main()
