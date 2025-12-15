import unittest

from sglang.srt.metrics.utils import generate_buckets, two_sides_exponential_buckets


class TestMetricsUtils(unittest.TestCase):
    """Test cases for metrics utility functions."""

    def test_two_sides_exponential_buckets_basic(self):
        """Test basic functionality of two_sides_exponential_buckets."""
        # Test with simple parameters
        count = 5
        buckets = two_sides_exponential_buckets(middle=10.0, base=2.0, count=count)

        # Should contain the middle value
        self.assertIn(10.0, buckets)

        # Should be sorted
        self.assertEqual(buckets, sorted(buckets))

        # Should have unique values (no duplicates)
        self.assertEqual(len(buckets), len(set(buckets)))

        # Should have reasonable number of buckets (not exactly count due to ceiling and deduplication)
        self.assertGreaterEqual(len(buckets), 3)
        self.assertLessEqual(len(buckets), count + 2)

    def test_two_sides_exponential_buckets_specific_values(self):
        """Test specific values for two_sides_exponential_buckets."""
        buckets = two_sides_exponential_buckets(middle=100.0, base=2.0, count=4)
        expected_values = [96.0, 98.0, 100.0, 102.0, 104.0]
        self.assertEqual(buckets, expected_values)

    def test_two_sides_exponential_buckets_negative_values(self):
        """Test two_sides_exponential_buckets with values that could go negative."""
        buckets = two_sides_exponential_buckets(middle=5.0, base=3.0, count=4)

        # Should not contain negative values (max(0, middle - distance))
        for bucket in buckets:
            self.assertGreaterEqual(bucket, 0.0)

        # Should contain the middle value
        self.assertIn(5.0, buckets)

    def test_two_sides_exponential_buckets_edge_cases(self):
        """Test edge cases for two_sides_exponential_buckets."""
        # Count = 1
        buckets = two_sides_exponential_buckets(middle=10.0, base=2.0, count=1)
        self.assertIn(10.0, buckets)

        # Very small middle value
        buckets = two_sides_exponential_buckets(middle=0.1, base=2.0, count=2)
        self.assertIn(0.1, buckets)
        for bucket in buckets:
            self.assertGreaterEqual(bucket, 0.0)

    def test_generate_buckets_default(self):
        """Test generate_buckets with default rule."""
        default_buckets = [1.0, 5.0, 10.0, 50.0, 100.0]

        # Test with "default" rule
        result = generate_buckets(["default"], default_buckets)
        self.assertEqual(result, default_buckets)

        # Test with None (should default to "default")
        result = generate_buckets(None, default_buckets)
        self.assertEqual(result, default_buckets)

        # Test with empty (should default to "default")
        result = generate_buckets(None, default_buckets)
        self.assertEqual(result, default_buckets)

    def test_generate_buckets_tse(self):
        """Test generate_buckets with tse (two sides exponential) rule."""
        default_buckets = [1.0, 5.0, 10.0]

        # Test with "tse" rule
        result = generate_buckets(["tse", "10", "2.0", "4"], default_buckets)

        # Should return the same as calling two_sides_exponential_buckets directly
        expected = two_sides_exponential_buckets(10.0, 2.0, 4)
        self.assertEqual(result, expected)

    def test_generate_buckets_custom(self):
        """Test generate_buckets with custom rule."""
        default_buckets = [1.0, 5.0, 10.0]

        # Test with "custom" rule
        result = generate_buckets(
            ["custom", "1.5", "3.2", "7.8", "15.6"], default_buckets
        )
        expected = [1.5, 3.2, 7.8, 15.6]
        self.assertEqual(result, expected)

    def test_generate_buckets_custom_with_integers(self):
        """Test generate_buckets with custom rule using integer strings."""
        default_buckets = [1.0, 5.0, 10.0]

        # Test with integer strings
        result = generate_buckets(["custom", "1", "5", "10", "50"], default_buckets)
        expected = [1.0, 5.0, 10.0, 50.0]
        self.assertEqual(result, expected)

    def test_generate_buckets_preserves_order_and_type(self):
        """Test that generate_buckets preserves order and returns floats."""
        default_buckets = [1, 5, 10, 50, 100]  # integers

        # Test default rule
        result = generate_buckets(["default"], default_buckets)
        self.assertEqual(result, default_buckets)
        self.assertIsInstance(result, list)

        # Test custom rule with proper float conversion
        result = generate_buckets(
            ["custom", "100", "50", "10", "5", "1"], default_buckets
        )
        expected = [1.0, 5.0, 10.0, 50.0, 100.0]
        self.assertEqual(result, expected)

        # All values should be floats
        for value in result:
            self.assertIsInstance(value, float)

    def test_integration_tse_through_generate_buckets(self):
        """Test integration of TSE buckets through generate_buckets function."""
        default_buckets = [1.0, 10.0, 100.0]

        # Generate buckets using both methods
        direct_result = two_sides_exponential_buckets(50.0, 1.5, 6)
        indirect_result = generate_buckets(["tse", "50.0", "1.5", "6"], default_buckets)

        # Results should be identical
        self.assertEqual(direct_result, indirect_result)


if __name__ == "__main__":
    unittest.main()
