"""Unit tests for RadixKey.match — #20865."""

import unittest
from array import array

from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _key(tokens, **kwargs) -> RadixKey:
    return RadixKey(array("q", tokens), **kwargs)


class TestRadixKeyMatch(unittest.TestCase):
    def test_identical_prefix(self):
        a = _key([1, 2, 3, 4, 5])
        b = _key([1, 2, 3, 9, 9])
        self.assertEqual(a.match(b), 3)

    def test_no_shared_prefix(self):
        a = _key([10, 11, 12])
        b = _key([20, 21, 22])
        self.assertEqual(a.match(b), 0)

    def test_long_shared_prefix(self):
        shared = list(range(200))
        a = _key(shared + [99, 100])
        b = _key(shared + [88, 77])
        self.assertEqual(a.match(b), 200)

    def test_page_size_rounds_down(self):
        a = _key([1, 2, 3, 4, 5, 6, 7])
        b = _key([1, 2, 3, 4, 5, 6, 9])
        self.assertEqual(a.match(b, page_size=4), 4)

    def test_extra_key_mismatch(self):
        a = _key([1, 2, 3], extra_key="lora_a")
        b = _key([1, 2, 3], extra_key="lora_b")
        with self.assertRaises(ValueError) as ctx:
            a.match(b)
        self.assertIn("extra_key", str(ctx.exception))

    def test_cache_salt_mismatch(self):
        a = _key([1, 2, 3], cache_salt="s1")
        b = _key([1, 2, 3], cache_salt="s2")
        with self.assertRaises(ValueError) as ctx:
            a.match(b)
        self.assertIn("cache_salt", str(ctx.exception))

    def test_bigram_counts_logical_units(self):
        # raw tokens [a,b,c,d] -> 3 bigrams; share first two bigrams -> match 2
        a = _key([1, 2, 3, 4], is_bigram=True)
        b = _key([1, 2, 3, 9], is_bigram=True)
        self.assertEqual(a.match(b), 2)
        self.assertEqual(len(a), 3)

    def test_limit_caps_reported_length(self):
        a = _key([1, 2, 3, 4, 5], limit=3)
        b = _key([1, 2, 3, 4, 5])
        self.assertEqual(len(a), 3)
        self.assertEqual(a.match(b), 3)


class TestRadixKeyPageAligned(unittest.TestCase):
    def test_truncates_to_page_boundary(self):
        k = _key(list(range(10)))
        aligned = k.page_aligned(4)
        self.assertEqual(len(aligned), 8)
        self.assertEqual(list(aligned.token_ids[:8]), list(range(8)))


if __name__ == "__main__":
    unittest.main()
