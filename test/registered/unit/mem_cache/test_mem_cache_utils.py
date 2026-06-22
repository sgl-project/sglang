"""Unit tests for mem_cache/utils.py — no server, no model loading."""

import hashlib
import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.mem_cache.evict_policy import (
    FIFOStrategy,
    FILOStrategy,
    LFUStrategy,
    LRUStrategy,
    MRUStrategy,
    PriorityStrategy,
    SLRUStrategy,
)
from sglang.srt.mem_cache.utils import (
    compute_node_hash_values,
    get_eviction_strategy,
    get_hash_str,
    hash_str_to_int64,
    maybe_init_custom_mem_pool,
    split_node_hash_value,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestGetEvictionStrategy(CustomTestCase):
    def test_lru(self):
        self.assertIsInstance(get_eviction_strategy("lru"), LRUStrategy)

    def test_lfu(self):
        self.assertIsInstance(get_eviction_strategy("lfu"), LFUStrategy)

    def test_fifo(self):
        self.assertIsInstance(get_eviction_strategy("fifo"), FIFOStrategy)

    def test_mru(self):
        self.assertIsInstance(get_eviction_strategy("mru"), MRUStrategy)

    def test_filo(self):
        self.assertIsInstance(get_eviction_strategy("filo"), FILOStrategy)

    def test_priority(self):
        self.assertIsInstance(get_eviction_strategy("priority"), PriorityStrategy)

    def test_slru(self):
        self.assertIsInstance(get_eviction_strategy("slru"), SLRUStrategy)

    def test_case_insensitive(self):
        self.assertIsInstance(get_eviction_strategy("LRU"), LRUStrategy)
        self.assertIsInstance(get_eviction_strategy("Lru"), LRUStrategy)
        self.assertIsInstance(get_eviction_strategy("FIFO"), FIFOStrategy)

    def test_unknown_policy_raises_valueerror(self):
        with self.assertRaises(ValueError) as ctx:
            get_eviction_strategy("nonexistent")
        msg = str(ctx.exception)
        self.assertIn("Unknown eviction policy", msg)
        self.assertIn("lru", msg)
        self.assertIn("lfu", msg)
        self.assertIn("fifo", msg)
        self.assertIn("mru", msg)
        self.assertIn("filo", msg)
        self.assertIn("priority", msg)
        self.assertIn("slru", msg)

    def test_each_call_creates_new_instance(self):
        s1 = get_eviction_strategy("lru")
        s2 = get_eviction_strategy("lru")
        self.assertIsNot(s1, s2)


class TestMaybeInitCustomMemPool(CustomTestCase):
    @patch("sglang.srt.mem_cache.utils.envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get")
    def test_disabled_by_default(self, mock_env_get):
        mock_env_get.return_value = None
        enabled, pool, pool_type = maybe_init_custom_mem_pool("cuda:0")
        self.assertFalse(enabled)
        self.assertIsNone(pool)
        self.assertIsNone(pool_type)

    @patch("sglang.srt.mem_cache.utils.envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get")
    @patch("sglang.srt.disaggregation.mooncake.utils.init_mooncake_custom_mem_pool")
    def test_enabled_via_env(self, mock_init, mock_env_get):
        mock_env_get.return_value = "enabled"
        mock_init.return_value = (True, "mock_pool_instance", "mooncake")

        enabled, pool, pool_type = maybe_init_custom_mem_pool("cuda:0")
        self.assertTrue(enabled)
        self.assertEqual(pool, "mock_pool_instance")
        self.assertEqual(pool_type, "mooncake")
        mock_init.assert_called_once_with("cuda:0")


class _TokenView:
    def __init__(self, tokens, extra_key):
        self.tokens = tokens
        self.extra_key = extra_key

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return _TokenView(self.tokens[item], self.extra_key)
        return self.tokens[item]


class TestGetHashStr(CustomTestCase):
    def test_empty_list(self):
        result = get_hash_str([])
        self.assertIsInstance(result, str)
        self.assertEqual(len(result), 64)
        expected = hashlib.sha256().hexdigest()
        self.assertEqual(result, expected)

    def test_different_sequence_different_hash(self):
        h1 = get_hash_str([1, 2, 3])
        h2 = get_hash_str([3, 2, 1])
        self.assertNotEqual(h1, h2)

    def test_different_values_different_hash(self):
        h1 = get_hash_str([100])
        h2 = get_hash_str([200])
        self.assertNotEqual(h1, h2)

    def test_bigram_vs_flat_token_equivalent(self):
        h_flat = get_hash_str([1, 2])
        h_bigram = get_hash_str([(1, 2)])
        self.assertEqual(h_flat, h_bigram)

    def test_bigram_order_matters(self):
        h1 = get_hash_str([(1, 2)])
        h2 = get_hash_str([(2, 1)])
        self.assertNotEqual(h1, h2)

    def test_prior_hash_chaining(self):
        chained = get_hash_str([3, 4], prior_hash=get_hash_str([1, 2]))
        # prior_hash must fold into the digest, so chaining differs from
        # hashing [3, 4] alone...
        self.assertNotEqual(chained, get_hash_str([3, 4]))
        # ...and a different prior_hash must yield a different chained digest.
        self.assertNotEqual(
            chained, get_hash_str([3, 4], prior_hash=get_hash_str([9, 9]))
        )

    def test_extra_key_namespaces_first_step(self):
        salted_a = get_hash_str([1, 2, 3, 4], extra_key="salt-A")
        salted_b = get_hash_str([1, 2, 3, 4], extra_key="salt-B")
        unsalted = get_hash_str([1, 2, 3, 4])

        self.assertNotEqual(salted_a, salted_b)
        self.assertNotEqual(salted_a, unsalted)
        self.assertNotEqual(get_hash_str([1, 2, 3, 4], extra_key=""), unsalted)

    def test_extra_key_is_carried_by_token_view_slices(self):
        token_view = _TokenView([1, 2, 3, 4, 5], "salt-A")

        first_hash = get_hash_str(token_view[:4])
        self.assertEqual(first_hash, get_hash_str([1, 2, 3, 4], extra_key="salt-A"))

        second_hash = get_hash_str(token_view[4:5], prior_hash=first_hash)
        self.assertEqual(
            second_hash, get_hash_str([5], prior_hash=first_hash, extra_key="salt-A")
        )

    def test_prior_hash_single_step(self):
        step1 = get_hash_str([1])
        step2 = get_hash_str([2], prior_hash=step1)
        direct = get_hash_str([1, 2])
        self.assertNotEqual(step2, direct)

    def test_returns_64_char_hex(self):
        for tokens in [[], [1], [1, 2, 3], [(1, 2)], [1, 2, 3, 4, 5]]:
            result = get_hash_str(tokens)
            self.assertRegex(result, r"^[0-9a-f]{64}$")


class TestHashStrToInt64(CustomTestCase):
    def test_zero_hash(self):
        result = hash_str_to_int64("0" * 64)
        self.assertEqual(result, 0)

    def test_small_positive_value(self):
        result = hash_str_to_int64("0000000000000001" + "0" * 48)
        self.assertEqual(result, 1)

    def test_large_positive_value(self):
        result = hash_str_to_int64("7fffffffffffffff" + "0" * 48)
        self.assertEqual(result, 2**63 - 1)

    def test_negative_overflow(self):
        result = hash_str_to_int64("8000000000000000" + "0" * 48)
        self.assertEqual(result, -(2**63))

    def test_max_unsigned_maps_to_minus_one(self):
        result = hash_str_to_int64("f" * 16 + "0" * 48)
        self.assertEqual(result, -1)

    def test_only_first_16_chars_matter(self):
        h1 = hash_str_to_int64("a" * 16 + "0" * 48)
        h2 = hash_str_to_int64("a" * 16 + "f" * 48)
        self.assertEqual(h1, h2)

    def test_roundtrip_with_get_hash_str(self):
        hash_hex = get_hash_str([42, 99])
        int64_val = hash_str_to_int64(hash_hex)
        self.assertIsInstance(int64_val, int)
        self.assertTrue(-(2**63) <= int64_val < 2**63)


class TestComputeNodeHashValues(CustomTestCase):
    def setUp(self):
        def mock_hash_page(start, end, parent_hash):
            parts = [f"p{start}-{end}"]
            if parent_hash is not None:
                parts.append(parent_hash)
            return "-".join(parts)

        self.mock_hash_page = mock_hash_page

    def _make_node(self, key_len, parent=None, parent_hash_values=None):
        node = MagicMock()
        node.key.__len__.return_value = key_len
        node.key.hash_page = self.mock_hash_page
        node.parent = parent
        if parent is not None:
            parent.hash_value = parent_hash_values
        return node

    def test_single_page_root(self):
        node = self._make_node(key_len=3)
        result = compute_node_hash_values(node, page_size=16)
        self.assertEqual(len(result), 1)
        self.assertIn("p0-3", result[0])

    def test_multiple_pages(self):
        node = self._make_node(key_len=30)
        result = compute_node_hash_values(node, page_size=16)
        self.assertEqual(len(result), 2)
        self.assertIn("p0-16", result[0])
        self.assertIn("p16-30", result[1])

    def test_page_aligned_boundary(self):
        node = self._make_node(key_len=32)
        result = compute_node_hash_values(node, page_size=8)
        self.assertEqual(len(result), 4)
        self.assertIn("p24-32", result[3])

    def test_key_shorter_than_page_size(self):
        node = self._make_node(key_len=5)
        result = compute_node_hash_values(node, page_size=16)
        self.assertEqual(result, ["p0-5"])

    def test_chained_parent_hash(self):
        parent = MagicMock()
        parent.key.__len__.return_value = 8
        parent.hash_value = ["parent_hash_0", "parent_hash_1"]
        parent.key.hash_page = self.mock_hash_page

        child = self._make_node(
            key_len=16, parent=parent, parent_hash_values=parent.hash_value
        )
        result = compute_node_hash_values(child, page_size=8)
        self.assertEqual(result, ["p0-8-parent_hash_1", "p8-16-p0-8-parent_hash_1"])

    def test_parent_with_empty_key(self):
        parent = MagicMock()
        parent.key.__len__.return_value = 0
        parent.hash_value = ["some_hash"]
        parent.key.hash_page = self.mock_hash_page

        child = self._make_node(
            key_len=8, parent=parent, parent_hash_values=parent.hash_value
        )
        result = compute_node_hash_values(child, page_size=8)
        self.assertEqual(len(result), 1)
        self.assertNotIn("some_hash", result[0])

    def test_parent_without_hash_value(self):
        parent = MagicMock()
        parent.key.__len__.return_value = 8
        parent.hash_value = []
        parent.key.hash_page = self.mock_hash_page

        child = self._make_node(key_len=8, parent=parent, parent_hash_values=[])
        result = compute_node_hash_values(child, page_size=8)
        self.assertEqual(result, ["p0-8"])

    def test_parent_with_none_hash_value(self):
        parent = MagicMock()
        parent.key.__len__.return_value = 8
        parent.hash_value = None
        parent.key.hash_page = self.mock_hash_page

        child = self._make_node(key_len=8, parent=parent, parent_hash_values=None)
        result = compute_node_hash_values(child, page_size=8)
        self.assertEqual(result, ["p0-8"])


class TestSplitNodeHashValue(CustomTestCase):
    def test_none_input_returns_none_tuple(self):
        result = split_node_hash_value(None, 10, 4)
        self.assertEqual(result, (None, None))

    def test_page_size_one_split(self):
        hash_values = [f"hash_{i}" for i in range(4)]
        new_hash, child_hash = split_node_hash_value(hash_values, 2, 1)
        self.assertEqual(new_hash, ["hash_0", "hash_1"])
        self.assertEqual(child_hash, ["hash_2", "hash_3"])

    def test_page_size_one_split_at_zero(self):
        hash_values = ["a", "b", "c"]
        new_hash, child_hash = split_node_hash_value(hash_values, 0, 1)
        self.assertEqual(new_hash, [])
        self.assertEqual(child_hash, ["a", "b", "c"])

    def test_page_size_one_split_at_len(self):
        hash_values = ["a", "b", "c"]
        new_hash, child_hash = split_node_hash_value(hash_values, 3, 1)
        self.assertEqual(new_hash, ["a", "b", "c"])
        self.assertEqual(child_hash, [])

    def test_page_size_larger_than_one(self):
        hash_values = ["p0", "p1", "p2", "p3", "p4", "p5"]
        new_hash, child_hash = split_node_hash_value(hash_values, 4, 2)
        self.assertEqual(new_hash, ["p0", "p1"])
        self.assertEqual(child_hash, ["p2", "p3", "p4", "p5"])

    def test_page_size_split_exact_page_boundary(self):
        hash_values = ["a", "b", "c", "d"]
        new_hash, child_hash = split_node_hash_value(hash_values, 4, 4)
        self.assertEqual(new_hash, ["a"])
        self.assertEqual(child_hash, ["b", "c", "d"])

    def test_page_size_split_within_page(self):
        hash_values = ["a", "b", "c"]
        new_hash, child_hash = split_node_hash_value(hash_values, 6, 4)
        self.assertEqual(new_hash, ["a"])
        self.assertEqual(child_hash, ["b", "c"])

    def test_total_length_preserved(self):
        hash_values = [f"h_{i}" for i in range(10)]
        for split_len in [0, 1, 3, 5, 7, 9, 10]:
            for page_size in [1, 2, 4]:
                new_hash, child_hash = split_node_hash_value(
                    hash_values, split_len, page_size
                )
                self.assertEqual(
                    len(new_hash) + len(child_hash),
                    len(hash_values),
                    f"split_len={split_len}, page_size={page_size}",
                )


if __name__ == "__main__":
    unittest.main()
