"""Unit tests for mem_cache/utils.py — no server, no model loading."""

import hashlib
import sys
import types
import unittest
from array import array
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

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _legacy_get_hash_str(token_ids, prior_hash=None):
    hasher = hashlib.sha256()
    if prior_hash:
        hasher.update(bytes.fromhex(prior_hash))
    for t in token_ids:
        if isinstance(t, tuple):
            for elem in t:
                hasher.update(elem.to_bytes(4, byteorder="little", signed=False))
        else:
            hasher.update(t.to_bytes(4, byteorder="little", signed=False))
    return hasher.hexdigest()


def _legacy_page_hashes(key, page_size, prior_hash=None):
    hashes = []
    running_hash = prior_hash
    for start in range(0, len(key), page_size):
        running_hash = _legacy_get_hash_str(
            key[start : start + page_size], running_hash
        )
        hashes.append(running_hash)
    return hashes


class _HashKey:
    def __init__(self, token_ids, is_bigram=False):
        self.token_ids = token_ids
        self.is_bigram = is_bigram

    def __len__(self):
        if self.is_bigram:
            return max(0, len(self.token_ids) - 1)
        return len(self.token_ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start = index.start or 0
            stop = index.stop if index.stop is not None else len(self)
            if self.is_bigram:
                return _HashKey(self.token_ids[start : stop + 1], is_bigram=True)
            return _HashKey(self.token_ids[start:stop])
        if self.is_bigram:
            return (self.token_ids[index], self.token_ids[index + 1])
        return self.token_ids[index]

    def raw_token_ids(self):
        return self.token_ids

    def hash_page(self, start, end, prior_hash=None):
        return _legacy_get_hash_str(self[start:end], prior_hash)


class TestOptimizedHashCompatibility(unittest.TestCase):
    def test_hash_str_matches_pre_optimization_per_token_loop(self):
        prior_hash = _legacy_get_hash_str([7, 8, 9])
        cases = [
            ("list", [1, 2, 3, 4, 5], None),
            ("array_q", _HashKey(array("q", range(1, 258))), None),
            ("array_i", _HashKey(array("I", range(1, 258))), prior_hash),
            (
                "tuple_bigram",
                [(10, 20), (20, 30), (30, 40), (40, 50)],
                prior_hash,
            ),
            (
                "eagle_bigram",
                _HashKey(
                    array("q", ((i * 2654435761) & 0x00FFFFFF for i in range(258))),
                    is_bigram=True,
                ),
                prior_hash,
            ),
        ]

        for name, tokens, prior_hash in cases:
            with self.subTest(name=name):
                self.assertEqual(
                    get_hash_str(tokens, prior_hash),
                    _legacy_get_hash_str(tokens, prior_hash),
                )

    def test_page_hashes_match_pre_optimization_per_token_loop(self):
        prior_hash = _legacy_get_hash_str([7, 8, 9])
        cases = [
            ("array_q_page_64", _HashKey(array("q", range(1, 258))), 64, None),
            (
                "array_i_page_64_with_prior",
                _HashKey(array("I", range(1, 258))),
                64,
                prior_hash,
            ),
            (
                "eagle_bigram_page_64",
                _HashKey(
                    array("q", ((i * 2654435761) & 0x00FFFFFF for i in range(258))),
                    is_bigram=True,
                ),
                64,
                prior_hash,
            ),
            (
                "eagle_bigram_page_1",
                _HashKey(array("q", [11, 22, 33, 44, 55]), is_bigram=True),
                1,
                prior_hash,
            ),
        ]

        for name, tokens, page_size, prior_hash in cases:
            with self.subTest(name=name):
                self.assertEqual(
                    get_hash_str(tokens, prior_hash, page_size=page_size),
                    _legacy_page_hashes(tokens, page_size, prior_hash),
                )


class TestGetEvictionStrategy(unittest.TestCase):
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


class TestMaybeInitCustomMemPool(unittest.TestCase):
    @patch("sglang.srt.mem_cache.utils.envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get")
    def test_disabled_by_default(self, mock_env_get):
        mock_env_get.return_value = None
        enabled, pool, pool_type = maybe_init_custom_mem_pool("cuda:0")
        self.assertFalse(enabled)
        self.assertIsNone(pool)
        self.assertIsNone(pool_type)

    @patch("sglang.srt.mem_cache.utils.envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get")
    def test_enabled_via_env(self, mock_env_get):
        mock_env_get.return_value = "enabled"
        mock_init = MagicMock()
        mock_init.return_value = (True, "mock_pool_instance", "mooncake")

        mooncake_pkg = types.ModuleType("sglang.srt.disaggregation.mooncake")
        mooncake_utils = types.ModuleType("sglang.srt.disaggregation.mooncake.utils")
        mooncake_utils.init_mooncake_custom_mem_pool = mock_init
        with patch.dict(
            sys.modules,
            {
                "sglang.srt.disaggregation.mooncake": mooncake_pkg,
                "sglang.srt.disaggregation.mooncake.utils": mooncake_utils,
            },
        ):
            enabled, pool, pool_type = maybe_init_custom_mem_pool("cuda:0")
        self.assertTrue(enabled)
        self.assertEqual(pool, "mock_pool_instance")
        self.assertEqual(pool_type, "mooncake")
        mock_init.assert_called_once_with("cuda:0")


class TestGetHashStr(unittest.TestCase):
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

    def test_prior_hash_single_step(self):
        step1 = get_hash_str([1])
        step2 = get_hash_str([2], prior_hash=step1)
        direct = get_hash_str([1, 2])
        self.assertNotEqual(step2, direct)

    def test_returns_64_char_hex(self):
        for tokens in [[], [1], [1, 2, 3], [(1, 2)], [1, 2, 3, 4, 5]]:
            result = get_hash_str(tokens)
            self.assertRegex(result, r"^[0-9a-f]{64}$")

    def test_hash_key_matches_legacy_loop(self):
        prior_hash = get_hash_str([7, 8, 9])
        cases = [
            ("unigram", _HashKey(array("q", range(1, 34))), None),
            (
                "bigram",
                _HashKey(array("q", [10, 20, 30, 40, 50, 60, 70]), is_bigram=True),
                None,
            ),
            (
                "prior_hash",
                _HashKey(array("q", [101, 102, 103, 104, 105])),
                prior_hash,
            ),
        ]

        for name, key, prior_hash in cases:
            with self.subTest(name=name):
                self.assertEqual(
                    get_hash_str(key, prior_hash),
                    _legacy_get_hash_str(key, prior_hash),
                )

    def test_hash_key_hash_page_matches_get_hash_str(self):
        key = _HashKey(array("q", [1, 2, 3, 4, 5, 6]), is_bigram=True)
        prior_hash = get_hash_str([(9, 10)])

        self.assertEqual(
            key.hash_page(1, 4, prior_hash),
            get_hash_str(key[1:4], prior_hash),
        )

    def test_empty_bigram_hash_matches_empty_hash(self):
        key = _HashKey(array("q"), is_bigram=True)
        self.assertEqual(get_hash_str(key), hashlib.sha256().hexdigest())


class TestGetHashStrPageMode(unittest.TestCase):
    def test_page_mode_matches_legacy_loop(self):
        prior_hash = get_hash_str([7, 8, 9])
        cases = [
            ("empty", [], 8, None),
            ("empty_bigram", _HashKey(array("q"), is_bigram=True), 8, None),
            ("unigram", _HashKey(array("q", range(1, 34))), 8, None),
            (
                "bigram",
                _HashKey(array("q", [10, 20, 30, 40, 50, 60, 70]), is_bigram=True),
                2,
                None,
            ),
            (
                "prior_hash",
                _HashKey(array("q", [101, 102, 103, 104, 105])),
                2,
                prior_hash,
            ),
        ]

        for name, key, page_size, prior_hash in cases:
            with self.subTest(name=name):
                self.assertEqual(
                    get_hash_str(key, prior_hash, page_size=page_size),
                    _legacy_page_hashes(key, page_size, prior_hash),
                )


class TestHashStrToInt64(unittest.TestCase):
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


class TestComputeNodeHashValues(unittest.TestCase):
    def _make_node(self, key, parent=None, parent_hash_values=None):
        node = MagicMock()
        node.key = key
        node.parent = parent
        if parent is not None:
            parent.hash_value = parent_hash_values
        return node

    def test_single_page_root(self):
        key = _HashKey(array("q", [1, 2, 3]))
        node = self._make_node(key)
        result = compute_node_hash_values(node, page_size=16)
        self.assertEqual(result, _legacy_page_hashes(key, page_size=16))

    def test_multiple_pages(self):
        key = _HashKey(array("q", range(1, 31)))
        node = self._make_node(key)
        result = compute_node_hash_values(node, page_size=16)
        self.assertEqual(result, _legacy_page_hashes(key, page_size=16))

    def test_page_aligned_boundary(self):
        key = _HashKey(array("q", range(1, 33)))
        node = self._make_node(key)
        result = compute_node_hash_values(node, page_size=8)
        self.assertEqual(result, _legacy_page_hashes(key, page_size=8))

    def test_key_shorter_than_page_size(self):
        key = _HashKey(array("q", [1, 2, 3, 4, 5]))
        node = self._make_node(key)
        result = compute_node_hash_values(node, page_size=16)
        self.assertEqual(result, _legacy_page_hashes(key, page_size=16))

    def test_chained_parent_hash(self):
        parent = MagicMock()
        parent.key = _HashKey(array("q", range(1, 17)))
        parent.hash_value = _legacy_page_hashes(parent.key, page_size=8)

        child_key = _HashKey(array("q", range(101, 117)))
        child = self._make_node(
            child_key, parent=parent, parent_hash_values=parent.hash_value
        )
        result = compute_node_hash_values(child, page_size=8)
        self.assertEqual(
            result,
            _legacy_page_hashes(
                child_key, page_size=8, prior_hash=parent.hash_value[-1]
            ),
        )

    def test_parent_with_empty_key(self):
        parent = MagicMock()
        parent.key = _HashKey(array("q"))
        parent.hash_value = [get_hash_str([1, 2, 3])]

        child_key = _HashKey(array("q", range(1, 9)))
        child = self._make_node(
            child_key, parent=parent, parent_hash_values=parent.hash_value
        )
        result = compute_node_hash_values(child, page_size=8)
        self.assertEqual(result, _legacy_page_hashes(child_key, page_size=8))

    def test_parent_without_hash_value(self):
        parent = MagicMock()
        parent.key = _HashKey(array("q", range(1, 9)))
        parent.hash_value = []

        child_key = _HashKey(array("q", range(101, 109)))
        child = self._make_node(child_key, parent=parent, parent_hash_values=[])
        result = compute_node_hash_values(child, page_size=8)
        self.assertEqual(result, _legacy_page_hashes(child_key, page_size=8))

    def test_parent_with_none_hash_value(self):
        parent = MagicMock()
        parent.key = _HashKey(array("q", range(1, 9)))
        parent.hash_value = None

        child_key = _HashKey(array("q", range(101, 109)))
        child = self._make_node(child_key, parent=parent, parent_hash_values=None)
        result = compute_node_hash_values(child, page_size=8)
        self.assertEqual(result, _legacy_page_hashes(child_key, page_size=8))


class TestSplitNodeHashValue(unittest.TestCase):
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
