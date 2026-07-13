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

register_cpu_ci(est_time=16, suite="base-a-test-cpu")


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


def _single_hash_compatibility_cases():
    prior_hash = _legacy_get_hash_str([7, 8, 9])
    return [
        ("empty_list", [], None),
        ("plain_list", [1, 2, 3, 4, 5], None),
        ("array_q", _HashKey(array("q", range(1, 258))), None),
        ("array_i_with_prior", _HashKey(array("I", range(1, 258))), prior_hash),
        (
            "tuple_bigram_with_prior",
            [(10, 20), (20, 30), (30, 40), (40, 50)],
            prior_hash,
        ),
        (
            "eagle_bigram_with_prior",
            _HashKey(
                array("q", ((i * 2654435761) & 0x00FFFFFF for i in range(258))),
                is_bigram=True,
            ),
            prior_hash,
        ),
        ("empty_bigram", _HashKey(array("q"), is_bigram=True), None),
    ]


def _page_hash_compatibility_cases():
    prior_hash = _legacy_get_hash_str([7, 8, 9])
    return [
        ("empty_list", [], 8, None),
        ("empty_bigram", _HashKey(array("q"), is_bigram=True), 8, None),
        ("array_q_page_64", _HashKey(array("q", range(1, 258))), 64, None),
        (
            "array_i_page_64_with_prior",
            _HashKey(array("I", range(1, 258))),
            64,
            prior_hash,
        ),
        (
            "eagle_bigram_page_64_with_prior",
            _HashKey(
                array("q", ((i * 2654435761) & 0x00FFFFFF for i in range(258))),
                is_bigram=True,
            ),
            64,
            prior_hash,
        ),
        (
            "eagle_bigram_page_1_with_prior",
            _HashKey(array("q", [11, 22, 33, 44, 55]), is_bigram=True),
            1,
            prior_hash,
        ),
    ]


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
    def test_hash_str_matches_pre_optimization_per_token_loop(self):
        for name, tokens, prior_hash in _single_hash_compatibility_cases():
            with self.subTest(name=name):
                self.assertEqual(
                    get_hash_str(tokens, prior_hash),
                    _legacy_get_hash_str(tokens, prior_hash),
                )

    def test_page_hashes_match_pre_optimization_per_token_loop(self):
        for name, tokens, page_size, prior_hash in _page_hash_compatibility_cases():
            with self.subTest(name=name):
                self.assertEqual(
                    get_hash_str(tokens, prior_hash, page_size=page_size),
                    _legacy_page_hashes(tokens, page_size, prior_hash),
                )

    def test_hash_properties(self):
        self.assertEqual(get_hash_str([]), hashlib.sha256().hexdigest())

        h1 = get_hash_str([1, 2, 3])
        h2 = get_hash_str([3, 2, 1])
        self.assertNotEqual(h1, h2)

        self.assertNotEqual(get_hash_str([100]), get_hash_str([200]))
        self.assertEqual(get_hash_str([1, 2]), get_hash_str([(1, 2)]))
        self.assertNotEqual(get_hash_str([(1, 2)]), get_hash_str([(2, 1)]))

        chained = get_hash_str([3, 4], prior_hash=get_hash_str([1, 2]))
        self.assertNotEqual(chained, get_hash_str([3, 4]))
        self.assertNotEqual(
            chained, get_hash_str([3, 4], prior_hash=get_hash_str([9, 9]))
        )
        for tokens in [[], [1], [1, 2, 3], [(1, 2)], [1, 2, 3, 4, 5]]:
            with self.subTest(tokens=tokens):
                self.assertRegex(get_hash_str(tokens), r"^[0-9a-f]{64}$")

    def test_hash_key_hash_page_matches_get_hash_str(self):
        key = _HashKey(array("q", [1, 2, 3, 4, 5, 6]), is_bigram=True)
        prior_hash = get_hash_str([(9, 10)])

        self.assertEqual(
            key.hash_page(1, 4, prior_hash),
            get_hash_str(key[1:4], prior_hash),
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

    def test_root_node_hashes_match_legacy_page_hashes(self):
        cases = [
            ("single_page", _HashKey(array("q", [1, 2, 3])), 16),
            ("multiple_pages", _HashKey(array("q", range(1, 31))), 16),
            ("page_aligned_boundary", _HashKey(array("q", range(1, 33))), 8),
            ("shorter_than_page", _HashKey(array("q", [1, 2, 3, 4, 5])), 16),
        ]

        for name, key, page_size in cases:
            with self.subTest(name=name):
                node = self._make_node(key)
                self.assertEqual(
                    compute_node_hash_values(node, page_size=page_size),
                    _legacy_page_hashes(key, page_size=page_size),
                )

    def test_parent_hash_is_used_only_when_parent_has_nonempty_key_and_hash(self):
        parent = MagicMock()
        parent.key = _HashKey(array("q", range(1, 17)))
        parent.hash_value = _legacy_page_hashes(parent.key, page_size=8)
        child_key = _HashKey(array("q", range(101, 109)))
        cases = [
            ("valid_parent_hash", parent, parent.hash_value, parent.hash_value[-1]),
            (
                "empty_parent_key",
                self._make_node(_HashKey(array("q"))),
                [get_hash_str([1, 2, 3])],
                None,
            ),
            (
                "empty_parent_hash",
                self._make_node(_HashKey(array("q", range(1, 9)))),
                [],
                None,
            ),
            (
                "none_parent_hash",
                self._make_node(_HashKey(array("q", range(1, 9)))),
                None,
                None,
            ),
        ]

        for name, parent, parent_hash_values, expected_prior in cases:
            with self.subTest(name=name):
                child = self._make_node(
                    child_key, parent=parent, parent_hash_values=parent_hash_values
                )
                self.assertEqual(
                    compute_node_hash_values(child, page_size=8),
                    _legacy_page_hashes(
                        child_key, page_size=8, prior_hash=expected_prior
                    ),
                )


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
