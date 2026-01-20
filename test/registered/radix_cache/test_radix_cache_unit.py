"""
Unit tests for the RadixCache implementation.

This module tests the core functionality of RadixCache, RadixKey, and TreeNode
following SGLang testing patterns.

Test Coverage:
- RadixKey: token ID management, slicing, iteration, representation
- TreeNode: node properties, reference counting, hash values
- RadixCache: insert/match operations, eviction, page alignment, error handling
- Cache events and request handling
- Boundary conditions with parameterized testing

Usage:
    python test_radix_cache_unit.py
    python -m pytest test_radix_cache_unit.py -v
    python -m pytest test_radix_cache_unit.py::TestRadixCache::test_insert_basic
"""

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

# CPU-based unit test, runs quickly on any GPU runner
register_cuda_ci(est_time=5, suite="stage-b-test-small-1-gpu")
register_amd_ci(est_time=5, suite="stage-b-test-small-1-gpu-amd")

import random
import time
import unittest
import unittest.mock

import torch

from sglang.srt.disaggregation.kv_events import BlockRemoved, BlockStored
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey, TreeNode

# Test constants
DEFAULT_PAGE_SIZE = 4


class TestRadixKey(unittest.TestCase):
    """Test cases for RadixKey class."""

    def test_init_basic(self):
        """Test basic initialization of RadixKey."""
        token_ids = [1, 2, 3, 4]
        key = RadixKey(token_ids)
        self.assertEqual(key.token_ids, token_ids)
        self.assertIsNone(key.extra_key)

    def test_init_with_extra_key(self):
        """Test initialization with extra_key."""
        token_ids = [1, 2, 3]
        extra_key = "test_key"
        key = RadixKey(token_ids, extra_key)
        self.assertEqual(key.token_ids, token_ids)
        self.assertEqual(key.extra_key, extra_key)

    def test_len(self):
        """Test __len__ method."""
        key = RadixKey([1, 2, 3])
        self.assertEqual(len(key), 3)

        empty_key = RadixKey([])
        self.assertEqual(len(empty_key), 0)

    def test_iter(self):
        """Test __iter__ method."""
        token_ids = [1, 2, 3, 4]
        key = RadixKey(token_ids)
        self.assertEqual(list(key), token_ids)

    def test_len_and_iter(self):
        """Test __len__ and __iter__ methods."""
        test_cases = [
            ([1, 2, 3], 3),
            ([], 0),
            ([42], 1),
        ]

        for tokens, expected in test_cases:
            with self.subTest(tokens=tokens):
                key = RadixKey(tokens)
                self.assertEqual(len(key), expected)
                self.assertEqual(list(key), tokens)

    def test_getitem_int(self):
        """Test __getitem__ with int index."""
        test_cases = [
            ([10, 20, 30], 0, [10]),
            ([10, 20, 30], -1, [30]),
            ([10, 20, 30], 2, [30]),
        ]

        for tokens, index, expected in test_cases:
            with self.subTest(tokens=tokens, index=index):
                key = RadixKey(tokens)
                result = key[index]
                self.assertIsInstance(result, RadixKey)
                self.assertEqual(result.token_ids, expected)

    def test_getitem_slice(self):
        """Test __getitem__ with slice and edge cases."""
        key = RadixKey([1, 2, 3, 4, 5], "extra")

        # Basic slice
        sliced = key[1:4]
        self.assertIsInstance(sliced, RadixKey)
        self.assertEqual(sliced.token_ids, [2, 3, 4])
        self.assertEqual(sliced.extra_key, "extra")

        # Edge cases
        self.assertEqual(key[2:2].token_ids, [])  # Empty slice
        self.assertEqual(key[:].token_ids, [1, 2, 3, 4, 5])  # Full slice

    def test_getitem_invalid_index(self):
        """Test __getitem__ with invalid indices."""
        key = RadixKey([1, 2, 3])
        with self.assertRaises(IndexError):
            _ = key[10]  # Out of bounds

    def test_repr(self):
        """Test __repr__ method."""
        key = RadixKey([1, 2, 3], "test")
        repr_str = repr(key)
        self.assertIn("RadixKey", repr_str)
        self.assertIn("extra_key='test'", repr_str)
        self.assertIn("[1, 2, 3]", repr_str)

    def test_repr_long_token_ids(self):
        """Test __repr__ with long token_ids."""
        long_tokens = list(range(15))
        key = RadixKey(long_tokens)
        repr_str = repr(key)
        self.assertIn("...", repr_str)  # Should be truncated


class TestTreeNode(unittest.TestCase):
    """Test cases for TreeNode class."""

    def setUp(self):
        """Reset the counter before each test."""
        TreeNode.counter = 0

    def test_init_basic(self):
        """Test basic initialization of TreeNode."""
        node = TreeNode()
        self.assertEqual(node.id, 0)
        self.assertEqual(len(node.children), 0)
        self.assertIsNone(node.parent)
        self.assertIsNone(node.key)
        self.assertIsNone(node.value)
        self.assertEqual(node.lock_ref, 0)
        self.assertEqual(node.hit_count, 0)
        self.assertEqual(node.host_ref_counter, 0)
        self.assertIsNone(node.host_value)
        self.assertIsNone(node.hash_value)

    def test_init_with_id(self):
        """Test initialization with custom ID."""
        node = TreeNode(id=42)
        self.assertEqual(node.id, 42)
        node2 = TreeNode()
        self.assertEqual(node2.id, 1)  # Counter was incremented

    def test_counter_increment(self):
        """Test that counter increments properly."""
        node1 = TreeNode()
        node2 = TreeNode()
        self.assertEqual(node1.id, 0)
        self.assertEqual(node2.id, 1)

    def test_evicted_backuped_properties(self):
        """Test evicted and backuped properties."""
        test_cases = [
            (False, False, True, False),
            (True, False, False, False),
            (True, True, False, True),
            (False, True, True, True),
        ]

        for (
            has_value,
            has_host_value,
            expected_evicted,
            expected_backuped,
        ) in test_cases:
            with self.subTest(has_value=has_value, has_host_value=has_host_value):
                node = TreeNode()

                if has_value:
                    node.value = torch.tensor([1, 2, 3])
                if has_host_value:
                    node.host_value = torch.tensor([4, 5, 6])

                self.assertEqual(node.evicted, expected_evicted)
                self.assertEqual(node.backuped, expected_backuped)

    def test_protect_release_host(self):
        """Test protect_host and release_host methods."""
        node = TreeNode()
        self.assertEqual(node.host_ref_counter, 0)

        node.protect_host()
        self.assertEqual(node.host_ref_counter, 1)

        node.release_host()
        self.assertEqual(node.host_ref_counter, 0)

        # Test error case
        with self.assertRaises(RuntimeError):
            node.release_host()

    def test_get_last_hash_value(self):
        """Test get_last_hash_value method."""
        node = TreeNode()
        self.assertIsNone(node.get_last_hash_value())

        node.hash_value = ["hash1", "hash2", "hash3"]
        self.assertEqual(node.get_last_hash_value(), "hash3")

    def test_lt_comparison(self):
        """Test less than comparison based on last_access_time."""
        node1 = TreeNode()
        time.sleep(0.001)  # Small delay to ensure different timestamps
        node2 = TreeNode()

        self.assertTrue(node1 < node2)
        self.assertFalse(node2 < node1)


class TestRadixCache(unittest.TestCase):
    """Test cases for RadixCache class."""

    def setUp(self):
        """Set up test fixtures."""
        TreeNode.counter = 0

    def test_init_variations(self):
        """Test cache initialization with different parameters."""
        test_cases = [
            (1, False, False),
            (4, False, True),
            (1, True, False),
        ]

        for page_size, disable, enable_events in test_cases:
            with self.subTest(
                page_size=page_size, disable=disable, enable_events=enable_events
            ):
                cache = RadixCache.create_simulated(
                    disable=disable,
                    page_size=page_size,
                    enable_kv_cache_events=enable_events,
                )

                self.assertEqual(cache.page_size, page_size)
                self.assertEqual(cache.disable, disable)
                self.assertEqual(cache.enable_kv_cache_events, enable_events)
                self.assertEqual(cache.device, torch.device("cpu"))
                self.assertIsNotNone(cache.root_node)
                self.assertEqual(len(cache.root_node.key), 0)

    def test_reset(self):
        """Test reset method."""
        cache = RadixCache.create_simulated()

        # Insert some data
        cache.insert(RadixKey([1, 2, 3]), torch.tensor([10, 20, 30], dtype=torch.int64))
        self.assertGreater(cache.total_size(), 0)

        # Reset
        cache.reset()
        self.assertEqual(cache.total_size(), 0)
        self.assertEqual(cache.evictable_size(), 0)
        self.assertEqual(cache.protected_size(), 0)

    def test_insert_and_match_basic(self):
        """Test basic insert and match operations."""
        for disable_cache in [False, True]:
            with self.subTest(disable_cache=disable_cache):
                cache = RadixCache.create_simulated(disable=disable_cache)

                key = RadixKey([1, 2, 3])
                value = torch.tensor([10, 20, 30], dtype=torch.int64)
                prefix_len = cache.insert(key, value)

                if disable_cache:
                    self.assertEqual(prefix_len, 0)
                    self.assertEqual(cache.total_size(), 0)
                    continue

                self.assertEqual(prefix_len, 0)  # No existing prefix
                self.assertEqual(cache.total_size(), 3)
                self.assertEqual(cache.evictable_size(), 3)

                # Test match_prefix
                result = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3])))
                self.assertEqual(len(result.device_indices), 3)
                torch.testing.assert_close(result.device_indices, value)

                # Test partial match
                result = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2])))
                self.assertEqual(len(result.device_indices), 2)
                torch.testing.assert_close(
                    result.device_indices, torch.tensor([10, 20], dtype=torch.int64)
                )

    def test_insert_with_none_value(self):
        """Test insert with None value (should use token_ids as list)."""
        cache = RadixCache.create_simulated()

        key = RadixKey([1, 2, 3])
        prefix_len = cache.insert(key, None)

        # When None is passed, it should create value from token_ids
        self.assertEqual(prefix_len, 0)
        self.assertEqual(cache.total_size(), 3)

    def test_total_size(self):
        """Test total_size calculation."""
        cache = RadixCache.create_simulated()

        self.assertEqual(cache.total_size(), 0)

        cache.insert(RadixKey([1, 2, 3]), torch.tensor([10, 20, 30], dtype=torch.int64))
        self.assertEqual(cache.total_size(), 3)

        cache.insert(RadixKey([4, 5]), torch.tensor([40, 50], dtype=torch.int64))
        self.assertEqual(cache.total_size(), 5)

    def test_kv_cache_events(self):
        """Test KV cache events functionality."""
        test_cases = [
            (1, True),
            (2, True),
            (1, False),
        ]

        for page_size, enable_events in test_cases:
            with self.subTest(page_size=page_size, enable_events=enable_events):
                cache = RadixCache.create_simulated(
                    page_size=page_size, enable_kv_cache_events=enable_events
                )

                # Insert data
                cache.insert(RadixKey([1, 2, 3, 4, 5]), None)

                # Take events
                events = cache.take_events()

                if enable_events:
                    self.assertGreater(len(events), 0)
                    # Verify events include BlockStored events (there might be other event types)
                    block_stored_events = [
                        e for e in events if isinstance(e, BlockStored)
                    ]
                    self.assertGreater(len(block_stored_events), 0)
                    for event in block_stored_events:
                        self.assertLessEqual(len(event.token_ids), page_size)
                else:
                    self.assertEqual(len(events), 0)

    def test_kv_cache_events_with_eviction(self):
        """Test KV cache events include removal events."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache.create_simulated(
            mock_allocator=mock_allocator, enable_kv_cache_events=True
        )

        # Insert and then evict data
        cache.insert(RadixKey([1, 2, 3]), torch.tensor([10, 20, 30], dtype=torch.int64))
        cache.evict(3)

        # Take events - should include both store and remove events
        events = cache.take_events()
        self.assertGreater(len(events), 0)

        # Check event types
        event_types = [type(event).__name__ for event in events]
        self.assertIn("BlockStored", event_types)

        # Verify BlockRemoved event content
        remove_events = [e for e in events if isinstance(e, BlockRemoved)]
        for event in remove_events:
            self.assertGreater(len(event.block_hashes), 0)

    def test_extra_key_isolation(self):
        """Test that keys with different extra_key values are isolated."""
        cache = RadixCache.create_simulated()

        # Insert same token sequence with different extra keys
        cache.insert(
            RadixKey([1, 2, 3], "key1"), torch.tensor([10, 20, 30], dtype=torch.int64)
        )
        cache.insert(
            RadixKey([1, 2, 3], "key2"), torch.tensor([40, 50, 60], dtype=torch.int64)
        )
        cache.insert(
            RadixKey([1, 2, 3], None), torch.tensor([70, 80, 90], dtype=torch.int64)
        )

        # Keys with different extra_key should not match each other
        result1 = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3], "key1")))
        result2 = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3], "key2")))
        result3 = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3], None)))
        result4 = cache.match_prefix(
            MatchPrefixParams(key=RadixKey([1, 2, 3], "nonexistent"))
        )

        # Each should match only its own data
        self.assertEqual(len(result1.device_indices), 3)
        torch.testing.assert_close(
            result1.device_indices, torch.tensor([10, 20, 30], dtype=torch.int64)
        )

        self.assertEqual(len(result2.device_indices), 3)
        torch.testing.assert_close(
            result2.device_indices, torch.tensor([40, 50, 60], dtype=torch.int64)
        )

        self.assertEqual(len(result3.device_indices), 3)
        torch.testing.assert_close(
            result3.device_indices, torch.tensor([70, 80, 90], dtype=torch.int64)
        )

        # Non-existent extra_key should not match
        self.assertEqual(len(result4.device_indices), 0)

    def test_lock_ref_operations(self):
        """Test lock reference counting operations."""
        cache = RadixCache.create_simulated()

        # Insert sequence
        cache.insert(RadixKey([1, 2, 3]), torch.tensor([10, 20, 30], dtype=torch.int64))

        # Get node
        result = cache.match_prefix(MatchPrefixParams(key=RadixKey([1, 2, 3])))
        node = result.last_device_node

        initial_evictable = cache.evictable_size()
        initial_protected = cache.protected_size()

        # Lock the node
        cache.inc_lock_ref(node)
        self.assertEqual(cache.protected_size(), initial_protected + 3)
        self.assertEqual(cache.evictable_size(), initial_evictable - 3)

        # Unlock the node
        cache.dec_lock_ref(node)
        self.assertEqual(cache.protected_size(), initial_protected)
        self.assertEqual(cache.evictable_size(), initial_evictable)

    def test_evict_functionality(self):
        """Test eviction functionality."""
        mock_allocator = unittest.mock.Mock()
        mock_allocator.device = torch.device("cpu")

        cache = RadixCache.create_simulated(mock_allocator=mock_allocator)

        # Insert sequences
        cache.insert(RadixKey([1, 2]), torch.tensor([10, 20], dtype=torch.int64))
        cache.insert(RadixKey([3, 4]), torch.tensor([30, 40], dtype=torch.int64))

        initial_size = cache.total_size()

        # Evict some tokens
        cache.evict(2)

        # Should have called free and reduced size
        mock_allocator.free.assert_called()
        self.assertLess(cache.total_size(), initial_size)

    def test_page_alignment_boundary(self):
        """Test page alignment with different sizes."""
        test_cases = [
            (1, 5),
            (2, 5),
            (4, 6),
        ]

        for page_size, sequence_length in test_cases:
            with self.subTest(page_size=page_size, sequence_length=sequence_length):
                cache = RadixCache.create_simulated(page_size=page_size)

                tokens = list(range(sequence_length))
                cache.insert(RadixKey(tokens), torch.tensor(tokens, dtype=torch.int64))

                result = cache.match_prefix(MatchPrefixParams(key=RadixKey(tokens)))
                self.assertGreater(len(result.device_indices), 0)

                # Match length should be page-aligned
                match_len = len(result.device_indices)
                self.assertEqual(match_len % page_size, 0)

    def test_pretty_print_basic(self):
        """Test pretty_print produces output."""
        cache = RadixCache.create_simulated()

        cache.insert(RadixKey([1, 2, 3]), torch.tensor([10, 20, 30], dtype=torch.int64))

        # Just test that it doesn't crash
        try:
            cache.pretty_print()
        except Exception as e:
            self.fail(f"pretty_print raised an exception: {e}")

    def test_all_values_flatten(self):
        """Test all_values_flatten method."""
        cache = RadixCache.create_simulated()

        cache.insert(RadixKey([1, 2]), torch.tensor([10, 20], dtype=torch.int64))
        cache.insert(RadixKey([3, 4]), torch.tensor([30, 40], dtype=torch.int64))

        all_values = cache.all_values_flatten()
        self.assertEqual(len(all_values), 4)
        # Values should contain all inserted values (order may vary)
        values_set = set(all_values.tolist())
        self.assertEqual(values_set, {10, 20, 30, 40})

    def test_advanced_prefix_match_with_node_splits(self):
        """Advanced prefix matching: splits inside nodes and across pages."""
        for page_size in [1, 2]:
            with self.subTest(page_size=page_size):
                cache = RadixCache.create_simulated(page_size=page_size)

                # Insert a long sequence that will be split later.
                seq1 = [1, 2, 3, 4, 5, 6, 7, 8]
                val1 = torch.tensor([x * 10 for x in seq1], dtype=torch.int64)
                cache.insert(RadixKey(seq1), val1)

                # Insert a diverging branch to create an internal node on the path.
                seq2 = [1, 2, 9, 10]
                val2 = torch.tensor([x * 10 for x in seq2], dtype=torch.int64)
                cache.insert(RadixKey(seq2), val2)
                print(cache.pretty_print())

                baseline_total = cache.total_size()
                expected_total = 10  # 8 + 2
                self.assertEqual(baseline_total, expected_total)

                # Match that causes a split inside an existing node:
                # take first 4 tokens of seq1, then diverge.
                query1 = [1, 2, 3, 4, 999, 1000]
                result1 = cache.match_prefix(MatchPrefixParams(key=RadixKey(query1)))
                torch.testing.assert_close(result1.device_indices, val1[:4])
                # No data change after structural split during matching.
                self.assertEqual(cache.total_size(), baseline_total)

                # Full match of the long sequence still returns the full indices.
                result_full = cache.match_prefix(MatchPrefixParams(key=RadixKey(seq1)))
                torch.testing.assert_close(result_full.device_indices, val1)

                # Another split deeper on the path (after matching 6 tokens, then diverge).
                query2 = [1, 2, 3, 4, 5, 6, 777, 888]
                result2 = cache.match_prefix(MatchPrefixParams(key=RadixKey(query2)))
                torch.testing.assert_close(result2.device_indices, val1[:6])
                self.assertEqual(cache.total_size(), baseline_total)

                # Matching the short diverging branch should return exactly its indices.
                result_branch = cache.match_prefix(
                    MatchPrefixParams(key=RadixKey(seq2))
                )
                torch.testing.assert_close(result_branch.device_indices, val2)

    def test_hash_value_storage(self):
        """Test that hash_value is stored correctly after insert operations."""
        cache = RadixCache.create_simulated(
            page_size=4,
            enable_kv_cache_events=True,
        )

        # Insert a sequence
        cache.insert(RadixKey([1, 2, 3, 4, 5, 6, 7, 8]), None)

        # Trigger event emission to compute hash_value lazily
        cache.take_events()

        # Find the inserted node (traverse from root)
        node = cache.root_node
        for i in range(0, 8, 4):  # page_size=4, so 2 pages
            child_key = tuple([1, 2, 3, 4][:4]) if i == 0 else tuple([5, 6, 7, 8][:4])
            if child_key in node.children:
                node = node.children[child_key]
                break

        # Verify hash_value is set (computed lazily during event emission)
        self.assertIsNotNone(node.hash_value)
        # Should have 2 pages (8 tokens / 4 page_size)
        self.assertEqual(len(node.hash_value), 2)

    def test_hash_value_repeating_tokens(self):
        """Test that repeating token patterns get different hash values."""
        cache = RadixCache.create_simulated(
            page_size=4,
            enable_kv_cache_events=True,
        )

        # Insert a sequence with repeating token pattern: [1,2,3,4, 1,2,3,4]
        cache.insert(RadixKey([1, 2, 3, 4, 1, 2, 3, 4]), None)

        events = cache.take_events()
        block_stored_events = [e for e in events if isinstance(e, BlockStored)]

        # Should have 2 blocks (2 pages of size 4)
        self.assertEqual(len(block_stored_events), 2)

        # Extract block hashes
        block_hash_1 = block_stored_events[0].block_hashes[0]
        block_hash_2 = block_stored_events[1].block_hashes[0]

        # The two blocks should have DIFFERENT hashes despite same content
        # because they are at different positions (sequence-aware hashing)
        self.assertNotEqual(
            block_hash_1,
            block_hash_2,
            "Repeating token patterns should get different sequence-aware hashes",
        )

        # First block should have no parent
        self.assertIsNone(block_stored_events[0].parent_block_hash)

        # Second block's parent should be the first block's hash
        self.assertEqual(block_stored_events[1].parent_block_hash, block_hash_1)

    def test_hash_value_split(self):
        """Test that hash_value is split correctly when nodes are split."""
        cache = RadixCache.create_simulated(
            page_size=2,
            enable_kv_cache_events=True,
        )

        # Insert a sequence that will cause a split
        cache.insert(RadixKey([1, 2, 3, 4]), None)
        cache.take_events()  # Clear events and compute hash_value for first node

        # Insert a diverging sequence that will cause a split at page boundary
        cache.insert(RadixKey([1, 2, 5, 6]), None)
        cache.take_events()  # Trigger event emission to compute hash_value

        # Find the split node
        node = cache.root_node
        child_key = tuple([1, 2])
        if child_key in node.children:
            node = node.children[child_key]
            # After split and event emission, hash_value should be computed
            # Note: If hash_value wasn't set before split, it will be computed lazily
            # during event emission. If it was set, it will be split.
            # Either way, after events are emitted, it should be set.
            self.assertIsNotNone(node.hash_value)
            # Should have 1 page (split at page_size=2)
            self.assertEqual(len(node.hash_value), 1)

    def test_memory_allocated(self):
        keys, values = [], []

        num_seqs = 10000
        vocab_size = 1000
        base_prefix_len = 10000
        suffix_len = 100

        torch_allocated_before = torch.cuda.memory_allocated()

        # build dataset with common prefix
        common_prefix = [random.randint(1, vocab_size) for _ in range(base_prefix_len)]
        for _ in range(num_seqs):
            suffix = [random.randint(1, vocab_size) for _ in range(suffix_len)]
            seq = common_prefix + suffix
            keys.append(seq)
            values.append(torch.zeros(len(seq), device="cuda", dtype=torch.int32))

        cache: RadixCache = RadixCache.create_simulated()

        for key, value in zip(keys, values):
            cache.insert(RadixKey(key), value)

        del values

        torch_allocated = torch.cuda.memory_allocated() - torch_allocated_before
        cache_size_bytes = cache.total_size() * 4
        print(f"\nCache size (MB): {cache_size_bytes / (1024 * 1024)}")
        print(f"Torch allocated (MB): {torch_allocated / (1024 * 1024)}")

        # The cache size should be within reasonable bounds of the actual allocated memory.
        self.assertLess(torch_allocated, cache_size_bytes * 2)


if __name__ == "__main__":
    unittest.main()
