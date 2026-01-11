#!/usr/bin/env python3

import unittest
from typing import List

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.valkey.hicache_valkey_storage import (
    HiCacheValkeyStorage,
)


class TestValkeyUnified(unittest.TestCase):
    """Unified test suite for Valkey HiCache storage."""

    def setUp(self):
        """Set up test environment."""
        # Create storage config for testing
        self.storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=2,
            pp_rank=0,
            pp_size=1,
            is_mla_model=False,
            is_page_first_layout=False,
            model_name="test_model",
        )

        try:
            self.hicache = HiCacheValkeyStorage(self.storage_config)
        except ImportError:
            self.skipTest("valkey-py not available, skipping Valkey storage tests")
        except ConnectionError:
            self.skipTest("Valkey server not available, skipping Valkey storage tests")

    def tearDown(self):
        """Clean up test data."""
        if hasattr(self, "hicache"):
            self.hicache.clear()

    def verify_tensors_equal(self, expected: torch.Tensor, actual: torch.Tensor):
        """Helper to verify tensor equality."""
        self.assertIsNotNone(actual, "Retrieved tensor is None")
        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-6),
            f"Tensors not equal:\nExpected: {expected}\nActual: {actual}",
        )

    def verify_tensor_lists_equal(
        self, expected: List[torch.Tensor], actual: List[torch.Tensor]
    ):
        """Helper to verify lists of tensors are equal."""
        self.assertEqual(len(expected), len(actual), "Lists have different lengths")
        for exp, act in zip(expected, actual):
            if act is not None:
                self.verify_tensors_equal(exp, act)

    # ============================================================================
    # HiCache Integration Tests
    # ============================================================================

    def test_single_set_get(self):
        """Test single tensor set/get operations."""
        key = "test_key"
        value = torch.randn(10, 10, device="cpu")

        # Test set
        self.assertTrue(self.hicache.set(key, value))
        self.assertTrue(self.hicache.exists(key))

        # Test get without target location
        retrieved_direct = self.hicache.get(key)
        self.assertIsNotNone(retrieved_direct)

    def test_batch_set_get(self):
        """Test batch tensor set/get operations."""
        keys = ["key1", "key2", "key3"]
        values = [
            torch.randn(5, 5, device="cpu"),
            torch.randn(3, 3, device="cpu"),
            torch.randn(7, 7, device="cpu"),
        ]

        # Test batch set
        self.assertTrue(self.hicache.batch_set(keys, values))
        self.assertTrue(all(self.hicache.exists(key) for key in keys))

        # Test batch get without target locations
        retrieved_direct = self.hicache.batch_get(keys)
        self.assertEqual(len(retrieved_direct), len(keys))
        self.assertTrue(all(r is not None for r in retrieved_direct))

    def test_mixed_operations(self):
        """Test mixing single and batch operations."""
        # Test interleaved set/get operations
        key1, key2 = "key1", "key2"
        value1 = torch.randn(4, 4, device="cpu")
        value2 = torch.randn(6, 6, device="cpu")

        # Single set/get
        self.assertTrue(self.hicache.set(key1, value1))
        retrieved1 = self.hicache.get(key1)
        self.assertIsNotNone(retrieved1)

        # Batch set/get
        self.assertTrue(self.hicache.batch_set([key2], [value2]))
        retrieved2 = self.hicache.batch_get([key2])
        self.assertEqual(len(retrieved2), 1)
        self.assertIsNotNone(retrieved2[0])

    def test_data_integrity(self):
        """Test data integrity across operations."""
        # Test with various tensor types and sizes
        test_cases = [
            ("float32", torch.randn(10, 10, dtype=torch.float32)),
            ("float64", torch.randn(5, 5, dtype=torch.float64)),
            ("int32", torch.randint(-100, 100, (8, 8), dtype=torch.int32)),
            ("int64", torch.randint(-100, 100, (6, 6), dtype=torch.int64)),
            ("bool", torch.randint(0, 2, (4, 4)).bool()),
        ]

        for name, tensor in test_cases:
            with self.subTest(tensor_type=name):
                key = f"test_{name}"

                # Set and immediately get
                self.assertTrue(self.hicache.set(key, tensor))
                retrieved1 = self.hicache.get(key)
                self.assertIsNotNone(retrieved1)

                # Get again to verify persistence
                retrieved2 = self.hicache.get(key)
                self.assertIsNotNone(retrieved2)

    def test_batch_exists(self):
        """Test batch exists functionality."""
        keys = ["exist1", "exist2", "exist3", "exist4"]
        values = [torch.randn(2, 2) for _ in range(3)]  # Only 3 values for 4 keys

        # Set first 3 keys
        self.assertTrue(self.hicache.batch_set(keys[:3], values))

        # Test batch exists - should return 3 (first 3 exist consecutively)
        exist_count = self.hicache.batch_exists(keys)
        self.assertEqual(exist_count, 3)

        # Test with all existing keys
        all_exist_count = self.hicache.batch_exists(keys[:3])
        self.assertEqual(all_exist_count, 3)

        # Test with no existing keys
        nonexistent_keys = ["nonexist1", "nonexist2"]
        no_exist_count = self.hicache.batch_exists(nonexistent_keys)
        self.assertEqual(no_exist_count, 0)

    def test_get_stats(self):
        """Test statistics retrieval."""
        # Set some test data
        self.hicache.set("stats_test", torch.randn(5, 5))

        stats = self.hicache.get_stats()
        self.assertIsNotNone(stats)
        self.assertEqual(stats.get("backend"), "valkey")
        self.assertGreaterEqual(stats.get("key_count", 0), 1)
        self.assertIn("memory_used", stats)
        self.assertIn("connected_clients", stats)

    def test_clear_storage(self):
        """Test storage clearing."""
        # Set some test data
        keys = ["clear1", "clear2", "clear3"]
        values = [torch.randn(2, 2) for _ in keys]
        self.assertTrue(self.hicache.batch_set(keys, values))

        # Verify data exists
        self.assertEqual(self.hicache.batch_exists(keys), len(keys))

        # Clear storage
        self.hicache.clear()

        # Verify data is gone
        self.assertEqual(self.hicache.batch_exists(keys), 0)

    def test_nonexistent_operations(self):
        """Test operations on nonexistent keys."""
        # Test get nonexistent key
        result = self.hicache.get("nonexistent_key")
        self.assertIsNone(result)

        # Test exists on nonexistent key
        self.assertFalse(self.hicache.exists("nonexistent_key"))

        # Test batch get with nonexistent keys
        results = self.hicache.batch_get(["nonexist1", "nonexist2"])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r is None for r in results))

    def test_empty_batch_operations(self):
        """Test batch operations with empty inputs."""
        # Empty batch get
        result = self.hicache.batch_get([])
        self.assertEqual(result, [])

        # Empty batch exists
        result = self.hicache.batch_exists([])
        self.assertEqual(result, 0)

        # Empty batch set
        result = self.hicache.batch_set([], [])
        self.assertFalse(result)

    def test_error_handling(self):
        """Test error handling in operations."""
        # Test set with None value
        result = self.hicache.set("test_key", None)
        self.assertFalse(result)

        # Test batch set with mismatched keys and values
        result = self.hicache.batch_set(["key1", "key2"], [torch.randn(2, 2)])
        self.assertFalse(result)

        # Test batch set with None values
        result = self.hicache.batch_set(["key1"], None)
        self.assertFalse(result)

    def test_key_prefixing(self):
        """Test automatic key prefixing."""
        # Verify the key prefix is set correctly
        expected_prefix = "hicache:test_model:0:2"
        self.assertEqual(self.hicache.key_prefix, expected_prefix)

        # Test that keys are properly prefixed
        key = "test_prefix"
        full_key = self.hicache._get_full_key(key)
        self.assertEqual(full_key, f"{expected_prefix}:{key}")

    def test_mla_model_key_prefix(self):
        """Test key prefix for MLA models."""
        mla_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            is_mla_model=True,
            is_page_first_layout=True,
            model_name="mla_test_model",
        )

        try:
            mla_storage = HiCacheValkeyStorage(mla_config)
            self.assertEqual(mla_storage.key_prefix, "hicache:mla_test_model")
            mla_storage.clear()  # Clean up
        except ConnectionError:
            self.skipTest("Valkey server not available")


if __name__ == "__main__":
    unittest.main()
