#!/usr/bin/env python3

import os
import unittest
from typing import List
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.nixl.hicache_nixl import HiCacheNixl
from sglang.srt.mem_cache.storage.nixl.nixl_utils import (
    NixlFileManager,
    NixlRegistration,
)


class TestNixlUnified(unittest.TestCase):
    """Unified test suite for all NIXL components."""

    def setUp(self):
        """Set up test environment."""
        # Create test directories
        self.test_dir = "/tmp/test_nixl_unified"
        os.makedirs(self.test_dir, exist_ok=True)

        # Mock NIXL agent for registration tests
        self.mock_agent = MagicMock()
        self.mock_agent.get_reg_descs.return_value = "mock_reg_descs"
        self.mock_agent.register_memory.return_value = "mock_registered_memory"

        # Create instances
        self.file_manager = NixlFileManager(self.test_dir)
        self.registration = NixlRegistration(self.mock_agent)

        # Create storage config for testing
        self.storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=2,
            is_mla_model=False,
            is_page_first_layout=False,
            model_name="test_model",
        )

        try:
            self.hicache = HiCacheNixl(
                storage_config=self.storage_config,
                file_path=self.test_dir,
                plugin="POSIX",
            )
        except ImportError:
            self.skipTest("NIXL not available, skipping NIXL storage tests")

    def tearDown(self):
        """Clean up test directories."""
        if os.path.exists(self.test_dir):
            import shutil

            shutil.rmtree(self.test_dir)

    def delete_test_file(self, file_path: str) -> bool:
        """Helper method to delete a test file.

        Args:
            file_path: Path to the file to delete

        Returns:
            bool: True if file was deleted or didn't exist, False on error
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            return True
        except Exception as e:
            return False

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
            self.verify_tensors_equal(exp, act)

    # ============================================================================
    # HiCache Integration Tests
    # ============================================================================

    def test_single_set_get(self):
        """Test single tensor set/get operations."""
        key = "test_key"
        value = torch.randn(10, 10, device="cpu")
        dst_tensor = torch.zeros_like(value, device="cpu")

        # Test set
        self.assertTrue(self.hicache.set(key, value))
        self.assertTrue(self.hicache.exists(key))

        # Test get
        retrieved = self.hicache.get(key, dst_tensor)
        self.verify_tensors_equal(value, dst_tensor)
        self.verify_tensors_equal(value, retrieved)

        # Same test in addr,len mode with another key and dst_tensor
        key2 = "test_key2"
        dst_tensor2 = torch.zeros_like(value, device="cpu")
        src_addr, src_len = value.data_ptr(), value.numel() * value.element_size()
        dst_addr, dst_len = (
            dst_tensor2.data_ptr(),
            dst_tensor2.numel() * dst_tensor2.element_size(),
        )

        # Test set
        self.assertTrue(self.hicache.set(key, None, src_addr, src_len))
        self.assertTrue(self.hicache.exists(key))

        # Test get
        retrieved2 = self.hicache.get(key, dst_addr, dst_len)
        self.assertTrue(retrieved2 == None)
        self.verify_tensors_equal(value, dst_tensor2)

    def test_batch_set_get(self):
        """Test batch tensor set/get operations."""
        keys = ["key1", "key2", "key3"]
        values = [
            torch.randn(5, 5, device="cpu"),
            torch.randn(3, 3, device="cpu"),
            torch.randn(7, 7, device="cpu"),
        ]
        dst_tensors = [torch.zeros_like(v, device="cpu") for v in values]

        # Test batch set
        self.assertTrue(self.hicache.batch_set(keys, values))
        self.assertTrue(all(self.hicache.exists(key) for key in keys))

        # Test batch get
        retrieved = self.hicache.batch_get(keys, dst_tensors)
        self.verify_tensor_lists_equal(values, retrieved)

        # Same test in addr,len mode with another key and dst_tensor
        keys2 = ["key4", "key5", "key6"]
        dst_tensors2 = [torch.zeros_like(v, device="cpu") for v in values]
        src_addrs = [v.data_ptr() for v in values]
        src_lens = [v.numel() * v.element_size() for v in values]
        dst_addrs = [dt.data_ptr() for dt in dst_tensors2]
        dst_lens = [dt.numel() * dt.element_size() for dt in dst_tensors2]

        # Test batch set
        self.assertTrue(self.hicache.batch_set(keys2, None, src_addrs, src_lens))
        self.assertTrue(all(self.hicache.exists(key) for key in keys2))

        # Test batch get
        retrieved2 = self.hicache.batch_get(keys, dst_addrs, dst_lens)
        self.assertTrue(all(ret == None for ret in retrieved2))
        self.verify_tensor_lists_equal(values, dst_tensors2)

    def test_mixed_operations(self):
        """Test mixing single and batch operations."""
        # Test interleaved set/get operations
        key1, key2 = "key1", "key2"
        value1 = torch.randn(4, 4, device="cpu")
        value2 = torch.randn(6, 6, device="cpu")
        dst1 = torch.zeros_like(value1)
        dst2 = torch.zeros_like(value2)

        # Single set/get
        self.assertTrue(self.hicache.set(key1, value1))
        retrieved1 = self.hicache.get(key1, dst1)
        self.verify_tensors_equal(value1, retrieved1)

        # Batch set/get
        self.assertTrue(self.hicache.batch_set([key2], [value2]))
        retrieved2 = self.hicache.batch_get([key2], [dst2])
        self.verify_tensors_equal(value2, retrieved2[0])

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
                dst_tensor = torch.zeros_like(tensor)

                # Set and immediately get
                self.assertTrue(self.hicache.set(key, tensor))
                retrieved1 = self.hicache.get(key, dst_tensor)
                self.verify_tensors_equal(tensor, retrieved1)

                # Get again to verify persistence
                dst_tensor.zero_()
                retrieved2 = self.hicache.get(key, dst_tensor)
                self.verify_tensors_equal(tensor, retrieved2)

    def test_basic_file_operations(self):
        """Test basic file operations."""
        test_file = os.path.join(self.test_dir, "test_file.bin")
        self.file_manager.create_file(test_file)
        self.assertTrue(os.path.exists(test_file))
        self.assertEqual(os.path.getsize(test_file), 0)  # Empty file

        # Test file deletion
        self.assertTrue(self.delete_test_file(test_file))
        self.assertFalse(os.path.exists(test_file))

    def test_create_nixl_tuples(self):
        """Test creation of NIXL tuples."""
        test_file = os.path.join(self.test_dir, "test_file.bin")
        self.file_manager.create_file(test_file)

        # Test tuple creation
        tuples = self.file_manager.files_to_nixl_tuples([test_file])
        self.assertIsNotNone(tuples)
        self.assertTrue(len(tuples) > 0)

    def test_error_handling(self):
        """Test error handling in file operations."""
        # Test non-existent file
        self.assertTrue(
            self.delete_test_file("nonexistent_file.bin")
        )  # Returns True if file doesn't exist

        # Test invalid file path
        self.assertFalse(self.file_manager.create_file(""))  # Empty path should fail

    def test_register_buffers(self):
        """Test registration of memory buffers."""
        # Create test tensor
        tensor = torch.randn(10, 10)

        # Test buffer registration
        self.assertIsNotNone(self.hicache.register_buffers(tensor))

        # Test batch registration
        tensors = [torch.randn(5, 5) for _ in range(3)]
        self.assertIsNotNone(self.hicache.register_buffers(tensors))

    def test_register_files_with_tuples(self):
        """Test registration of files using NIXL tuples."""
        files = [os.path.join(self.test_dir, f"test_file_{i}.bin") for i in range(3)]
        for file in files:
            self.file_manager.create_file(file)

        # Create tuples and register
        tuples = self.file_manager.files_to_nixl_tuples(files)
        self.hicache.register_files(tuples)

        # Verify tuples
        self.assertEqual(len(tuples), len(files))
        for t, f in zip(tuples, files):
            self.assertEqual(t[3], f)  # Check file path


if __name__ == "__main__":
    unittest.main()
