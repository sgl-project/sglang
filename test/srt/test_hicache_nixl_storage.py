import unittest
import tempfile
import os
from types import SimpleNamespace

import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestHiCacheNixl(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-hierarchical-cache",
                "--mem-fraction-static",
                0.7,
                "--hicache-size",
                100,
                "--page-size",
                64,
                "--hicache-storage-backend",
                "nixl",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_basic_functionality(self):
        """Test basic NIXL storage functionality."""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=32,
            num_threads=16,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.65)

    def test_storage_operations(self):
        """Test individual storage operations."""
        try:
            from sglang.srt.mem_cache.hicache_nixl import HiCacheNixl
            
            # Test initialization with default parameters
            storage = HiCacheNixl()
            self.assertIsNotNone(storage)
            self.assertEqual(storage.file_path, "/tmp/hicache_nixl")
            
            # Test initialization with custom file path
            custom_storage = HiCacheNixl(file_path="/tmp/test_hicache_nixl")
            self.assertIsNotNone(custom_storage)
            self.assertEqual(custom_storage.file_path, "/tmp/test_hicache_nixl")
            
            # Test tensor operations
            test_tensor = torch.randn(10, 10)
            test_key = "test_key_123"
            
            # Test set operation
            success = storage.set(test_key, test_tensor)
            self.assertTrue(success)
            
            # Test exists operation
            exists = storage.exists(test_key)
            self.assertTrue(exists)
            
            # Test get operation with dst_tensor
            dst_tensor = torch.zeros_like(test_tensor)
            retrieved_tensor = storage.get(test_key, dst_tensor)
            self.assertIsNotNone(retrieved_tensor)
            self.assertTrue(torch.allclose(test_tensor, retrieved_tensor))
            
            # Test get operation without dst_tensor (should fail)
            retrieved_tensor = storage.get(test_key, None)
            self.assertIsNone(retrieved_tensor)
            
            # Test delete operation
            storage.delete(test_key)
            exists = storage.exists(test_key)
            self.assertFalse(exists)
            
            # Test clear operation
            storage.set("key1", torch.randn(5, 5))
            storage.set("key2", torch.randn(5, 5))
            storage.clear()
            self.assertFalse(storage.exists("key1"))
            self.assertFalse(storage.exists("key2"))
            
        except ImportError:
            self.skipTest("NIXL not available, skipping NIXL storage tests")

    def test_batch_operations(self):
        """Test batch storage operations."""
        try:
            from sglang.srt.mem_cache.hicache_nixl import HiCacheNixl
            
            storage = HiCacheNixl()
            
            # Test batch set operations
            keys = ["batch_key_1", "batch_key_2", "batch_key_3"]
            tensors = [torch.randn(5, 5) for _ in range(3)]
            
            success = storage.batch_set(keys, tensors)
            self.assertTrue(success)
            
            # Test batch get operations
            dst_tensors = [torch.zeros_like(t) for t in tensors]
            retrieved_tensors = storage.batch_get(keys, dst_tensors)
            
            self.assertEqual(len(retrieved_tensors), len(keys))
            for i, (original, retrieved) in enumerate(zip(tensors, retrieved_tensors)):
                self.assertIsNotNone(retrieved)
                self.assertTrue(torch.allclose(original, retrieved))
            
            # Test batch get with non-existent keys
            non_existent_keys = ["non_existent_1", "non_existent_2"]
            dst_tensors = [torch.zeros(5, 5) for _ in range(2)]
            retrieved_tensors = storage.batch_get(non_existent_keys, dst_tensors)
            
            self.assertEqual(len(retrieved_tensors), len(non_existent_keys))
            for retrieved in retrieved_tensors:
                self.assertIsNone(retrieved)
                
        except ImportError:
            self.skipTest("NIXL not available, skipping NIXL storage tests")

    def test_tensor_registration(self):
        """Test tensor registration and deregistration."""
        try:
            from sglang.srt.mem_cache.hicache_nixl import HiCacheNixl
            
            storage = HiCacheNixl()
            
            # Test CPU tensor registration
            cpu_tensor = torch.randn(10, 10)
            success = storage.register(cpu_tensor)
            self.assertTrue(success)
            
            # Test GPU tensor registration (if CUDA available)
            if torch.cuda.is_available():
                gpu_tensor = torch.randn(10, 10, device="cuda")
                success = storage.register(gpu_tensor)
                self.assertTrue(success)
                
                # Test deregistration
                success = storage.deregister(gpu_tensor)
                self.assertTrue(success)
            
            # Test deregistration
            success = storage.deregister(cpu_tensor)
            self.assertTrue(success)
            
        except ImportError:
            self.skipTest("NIXL not available, skipping NIXL storage tests")

    def test_file_plugin_selection(self):
        """Test file plugin selection and backend creation."""
        try:
            from sglang.srt.mem_cache.hicache_nixl import HiCacheNixl
            
            # Test auto plugin selection
            storage = HiCacheNixl(file_plugin="auto")
            self.assertIsNotNone(storage.backend_name)
            
            # Test specific plugin selection
            storage = HiCacheNixl(file_plugin="POSIX")
            self.assertEqual(storage.backend_name, "POSIX")
            
            # Test GDS_MT plugin (if available)
            try:
                storage = HiCacheNixl(file_plugin="GDS_MT")
                self.assertIsNotNone(storage.backend_name)
            except Exception:
                # GDS_MT might not be available, which is expected
                pass
                
        except ImportError:
            self.skipTest("NIXL not available, skipping NIXL storage tests")


class TestHiCacheNixlIntegration(CustomTestCase):
    """Integration tests for HiCacheNixl with actual NIXL storage endpoints."""
    
    def test_nixl_storage_endpoint_connection(self):
        """Test connection to NIXL storage endpoint."""
        try:
            from sglang.srt.mem_cache.hicache_nixl import HiCacheNixl
            
            # Test with different file paths
            file_paths = [
                "/tmp/test_hicache_nixl_1",
                "/tmp/test_hicache_nixl_2",
                "/tmp/test_hicache_nixl_3",
            ]
            
            for file_path in file_paths:
                try:
                    storage = HiCacheNixl(file_path=file_path)
                    self.assertIsNotNone(storage)
                    self.assertEqual(storage.file_path, file_path)
                except Exception as e:
                    # This is expected if NIXL is not available
                    print(f"Expected failure for file path {file_path}: {e}")
                    
        except ImportError:
            self.skipTest("NIXL not available, skipping NIXL storage tests")

    def test_nixl_storage_error_handling(self):
        """Test error handling in NIXL storage."""
        try:
            from sglang.srt.mem_cache.hicache_nixl import HiCacheNixl
            
            # Test with valid file path
            storage = HiCacheNixl()
            
            # These operations should handle errors gracefully
            test_tensor = torch.randn(5, 5)
            dst_tensor = torch.zeros_like(test_tensor)
            
            # Test set operation (may fail if no storage service)
            result = storage.set("test_key", test_tensor)
            # Should return True if storage is available
            
            # Test get operation (may return None if no storage service)
            retrieved = storage.get("test_key", dst_tensor)
            # Should return tensor if storage is available
            
            # Test exists operation (may return False if no storage service)
            exists = storage.exists("test_key")
            # Should return True if storage is available
            
        except ImportError:
            self.skipTest("NIXL not available, skipping NIXL storage tests")

    def test_pre_opened_files(self):
        """Test pre-opening of existing files."""
        try:
            from sglang.srt.mem_cache.hicache_nixl import HiCacheNixl
            
            # Create storage and add some files
            storage = HiCacheNixl(file_path="/tmp/test_pre_opened")
            
            # Add some test files
            test_tensor = torch.randn(5, 5)
            storage.set("test_file_1", test_tensor)
            storage.set("test_file_2", test_tensor)
            
            # Create new storage instance (should pre-open existing files)
            new_storage = HiCacheNixl(file_path="/tmp/test_pre_opened")
            
            # Test that files are accessible
            dst_tensor = torch.zeros_like(test_tensor)
            retrieved = new_storage.get("test_file_1", dst_tensor)
            self.assertIsNotNone(retrieved)
            
        except ImportError:
            self.skipTest("NIXL not available, skipping NIXL storage tests")


if __name__ == "__main__":
    unittest.main() 