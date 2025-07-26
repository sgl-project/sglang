#!/usr/bin/env python3

import os
import torch
import unittest
from unittest.mock import MagicMock
from typing import List, Optional

from sglang.srt.mem_cache.nixl.hicache_nixl import HiCacheNixl
from sglang.srt.mem_cache.nixl.nixl_utils import NixlRegistration, NixlFileManager


class TestNixlUnified(unittest.TestCase):
    """Unified test suite for all NIXL components."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test directories
        self.test_dir = "/tmp/test_nixl_unified"
        self.hicache_test_dir = "/tmp/hicache_nixl_test"
        os.makedirs(self.test_dir, exist_ok=True)
        os.makedirs(self.hicache_test_dir, exist_ok=True)
        
        # Mock NIXL agent for registration tests
        self.mock_agent = MagicMock()
        self.mock_agent.get_reg_descs.return_value = "mock_reg_descs"
        self.mock_agent.register_memory.return_value = "mock_registered_memory"
        
        # Create instances
        self.file_manager = NixlFileManager(self.test_dir)
        self.registration = NixlRegistration(self.mock_agent)
        self.hicache = HiCacheNixl(file_path=self.hicache_test_dir, file_plugin="POSIX")

    def tearDown(self):
        """Clean up test directories."""
        for test_dir in [self.test_dir, self.hicache_test_dir]:
            if os.path.exists(test_dir):
                for root, dirs, files in os.walk(test_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(test_dir)

    def verify_tensors_equal(self, expected: torch.Tensor, actual: torch.Tensor):
        """Helper to verify tensor equality."""
        self.assertIsNotNone(actual, "Retrieved tensor is None")
        self.assertTrue(
            torch.allclose(expected, actual, atol=1e-6),
            f"Tensors not equal:\nExpected: {expected}\nActual: {actual}"
        )

    def verify_tensor_lists_equal(self, expected: List[torch.Tensor], actual: List[torch.Tensor]):
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
        self.verify_tensors_equal(value, retrieved)

    def test_batch_set_get(self):
        """Test batch tensor set/get operations."""
        keys = ["key1", "key2", "key3"]
        values = [
            torch.randn(5, 5, device="cpu"),
            torch.randn(3, 3, device="cpu"),
            torch.randn(7, 7, device="cpu")
        ]
        dst_tensors = [torch.zeros_like(v, device="cpu") for v in values]

        # Test batch set
        self.assertTrue(self.hicache.batch_set(keys, values))
        self.assertTrue(all(self.hicache.exists(key) for key in keys))

        # Test batch get
        retrieved = self.hicache.batch_get(keys, dst_tensors)
        self.verify_tensor_lists_equal(values, retrieved)

    def test_mixed_operations(self):
        """Test mixing single and batch operations."""
        # Set individual tensors
        key1, value1 = "key1", torch.randn(2, 2, device="cpu")
        key2, value2 = "key2", torch.randn(3, 3, device="cpu")
        self.assertTrue(self.hicache.set(key1, value1))
        self.assertTrue(self.hicache.set(key2, value2))

        # Batch get them
        dst_tensors = [torch.zeros_like(value1, device="cpu"), torch.zeros_like(value2, device="cpu")]
        retrieved = self.hicache.batch_get([key1, key2], dst_tensors)
        self.verify_tensor_lists_equal([value1, value2], retrieved)

        # Batch set new values
        new_values = [torch.randn_like(value1, device="cpu"), torch.randn_like(value2, device="cpu")]
        self.assertTrue(self.hicache.batch_set([key1, key2], new_values))

        # Get them individually
        dst1, dst2 = torch.zeros_like(value1, device="cpu"), torch.zeros_like(value2, device="cpu")
        retrieved1 = self.hicache.get(key1, dst1)
        retrieved2 = self.hicache.get(key2, dst2)
        self.verify_tensors_equal(new_values[0], retrieved1)
        self.verify_tensors_equal(new_values[1], retrieved2)

    def test_error_cases(self):
        """Test error handling in set/get operations."""
        # Test get with non-existent key
        dst_tensor = torch.zeros(5, 5, device="cpu")
        self.assertIsNone(self.hicache.get("nonexistent", dst_tensor))

        # Test batch get with some non-existent keys
        key1, value1 = "key1", torch.randn(2, 2, device="cpu")
        self.assertTrue(self.hicache.set(key1, value1))
        dst_tensors = [torch.zeros_like(value1, device="cpu"), torch.zeros(3, 3, device="cpu")]
        retrieved = self.hicache.batch_get([key1, "nonexistent"], dst_tensors)
        # Since one file doesn't exist, the entire batch operation fails
        self.assertEqual(retrieved, [None, None])

        # Test empty batch operations
        self.assertTrue(self.hicache.batch_set([], []))
        self.assertEqual([], self.hicache.batch_get([], []))

    # ============================================================================
    # File Management Tests
    # ============================================================================

    def test_basic_file_operations(self):
        """Test basic file operations."""
        # Test file creation
        test_file = os.path.join(self.test_dir, "test.bin")
        self.assertTrue(self.file_manager.create_file(test_file))
        self.assertTrue(os.path.exists(test_file))

        # Test file opening and closing
        file_tuple = self.file_manager.create_nixl_tuples([(test_file, 0, 1024)])[0]
        self.assertEqual(len(file_tuple), 4)  # (offset, length, fd, "")
        self.assertIsInstance(file_tuple[2], int)  # fd should be an integer
        
        # Test file descriptor cleanup
        self.assertTrue(self.file_manager.close_file(file_tuple[2]))

    def test_create_nixl_tuples(self):
        """Test creation of NIXL tuples with proper offsets and lengths."""
        # Create test files
        file_paths = [
            os.path.join(self.test_dir, "test1.bin"),
            os.path.join(self.test_dir, "test2.bin")
        ]
        
        # Create files
        for path in file_paths:
            self.assertTrue(self.file_manager.create_file(path))
        
        # Test tuple creation with different offsets and lengths
        file_info = [
            (file_paths[0], 0, 1024),      # First file: offset 0, length 1024
            (file_paths[1], 2048, 4096),   # Second file: offset 2048, length 4096
        ]
        
        file_tuples = self.file_manager.create_nixl_tuples(file_info)
        self.assertEqual(len(file_tuples), 2)
        
        # Verify tuple structure: (offset, length, fd, "")
        for i, (file_path, expected_offset, expected_length) in enumerate(file_info):
            offset, length, fd, meta = file_tuples[i]
            self.assertEqual(offset, expected_offset)
            self.assertEqual(length, expected_length)
            self.assertIsInstance(fd, int)
            self.assertEqual(meta, "")
        
        # Clean up file descriptors
        for file_tuple in file_tuples:
            self.assertTrue(self.file_manager.close_file(file_tuple[2]))

    def test_get_nixl_tuples_for_transfer(self):
        """Test conversion of 4-element tuples to 3-element tuples for transfer."""
        # Create test file
        test_file = os.path.join(self.test_dir, "test.bin")
        self.assertTrue(self.file_manager.create_file(test_file))
        
        # Create 4-element tuples
        file_info = [(test_file, 0, 1024)]
        file_tuples = self.file_manager.create_nixl_tuples(file_info)
        
        # Test conversion without tensor sizes (use original lengths)
        transfer_tuples = self.file_manager.get_nixl_tuples_for_transfer(file_tuples)
        self.assertEqual(len(transfer_tuples), 1)
        self.assertEqual(len(transfer_tuples[0]), 3)  # (offset, length, fd)
        self.assertEqual(transfer_tuples[0][0], 0)    # offset
        self.assertEqual(transfer_tuples[0][1], 1024) # length
        self.assertEqual(transfer_tuples[0][2], file_tuples[0][2])  # fd
        
        # Test conversion with tensor sizes
        tensor_sizes = [2048]  # Different size than file length
        transfer_tuples = self.file_manager.get_nixl_tuples_for_transfer(file_tuples, tensor_sizes)
        self.assertEqual(len(transfer_tuples), 1)
        self.assertEqual(transfer_tuples[0][1], 2048)  # Should use tensor size
        
        # Clean up
        self.assertTrue(self.file_manager.close_file(file_tuples[0][2]))

    def test_error_handling(self):
        """Test error handling in file operations."""
        # Test non-existent file
        non_existent_file = "/nonexistent/file"
        file_tuples = self.file_manager.create_nixl_tuples([(non_existent_file, 0, 1024)])
        self.assertEqual(file_tuples, [])
        
        # Test invalid file path (root directory)
        invalid_file = "/root/test.bin"
        file_tuples = self.file_manager.create_nixl_tuples([(invalid_file, 0, 1024)])
        self.assertEqual(file_tuples, [])

    def test_file_descriptor_cleanup(self):
        """Test that file descriptors are properly cleaned up on failure."""
        # Create a file
        test_file = os.path.join(self.test_dir, "test3.bin")
        self.assertTrue(self.file_manager.create_file(test_file))
        
        # Create tuples
        file_tuples = self.file_manager.create_nixl_tuples([(test_file, 0, 1024)])
        self.assertEqual(len(file_tuples), 1)
        
        # Simulate a failure scenario
        fd = file_tuples[0][2]
        self.assertTrue(self.file_manager.close_file(fd))
        
        # Try to close again (should fail gracefully)
        self.assertFalse(self.file_manager.close_file(fd))

    # ============================================================================
    # Registration Tests
    # ============================================================================

    def test_register_buffers(self):
        """Test tensor registration."""
        # Test single tensor
        tensor = torch.randn(2, 3)
        reg_mem = self.registration.register_buffers(tensor)
        self.assertIsNotNone(reg_mem)
        self.mock_agent.get_reg_descs.assert_called_with([tensor], "DRAM")
        
        # Test list of tensors
        tensors = [torch.randn(2, 3), torch.randn(4, 5)]
        reg_mem = self.registration.register_buffers(tensors)
        self.assertIsNotNone(reg_mem)
        self.mock_agent.get_reg_descs.assert_called_with(tensors, "DRAM")

    def test_register_files_with_tuples(self):
        """Test file registration using NIXL tuples with offsets and lengths."""
        # Create test files
        file_paths = [
            os.path.join(self.test_dir, "test1.bin"),
            os.path.join(self.test_dir, "test2.bin")
        ]
        
        # Create files and prepare test data
        file_info = [
            (file_paths[0], 0, 1024),      # First file: offset 0, length 1024
            (file_paths[1], 2048, 4096),   # Second file: offset 2048, length 4096
        ]
        
        # Create the files
        for path in file_paths:
            self.assertTrue(self.file_manager.create_file(path))
        
        # Create NIXL tuples
        file_tuples = self.file_manager.create_nixl_tuples(file_info)
        self.assertEqual(len(file_tuples), 2)
        
        # Test registration with tuples
        reg_mem = self.registration.register_files(file_tuples)
        self.assertIsNotNone(reg_mem)
        
        # Verify NIXL agent was called correctly
        self.mock_agent.get_reg_descs.assert_called_with(file_tuples, "FILE")
        self.mock_agent.register_memory.assert_called_with("mock_reg_descs")
        
        # Clean up file descriptors
        for file_tuple in file_tuples:
            self.assertTrue(self.file_manager.close_file(file_tuple[2]))

    # ============================================================================
    # Backend Selection Tests
    # ============================================================================

    def test_backend_selection(self):
        """Test NixlBackendSelection class."""
        from sglang.srt.mem_cache.nixl.nixl_utils import NixlBackendSelection
        from nixl._api import nixl_agent, nixl_agent_config
        import uuid
        
        # Create NIXL agent
        agent_config = nixl_agent_config(backends=[])
        agent = nixl_agent(str(uuid.uuid4()), agent_config)
        
        # Test auto backend selection
        backend_selector = NixlBackendSelection("auto")
        result = backend_selector.create_backend(agent)
        self.assertTrue(result, "Auto backend selection should succeed")
        
        # Get available plugins
        plugin_list = agent.get_plugin_list()
        self.assertIsInstance(plugin_list, list, "Plugin list should be a list")
        
        # Check which backend was selected
        self.assertTrue(hasattr(agent, 'backends'), "Agent should have backends attribute")
        self.assertTrue(agent.backends, "Agent should have at least one backend")
        
        selected_backend = list(agent.backends.keys())[0]
        self.assertIsInstance(selected_backend, str, "Selected backend should be a string")
        
        # Verify priority order (3FS > POSIX > GDS_MT > GDS)
        if "3FS" in plugin_list:
            # If 3FS is available, it should be selected first
            if selected_backend == "3FS":
                pass  # Correct priority
            elif "POSIX" in plugin_list and selected_backend == "POSIX":
                pass  # 3FS not available, POSIX is correct
            elif "GDS_MT" in plugin_list and selected_backend == "GDS_MT":
                pass  # 3FS and POSIX not available, GDS_MT is correct
            elif "GDS" in plugin_list and selected_backend == "GDS":
                pass  # Only GDS available
            else:
                self.fail(f"Unexpected backend selected: {selected_backend}")
        
        # Test specific backend selection (if POSIX is available)
        if "POSIX" in plugin_list:
            agent2 = nixl_agent(str(uuid.uuid4()), agent_config)
            backend_selector2 = NixlBackendSelection("POSIX")
            result2 = backend_selector2.create_backend(agent2)
            self.assertTrue(result2, "Specific POSIX backend selection should succeed")
            
            # Verify the specific backend was created
            self.assertIn("POSIX", agent2.backends, "POSIX backend should be created")


if __name__ == "__main__":
    unittest.main() 