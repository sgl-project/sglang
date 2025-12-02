#!/usr/bin/env python3
"""
Performance comparison test for HiCacheFile storage backend.
Compares batch_set/batch_get (old) vs batch_set_v1/batch_get_v1 (new zero-copy batch).

Usage:
    python3 test/srt/hicache/test_hicache_storage_performance_comparison.py
"""

import os
import shutil
import tempfile
import time
import unittest
from typing import List, Tuple

import torch

from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig


class MockHostKVCache:
    """Mock HostKVCache for testing purposes.

    For MHA models, we store K and V separately but need to return them as a single
    contiguous tensor for zero-copy operations. We use a pre-allocated buffer to avoid
    repeated allocations during performance testing.
    """

    def __init__(self, page_size: int = 64, is_mla: bool = False):
        self.page_size = page_size
        self.layout = "page_first"
        self.is_mla = is_mla

        # Create a simple memory pool: 100 pages for performance testing
        self.num_pages = 100
        self.dtype = torch.float16
        self.size_per_token = 2 if is_mla else 4  # MLA: 2 (K), MHA: 4 (K+V)
        self.numel_per_page = page_size * self.size_per_token

        # Allocate memory pool
        if is_mla:
            # MLA: single tensor per page
            self.kv_buffer = torch.randn(
                self.num_pages, self.numel_per_page, dtype=self.dtype
            )
        else:
            # MHA: Store as flat contiguous buffer for zero-copy access
            # In real implementation, page_first layout stores K+V contiguously
            # For MHA, we need to simulate K and V separately for get_page_buffer_meta
            # Each page has K (first half) and V (second half)
            self.kv_buffer = torch.randn(
                self.num_pages, self.numel_per_page, dtype=self.dtype
            )

    def get_data_page(self, index: int, flat: bool = True) -> torch.Tensor:
        """Get a flat data page from the memory pool.

        Returns a view of the buffer for zero-copy operations.
        """
        # Return a view of the buffer (zero-copy)
        page = self.kv_buffer[index]

        if flat:
            # Already flat, just return the view
            return page
        return page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        """Get a dummy flat data page for batch_get (matching cache_controller logic)."""
        # Create a dummy tensor with the same shape as a page
        return torch.zeros(self.numel_per_page, dtype=self.dtype)

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        """Set a flat data page to the memory pool."""
        # Direct assignment for zero-copy (data_page should match buffer shape)
        if data_page.shape != self.kv_buffer[index].shape:
            self.kv_buffer[index] = data_page.reshape(self.kv_buffer[index].shape)
        else:
            self.kv_buffer[index].copy_(data_page)

    def get_page_buffer_meta(self, indices):
        """Get buffer metadata (pointers and sizes) for zero-copy operations.

        Returns:
            ptr_list: List of memory pointers
            element_size_list: List of sizes for each pointer
        """
        assert len(indices) % self.page_size == 0
        ptr_list = []
        indices = indices.tolist()

        if self.is_mla:
            # MLA: one pointer per page
            for index in range(0, len(indices), self.page_size):
                page_idx = indices[index]
                # Get the specific page tensor and its data pointer
                page_tensor = self.kv_buffer[page_idx]
                ptr = page_tensor.data_ptr()
                ptr_list.append(ptr)
            element_size = self.numel_per_page * self.dtype.itemsize
            element_size_list = [element_size] * len(ptr_list)
        else:
            # MHA: K and V pointer pairs per page
            # For page_first layout, K and V are stored contiguously in the page
            # K is first half, V is second half of each page
            k_size = self.numel_per_page // 2  # Half page for K
            v_size = self.numel_per_page // 2  # Half page for V

            for index in range(0, len(indices), self.page_size):
                page_idx = indices[index]
                # Get the page tensor
                page_tensor = self.kv_buffer[page_idx]
                k_ptr = page_tensor.data_ptr()
                # V pointer is offset by half page
                v_ptr = k_ptr + k_size * self.dtype.itemsize
                ptr_list.append(k_ptr)
                ptr_list.append(v_ptr)

            # Each K or V is half page size
            element_size = k_size * self.dtype.itemsize
            element_size_list = [element_size] * len(ptr_list)

        return ptr_list, element_size_list


class TestHiCacheFilePerformanceComparison(unittest.TestCase):
    """Performance comparison tests for HiCacheFile storage backend."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp(prefix="hicache_perf_test_")

        # Create storage config
        self.storage_config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            is_mla_model=False,  # Test MHA
            is_page_first_layout=True,
            model_name="test_model",
        )

        self.page_size = 64
        self.mem_pool_host = MockHostKVCache(page_size=self.page_size, is_mla=False)

        # Flag to control which interface to use in tests
        # Can be set to 'old', 'new', or 'both' for comparison
        # Check if class variable is set (from command line), otherwise use default
        if hasattr(TestHiCacheFilePerformanceComparison, "use_interface"):
            self.use_interface = TestHiCacheFilePerformanceComparison.use_interface
        else:
            self.use_interface = "both"  # Default: compare both

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_test_data(self, num_keys: int) -> Tuple[List[str], torch.Tensor]:
        """Create test keys and host_indices."""
        keys = [f"perf_test_key_{i}" for i in range(num_keys)]

        # Create host_indices: [page0_idx, page0_idx, ..., page1_idx, page1_idx, ...]
        host_indices_list = []
        for i in range(num_keys):
            indices = torch.full((self.page_size,), i, dtype=torch.int64)
            host_indices_list.append(indices)
        host_indices = torch.cat(host_indices_list)

        # Initialize memory pool with test data
        for i in range(num_keys):
            test_data = torch.randn(
                self.mem_pool_host.numel_per_page, dtype=self.mem_pool_host.dtype
            )
            # Add unique marker
            test_data[0] = float(i + 1)
            self.mem_pool_host.set_from_flat_data_page(i, test_data)

        return keys, host_indices

    def _create_storage(self) -> HiCacheFile:
        """Create a fresh storage instance."""
        storage = HiCacheFile(
            storage_config=self.storage_config,
            file_path=self.test_dir,
        )
        storage.register_mem_pool_host(self.mem_pool_host)
        return storage

    def _measure_batch_set_performance(
        self, storage: HiCacheFile, keys: List[str], host_indices: torch.Tensor
    ) -> float:
        """Measure batch_set performance (including get_data_page, matching cache_controller logic)."""
        # Clear files first
        for key in keys:
            suffixed_key = storage._get_suffixed_key(key)
            file_path = os.path.join(self.test_dir, f"{suffixed_key}.bin")
            if os.path.exists(file_path):
                os.remove(file_path)

        # Warm up (matching cache_controller._generic_page_set logic)
        data = [
            self.mem_pool_host.get_data_page(
                host_indices[i * self.mem_pool_host.page_size]
            )
            for i in range(len(keys))
        ]
        storage.batch_set(keys=keys, values=data)

        # Clear files for measurement
        for key in keys:
            suffixed_key = storage._get_suffixed_key(key)
            file_path = os.path.join(self.test_dir, f"{suffixed_key}.bin")
            if os.path.exists(file_path):
                os.remove(file_path)

        # Measure - include get_data_page time (matching cache_controller._generic_page_set)
        start_time = time.perf_counter()
        data = [
            self.mem_pool_host.get_data_page(
                host_indices[i * self.mem_pool_host.page_size]
            )
            for i in range(len(keys))
        ]
        storage.batch_set(keys=keys, values=data)
        end_time = time.perf_counter()

        return end_time - start_time

    def _measure_batch_set_v1_performance(
        self, storage: HiCacheFile, keys: List[str], host_indices: torch.Tensor
    ) -> float:
        """Measure batch_set_v1 performance (matching cache_controller logic)."""
        # Clear batch chunk files first (new format: {key}_batch_{count}_chunk_{idx}.bin)
        batch_size = int(os.getenv("SGLANG_HICACHE_FILE_BATCH_SIZE", "4"))
        num_batches = (len(keys) + batch_size - 1) // batch_size
        suffixed_key = storage._get_suffixed_key(keys[0])

        # Clear all possible chunk files
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(keys))
            batch_count = end_idx - start_idx
            batch_file_path = os.path.join(
                self.test_dir,
                f"{suffixed_key}_batch_{batch_count}_chunk_{batch_idx}.bin",
            )
            if os.path.exists(batch_file_path):
                os.remove(batch_file_path)

        # Warm up
        storage.batch_set_v1(keys=keys, host_indices=host_indices)

        # Clear files for measurement
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(keys))
            batch_count = end_idx - start_idx
            batch_file_path = os.path.join(
                self.test_dir,
                f"{suffixed_key}_batch_{batch_count}_chunk_{batch_idx}.bin",
            )
            if os.path.exists(batch_file_path):
                os.remove(batch_file_path)

        # Measure - use perf_counter for high precision
        start_time = time.perf_counter()
        storage.batch_set_v1(keys=keys, host_indices=host_indices)
        end_time = time.perf_counter()

        return end_time - start_time

    def _measure_batch_get_performance(
        self, storage: HiCacheFile, keys: List[str]
    ) -> float:
        """Measure batch_get performance (including get_dummy_flat_data_page, matching cache_controller logic)."""
        # Warm up (matching cache_controller._generic_page_get logic)
        dummy_page_dst = [self.mem_pool_host.get_dummy_flat_data_page() for _ in keys]
        storage.batch_get(keys=keys, target_locations=dummy_page_dst)

        # Measure - include get_dummy_flat_data_page time (matching cache_controller._generic_page_get)
        start_time = time.perf_counter()
        dummy_page_dst = [self.mem_pool_host.get_dummy_flat_data_page() for _ in keys]
        storage.batch_get(keys=keys, target_locations=dummy_page_dst)
        end_time = time.perf_counter()

        return end_time - start_time

    def _measure_batch_get_v1_performance(
        self, storage: HiCacheFile, keys: List[str], host_indices: torch.Tensor
    ) -> float:
        """Measure batch_get_v1 performance (matching cache_controller logic)."""
        # Warm up
        storage.batch_get_v1(keys=keys, host_indices=host_indices)

        # Measure - use perf_counter for high precision
        start_time = time.perf_counter()
        storage.batch_get_v1(keys=keys, host_indices=host_indices)
        end_time = time.perf_counter()

        return end_time - start_time

    def test_get_page_buffer_meta_mha(self):
        """Test get_page_buffer_meta for MHA model."""
        print("\n" + "=" * 80)
        print("Testing get_page_buffer_meta for MHA model")
        print("=" * 80)

        # Create MHA memory pool
        mem_pool = MockHostKVCache(page_size=self.page_size, is_mla=False)

        # Create test indices: 2 pages
        num_pages = 2
        host_indices = torch.cat(
            [
                torch.full((self.page_size,), 0, dtype=torch.int64),
                torch.full((self.page_size,), 1, dtype=torch.int64),
            ]
        )

        # Get buffer metadata
        ptr_list, element_size_list = mem_pool.get_page_buffer_meta(host_indices)

        # Verify results
        # For MHA: should have K and V pairs (2 pages * 2 = 4 pointers)
        expected_ptr_count = num_pages * 2  # K and V for each page
        self.assertEqual(len(ptr_list), expected_ptr_count)
        self.assertEqual(len(element_size_list), expected_ptr_count)

        # Verify all sizes are the same (half page for K or V)
        expected_size = (mem_pool.numel_per_page // 2) * mem_pool.dtype.itemsize
        for size in element_size_list:
            self.assertEqual(size, expected_size)

        # Verify pointers are valid (non-zero)
        for ptr in ptr_list:
            self.assertGreater(ptr, 0)

        # Verify K and V pointers for each page are different
        for i in range(num_pages):
            k_idx = i * 2
            v_idx = i * 2 + 1
            k_ptr = ptr_list[k_idx]
            v_ptr = ptr_list[v_idx]
            self.assertNotEqual(k_ptr, v_ptr)
            # V should be after K
            self.assertGreater(v_ptr, k_ptr)

        print(
            f"✓ MHA test passed: {len(ptr_list)} pointers, size={expected_size} bytes each"
        )
        print(f"  Page 0: K_ptr={ptr_list[0]}, V_ptr={ptr_list[1]}")
        print(f"  Page 1: K_ptr={ptr_list[2]}, V_ptr={ptr_list[3]}")

    def test_get_page_buffer_meta_mla(self):
        """Test get_page_buffer_meta for MLA model."""
        print("\n" + "=" * 80)
        print("Testing get_page_buffer_meta for MLA model")
        print("=" * 80)

        # Create MLA memory pool
        mem_pool = MockHostKVCache(page_size=self.page_size, is_mla=True)

        # Create test indices: 2 pages
        num_pages = 2
        host_indices = torch.cat(
            [
                torch.full((self.page_size,), 0, dtype=torch.int64),
                torch.full((self.page_size,), 1, dtype=torch.int64),
            ]
        )

        # Get buffer metadata
        ptr_list, element_size_list = mem_pool.get_page_buffer_meta(host_indices)

        # Verify results
        # For MLA: should have one pointer per page
        expected_ptr_count = num_pages
        self.assertEqual(len(ptr_list), expected_ptr_count)
        self.assertEqual(len(element_size_list), expected_ptr_count)

        # Verify all sizes are the same (full page)
        expected_size = mem_pool.numel_per_page * mem_pool.dtype.itemsize
        for size in element_size_list:
            self.assertEqual(size, expected_size)

        # Verify pointers are valid (non-zero)
        for ptr in ptr_list:
            self.assertGreater(ptr, 0)

        # Verify pointers are different for different pages
        self.assertNotEqual(ptr_list[0], ptr_list[1])

        print(
            f"✓ MLA test passed: {len(ptr_list)} pointers, size={expected_size} bytes each"
        )
        print(f"  Page 0: ptr={ptr_list[0]}")
        print(f"  Page 1: ptr={ptr_list[1]}")

    def test_get_page_buffer_meta_integration(self):
        """Test get_page_buffer_meta integration with batch_set_v1."""
        print("\n" + "=" * 80)
        print("Testing get_page_buffer_meta integration with batch_set_v1")
        print("=" * 80)

        num_keys = 5
        keys, host_indices = self._create_test_data(num_keys)
        storage = self._create_storage()

        # Verify that batch_set_v1 uses get_page_buffer_meta internally
        # by checking that it works correctly
        results = storage.batch_set_v1(keys=keys, host_indices=host_indices)

        # All writes should succeed
        self.assertEqual(len(results), num_keys)
        self.assertTrue(all(results), "All batch_set_v1 operations should succeed")

        # Verify files were created
        for key in keys:
            batch_file_path = os.path.join(
                self.test_dir, f"{storage._get_suffixed_key(key)}.batch.bin"
            )
            self.assertTrue(
                os.path.exists(batch_file_path), f"File should exist: {batch_file_path}"
            )

            # Verify file size is correct (K + V for MHA)
            expected_size = (
                self.mem_pool_host.numel_per_page * self.mem_pool_host.dtype.itemsize
            )
            actual_size = os.path.getsize(batch_file_path)
            self.assertEqual(
                actual_size,
                expected_size,
                f"File size mismatch for {key}: expected {expected_size}, got {actual_size}",
            )

        print(f"✓ Integration test passed: {num_keys} keys written successfully")
        print(f"  All files created with correct sizes")

    def test_performance_comparison_small_batch(self):
        """Compare performance with small batch (10 keys)."""
        print("\n" + "=" * 80)
        print("Performance Comparison: Small Batch (10 keys)")
        print("=" * 80)

        num_keys = 10
        keys, host_indices = self._create_test_data(num_keys)

        write_time_old = None
        read_time_old = None
        write_time_new = None
        read_time_new = None

        # Test old interface (batch_set/batch_get) if needed
        if self.use_interface in ["old", "both"]:
            storage_old = self._create_storage()

            # Write performance (matching cache_controller._generic_page_set logic)
            write_time_old = self._measure_batch_set_performance(
                storage_old, keys, host_indices
            )
            print(f"\n[Old Interface] batch_set: {write_time_old*1000:.3f} ms")

            # Read performance (matching cache_controller._generic_page_get logic)
            read_time_old = self._measure_batch_get_performance(storage_old, keys)
            print(f"[Old Interface] batch_get: {read_time_old*1000:.3f} ms")

        # Test new interface (batch_set_v1/batch_get_v1) if needed
        if self.use_interface in ["new", "both"]:
            # Re-initialize memory pool
            for i in range(num_keys):
                test_data = torch.randn(
                    self.mem_pool_host.numel_per_page, dtype=self.mem_pool_host.dtype
                )
                test_data[0] = float(i + 1)
                self.mem_pool_host.set_from_flat_data_page(i, test_data)

            storage_new = self._create_storage()

            # Write performance
            write_time_new = self._measure_batch_set_v1_performance(
                storage_new, keys, host_indices
            )
            print(f"\n[New Interface] batch_set_v1: {write_time_new*1000:.3f} ms")

            # Read performance
            read_time_new = self._measure_batch_get_v1_performance(
                storage_new, keys, host_indices
            )
            print(f"[New Interface] batch_get_v1: {read_time_new*1000:.3f} ms")

        # Calculate speedup if both were tested
        if self.use_interface == "both" and write_time_old and write_time_new:
            write_speedup = write_time_old / write_time_new if write_time_new > 0 else 0
            read_speedup = read_time_old / read_time_new if read_time_new > 0 else 0

            print(f"\n{'='*80}")
            print(
                f"Write Speedup: {write_speedup:.2f}x ({'faster' if write_speedup > 1 else 'slower'})"
            )
            print(
                f"Read Speedup:  {read_speedup:.2f}x ({'faster' if read_speedup > 1 else 'slower'})"
            )
            print(f"{'='*80}\n")

    def test_performance_comparison_medium_batch(self):
        """Compare performance with medium batch (50 keys)."""
        print("\n" + "=" * 80)
        print("Performance Comparison: Medium Batch (50 keys)")
        print("=" * 80)

        num_keys = 50
        keys, host_indices = self._create_test_data(num_keys)

        write_time_old = None
        read_time_old = None
        write_time_new = None
        read_time_new = None

        # Test old interface (batch_set/batch_get) if needed
        if self.use_interface in ["old", "both"]:
            storage_old = self._create_storage()
            write_time_old = self._measure_batch_set_performance(
                storage_old, keys, host_indices
            )
            read_time_old = self._measure_batch_get_performance(storage_old, keys)
            print(f"\n[Old Interface] batch_set: {write_time_old*1000:.3f} ms")
            print(f"[Old Interface] batch_get: {read_time_old*1000:.3f} ms")

        # Test new interface (batch_set_v1/batch_get_v1) if needed
        if self.use_interface in ["new", "both"]:
            # Re-initialize memory pool
            for i in range(num_keys):
                test_data = torch.randn(
                    self.mem_pool_host.numel_per_page, dtype=self.mem_pool_host.dtype
                )
                test_data[0] = float(i + 1)
                self.mem_pool_host.set_from_flat_data_page(i, test_data)

            storage_new = self._create_storage()
            write_time_new = self._measure_batch_set_v1_performance(
                storage_new, keys, host_indices
            )
            read_time_new = self._measure_batch_get_v1_performance(
                storage_new, keys, host_indices
            )
            print(f"\n[New Interface] batch_set_v1: {write_time_new*1000:.3f} ms")
            print(f"[New Interface] batch_get_v1: {read_time_new*1000:.3f} ms")

        # Calculate speedup if both were tested
        if self.use_interface == "both" and write_time_old and write_time_new:
            write_speedup = write_time_old / write_time_new if write_time_new > 0 else 0
            read_speedup = read_time_old / read_time_new if read_time_new > 0 else 0

            print(f"\n{'='*80}")
            print(
                f"Write Speedup: {write_speedup:.2f}x ({'faster' if write_speedup > 1 else 'slower'})"
            )
            print(
                f"Read Speedup:  {read_speedup:.2f}x ({'faster' if read_speedup > 1 else 'slower'})"
            )
            print(f"{'='*80}\n")

    def test_performance_comparison_large_batch(self):
        """Compare performance with large batch (100 keys)."""
        print("\n" + "=" * 80)
        print("Performance Comparison: Large Batch (100 keys)")
        print("=" * 80)

        num_keys = 100
        keys, host_indices = self._create_test_data(num_keys)

        write_time_old = None
        read_time_old = None
        write_time_new = None
        read_time_new = None

        # Test old interface (batch_set/batch_get) if needed
        if self.use_interface in ["old", "both"]:
            storage_old = self._create_storage()
            write_time_old = self._measure_batch_set_performance(
                storage_old, keys, host_indices
            )
            read_time_old = self._measure_batch_get_performance(storage_old, keys)
            print(f"\n[Old Interface] batch_set: {write_time_old*1000:.3f} ms")
            print(f"[Old Interface] batch_get: {read_time_old*1000:.3f} ms")

        # Test new interface (batch_set_v1/batch_get_v1) if needed
        if self.use_interface in ["new", "both"]:
            # Re-initialize memory pool
            for i in range(num_keys):
                test_data = torch.randn(
                    self.mem_pool_host.numel_per_page, dtype=self.mem_pool_host.dtype
                )
                test_data[0] = float(i + 1)
                self.mem_pool_host.set_from_flat_data_page(i, test_data)

            storage_new = self._create_storage()
            write_time_new = self._measure_batch_set_v1_performance(
                storage_new, keys, host_indices
            )
            read_time_new = self._measure_batch_get_v1_performance(
                storage_new, keys, host_indices
            )
            print(f"\n[New Interface] batch_set_v1: {write_time_new*1000:.3f} ms")
            print(f"[New Interface] batch_get_v1: {read_time_new*1000:.3f} ms")

        # Calculate speedup if both were tested
        if self.use_interface == "both" and write_time_old and write_time_new:
            write_speedup = write_time_old / write_time_new if write_time_new > 0 else 0
            read_speedup = read_time_old / read_time_new if read_time_new > 0 else 0

            print(f"\n{'='*80}")
            print(
                f"Write Speedup: {write_speedup:.2f}x ({'faster' if write_speedup > 1 else 'slower'})"
            )
            print(
                f"Read Speedup:  {read_speedup:.2f}x ({'faster' if read_speedup > 1 else 'slower'})"
            )
            print(f"{'='*80}\n")

    def test_performance_comparison_multiple_runs(self):
        """Compare performance with multiple runs for statistical accuracy."""
        print("\n" + "=" * 80)
        print("Performance Comparison: Multiple Runs (10 keys, 5 runs)")
        print("=" * 80)

        num_keys = 10
        num_runs = 5

        write_times_old = []
        read_times_old = []
        write_times_new = []
        read_times_new = []

        for run in range(num_runs):
            # Recreate test data for each run
            keys, host_indices = self._create_test_data(num_keys)

            # Test old interface (matching cache_controller logic) if needed
            if self.use_interface in ["old", "both"]:
                storage_old = self._create_storage()
                write_time_old = self._measure_batch_set_performance(
                    storage_old, keys, host_indices
                )
                read_time_old = self._measure_batch_get_performance(storage_old, keys)
                write_times_old.append(write_time_old)
                read_times_old.append(read_time_old)

            # Test new interface if needed
            if self.use_interface in ["new", "both"]:
                # Re-initialize for new interface
                for i in range(num_keys):
                    test_data = torch.randn(
                        self.mem_pool_host.numel_per_page,
                        dtype=self.mem_pool_host.dtype,
                    )
                    test_data[0] = float(i + 1)
                    self.mem_pool_host.set_from_flat_data_page(i, test_data)

                storage_new = self._create_storage()
                write_time_new = self._measure_batch_set_v1_performance(
                    storage_new, keys, host_indices
                )
                read_time_new = self._measure_batch_get_v1_performance(
                    storage_new, keys, host_indices
                )
                write_times_new.append(write_time_new)
                read_times_new.append(read_time_new)

            # Clean up for next run
            shutil.rmtree(self.test_dir)
            self.test_dir = tempfile.mkdtemp(prefix="hicache_perf_test_")

        # Calculate statistics
        if self.use_interface in ["old", "both"] and write_times_old:
            avg_write_old = sum(write_times_old) / len(write_times_old)
            avg_read_old = sum(read_times_old) / len(read_times_old)
            print(f"\n[Old Interface] Average batch_set: {avg_write_old*1000:.3f} ms")
            print(f"[Old Interface] Average batch_get: {avg_read_old*1000:.3f} ms")

        if self.use_interface in ["new", "both"] and write_times_new:
            avg_write_new = sum(write_times_new) / len(write_times_new)
            avg_read_new = sum(read_times_new) / len(read_times_new)
            print(
                f"\n[New Interface] Average batch_set_v1: {avg_write_new*1000:.3f} ms"
            )
            print(f"[New Interface] Average batch_get_v1: {avg_read_new*1000:.3f} ms")

        # Calculate speedup if both were tested
        if self.use_interface == "both" and write_times_old and write_times_new:
            avg_write_old = sum(write_times_old) / len(write_times_old)
            avg_read_old = sum(read_times_old) / len(read_times_old)
            avg_write_new = sum(write_times_new) / len(write_times_new)
            avg_read_new = sum(read_times_new) / len(read_times_new)

            write_speedup = avg_write_old / avg_write_new if avg_write_new > 0 else 0
            read_speedup = avg_read_old / avg_read_new if avg_read_new > 0 else 0

            print(f"\n{'='*80}")
            print(
                f"Average Write Speedup: {write_speedup:.2f}x ({'faster' if write_speedup > 1 else 'slower'})"
            )
            print(
                f"Average Read Speedup:  {read_speedup:.2f}x ({'faster' if read_speedup > 1 else 'slower'})"
            )
            print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys

    # Allow command line argument to select interface
    # Usage: python3 test_hicache_storage_performance_comparison.py [old|new|both]
    if len(sys.argv) > 1:
        interface_choice = sys.argv[1].lower()
        if interface_choice in ["old", "new", "both"]:
            # Set the interface choice for all tests
            TestHiCacheFilePerformanceComparison.use_interface = interface_choice
            print(f"\n{'='*80}")
            print(f"Testing with interface: {interface_choice}")
            print(f"{'='*80}\n")
            # Remove the argument so unittest doesn't try to parse it
            sys.argv = [sys.argv[0]] + sys.argv[2:]
        else:
            print(f"Invalid interface choice: {interface_choice}")
            print(
                "Usage: python3 test_hicache_storage_performance_comparison.py [old|new|both]"
            )
            sys.exit(1)

    unittest.main(verbosity=2)
