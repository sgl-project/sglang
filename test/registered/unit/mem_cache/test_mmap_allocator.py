from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import sys

sys.modules["libtpu"] = None
import mmap
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host.common import (
    HostTensorAllocator,
    ShmHostTensorAllocator,
)
from sglang.srt.mem_cache.storage.mmap import alloc_mmap, alloc_shm, mmap_allocator


class TestMmapAllocator(unittest.TestCase):
    def test_alloc_mmap(self):
        dims = (10, 1024)
        dtype = torch.float32
        tensor = alloc_mmap(dims, dtype)
        self.assertEqual(tensor.shape, dims)
        self.assertEqual(tensor.dtype, dtype)
        # Verify it has mapped memory address
        self.assertGreater(tensor.data_ptr(), 0)

    def test_alloc_shm(self):
        dims = (10, 1024)
        dtype = torch.float32
        tensor, fd, mm = alloc_shm(dims, dtype)

        self.assertEqual(tensor.shape, dims)
        self.assertEqual(tensor.dtype, dtype)
        self.assertGreater(tensor.data_ptr(), 0)
        self.assertGreaterEqual(fd, 0)
        self.assertIsInstance(mm, mmap.mmap)

        # Check that we can write to the tensor
        tensor[0, 0] = 42.0
        self.assertEqual(tensor[0, 0].item(), 42.0)

        # Check that the FD is open and valid
        try:
            os.lseek(fd, 0, os.SEEK_SET)
        except OSError:
            self.fail("FD is not valid or closed")

        # Cleanup
        mm.close()
        os.close(fd)

    def test_shm_host_tensor_allocator(self):
        allocator = ShmHostTensorAllocator()
        dims = (2, 512)
        dtype = torch.int32

        tensor = allocator.allocate(dims, dtype, "cpu")
        self.assertEqual(tensor.shape, dims)
        self.assertEqual(tensor.dtype, dtype)
        self.assertIsNotNone(allocator.fd)
        self.assertGreaterEqual(allocator.fd, 0)

        # Write data and check
        tensor[1, 1] = 99
        self.assertEqual(tensor[1, 1].item(), 99)

        # Test destructor cleans up fd
        fd = allocator.fd
        # Trigger GC / deletion
        del allocator

        # Verify fd is closed
        with self.assertRaises(OSError):
            os.fstat(fd)

    def test_alloc_shm_unlinked(self):
        dims = (4, 256)
        dtype = torch.float32
        tensor, fd, mm = alloc_shm(dims, dtype)

        # On Linux, the path of an unlinked fd shows up in /proc/self/fd/
        # with a ' (deleted)' suffix.
        fd_path = f"/proc/self/fd/{fd}"
        try:
            resolved_path = os.readlink(fd_path)
            self.assertIn("sglang_host_pool_", resolved_path)
            self.assertTrue(resolved_path.endswith(" (deleted)"))
        except OSError:
            # If procfs is not available or readlink fails, fallback to direct path existence check
            self.assertFalse(os.path.exists(f"/dev/shm/sglang_host_pool_"))

        # Cleanup
        mm.close()
        os.close(fd)

    def test_alloc_shm_hugepage_warning(self):
        from sglang.srt.environ import envs

        envs.SGLANG_HUGEPAGE_SIZE.override("2MB")
        try:
            # Should succeed by falling back to plain page size mapping
            dims = (2, 2)
            tensor, fd, mm = alloc_shm(dims, torch.float32)
            self.assertEqual(tensor.shape, dims)
            mm.close()
            os.close(fd)
        finally:
            envs.SGLANG_HUGEPAGE_SIZE.override(None)

    def test_shm_host_tensor_allocator_invalid_device(self):
        allocator = ShmHostTensorAllocator()
        with self.assertRaises(AssertionError) as ctx:
            allocator.allocate((2, 2), torch.float32, device="cuda")
        self.assertIn("only supports CPU allocations", str(ctx.exception))


class TestFreeHugepageBytes(unittest.TestCase):
    """free_hugepage_bytes() reports capacity MemAvailable deliberately hides."""

    @staticmethod
    def _fake_pool(root, page_kb, free_count):
        pool_dir = os.path.join(root, f"hugepages-{page_kb}kB")
        os.makedirs(pool_dir)
        with open(os.path.join(pool_dir, "free_hugepages"), "w") as f:
            f.write(f"{free_count}\n")

    def _read(self, hugepage_size, root):
        with envs.SGLANG_HUGEPAGE_SIZE.override(hugepage_size):
            with patch.object(mmap_allocator, "_HUGEPAGE_SYSFS_DIR", root):
                return mmap_allocator.free_hugepage_bytes()

    def test_zero_when_hugepages_not_requested(self):
        with tempfile.TemporaryDirectory() as root:
            self._fake_pool(root, 2048, 1024)
            # No credit unless the allocator was actually told to use hugetlb.
            self.assertEqual(self._read(None, root), 0)

    def test_zero_for_unrecognized_size(self):
        with tempfile.TemporaryDirectory() as root:
            self._fake_pool(root, 2048, 1024)
            self.assertEqual(self._read("4MB", root), 0)

    def test_reads_2mb_pool(self):
        with tempfile.TemporaryDirectory() as root:
            self._fake_pool(root, 2048, 1024)
            self.assertEqual(self._read("2MB", root), 1024 * 2 * 1024 * 1024)

    def test_reads_1gb_pool(self):
        with tempfile.TemporaryDirectory() as root:
            self._fake_pool(root, 1048576, 3)
            self.assertEqual(self._read("1GB", root), 3 * 1024**3)

    def test_does_not_cross_read_other_page_sizes(self):
        with tempfile.TemporaryDirectory() as root:
            # Only a 2MB pool exists; a 1GB request must not borrow from it.
            self._fake_pool(root, 2048, 1024)
            self.assertEqual(self._read("1GB", root), 0)

    def test_zero_when_no_hugetlb_support(self):
        # Non-Linux hosts (and kernels without hugetlb) have no sysfs pool.
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(self._read("2MB", os.path.join(root, "missing")), 0)

    def test_zero_on_unreadable_counter(self):
        with tempfile.TemporaryDirectory() as root:
            pool_dir = os.path.join(root, "hugepages-2048kB")
            os.makedirs(pool_dir)
            with open(os.path.join(pool_dir, "free_hugepages"), "w") as f:
                f.write("not-a-number\n")
            self.assertEqual(self._read("2MB", root), 0)


class TestAllocatorHugetlbCapability(unittest.TestCase):
    def test_mmap_allocator_can_use_hugetlb(self):
        self.assertTrue(HostTensorAllocator.uses_hugetlb)

    def test_shm_allocator_cannot_use_hugetlb(self):
        # alloc_shm warns and falls back to plain pages, so crediting the
        # hugetlb pool to it would let an oversized pool past the pre-check.
        self.assertFalse(ShmHostTensorAllocator.uses_hugetlb)


if __name__ == "__main__":
    unittest.main()
