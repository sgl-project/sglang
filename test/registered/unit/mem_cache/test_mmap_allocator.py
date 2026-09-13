from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import sys

sys.modules["libtpu"] = None
import ctypes
import ctypes.util
import mmap
import os
import tempfile
import unittest
import unittest.mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host.common import ShmHostTensorAllocator
from sglang.srt.mem_cache.storage.mmap import alloc_mmap, alloc_shm, mmap_allocator
from sglang.srt.mem_cache.storage.mmap.mmap_allocator import _mmap_prefaulted


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

    def _assert_resident(self, mm, alloc_bytes):
        # mincore() reports per-page residency, so the invariant is checked
        # rather than inferred from the flags that were passed.
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        npages = alloc_bytes // mmap.PAGESIZE
        vec = (ctypes.c_ubyte * npages)()
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        if libc.mincore(ctypes.c_void_p(addr), ctypes.c_size_t(alloc_bytes), vec) != 0:
            self.skipTest("mincore unavailable")
        self.assertTrue(all(v & 1 for v in vec), "mapping was not fully pre-faulted")

    def test_mmap_prefaulted_leaves_no_lazy_page(self):
        """Both populate paths must return a mapping with every page resident.

        cudaHostRegister pins these buffers, so a page still unfaulted at
        registration lets the device read memory that is not backed yet.
        """
        alloc_bytes = 64 * mmap.PAGESIZE
        flags = mmap.MAP_SHARED | mmap.MAP_ANONYMOUS

        with self.subTest(path="madvise"):
            mm = _mmap_prefaulted(-1, alloc_bytes, flags)
            try:
                self._assert_resident(mm, alloc_bytes)
            finally:
                mm.close()

        # MAP_POPULATE is unreachable on a 5.14+ kernel, so CI never runs it;
        # force the branch or it ships untested.
        with (
            self.subTest(path="map_populate"),
            unittest.mock.patch(
                "sglang.srt.mem_cache.storage.mmap.mmap_allocator._has_madv_populate_write",
                return_value=False,
            ),
        ):
            mm = _mmap_prefaulted(-1, alloc_bytes, flags)
            try:
                self._assert_resident(mm, alloc_bytes)
            finally:
                mm.close()

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


class TestHugetlbPool(unittest.TestCase):
    """What the host-pool preflight may credit from the kernel's hugetlb pool."""

    @staticmethod
    def _sysfs(root, free_2mb=None, free_1gb=None, resv_2mb=0, resv_1gb=0):
        for name, free, resv in (
            ("hugepages-2048kB", free_2mb, resv_2mb),
            ("hugepages-1048576kB", free_1gb, resv_1gb),
        ):
            if free is None:
                continue
            os.makedirs(os.path.join(root, name))
            for counter, value in (("free_hugepages", free), ("resv_hugepages", resv)):
                with open(os.path.join(root, name, counter), "w") as f:
                    f.write(f"{value}\n")

    def test_hugepage_size_requested_parses_the_env(self):
        for raw, expected in {"": 0, "2MB": 2 * 1024**2, " 1gb ": 1024**3}.items():
            with self.subTest(raw=raw), envs.SGLANG_HUGEPAGE_SIZE.override(raw):
                self.assertEqual(mmap_allocator.hugepage_size_requested(), expected)

    def test_unrecognized_hugepage_size_means_plain_pages(self):
        with (
            envs.SGLANG_HUGEPAGE_SIZE.override("4MB"),
            self.assertLogs(mmap_allocator.logger, "WARNING"),
        ):
            self.assertEqual(mmap_allocator.hugepage_size_requested(), 0)

    def test_hugetlb_pool_free_bytes_reads_the_requested_pool(self):
        # sysfs keeps one pool per page size; MemAvailable excludes all of them.
        with tempfile.TemporaryDirectory() as root:
            self._sysfs(root, free_2mb=6144, free_1gb=3)
            with unittest.mock.patch.object(
                mmap_allocator, "_HUGEPAGE_SYSFS_DIR", root
            ):
                with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"):
                    self.assertEqual(
                        mmap_allocator.hugetlb_pool_free_bytes(), 6144 * 2 * 1024**2
                    )
                with envs.SGLANG_HUGEPAGE_SIZE.override("1GB"):
                    self.assertEqual(
                        mmap_allocator.hugetlb_pool_free_bytes(), 3 * 1024**3
                    )

    def test_hugetlb_pool_free_bytes_excludes_pages_reserved_by_other_mappings(self):
        # A mapping reserves its pages at mmap time and faults them in
        # afterwards; until then sysfs still counts them as free. Co-located
        # ranks populate their pools concurrently, so only free - resv can back
        # a new mapping.
        with tempfile.TemporaryDirectory() as root:
            self._sysfs(root, free_2mb=6144, resv_2mb=1000)
            with (
                unittest.mock.patch.object(mmap_allocator, "_HUGEPAGE_SYSFS_DIR", root),
                envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
            ):
                self.assertEqual(
                    mmap_allocator.hugetlb_pool_free_bytes(),
                    (6144 - 1000) * 2 * 1024**2,
                )

    def test_hugetlb_pool_free_bytes_is_zero_when_mmap_would_not_use_hugetlb(self):
        with tempfile.TemporaryDirectory() as root:
            self._sysfs(root, free_2mb=6144)
            with unittest.mock.patch.object(
                mmap_allocator, "_HUGEPAGE_SYSFS_DIR", root
            ):
                with (
                    self.subTest(why="not requested"),
                    envs.SGLANG_HUGEPAGE_SIZE.override(""),
                ):
                    self.assertEqual(mmap_allocator.hugetlb_pool_free_bytes(), 0)
                with (
                    self.subTest(why="libc unavailable"),
                    envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
                    unittest.mock.patch.object(mmap_allocator, "_libc", None),
                ):
                    self.assertEqual(mmap_allocator.hugetlb_pool_free_bytes(), 0)
                with (
                    self.subTest(why="no pool of that size"),
                    envs.SGLANG_HUGEPAGE_SIZE.override("1GB"),
                    self.assertLogs(mmap_allocator.logger, "WARNING"),
                ):
                    self.assertEqual(mmap_allocator.hugetlb_pool_free_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
