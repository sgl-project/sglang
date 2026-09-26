from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import sys

sys.modules["libtpu"] = None
import ctypes
import ctypes.util
import mmap
import os
import unittest
import unittest.mock

import torch

from sglang.srt.mem_cache.pool_host.common import ShmHostTensorAllocator
from sglang.srt.mem_cache.storage.mmap import alloc_mmap, alloc_shm
from sglang.srt.mem_cache.storage.mmap.mmap_allocator import (
    _has_madv_populate_write,
    _mmap_prefaulted,
)


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

    @unittest.skipUnless(
        _has_madv_populate_write(), "THP needs MADV_POPULATE_WRITE (Linux 5.14+)"
    )
    def test_alloc_mmap_thp_is_resident_and_advised_huge(self):
        """THP needs no hugetlbfs reservation, so it must work on any host.

        The region is 2 MiB aligned in size, fully pre-faulted, and carries
        MADV_HUGEPAGE ("hg" in smaps) -- whether the kernel then backs it with
        huge pages depends on fragmentation, so that part is not asserted.
        """
        from sglang.srt.environ import envs

        dims = (3, 1024 * 1024)  # 3 MiB of uint8, rounded up to 4 MiB
        with envs.SGLANG_HUGEPAGE_SIZE.override("THP"):
            tensor = alloc_mmap(dims, torch.uint8)
        self.assertEqual(tensor.shape, dims)
        tensor[-1, -1] = 7
        self.assertEqual(tensor[-1, -1].item(), 7)

        start = tensor.data_ptr()
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
        npages = 4 * 1024 * 1024 // mmap.PAGESIZE
        vec = (ctypes.c_ubyte * npages)()
        if (
            libc.mincore(
                ctypes.c_void_p(start), ctypes.c_size_t(npages * mmap.PAGESIZE), vec
            )
            != 0
        ):
            self.skipTest("mincore unavailable")
        self.assertTrue(
            all(v & 1 for v in vec), "THP mapping was not fully pre-faulted"
        )

        with open("/proc/self/smaps") as smaps:
            region = None
            for line in smaps:
                head = line.split()[0]
                if "-" in head and ":" not in head:
                    low, high = (int(x, 16) for x in head.split("-"))
                    region = low <= start < high
                elif region and line.startswith("VmFlags:"):
                    self.assertIn("hg", line.split()[1:])
                    break
            else:
                self.fail("THP mapping not found in /proc/self/smaps")

    def test_alloc_mmap_thp_refuses_without_populate_write(self):
        from sglang.srt.environ import envs

        with (
            envs.SGLANG_HUGEPAGE_SIZE.override("THP"),
            unittest.mock.patch(
                "sglang.srt.mem_cache.storage.mmap.mmap_allocator._has_madv_populate_write",
                return_value=False,
            ),
            self.assertRaisesRegex(OSError, "MADV_POPULATE_WRITE"),
        ):
            alloc_mmap((1024,), torch.uint8)

    def test_small_page_bytes_counts_overlapping_mappings(self):
        from sglang.srt.mem_cache.storage.mmap.mmap_allocator import _small_page_bytes

        smaps = """\
7f0000000000-7f0000800000 rw-p 00000000 00:00 0
Rss:                8192 kB
AnonHugePages:      6144 kB
VmFlags: rd wr mr mw me ac hg
7f0000800000-7f0001000000 rw-p 00000000 00:00 0
Rss:                8192 kB
AnonHugePages:      8192 kB
7f0002000000-7f0002200000 rw-p 00000000 00:00 0
Rss:                2048 kB
AnonHugePages:         0 kB
""".splitlines()
        start, end = 0x7F0000000000, 0x7F0001000000
        # 2 MiB of the first mapping is on 4 KiB pages; the third is outside.
        self.assertEqual(_small_page_bytes(smaps, start, end), 2 * 2**20)
        self.assertEqual(_small_page_bytes(smaps, 0x7F0000800000, end), 0)
        self.assertEqual(
            _small_page_bytes(smaps, 0x7F0002000000, 0x7F0002100000), 2 * 2**20
        )

    def _ensure(self, small_bytes, *, limit_mb=256, collapse_error=None):
        """Run _ensure_thp_backed with smaps readings ``small_bytes`` (before, after)."""
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.storage.mmap import mmap_allocator

        collapse = unittest.mock.Mock(side_effect=collapse_error)
        with (
            envs.SGLANG_HUGEPAGE_THP_MAX_SMALL_MB.override(limit_mb),
            unittest.mock.patch.object(
                mmap_allocator, "_thp_small_page_bytes", side_effect=list(small_bytes)
            ),
            unittest.mock.patch.object(mmap_allocator, "_madvise_collapse", collapse),
            self.assertLogs(mmap_allocator.logger, "INFO") as logs,
        ):
            try:
                mmap_allocator._ensure_thp_backed(None, 0, 4 * 2**30)
            finally:
                self.logs = logs.output
        return collapse

    def test_thp_fully_huge_needs_no_collapse(self):
        collapse = self._ensure([0])
        collapse.assert_not_called()
        self.assertIn("0.0 MiB on 4 KiB pages", self.logs[0])

    def test_thp_shortfall_is_collapsed(self):
        collapse = self._ensure([4 * 2**30 // 8, 0])
        collapse.assert_called_once()
        self.assertIn("512.0 MiB on 4 KiB pages after the pre-fault", self.logs[0])
        self.assertIn("MADV_COLLAPSE", self.logs[0])

    def test_thp_shortfall_left_after_collapse_fails_fast(self):
        with self.assertRaisesRegex(OSError, "still on 4 KiB pages"):
            self._ensure([2**30, 300 * 2**20])
        # Within the limit, or with no limit, the pool is kept.
        self._ensure([2**30, 200 * 2**20])
        self._ensure([2**30, 2**30], limit_mb=-1)

    def test_thp_collapse_error_still_checks(self):
        # e.g. EINVAL on a kernel without MADV_COLLAPSE, EAGAIN on a busy range.
        self._ensure([2**20, 2**20], collapse_error=OSError(22, "Invalid argument"))
        self.assertIn("Invalid argument", self.logs[0])
        with self.assertRaisesRegex(OSError, "still on 4 KiB pages"):
            self._ensure([2**30, 2**30], collapse_error=OSError(11, "busy"))

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


if __name__ == "__main__":
    unittest.main()
