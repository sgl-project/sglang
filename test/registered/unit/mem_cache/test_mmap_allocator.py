from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

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
from sglang.srt.mem_cache.storage.mmap import alloc_shm
from sglang.srt.mem_cache.storage.mmap.mmap_allocator import _mmap_prefaulted


class TestMmapAllocator(unittest.TestCase):
    def _alloc_with_hugetlb_failure(self, **patches):
        # MAP_HUGETLB fails the way it does on a host with nr_hugepages=0.
        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.storage.mmap import alloc_mmap, mmap_allocator

        if mmap_allocator._libc is None:
            self.skipTest("libc is not available")
        with (
            envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
            unittest.mock.patch.object(
                mmap_allocator,
                "_alloc_hugepage",
                side_effect=OSError(12, "Cannot allocate memory"),
            ),
            unittest.mock.patch.multiple(mmap_allocator, **patches),
        ):
            return alloc_mmap((4, 1024), torch.float32)

    def test_alloc_mmap_uses_thp_when_hugetlb_has_no_reservation(self):
        n_bytes = 4 * 1024 * 4
        backing = (ctypes.c_uint8 * n_bytes)()
        thp = unittest.mock.Mock(return_value=backing)
        tensor = self._alloc_with_hugetlb_failure(
            _thp_page_size=lambda: 2 * 1024 * 1024, _alloc_thp=thp
        )
        thp.assert_called_once_with(n_bytes, 2 * 1024 * 1024)
        self.assertEqual(tensor.shape, (4, 1024))
        tensor[0, 0] = 7.0
        self.assertEqual(tensor.data_ptr(), ctypes.addressof(backing))

    def test_alloc_mmap_falls_back_to_plain_when_thp_is_off(self):
        tensor = self._alloc_with_hugetlb_failure(_thp_page_size=lambda: 0)
        self.assertEqual(tensor.shape, (4, 1024))
        tensor[:] = 1.0
        self.assertEqual(tensor.sum().item(), 4 * 1024)

    def test_alloc_mmap_falls_back_to_plain_when_thp_alloc_fails(self):
        thp = unittest.mock.Mock(side_effect=OSError(22, "Invalid argument"))
        tensor = self._alloc_with_hugetlb_failure(
            _thp_page_size=lambda: 2 * 1024 * 1024, _alloc_thp=thp
        )
        thp.assert_called_once()
        self.assertEqual(tensor.shape, (4, 1024))

    @unittest.skipUnless(
        sys.platform == "linux", "transparent hugepages are Linux-only"
    )
    def test_alloc_thp_is_aligned_and_writable(self):
        from sglang.srt.mem_cache.storage.mmap import mmap_allocator

        page = mmap_allocator._thp_page_size()
        if not page or mmap_allocator._libc is None:
            self.skipTest("transparent hugepages are not available on this host")
        n_bytes = 3 * page + 4096
        try:
            array = mmap_allocator._alloc_thp(n_bytes, page)
        except OSError as e:
            self.skipTest(f"madvise(MADV_HUGEPAGE) not permitted here: {e}")
        self.assertEqual(ctypes.addressof(array) % page, 0)
        array[0] = 1
        array[n_bytes - 1] = 2
        self.assertEqual((array[0], array[n_bytes - 1]), (1, 2))

    @unittest.skipUnless(
        sys.platform == "linux", "transparent hugepages are Linux-only"
    )
    def test_alloc_thp_warns_when_faults_fall_back_to_4k(self):
        from sglang.srt.mem_cache.storage.mmap import mmap_allocator

        page = mmap_allocator._thp_page_size()
        if not page or mmap_allocator._libc is None:
            self.skipTest("transparent hugepages are not available on this host")
        with (
            unittest.mock.patch.object(
                mmap_allocator, "_thp_fault_fallbacks", side_effect=[10, 15]
            ),
            self.assertLogs(mmap_allocator.logger, level="WARNING") as logs,
        ):
            try:
                mmap_allocator._alloc_thp(page, page)
            except OSError as e:
                self.skipTest(f"madvise(MADV_HUGEPAGE) not permitted here: {e}")
        self.assertIn("5 transparent hugepage faults fell back", logs.output[0])

    def test_thp_fault_fallbacks_is_a_count_or_unknown(self):
        from sglang.srt.mem_cache.storage.mmap import mmap_allocator

        self.assertGreaterEqual(mmap_allocator._thp_fault_fallbacks(), -1)

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
