from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

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
