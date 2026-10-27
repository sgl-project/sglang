from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=9, suite="base-a-test-cpu")

import sys

sys.modules["libtpu"] = None
import ctypes
import ctypes.util
import mmap
import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import torch

from sglang.srt.mem_cache import numa_memory
from sglang.srt.mem_cache.pool_host.common import ShmHostTensorAllocator
from sglang.srt.mem_cache.storage.mmap import alloc_shm, mmap_allocator
from sglang.srt.mem_cache.storage.mmap.mmap_allocator import _mmap_prefaulted
from sglang.test.test_utils import CustomTestCase


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


class TestNumaAllocationGuard(CustomTestCase):
    def test_rejects_oversized_mapping_before_prefaulting(self):
        # The synthetic limit is 32 KiB. Even a regression requests only 36 KiB;
        # sentinels prevent mapping or sizing a backing file.
        for allocate in (mmap_allocator.alloc_mmap, mmap_allocator.alloc_shm):
            for hugepages in ("", "2MB", "1GB"):
                with (
                    self.subTest(allocator=allocate.__name__, hugepages=hugepages),
                    unittest.mock.patch.object(
                        numa_memory, "_numa_memory_capacity", return_value=([2], 32768)
                    ),
                    unittest.mock.patch.object(
                        mmap_allocator.envs.SGLANG_HUGEPAGE_SIZE,
                        "get",
                        return_value=hugepages,
                    ),
                    unittest.mock.patch.object(
                        mmap_allocator,
                        "_mmap_prefaulted",
                        side_effect=AssertionError("reached mmap"),
                    ) as plain,
                    unittest.mock.patch.object(
                        mmap_allocator,
                        "_alloc_hugepage",
                        side_effect=AssertionError("reached hugepage mmap"),
                    ) as huge,
                    unittest.mock.patch.object(
                        mmap_allocator.os,
                        "memfd_create",
                        return_value=12345,
                        create=True,
                    ) as shm,
                    unittest.mock.patch.object(mmap_allocator.os, "close") as close,
                    unittest.mock.patch.object(
                        mmap_allocator.os,
                        "ftruncate",
                        side_effect=AssertionError("reached ftruncate"),
                    ) as truncate,
                ):
                    with self.assertRaisesRegex(ValueError, "permitted NUMA nodes"):
                        allocate((9, 1024), torch.float32)
                    plain.assert_not_called()
                    huge.assert_not_called()
                    truncate.assert_not_called()
                    if allocate is mmap_allocator.alloc_shm:
                        shm.assert_called_once()
                        close.assert_called_once_with(12345)
                    else:
                        shm.assert_not_called()
                        close.assert_not_called()

    def test_shm_filesystem_fallback_uses_only_cpuset_capacity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "has_memory").write_text("0-2")
            for node, kib in enumerate((32, 64, 128)):
                path = root / f"node{node}"
                path.mkdir()
                (path / "meminfo").write_text(f"Node {node} MemTotal: {kib} kB")
            capacity = numa_memory._numa_memory_capacity
            # Task bind permits 32 KiB, cpuset permits 96 KiB. A tmpfs mount
            # policy can use the latter. Neither trial reaches a real mapping.
            for memfd_error in (AttributeError("unavailable"), OSError("denied")):
                for kib in (36, 100):
                    with (
                        self.subTest(error=memfd_error, kib=kib),
                        unittest.mock.patch.object(
                            numa_memory,
                            "_get_numa_policy",
                            return_value=(2, {0}, {0, 1}),
                        ),
                        unittest.mock.patch.object(
                            numa_memory,
                            "_numa_memory_capacity",
                            side_effect=lambda **kw: capacity(root, **kw),
                        ),
                        unittest.mock.patch.object(
                            mmap_allocator.os,
                            "memfd_create",
                            side_effect=memfd_error,
                            create=True,
                        ),
                        unittest.mock.patch.object(
                            mmap_allocator.os,
                            "open",
                            return_value=12345,
                        ),
                        unittest.mock.patch.object(mmap_allocator.os, "unlink"),
                        unittest.mock.patch.object(mmap_allocator.os, "close") as close,
                        unittest.mock.patch.object(
                            mmap_allocator.os,
                            "ftruncate",
                        ) as truncate,
                        unittest.mock.patch.object(
                            mmap_allocator,
                            "_mmap_prefaulted",
                            side_effect=RuntimeError("mapping intercepted"),
                        ) as mapping,
                    ):
                        error = RuntimeError if kib == 36 else ValueError
                        message = "mapping intercepted" if kib == 36 else "total RAM"
                        with self.assertRaisesRegex(error, message):
                            mmap_allocator.alloc_shm((kib * 256,), torch.float32)
                        close.assert_called_once_with(12345)
                        if kib == 36:
                            truncate.assert_called_once_with(12345, 36 * 1024)
                            mapping.assert_called_once()
                        else:
                            truncate.assert_not_called()
                            mapping.assert_not_called()

    def test_small_mapping_still_allocates(self):
        with (
            unittest.mock.patch.object(
                numa_memory, "_numa_memory_capacity", return_value=([2], 32768)
            ),
            unittest.mock.patch.object(
                mmap_allocator.envs.SGLANG_HUGEPAGE_SIZE, "get", return_value=""
            ),
        ):
            tensor = mmap_allocator.alloc_mmap((1024,), torch.float32)
            tensor.fill_(7)
            self.assertEqual(tensor.sum().item(), 7168)
            tensor, fd, mm = mmap_allocator.alloc_shm((1024,), torch.float32)
            try:
                tensor.fill_(7)
                self.assertEqual(tensor.sum().item(), 7168)
            finally:
                del tensor
                mm.close()
                os.close(fd)

    def test_hugepage_rounding_does_not_reject_plain_page_fallback(self):
        with (
            unittest.mock.patch.object(
                numa_memory, "_numa_memory_capacity", return_value=([2], 32768)
            ),
            unittest.mock.patch.object(
                mmap_allocator.envs.SGLANG_HUGEPAGE_SIZE, "get", return_value="1GB"
            ),
            unittest.mock.patch.object(
                mmap_allocator, "_alloc_hugepage", side_effect=OSError("no hugepages")
            ) as huge,
        ):
            tensor = mmap_allocator.alloc_mmap((1024,), torch.float32)
            self.assertEqual(tensor.numel(), 1024)
            huge.assert_called_once()


if __name__ == "__main__":
    unittest.main()
