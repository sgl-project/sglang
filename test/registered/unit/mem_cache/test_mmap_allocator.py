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

    def test_alloc_mmap_hugepage_modes(self):
        import ctypes
        from unittest.mock import patch

        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.storage.mmap import mmap_allocator
        from sglang.srt.mem_cache.storage.mmap.mmap_allocator import (
            MEM_BACKEND_HUGEPAGE,
            MEM_BACKEND_MMAP,
            tensor_mem_backend,
        )

        with envs.SGLANG_HUGEPAGE_MODE.override("off"):
            with envs.SGLANG_HUGEPAGE_SIZE.override(""):
                buf = mmap_allocator.alloc_mmap((4,), torch.float32)
                self.assertEqual(tensor_mem_backend(buf), MEM_BACKEND_MMAP)

        with envs.SGLANG_HUGEPAGE_MODE.override("prefer"):
            with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"):
                with patch.object(mmap_allocator, "_alloc_hugepage") as mock_hp:
                    mock_hp.return_value = (ctypes.c_uint8 * 16)()
                    buf = mmap_allocator.alloc_mmap((4,), torch.float32)
                self.assertEqual(tensor_mem_backend(buf), MEM_BACKEND_HUGEPAGE)

        with envs.SGLANG_HUGEPAGE_MODE.override("prefer"):
            with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"):
                with patch.object(
                    mmap_allocator,
                    "_alloc_hugepage",
                    side_effect=OSError("no hugepages"),
                ):
                    with patch.object(mmap_allocator, "_libc", object()):
                        buf = mmap_allocator.alloc_mmap((4,), torch.float32)
                self.assertEqual(tensor_mem_backend(buf), MEM_BACKEND_MMAP)

        with envs.SGLANG_HUGEPAGE_MODE.override("required"):
            with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"):
                with patch.object(
                    mmap_allocator,
                    "_alloc_hugepage",
                    side_effect=OSError("no hugepages"),
                ):
                    with patch.object(mmap_allocator, "_libc", object()):
                        with patch.object(mmap_allocator.mmap, "mmap") as plain_mmap:
                            with self.assertRaisesRegex(RuntimeError, "no hugepages"):
                                mmap_allocator.alloc_mmap((4,), torch.float32)
                            plain_mmap.assert_not_called()

        with envs.SGLANG_HUGEPAGE_MODE.override("required"):
            with envs.SGLANG_HUGEPAGE_SIZE.override(""):
                with self.assertRaisesRegex(ValueError, "SGLANG_HUGEPAGE_SIZE"):
                    mmap_allocator.alloc_mmap((4,), torch.float32)

    def test_memory_available_bytes_hugepage_modes(self):
        from unittest.mock import MagicMock, patch

        from sglang.srt.environ import envs
        from sglang.srt.mem_cache.storage.mmap import mmap_allocator
        from sglang.srt.mem_cache.storage.mmap.mmap_allocator import (
            HICACHE_HOST_MEMORY_RESERVE_BYTES,
            HUGEPAGE_BYTES_2MB,
            memory_available_bytes,
        )

        normal_usable_100mb = 100 * 1024 * 1024
        normal_ram_with_reserve = (
            HICACHE_HOST_MEMORY_RESERVE_BYTES + normal_usable_100mb
        )
        hugetlb_200mb = 100 * HUGEPAGE_BYTES_2MB
        low_ram = MagicMock(available=normal_ram_with_reserve)
        with patch.object(
            mmap_allocator.psutil, "virtual_memory", return_value=low_ram
        ):
            with envs.SGLANG_HUGEPAGE_MODE.override("off"):
                with envs.SGLANG_HUGEPAGE_SIZE.override(""):
                    self.assertEqual(memory_available_bytes(), normal_usable_100mb)
            with envs.SGLANG_HUGEPAGE_MODE.override("prefer"):
                with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"):
                    with patch.object(
                        mmap_allocator, "hugepage_available_bytes", return_value=0
                    ):
                        self.assertEqual(memory_available_bytes(), normal_usable_100mb)
                with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"):
                    with patch.object(
                        mmap_allocator,
                        "hugepage_available_bytes",
                        return_value=hugetlb_200mb,
                    ):
                        self.assertEqual(memory_available_bytes(), hugetlb_200mb)

        hugetlb_bytes = 3 * HUGEPAGE_BYTES_2MB
        with patch.object(
            mmap_allocator.psutil,
            "virtual_memory",
            return_value=MagicMock(available=1 << 50),
        ):
            with envs.SGLANG_HUGEPAGE_MODE.override("required"):
                with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"):
                    with patch.object(
                        mmap_allocator,
                        "hugepage_available_bytes",
                        return_value=hugetlb_bytes,
                    ):
                        self.assertEqual(memory_available_bytes(), hugetlb_bytes)

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
        with self.subTest(path="map_populate"), unittest.mock.patch(
            "sglang.srt.mem_cache.storage.mmap.mmap_allocator._has_madv_populate_write",
            return_value=False,
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


if __name__ == "__main__":
    unittest.main()
