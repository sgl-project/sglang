from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import sys

sys.modules["libtpu"] = None
import importlib
import mmap
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host.common import (
    HostTensorAllocator,
    ShmHostTensorAllocator,
)
from sglang.srt.mem_cache.storage.mmap import alloc_mmap, alloc_shm, mmap_allocator

# Optional storage backends: importable without their third-party packages
# (those are imported inside __init__), but not worth failing this file over.
_OPTIONAL_ALLOCATORS = (
    (
        "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store",
        "MooncakeHostTensorAllocator",
    ),
    (
        "sglang.srt.mem_cache.storage.umbp.umbp_host_allocator",
        "UMBPHostTensorAllocator",
    ),
)


def _try_import(module_name, class_name):
    try:
        return getattr(importlib.import_module(module_name), class_name)
    except Exception:
        return None


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
    def _fake_pool(root, page_kb, free_count, reserved_count=0):
        pool_dir = os.path.join(root, f"hugepages-{page_kb}kB")
        os.makedirs(pool_dir)
        with open(os.path.join(pool_dir, "free_hugepages"), "w") as f:
            f.write(f"{free_count}\n")
        with open(os.path.join(pool_dir, "resv_hugepages"), "w") as f:
            f.write(f"{reserved_count}\n")

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

    def test_subtracts_reserved_pages(self):
        # free_hugepages counts pages already promised to another mapping but
        # not yet faulted; handing them out again would pass the check and then
        # fail the mmap.
        with tempfile.TemporaryDirectory() as root:
            self._fake_pool(root, 2048, 1024, reserved_count=1000)
            self.assertEqual(self._read("2MB", root), 24 * 2 * 1024 * 1024)

    def test_never_reports_negative_capacity(self):
        with tempfile.TemporaryDirectory() as root:
            self._fake_pool(root, 2048, 4, reserved_count=9)
            self.assertEqual(self._read("2MB", root), 0)

    def test_missing_reservation_counter_is_treated_as_zero(self):
        with tempfile.TemporaryDirectory() as root:
            self._fake_pool(root, 2048, 8)
            os.remove(os.path.join(root, "hugepages-2048kB", "resv_hugepages"))
            self.assertEqual(self._read("2MB", root), 8 * 2 * 1024 * 1024)

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

    def test_explicit_page_size_overrides_the_env(self):
        # Callers that map hugepages themselves pick their own pool.
        with tempfile.TemporaryDirectory() as root:
            self._fake_pool(root, 1048576, 2)
            with envs.SGLANG_HUGEPAGE_SIZE.override(None):
                with patch.object(mmap_allocator, "_HUGEPAGE_SYSFS_DIR", root):
                    self.assertEqual(
                        mmap_allocator.free_hugepage_bytes(1024**3), 2 * 1024**3
                    )


class TestAllocatorHugetlbCapacity(unittest.TestCase):
    """Only allocators that really map MAP_HUGETLB may claim the pool."""

    def _pool_of(self, root, page_kb, free_count):
        TestFreeHugepageBytes._fake_pool(root, page_kb, free_count)
        return patch.object(mmap_allocator, "_HUGEPAGE_SYSFS_DIR", root)

    def test_mmap_allocator_reports_the_pool(self):
        with tempfile.TemporaryDirectory() as root:
            with self._pool_of(root, 2048, 8):
                with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"):
                    self.assertEqual(
                        HostTensorAllocator().free_hugetlb_bytes(), 8 * 2 * 1024 * 1024
                    )

    def test_shm_allocator_reports_zero(self):
        # alloc_shm warns and falls back to plain pages, so crediting the
        # hugetlb pool to it would let an oversized pool past the pre-check.
        with tempfile.TemporaryDirectory() as root:
            with self._pool_of(root, 2048, 8):
                with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"):
                    self.assertEqual(ShmHostTensorAllocator().free_hugetlb_bytes(), 0)

    def test_mooncake_allocator_reports_zero(self):
        # Mooncake allocates through its own host allocator, never alloc_mmap,
        # so the kernel pool is not capacity it can spend.
        cls = _try_import(*_OPTIONAL_ALLOCATORS[0])
        if cls is None:
            self.skipTest("mooncake backend is not importable")
        with tempfile.TemporaryDirectory() as root:
            with self._pool_of(root, 2048, 8):
                with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"):
                    self.assertEqual(cls.free_hugetlb_bytes(None), 0)

    def test_umbp_allocator_reads_its_own_page_size(self):
        # mori maps hugepages, but selects the pool with its own knobs, so a
        # 2MB pool must not be credited to a 1GB-backed allocator.
        cls = _try_import(*_OPTIONAL_ALLOCATORS[1])
        if cls is None:
            self.skipTest("umbp backend is not importable")
        read = cls.free_hugetlb_bytes
        with tempfile.TemporaryDirectory() as root:
            with self._pool_of(root, 2048, 8):
                two_mb = SimpleNamespace(_use_hugepage=True, _hugepage_size=2 << 20)
                one_gb = SimpleNamespace(_use_hugepage=True, _hugepage_size=1024**3)
                disabled = SimpleNamespace(_use_hugepage=False, _hugepage_size=2 << 20)
                self.assertEqual(read(two_mb), 8 * 2 * 1024 * 1024)
                self.assertEqual(read(one_gb), 0)
                self.assertEqual(read(disabled), 0)

    def test_every_allocator_answers_for_its_own_backing(self):
        # Inheriting the mmap-backed answer would credit the hugetlb pool to a
        # backend that never maps MAP_HUGETLB, letting an oversized host pool
        # past the pre-flight check and turning a clear error into an OOM.
        for module_name, class_name in _OPTIONAL_ALLOCATORS:
            _try_import(module_name, class_name)
        pending, subclasses = list(HostTensorAllocator.__subclasses__()), []
        while pending:
            cls = pending.pop()
            subclasses.append(cls)
            pending.extend(cls.__subclasses__())
        self.assertTrue(subclasses, "no allocator subclasses were imported")
        for cls in subclasses:
            if "allocate" in cls.__dict__:
                self.assertIn(
                    "free_hugetlb_bytes",
                    cls.__dict__,
                    f"{cls.__name__} overrides allocate() but inherits "
                    "free_hugetlb_bytes()",
                )


if __name__ == "__main__":
    unittest.main()
