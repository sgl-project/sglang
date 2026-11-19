from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import sys

sys.modules["libtpu"] = None
import ctypes
import ctypes.util
import mmap
import os
import signal
import tempfile
import unittest
import unittest.mock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.pool_host.common import ShmHostTensorAllocator
from sglang.srt.mem_cache.storage.mmap import (
    alloc_mmap,
    alloc_shm,
    mmap_allocator,
    tensor_uses_hugepage,
)
from sglang.srt.mem_cache.storage.mmap.mmap_allocator import (
    _mmap_prefaulted,
    _mmap_private_parallel_prefaulted,
)


class TestMmapAllocator(unittest.TestCase):
    def test_alloc_mmap_hugepage_modes(self):
        with (
            self.subTest(mode="off"),
            envs.SGLANG_HUGEPAGE_MODE.override("off"),
            envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
            unittest.mock.patch.object(mmap_allocator, "_alloc_hugepage") as hugepage,
        ):
            self.assertEqual(mmap_allocator.alloc_mmap((4,), torch.float32).shape, (4,))
            hugepage.assert_not_called()

        with (
            self.subTest(mode="prefer", result="hugetlb"),
            envs.SGLANG_HUGEPAGE_MODE.override("prefer"),
            envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
            unittest.mock.patch.object(
                mmap_allocator,
                "_alloc_hugepage",
                return_value=(ctypes.c_uint8 * 16)(),
            ) as hugepage,
            unittest.mock.patch.object(mmap_allocator, "_mmap_prefaulted") as plain,
        ):
            self.assertEqual(mmap_allocator.alloc_mmap((4,), torch.float32).shape, (4,))
            hugepage.assert_called_once()
            plain.assert_not_called()

        with (
            self.subTest(mode="prefer", result="fallback"),
            envs.SGLANG_HUGEPAGE_MODE.override("prefer"),
            envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
            unittest.mock.patch.object(mmap_allocator, "_libc", object()),
            unittest.mock.patch.object(
                mmap_allocator,
                "_alloc_hugepage",
                side_effect=OSError("no hugepages"),
            ),
            unittest.mock.patch.object(
                mmap_allocator,
                "_mmap_prefaulted",
                wraps=mmap_allocator._mmap_prefaulted,
            ) as plain,
        ):
            self.assertEqual(mmap_allocator.alloc_mmap((4,), torch.float32).shape, (4,))
            plain.assert_called_once()

        with (
            self.subTest(mode="required", result="failure"),
            envs.SGLANG_HUGEPAGE_MODE.override("required"),
            envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
            unittest.mock.patch.object(mmap_allocator, "_libc", object()),
            unittest.mock.patch.object(
                mmap_allocator,
                "_alloc_hugepage",
                side_effect=OSError("no hugepages"),
            ),
            unittest.mock.patch.object(mmap_allocator, "_mmap_prefaulted") as plain,
        ):
            with self.assertRaisesRegex(RuntimeError, "no hugepages"):
                mmap_allocator.alloc_mmap((4,), torch.float32)
            plain.assert_not_called()

        with (
            self.subTest(mode="required", result="missing-size"),
            envs.SGLANG_HUGEPAGE_MODE.override("required"),
            envs.SGLANG_HUGEPAGE_SIZE.override(""),
            self.assertRaisesRegex(ValueError, "SGLANG_HUGEPAGE_SIZE"),
        ):
            mmap_allocator.alloc_mmap((4,), torch.float32)

        with (
            self.subTest(mode="required", result="libc-unavailable"),
            envs.SGLANG_HUGEPAGE_MODE.override("required"),
            envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
            unittest.mock.patch.object(mmap_allocator, "_libc", None),
            self.assertRaisesRegex(RuntimeError, "C library"),
        ):
            mmap_allocator.alloc_mmap((4,), torch.float32)

    def test_alloc_mmap_marks_only_hugepage_mappings(self):
        with (
            envs.SGLANG_HUGEPAGE_MODE.override("off"),
            envs.SGLANG_HUGEPAGE_SIZE.override(""),
        ):
            buf = mmap_allocator.alloc_mmap((4,), torch.float32)
            self.assertFalse(tensor_uses_hugepage(buf))

        with (
            envs.SGLANG_HUGEPAGE_MODE.override("prefer"),
            envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
            unittest.mock.patch.object(
                mmap_allocator,
                "_alloc_hugepage",
                return_value=(ctypes.c_uint8 * 16)(),
            ),
        ):
            buf = mmap_allocator.alloc_mmap((4,), torch.float32)
            self.assertTrue(tensor_uses_hugepage(buf))
        # A freed buffer must not stay marked as hugepage-backed.
        del buf
        self.assertEqual(mmap_allocator._hugepage_storage_keys, set())

        # Prefer-mode fallback to plain pages must not register as hugepage.
        with (
            envs.SGLANG_HUGEPAGE_MODE.override("prefer"),
            envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
            unittest.mock.patch.object(mmap_allocator, "_libc", object()),
            unittest.mock.patch.object(
                mmap_allocator,
                "_alloc_hugepage",
                side_effect=OSError("no hugepages"),
            ),
        ):
            buf = mmap_allocator.alloc_mmap((4,), torch.float32)
            self.assertFalse(tensor_uses_hugepage(buf))

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

    def test_private_parallel_prefaulted_is_resident_and_zeroed(self):
        """Fully resident and zeroed, including a ragged non-THP-aligned tail."""
        alloc_bytes = 9 * 2 * 1024 * 1024 + 3 * mmap.PAGESIZE
        for threads in (1, 4, 64):
            with self.subTest(threads=threads):
                mm = _mmap_private_parallel_prefaulted(alloc_bytes, threads)
                try:
                    self.assertEqual(len(mm), alloc_bytes)
                    self._assert_resident(mm, alloc_bytes)
                    self.assertEqual(mm[:].count(0), alloc_bytes)
                finally:
                    mm.close()

    def test_private_parallel_mapping_is_not_inherited_by_fork(self):
        """MADV_DONTFORK: a forked child touching the buffer must SIGSEGV."""
        alloc_bytes = 4 * mmap.PAGESIZE
        mm = _mmap_private_parallel_prefaulted(alloc_bytes, 2)
        addr = ctypes.addressof(ctypes.c_char.from_buffer(mm))
        try:
            pid = os.fork()
            if pid == 0:  # child: the mapping must not exist here
                ctypes.c_char.from_address(addr).value  # noqa: B018
                os._exit(0)
            _, status = os.waitpid(pid, 0)
            self.assertTrue(os.WIFSIGNALED(status), f"child exited with {status}")
            self.assertEqual(os.WTERMSIG(status), signal.SIGSEGV)
        finally:
            mm.close()

    def test_alloc_mmap_parallel_flag(self):
        from sglang.srt.environ import envs

        dims = (3, 1024 * 1024)
        with envs.SGLANG_HICACHE_HOST_PARALLEL_ALLOC_THREADS.override(4):
            with unittest.mock.patch(
                "sglang.srt.mem_cache.storage.mmap.mmap_allocator._mmap_prefaulted",
                side_effect=AssertionError("shared path must not run"),
            ):
                tensor = alloc_mmap(dims, torch.uint8)
            self.assertEqual(tensor.shape, dims)
            self.assertTrue(bool(tensor.eq(0).all()))
            tensor[2, -1] = 7
            self.assertEqual(int(tensor[2, -1]), 7)

            # A failure on the new path falls back to the shared mapping.
            with unittest.mock.patch(
                "sglang.srt.mem_cache.storage.mmap.mmap_allocator._mmap_private_parallel_prefaulted",
                side_effect=OSError("boom"),
            ):
                tensor = alloc_mmap(dims, torch.uint8)
            self.assertEqual(tensor.shape, dims)

    def test_alloc_mmap_parallel_flag_off_by_default(self):
        """Default off: the shared prefaulted mapping is used unchanged."""
        with unittest.mock.patch(
            "sglang.srt.mem_cache.storage.mmap.mmap_allocator._mmap_private_parallel_prefaulted",
            side_effect=AssertionError("parallel path must not run by default"),
        ):
            tensor = alloc_mmap((2, 1024), torch.uint8)
        self.assertEqual(tensor.shape, (2, 1024))

    def test_alloc_shm_hugepage_warning(self):
        for mode in ("prefer", "required"):
            with (
                self.subTest(mode=mode),
                envs.SGLANG_HUGEPAGE_MODE.override(mode),
                envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
                self.assertLogs(mmap_allocator.logger, "WARNING"),
            ):
                # SHM does not dispatch to alloc_mmap, so the mode is ignored.
                dims = (2, 2)
                tensor, fd, mm = alloc_shm(dims, torch.float32)
                self.assertEqual(tensor.shape, dims)
                mm.close()
                os.close(fd)

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

    def test_hugepage_env_parsing(self):
        for raw, expected in {"": 0, "2MB": 2 * 1024**2, " 1gb ": 1024**3}.items():
            with self.subTest(raw=raw), envs.SGLANG_HUGEPAGE_SIZE.override(raw):
                self.assertEqual(mmap_allocator.hugepage_size_requested(), expected)
        with (
            envs.SGLANG_HUGEPAGE_SIZE.override("4MB"),
            self.assertLogs(mmap_allocator.logger, "WARNING"),
        ):
            self.assertEqual(mmap_allocator.hugepage_size_requested(), 0)

        for raw_mode, size, expected in (
            ("", 0, mmap_allocator.HUGEPAGE_MODE_OFF),
            ("", 2 * 1024**2, mmap_allocator.HUGEPAGE_MODE_PREFER),
            (" Required ", 2 * 1024**2, mmap_allocator.HUGEPAGE_MODE_REQUIRED),
        ):
            with (
                self.subTest(raw_mode=raw_mode, size=size),
                envs.SGLANG_HUGEPAGE_MODE.override(raw_mode),
            ):
                self.assertEqual(mmap_allocator.hugepage_mode(size), expected)

    def test_hugetlb_pool_free_bytes(self):
        # sysfs keeps one pool per page size, and MemAvailable excludes all of
        # them. A mapping reserves its pages at mmap time and faults them in
        # afterwards; until then sysfs still counts them as free, and
        # co-located ranks populate their pools concurrently, so only
        # free - resv can back a new mapping.
        pool_2mb = (6144 - 1000) * 2 * 1024**2
        with tempfile.TemporaryDirectory() as root:
            self._sysfs(root, free_2mb=6144, free_1gb=3, resv_2mb=1000)
            with unittest.mock.patch.object(
                mmap_allocator, "_HUGEPAGE_SYSFS_DIR", root
            ):
                # The requested page size selects its pool, net of reservations.
                with (
                    self.subTest(why="2MB pool"),
                    envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
                ):
                    self.assertEqual(mmap_allocator.hugetlb_pool_free_bytes(), pool_2mb)
                with (
                    self.subTest(why="1GB pool"),
                    envs.SGLANG_HUGEPAGE_SIZE.override("1GB"),
                ):
                    self.assertEqual(
                        mmap_allocator.hugetlb_pool_free_bytes(), 3 * 1024**3
                    )
                # No credit where a new mapping would not take MAP_HUGETLB.
                with (
                    self.subTest(why="not requested"),
                    envs.SGLANG_HUGEPAGE_SIZE.override(""),
                ):
                    self.assertEqual(mmap_allocator.hugetlb_pool_free_bytes(), 0)
                with (
                    self.subTest(why="mode off"),
                    envs.SGLANG_HUGEPAGE_MODE.override("off"),
                    envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
                ):
                    self.assertEqual(mmap_allocator.hugetlb_pool_free_bytes(), 0)
                with (
                    self.subTest(why="libc unavailable"),
                    envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
                    unittest.mock.patch.object(mmap_allocator, "_libc", None),
                ):
                    self.assertEqual(mmap_allocator.hugetlb_pool_free_bytes(), 0)
                # A cgroup limit below the pool bounds the credit.
                for cgroup, expected in [
                    (None, pool_2mb),  # no container limit: the sysfs pool as-is
                    (pool_2mb + 1, pool_2mb),  # room above the pool: the pool bounds
                    (1024**3, 1024**3),  # in-container: the cgroup bounds
                ]:
                    with (
                        self.subTest(why="cgroup bound", cgroup=cgroup),
                        envs.SGLANG_HUGEPAGE_SIZE.override("2MB"),
                        unittest.mock.patch.object(
                            mmap_allocator,
                            "cgroup_hugetlb_headroom_bytes",
                            return_value=cgroup,
                        ),
                    ):
                        self.assertEqual(
                            mmap_allocator.hugetlb_pool_free_bytes(), expected
                        )

        # No pool of the requested size: warn and report zero.
        with tempfile.TemporaryDirectory() as empty_root:
            with (
                self.subTest(why="no pool of that size"),
                unittest.mock.patch.object(
                    mmap_allocator, "_HUGEPAGE_SYSFS_DIR", empty_root
                ),
                envs.SGLANG_HUGEPAGE_SIZE.override("1GB"),
                self.assertLogs(mmap_allocator.logger, "WARNING"),
            ):
                self.assertEqual(mmap_allocator.hugetlb_pool_free_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
