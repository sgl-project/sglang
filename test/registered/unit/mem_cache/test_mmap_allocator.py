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
from unittest.mock import mock_open, patch

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

    def test_missing_reservation_counter_fails_closed(self):
        # Without resv_hugepages there is no way to tell committed pages apart
        # from spendable ones, and crediting the raw free count would admit an
        # allocation the pool has already promised elsewhere.
        with tempfile.TemporaryDirectory() as root:
            self._fake_pool(root, 2048, 8)
            os.remove(os.path.join(root, "hugepages-2048kB", "resv_hugepages"))
            self.assertEqual(self._read("2MB", root), 0)

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


class TestHostKVCacheHugetlbAdmission(unittest.TestCase):
    """The real HostKVCache.__init__ admission decision, not just its helpers.

    Reverting the base.py gate must break something here; the helper tests alone
    do not cover the site where the credit is actually spent.
    """

    GB = 1024**3
    # 1 GiB per token keeps the pool a handful of tokens wide, so clear()
    # allocates tiny bookkeeping tensors while the byte budgets stay realistic.
    BYTES_PER_TOKEN = GB

    def _pool_class(self, device="cuda", single_mapping=True):
        from sglang.srt.mem_cache.pool_host.base import HostKVCache

        outer = self

        class _FakeHostKVCache(HostKVCache):
            layer_num = 1
            target_layer_num = 1

            def get_size_per_token(self):
                return outer.BYTES_PER_TOKEN

            def init_kv_buffer(self):
                outer.allocated = True
                return torch.zeros(1)

            def _uses_single_host_mapping(self):
                return single_mapping

            # Transfer-path abstract methods, unused by the admission check.
            def get_data_page(self, *args, **kwargs):
                raise NotImplementedError

            def get_dummy_flat_data_page(self, *args, **kwargs):
                raise NotImplementedError

            def set_from_flat_data_page(self, *args, **kwargs):
                raise NotImplementedError

            def load_to_device_per_layer(self, *args, **kwargs):
                raise NotImplementedError

            def backup_from_device_all_layer(self, *args, **kwargs):
                raise NotImplementedError

        return _FakeHostKVCache

    def _build(self, requested_bytes, available_bytes, hugetlb_bytes, **kwargs):
        """Construct a pool with fully controlled memory budgets."""
        from sglang.srt.mem_cache.pool_host import base as base_mod

        device = kwargs.pop("device", "cuda")
        cls = self._pool_class(device=device, **kwargs)
        self.allocated = False

        tokens = requested_bytes // self.BYTES_PER_TOKEN
        assert tokens >= 1, "requested_bytes must be a whole number of tokens"
        device_pool = SimpleNamespace(
            store_dtype=torch.uint8,
            # page_size 1 rounds the pool up by exactly one page/token.
            size=tokens - 1,
            start_layer=0,
            end_layer=1,
            layer_num=1,
            layer_shard_enabled=False,
            device=device,
        )
        allocator = SimpleNamespace(free_hugetlb_bytes=lambda: hugetlb_bytes, fd=None)

        # available_bytes = MemAvailable - HICACHE_HOST_MEMORY_RESERVE_BYTES
        vm = SimpleNamespace(
            available=available_bytes + base_mod.HICACHE_HOST_MEMORY_RESERVE_BYTES
        )
        with patch.object(base_mod.psutil, "virtual_memory", return_value=vm), patch(
            "sglang.srt.mem_cache.pool_host.base.get_allocator_from_storage",
            return_value=allocator,
        ):
            return cls(
                device_pool=device_pool,
                host_to_device_ratio=1.0,
                host_size=0,
                page_size=1,
                layout="page_first",
                pin_memory=False,
                device="cpu",
            )

    def test_hugetlb_pool_admits_when_plain_memory_is_short(self):
        # Eligible single-mapping path on a device that dispatches to the
        # allocator: the hugetlb pool is the memory the mmap will really use.
        pool = self._build(
            requested_bytes=8 * self.GB,
            available_bytes=1 * self.GB,
            hugetlb_bytes=64 * self.GB,
        )
        self.assertTrue(self.allocated)
        self.assertGreater(pool.size, 0)

    def test_rejects_and_reports_the_pool_when_both_budgets_are_short(self):
        with self.assertRaises(ValueError) as ctx:
            self._build(
                requested_bytes=8 * self.GB,
                available_bytes=1 * self.GB,
                hugetlb_bytes=2 * self.GB,
            )
        message = str(ctx.exception)
        self.assertIn("Not enough host memory available", message)
        self.assertIn("hugetlb pool", message)

    def test_no_hugetlb_note_when_the_pool_is_empty(self):
        with self.assertRaises(ValueError) as ctx:
            self._build(
                requested_bytes=8 * self.GB,
                available_bytes=1 * self.GB,
                hugetlb_bytes=0,
            )
        self.assertNotIn("hugetlb pool", str(ctx.exception))

    def test_budgets_are_alternatives_not_additive(self):
        # The whole premise: the pool is ONE mapping that either fits the
        # hugetlb pool or falls back to plain pages. Summing the two budgets
        # would admit a request that neither path alone can satisfy.
        with self.assertRaises(ValueError):
            self._build(
                requested_bytes=8 * self.GB,
                available_bytes=5 * self.GB,
                hugetlb_bytes=6 * self.GB,
            )

    def test_admission_is_exclusive_at_the_boundary(self):
        # requested == budget must be admitted, not rejected.
        pool = self._build(
            requested_bytes=8 * self.GB,
            available_bytes=1 * self.GB,
            hugetlb_bytes=8 * self.GB,
        )
        self.assertTrue(self.allocated)
        self.assertGreater(pool.size, 0)

    def test_the_diagnostic_reports_the_hugetlb_figure_not_the_plain_one(self):
        with self.assertRaises(ValueError) as ctx:
            self._build(
                requested_bytes=8 * self.GB,
                available_bytes=1 * self.GB,
                hugetlb_bytes=2 * self.GB,
            )
        message = str(ctx.exception)
        # 2 GB free in the pool, 1 GB of plain memory: the note must not quote
        # the plain figure back to the operator.
        self.assertIn(f"{2 * self.GB / 1e9:.2f} GB free in the hugetlb pool", message)

    def test_npu_rejects_because_it_never_reaches_the_allocator(self):
        # ALLOC_MEMORY_FUNCS maps npu/musa to alloc_with_pin_memory, which calls
        # torch.empty(pin_memory=True) and ignores the selected allocator, so
        # whatever the allocator reports does not describe this allocation.
        for device in ("npu", "musa"):
            with self.subTest(device=device):
                with self.assertRaises(ValueError):
                    self._build(
                        requested_bytes=8 * self.GB,
                        available_bytes=1 * self.GB,
                        hugetlb_bytes=64 * self.GB,
                        device=device,
                    )

    def test_split_mapping_rejects_on_aggregate_logical_bytes(self):
        # Two mappings are each rounded up to a whole hugepage, so the sum of
        # the logical sizes fitting the pool does not mean both mappings fit.
        with self.assertRaises(ValueError):
            self._build(
                requested_bytes=8 * self.GB,
                available_bytes=1 * self.GB,
                hugetlb_bytes=64 * self.GB,
                single_mapping=False,
            )


class TestSplitMappingPoolsAreExcluded(unittest.TestCase):
    """Pools that issue more than one host mapping must declare it."""

    def test_ordinary_single_mapping_pools_keep_the_credit(self):
        # The other half of the gate: the default must stay True, or the fix
        # silently stops helping the very configurations it targets. Asserted
        # on the real class, not on the base or on a test double.
        from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost

        for layout in ("layer_first", "page_first", "page_first_direct", "page_head"):
            self.assertTrue(
                MHATokenToKVPoolHost._uses_single_host_mapping(
                    SimpleNamespace(layout=layout)
                ),
                f"layout={layout!r}",
            )

    def test_the_override_is_on_the_asymmetric_class_itself(self):
        # Calling the method through the MRO would still pass if the override
        # were moved up to the symmetric parent, which would wrongly zero the
        # credit for every ordinary MHA pool.
        from sglang.srt.mem_cache.pool_host.mha import (
            AsymmetricMHATokenToKVPoolHost,
            MHATokenToKVPoolHost,
        )

        self.assertIn(
            "_uses_single_host_mapping", AsymmetricMHATokenToKVPoolHost.__dict__
        )
        self.assertNotIn("_uses_single_host_mapping", MHATokenToKVPoolHost.__dict__)

    def test_multi_mapping_pools_do_not_silently_inherit_the_credit(self):
        # Mirrors test_every_allocator_answers_for_its_own_backing: any pool
        # that reaches the base preflight must answer for its own mapping
        # count rather than inherit the single-mapping default.
        import sglang.srt.mem_cache.memory_pool_host  # noqa: F401  (registers subclasses)
        from sglang.srt.mem_cache.pool_host.base import HostKVCache

        pending, subclasses = list(HostKVCache.__subclasses__()), []
        while pending:
            cls = pending.pop()
            subclasses.append(cls)
            pending.extend(cls.__subclasses__())
        self.assertTrue(subclasses, "no HostKVCache subclasses were imported")
        for cls in subclasses:
            init = cls.__dict__.get("__init__")
            if init is None:
                continue
            names = init.__code__.co_names
            # Pools with their own preflight never reach the base gate.
            if "virtual_memory" in names:
                continue
            self.assertIn(
                "_uses_single_host_mapping",
                {n for klass in cls.__mro__ for n in klass.__dict__},
                f"{cls.__name__} reaches the base preflight",
            )

    def test_asymmetric_mha_is_not_single_mapping(self):
        from sglang.srt.mem_cache.pool_host.mha import AsymmetricMHATokenToKVPoolHost

        self.assertFalse(
            AsymmetricMHATokenToKVPoolHost._uses_single_host_mapping(
                SimpleNamespace(layout="page_first")
            )
        )

    def test_mla_kv_split_layout_is_not_single_mapping(self):
        from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost

        read = MLATokenToKVPoolHost._uses_single_host_mapping
        self.assertFalse(read(SimpleNamespace(layout="page_first_kv_split")))
        self.assertTrue(read(SimpleNamespace(layout="page_first")))


class TestDispatchCapability(unittest.TestCase):
    def test_only_pin_memory_devices_bypass_the_allocator(self):
        from sglang.srt.mem_cache.pool_host.common import device_uses_allocator

        self.assertFalse(device_uses_allocator("npu"))
        self.assertFalse(device_uses_allocator("musa"))
        self.assertTrue(device_uses_allocator("cuda"))
        self.assertTrue(device_uses_allocator("cpu"))

    def test_an_unknown_dispatch_function_gets_no_credit(self):
        # Allowlist, not denylist: a future bypassing dispatch function must
        # not inherit the credit just by not being alloc_with_pin_memory.
        from sglang.srt.mem_cache.pool_host.common import alloc_func_uses_allocator

        def some_future_direct_allocator(*args, **kwargs):
            raise NotImplementedError

        self.assertFalse(alloc_func_uses_allocator(some_future_direct_allocator))


class TestHugepageMmapRequiresLibc(unittest.TestCase):
    def test_default_allocator_reports_zero_without_libc(self):
        # Without libc, alloc_mmap() cannot pass MAP_HUGETLB and silently falls
        # back to ordinary pages, so the pool is not capacity it can spend.
        with tempfile.TemporaryDirectory() as root:
            TestFreeHugepageBytes._fake_pool(root, 2048, 8)
            with patch.object(mmap_allocator, "_HUGEPAGE_SYSFS_DIR", root):
                with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"):
                    with patch.object(mmap_allocator, "_libc", None):
                        self.assertEqual(HostTensorAllocator().free_hugetlb_bytes(), 0)
                    self.assertEqual(
                        HostTensorAllocator().free_hugetlb_bytes(), 8 * 2 * 1024 * 1024
                    )


class TestNumaPolicyGate(unittest.TestCase):
    """A membound process must not be credited pages on nodes it cannot use."""

    def _pool(self, root, free_count=8):
        TestFreeHugepageBytes._fake_pool(root, 2048, free_count)
        return patch.object(mmap_allocator, "_HUGEPAGE_SYSFS_DIR", root)

    def _read(self, root, allowed, with_pages):
        with self._pool(root):
            with envs.SGLANG_HUGEPAGE_SIZE.override("2MB"):
                with patch.object(
                    mmap_allocator, "_mems_allowed_nodes", return_value=allowed
                ), patch.object(
                    mmap_allocator, "_hugepage_numa_nodes", return_value=with_pages
                ):
                    return mmap_allocator.free_hugepage_bytes()

    def test_credits_when_every_hugepage_node_is_allowed(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(
                self._read(root, allowed={0, 1}, with_pages={0}), 8 * 2 * 1024 * 1024
            )

    def test_zero_when_pages_sit_on_a_forbidden_node(self):
        # --membind=0 while the pool lives on node 1: the global counter says
        # the pages are free, but this process can never allocate them.
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(self._read(root, allowed={0}, with_pages={0, 1}), 0)

    def test_zero_when_the_policy_cannot_be_determined(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(self._read(root, allowed=None, with_pages={0}), 0)

    def test_multi_node_without_per_node_counters_is_not_credited(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(self._read(root, allowed={0, 1}, with_pages=None), 0)
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(
                self._read(root, allowed={0}, with_pages=None), 8 * 2 * 1024 * 1024
            )

    def test_mems_allowed_list_parsing(self):
        read = mmap_allocator._mems_allowed_nodes
        for value, expected in (
            ("0", {0}),
            ("0-3", {0, 1, 2, 3}),
            ("0,2", {0, 2}),
            ("0-1,4", {0, 1, 4}),
        ):
            with patch(
                "builtins.open",
                mock_open(read_data=f"Mems_allowed_list:\t{value}\n"),
            ):
                self.assertEqual(read(), expected, f"value={value!r}")


if __name__ == "__main__":
    unittest.main()
