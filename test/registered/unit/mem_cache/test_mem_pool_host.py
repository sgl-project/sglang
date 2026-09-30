"""Unit tests for host-pool allocation and free-list bookkeeping."""

import threading
import unittest
import unittest.mock
from functools import partial
from types import SimpleNamespace

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.memory_pool import MHATokenToKOnlyPool, MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    LogicalHostPool,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry, base, common
from sglang.srt.mem_cache.pool_host.common import ALLOC_MEMORY_FUNCS
from sglang.srt.mem_cache.pool_host.dsa import (
    DSAIndexerPoolHost,
    make_dsa_indexer_pool_decl,
)
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKOnlyPoolHost,
    MHATokenToKVPoolHost,
)
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestHostKVCache(CustomTestCase):
    def setUp(self):
        self.page_size = 2
        # Small device pool is enough to construct the host pool.
        self.device_pool = MHATokenToKVPool(
            size=self.page_size * 2,
            page_size=self.page_size,
            dtype=torch.float16,
            head_num=2,
            head_dim=4,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        self.host_pool = MHATokenToKVPoolHost(
            device_pool=self.device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=self.page_size,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )

    def test_index_k_host_pool_joins_a_host_pool_group(self):
        """Grouping a main KV pool with a K-only index pool raised AttributeError."""
        index_pool = MHATokenToKOnlyPool(
            size=self.page_size * 2,
            page_size=self.page_size,
            dtype=torch.float16,
            head_num=1,
            head_dim=4,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )
        index_host = MHATokenToKOnlyPoolHost(
            index_pool, self.host_pool, "layer_first", pin_memory=False
        )
        group = HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName.KV,
                    host_pool=self.host_pool,
                    device_pool=self.device_pool,
                    layer_mapper=lambda layer_id: layer_id,
                    is_primary_index_anchor=True,
                ),
                PoolEntry(
                    name=PoolName.INDEXER,
                    host_pool=index_host,
                    device_pool=index_pool,
                    layer_mapper=lambda layer_id: layer_id,
                ),
            ]
        )
        self.assertFalse(group.can_use_write_back_jit)

    def test_multiple_attention_rows_per_token(self):
        for rows_per_token in (1, 3):
            device_pool = MHATokenToKVPool(
                size=4,
                page_size=self.page_size,
                dtype=torch.float16,
                head_num=2 * rows_per_token,
                head_dim=4,
                layer_num=2,
                device="cpu",
                enable_memory_saver=False,
            )
            # Report logical heads while retaining the wider physical rows.
            device_pool.head_num = 2
            for layout in (
                "layer_first",
                "page_first",
                "page_first_direct",
                "page_head",
            ):
                with self.subTest(rows_per_token=rows_per_token, layout=layout):
                    host_pool = MHATokenToKVPoolHost(
                        device_pool=device_pool,
                        host_to_device_ratio=2.0,
                        host_size=0,
                        page_size=self.page_size,
                        layout=layout,
                        pin_memory=False,
                    )
                    device_row = device_pool.k_buffer[0][0]
                    row_bytes = device_row.numel() * device_row.element_size()
                    self.assertEqual(host_pool.element_dim, device_row.numel())
                    self.assertEqual(host_pool.token_stride_size, row_bytes)
                    self.assertEqual(
                        host_pool.size_per_token, 2 * device_pool.layer_num * row_bytes
                    )
                    self.assertEqual(
                        host_pool.kv_buffer.nbytes,
                        host_pool.size * host_pool.size_per_token,
                    )

    def test_double_alloc(self):
        indices = self.host_pool.alloc(4)
        self.assertEqual(len(indices), 4)
        # Mimic bookkeeping corruption: push an already-used slot back to the
        # head of free_slots so the next alloc would hand out an in-use slot.
        leak = torch.tensor([int(indices[0])])
        self.host_pool.free_slots = torch.cat([leak, self.host_pool.free_slots])
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.alloc(4)
        msg = str(ctx.exception)
        self.assertIn("Double-alloc", msg)
        self.assertIn(f"[{int(leak[0])}]", msg)

    def test_double_free(self):
        indices = self.host_pool.alloc(4)
        self.assertEqual(len(indices), 4)
        self.host_pool.free(indices[:2])
        # indices[1] is double freed.
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.free(indices[1:])
        msg = str(ctx.exception)
        self.assertIn("Double-free", msg)
        self.assertIn(f"[{int(indices[1])}]", msg)

    def test_free_unallocated(self):
        indices = torch.tensor([1])
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.free(indices)
        msg = str(ctx.exception)
        self.assertIn("Double-free", msg)
        self.assertIn(f"[{int(indices[0])}]", msg)

    def test_free_after_clear(self):
        indices = self.host_pool.alloc(4)
        self.host_pool.clear()
        with self.assertRaises(AssertionError) as ctx:
            self.host_pool.free(indices)
        msg = str(ctx.exception)
        self.assertIn("Double-free", msg)
        self.assertIn(str(indices.tolist()), msg)

    def test_shm_allocator(self):
        shm_host_pool = MHATokenToKVPoolHost(
            device_pool=self.device_pool,
            host_to_device_ratio=2.0,
            host_size=0,
            page_size=self.page_size,
            layout="layer_first",
            pin_memory=False,
            device="cpu",
            allocator_type="shm",
        )
        self.assertIsNotNone(shm_host_pool.fd)
        self.assertGreaterEqual(shm_host_pool.fd, 0)

        indices = shm_host_pool.alloc(4)
        self.assertEqual(len(indices), 4)
        shm_host_pool.free(indices)

    def test_empty_free_keeps_release_list_empty(self):
        self.assertEqual(self.host_pool.free(torch.empty(0, dtype=torch.int64)), 0)
        self.assertEqual(self.host_pool.num_release_slots, 0)
        self.assertEqual(self.host_pool.release_slots, [])


class TestLazyHostPoolRelease(CustomTestCase):
    @staticmethod
    def _make_mamba_pool():
        pool = MambaPoolHost.__new__(MambaPoolHost)
        pool.size = 8
        pool.page_size = 1
        pool.device = "cpu"
        pool.lock = threading.RLock()
        pool.clear()
        return pool

    @staticmethod
    def _make_deepseek_v4_pool():
        pool = DeepSeekV4PagedHostPool.__new__(DeepSeekV4PagedHostPool)
        pool.size = 8
        pool.slot_page_size = 2
        pool.lock = threading.RLock()
        pool.clear()
        return pool

    @staticmethod
    def _make_logical_pool():
        return LogicalHostPool(size=8, page_size=2)

    @staticmethod
    def _make_transfer_pool(*, page_aligned_only):
        pool = DeepSeekV4PagedHostPool.__new__(DeepSeekV4PagedHostPool)
        pool.pool_name = str(PoolName.DEEPSEEK_V4_C4_INDEXER)
        pool.slot_page_size = 4
        pool.layer_num = 1
        pool.page_aligned_only = page_aligned_only
        pool.device_ptrs = [0]
        pool.data_ptrs = [0]
        return pool

    def _assert_lazy_release(self, pool):
        self.assertEqual(pool.free(torch.empty(0, dtype=torch.int64)), 0)
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])

        allocated = pool.alloc(6)
        free_slots_before = pool.free_slots

        pool.free(allocated[:2])

        # free() should keep the primary free-list untouched and only record
        # the released chunk for a later merge.
        self.assertIs(pool.free_slots, free_slots_before)
        self.assertEqual(pool.num_release_slots, 2)
        self.assertEqual(len(pool.release_slots), 1)
        self.assertEqual(pool.available_size(), 4)

        # Consume the primary free-list first without merging pending slots.
        self.assertTrue(torch.equal(pool.alloc(2), torch.tensor([6, 7])))
        self.assertEqual(pool.num_release_slots, 2)

        # Once the primary free-list is exhausted, alloc() merges and reuses
        # the pending slots.
        self.assertTrue(torch.equal(pool.alloc(2), torch.tensor([0, 1])))
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])
        self.assertEqual(pool.available_size(), 0)

        pool.free(torch.tensor([0, 1]))
        pool.clear()
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])
        self.assertEqual(pool.available_size(), 8)

        # Exercise the general merge path with multiple released chunks.
        allocated = pool.alloc(8)
        pool.free(allocated[:2])
        pool.free(allocated[2:4])
        self.assertEqual(len(pool.release_slots), 2)
        self.assertTrue(torch.equal(pool.alloc(4), torch.tensor([0, 1, 2, 3])))
        self.assertEqual(pool.num_release_slots, 0)
        self.assertEqual(pool.release_slots, [])

    def test_mamba_pool_lazy_release(self):
        self._assert_lazy_release(self._make_mamba_pool())

    def test_deepseek_v4_pool_lazy_release(self):
        pool = self._make_deepseek_v4_pool()
        self._assert_lazy_release(pool)

        # Preserve the pool's page-aligned allocation behavior.
        pool.clear()
        self.assertEqual(len(pool.alloc(1)), 2)

    def test_grouped_page_rows_reject_unaligned_transfers(self):
        # FP4 indexer rows group their slots, so a partial page has no
        # well-defined token-granular copy and must not silently fall back.
        pool = self._make_transfer_pool(page_aligned_only=True)
        unaligned = torch.arange(3, dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "page-aligned"):
            pool.backup_from_device_all_layer(None, unaligned, unaligned, "direct")
        with self.assertRaisesRegex(ValueError, "page-aligned"):
            pool.load_to_device_per_layer(None, unaligned, unaligned, 0, "direct")

    def test_fused_page_rows_keep_token_granular_transfers(self):
        pool = self._make_transfer_pool(page_aligned_only=False)
        unaligned = torch.arange(3, dtype=torch.int64)
        with unittest.mock.patch(
            "sglang.srt.mem_cache.memory_pool_host.transfer_cache_dsv4_mla"
        ) as transfer:
            pool.backup_from_device_all_layer(None, unaligned, unaligned, "direct")
            pool.load_to_device_per_layer(None, unaligned, unaligned, 0, "direct")
        self.assertEqual(transfer.call_count, 2)

    def test_logical_pool_lazy_release(self):
        pool = self._make_logical_pool()
        self._assert_lazy_release(pool)

        # Preserve the logical pool's strict page-alignment checks.
        pool.clear()
        with self.assertRaises(ValueError):
            pool.alloc(1)
        with self.assertRaises(ValueError):
            pool.free(torch.tensor([0]))


class TestHostMemoryBudget(CustomTestCase):
    # Pinned so the two budget reads below see identical free memory; the real
    # psutil value drifts between calls and would flake the equality checks.
    _AVAILABLE = base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 64 * (1024**3)

    def _budget_with_ranks(self, ranks):
        # Deliberate single-accessor stub: isolates the budget math from the
        # topology derivation, which the ranks_per_host case below covers.
        with (
            unittest.mock.patch.object(base, "ranks_per_host", return_value=ranks),
            unittest.mock.patch.object(
                base, "available_host_memory_bytes", return_value=self._AVAILABLE
            ),
            envs.SGLANG_HUGEPAGE_MODE.override("off"),
        ):
            return base.host_memory_budget_bytes()

    def test_budget_is_split_across_co_located_ranks(self):
        solo = self._budget_with_ranks(1)
        self.assertEqual(self._budget_with_ranks(4), solo // 4)

    def test_reserve_is_taken_before_the_split(self):
        # Each rank must not get its own copy of the reserve.
        budget = self._budget_with_ranks(8)
        self.assertLessEqual(
            budget * 8, self._AVAILABLE - base.HICACHE_HOST_MEMORY_RESERVE_BYTES
        )

    def test_ranks_per_host_divides_world_size_by_nodes(self):
        with (
            get_context().override_server_args(nnodes=2, tp_size=16),
            unittest.mock.patch.object(
                torch.distributed, "is_initialized", return_value=True
            ),
        ):
            self.assertEqual(base.ranks_per_host(), 8)

    def _budget_for(
        self, allocator, device, available, ranks=8, mode="prefer", size="2MB"
    ):
        with (
            unittest.mock.patch.object(base, "ranks_per_host", return_value=ranks),
            unittest.mock.patch.object(
                base, "available_host_memory_bytes", return_value=available
            ),
            envs.SGLANG_HUGEPAGE_MODE.override(mode),
            envs.SGLANG_HUGEPAGE_SIZE.override(size),
        ):
            return base.host_memory_budget_bytes(allocator=allocator, device=device)

    def test_hugetlb_pool_is_an_alternative_budget(self):
        # One mapping is served entirely by the hugetlb pool or entirely by
        # plain pages, so the larger of the two is the budget; the reserve is
        # for the OS and applies to plain RAM only.
        gib = 1024**3
        reserve = base.HICACHE_HOST_MEMORY_RESERVE_BYTES
        allocator = unittest.mock.Mock(
            free_hugetlb_bytes=unittest.mock.Mock(return_value=96 * gib),
            supports_hugetlb=unittest.mock.Mock(return_value=True),
        )
        budget = self._budget_for(allocator, "cuda", available=reserve + 64 * gib)
        self.assertEqual(budget, 96 * gib // 8)

    def test_plain_pages_win_when_the_hugetlb_pool_is_smaller(self):
        gib = 1024**3
        reserve = base.HICACHE_HOST_MEMORY_RESERVE_BYTES
        allocator = unittest.mock.Mock(
            free_hugetlb_bytes=unittest.mock.Mock(return_value=16 * gib),
            supports_hugetlb=unittest.mock.Mock(return_value=True),
        )
        budget = self._budget_for(allocator, "cuda", available=reserve + 64 * gib)
        self.assertEqual(budget, 64 * gib // 8)

    def test_no_hugetlb_credit_for_pin_memory_devices(self):
        # npu/musa allocate with torch.empty(pin_memory=True) and never see the
        # allocator, so what it could map from hugetlb does not apply.
        gib = 1024**3
        reserve = base.HICACHE_HOST_MEMORY_RESERVE_BYTES
        allocator = unittest.mock.Mock(
            free_hugetlb_bytes=unittest.mock.Mock(return_value=96 * gib),
            supports_hugetlb=unittest.mock.Mock(return_value=True),
        )
        for device in ("npu", torch.device("xpu")):
            with self.subTest(device=repr(device)):
                budget = self._budget_for(
                    allocator, device, available=reserve + 64 * gib
                )
                self.assertEqual(budget, 64 * gib // 8)
                allocator.free_hugetlb_bytes.assert_not_called()

    def test_request_rounds_each_mapping_only_on_the_hugetlb_path(self):
        # Two 1.1 GiB mappings reserve 4 pages of a 1 GiB-page pool, not the
        # 3 pages their plain sum suggests: each MAP_HUGETLB mapping rounds up
        # on its own.
        gib = 1024**3
        mappings = [int(1.1 * gib), int(1.1 * gib)]
        allocator = unittest.mock.Mock(
            supports_hugetlb=unittest.mock.Mock(return_value=True)
        )
        with (
            envs.SGLANG_HUGEPAGE_MODE.override("prefer"),
            envs.SGLANG_HUGEPAGE_SIZE.override("1GB"),
        ):
            self.assertEqual(
                base.host_memory_requested_bytes(mappings, allocator, "cuda"),
                4 * gib,
            )
            # npu pins through torch and never maps MAP_HUGETLB: plain sum.
            self.assertEqual(
                base.host_memory_requested_bytes(mappings, allocator, "npu"),
                2 * int(1.1 * gib),
            )
        # Without SGLANG_HUGEPAGE_SIZE there is no page size to round to.
        with envs.SGLANG_HUGEPAGE_SIZE.override(""):
            self.assertEqual(
                base.host_memory_requested_bytes(mappings, allocator, "cuda"),
                2 * int(1.1 * gib),
            )
        # mode=off opts out of rounding even with a page size set.
        with (
            envs.SGLANG_HUGEPAGE_MODE.override("off"),
            envs.SGLANG_HUGEPAGE_SIZE.override("1GB"),
        ):
            self.assertEqual(
                base.host_memory_requested_bytes(mappings, allocator, "cuda"),
                2 * int(1.1 * gib),
            )

    def test_plain_budget_is_reported_as_is_without_a_hugetlb_pool(self):
        # A host below the reserve keeps its negative budget, and the failure
        # message that shows it, when there is no hugetlb pool to credit.
        gib = 1024**3
        allocator = unittest.mock.Mock(
            free_hugetlb_bytes=unittest.mock.Mock(return_value=0),
            supports_hugetlb=unittest.mock.Mock(return_value=True),
        )
        budget = self._budget_for(allocator, "cuda", available=4 * gib)
        self.assertEqual(
            budget, (4 * gib - base.HICACHE_HOST_MEMORY_RESERVE_BYTES) // 8
        )

    def test_off_mode_does_not_query_or_credit_hugetlb(self):
        gib = 1024**3
        reserve = base.HICACHE_HOST_MEMORY_RESERVE_BYTES
        allocator = unittest.mock.Mock(
            free_hugetlb_bytes=unittest.mock.Mock(return_value=96 * gib),
            supports_hugetlb=unittest.mock.Mock(return_value=True),
        )
        budget = self._budget_for(
            allocator, "cuda", available=reserve + 64 * gib, mode="off"
        )
        self.assertEqual(budget, 64 * gib // 8)
        allocator.free_hugetlb_bytes.assert_not_called()

    def test_required_mode_uses_only_hugetlb(self):
        gib = 1024**3
        allocator = unittest.mock.Mock(
            free_hugetlb_bytes=unittest.mock.Mock(return_value=24 * gib),
            supports_hugetlb=unittest.mock.Mock(return_value=True),
        )
        budget = self._budget_for(allocator, "cuda", available=1 << 50, mode="required")
        self.assertEqual(budget, 24 * gib // 8)

    def test_required_mode_is_ignored_by_unsupported_allocation_paths(self):
        gib = 1024**3
        reserve = base.HICACHE_HOST_MEMORY_RESERVE_BYTES
        allocator = unittest.mock.Mock(
            free_hugetlb_bytes=unittest.mock.Mock(return_value=96 * gib),
            supports_hugetlb=unittest.mock.Mock(return_value=False),
        )
        with unittest.mock.patch.object(base, "hugepage_size_requested") as parser:
            budget = self._budget_for(
                allocator, "cuda", available=reserve + 64 * gib, mode="required"
            )
            self.assertEqual(budget, 64 * gib // 8)
            parser.assert_not_called()
        allocator.free_hugetlb_bytes.assert_not_called()

    def test_required_mode_requires_a_hugepage_size(self):
        allocator = unittest.mock.Mock(
            supports_hugetlb=unittest.mock.Mock(return_value=True)
        )
        with self.assertRaisesRegex(ValueError, "SGLANG_HUGEPAGE_SIZE"):
            self._budget_for(
                allocator,
                "cuda",
                available=1 << 50,
                mode="required",
                size="",
            )

    def test_guard_passes_its_allocator_and_device(self):
        device_pool = MHATokenToKVPool(
            size=4,
            page_size=2,
            dtype=torch.float16,
            head_num=2,
            head_dim=4,
            layer_num=2,
            device="cpu",
            enable_memory_saver=False,
        )

        def plain_alloc(dims, dtype, device, pin_memory, allocator, **kwargs):
            return torch.empty(dims, dtype=dtype)

        with (
            unittest.mock.patch.object(
                base, "host_memory_budget_bytes", return_value=1024**3
            ) as budget,
            unittest.mock.patch.dict(ALLOC_MEMORY_FUNCS, {"cpu": plain_alloc}),
        ):
            pool = MHATokenToKVPoolHost(
                device_pool=device_pool,
                host_to_device_ratio=2.0,
                host_size=0,
                page_size=2,
                layout="layer_first",
                pin_memory=False,
                device="cpu",
                allocator_type="default",
            )
        budget.assert_called_once_with(
            pool.size * pool.size_per_token, pool.allocator, device_pool.device
        )


class TestHostTensorAllocatorHugetlb(CustomTestCase):
    def test_only_the_mmap_allocator_reports_the_hugetlb_pool(self):
        # Only the base allocate() maps MAP_HUGETLB; an allocator that gets its
        # memory elsewhere must not be credited with a pool it never touches.
        gib = 1024**3

        class Elsewhere(common.HostTensorAllocator):
            def allocate(self, dims, dtype, device):
                raise NotImplementedError

        with unittest.mock.patch.object(
            common, "hugetlb_pool_free_bytes", return_value=4 * gib
        ):
            self.assertEqual(common.HostTensorAllocator().free_hugetlb_bytes(), 4 * gib)
            self.assertEqual(common.ShmHostTensorAllocator().free_hugetlb_bytes(), 0)
            self.assertEqual(Elsewhere().free_hugetlb_bytes(), 0)


class TestHostPoolGroup(CustomTestCase):
    @staticmethod
    def _backup_under_host_pressure(
        order: tuple[PoolName, ...],
    ) -> tuple[list[str], set[str], dict[PoolName, list[int]]]:
        pools = {
            name: LogicalHostPool(2, page_size=1)
            for name in (PoolName.SWA, PoolName.MAMBA)
        }
        leaves: dict[str, dict[PoolName, torch.Tensor]] = {"a": {}, "b": {}}
        for slots in leaves.values():
            for name, pool in pools.items():
                indices = pool.alloc(1)
                assert indices is not None
                slots[name] = indices

        # Independent component evictions can give SWA and Mamba opposite host LRUs.
        lru = {PoolName.SWA: ["a", "b"], PoolName.MAMBA: ["b", "a"]}
        victims: list[str] = []

        def evict(name: PoolName, size: int) -> None:
            for leaf in lru[name]:
                if pools[name].available_size() >= size:
                    break
                if leaf in leaves:
                    victims.append(leaf)
                    # A host-leaf eviction releases every component, not just name.
                    for pool_name, indices in leaves.pop(leaf).items():
                        pools[pool_name].free(indices)

        group = HostPoolGroup(
            [
                PoolEntry(
                    name=name,
                    host_pool=pool,
                    device_pool=None,
                    layer_mapper=lambda layer: layer,
                    host_evict_fn=partial(evict, name),
                )
                for name, pool in pools.items()
            ]
        )
        transfers = [
            PoolTransfer(name=name, device_indices=torch.tensor([0])) for name in order
        ]
        assert group.resolve_host_transfers(transfers) is transfers
        assert tuple(transfer.name for transfer in transfers) == order
        assert len(victims) == 1
        assert all(pool.available_size() == 0 for pool in pools.values())

        allocations = {}
        for transfer in transfers:
            assert transfer.host_indices is not None
            allocations[transfer.name] = transfer.host_indices.tolist()
        return victims, set(leaves), allocations

    def test_host_reclamation_is_independent_of_transfer_order(self):
        # Rust HashMap iteration can deliver these two orders to different TP ranks.
        swa_first = self._backup_under_host_pressure((PoolName.SWA, PoolName.MAMBA))
        mamba_first = self._backup_under_host_pressure((PoolName.MAMBA, PoolName.SWA))
        self.assertEqual(swa_first, mamba_first)

    @staticmethod
    def _group(**sizes):
        return HostPoolGroup(
            [
                PoolEntry(
                    name=PoolName(name),
                    host_pool=LogicalHostPool(size=size, page_size=1),
                    device_pool=None,
                    layer_mapper=lambda layer_id: layer_id,
                    is_primary_index_anchor=name == PoolName.KV.value,
                )
                for name, size in sizes.items()
            ]
        )

    def test_resolve_and_release_multi_pool_allocation(self):
        group = self._group(kv=4, swa=2)
        primary = group.alloc(2)
        transfers = [
            PoolTransfer(name=PoolName.SWA, device_indices=torch.arange(2)),
            PoolTransfer(name=PoolName.INDEXER, indices_from_pool=PoolName.SWA),
        ]

        self.assertIsNotNone(
            group.resolve_host_transfers(
                transfers,
                primary_device_indices=torch.arange(2),
                primary_host_indices=primary,
            )
        )
        self.assertIs(transfers[1].host_indices, transfers[0].host_indices)
        group.free(primary)
        group.release_transfers(transfers)
        self.assertEqual(group.available_size(), 4)
        self.assertEqual(group.available_size(PoolName.SWA), 2)

    def test_resolve_rolls_back_partial_allocation(self):
        group = self._group(kv=4, swa=1, mamba=2)
        transfers = [
            PoolTransfer(name=PoolName.SWA, device_indices=torch.arange(2)),
            PoolTransfer(name=PoolName.MAMBA, device_indices=torch.arange(2)),
        ]

        self.assertIsNone(group.resolve_host_transfers(transfers))
        self.assertIsNone(transfers[0].host_indices)
        self.assertIsNone(transfers[1].host_indices)
        self.assertEqual(group.available_size(PoolName.SWA), 1)
        self.assertEqual(group.available_size(PoolName.MAMBA), 2)


class TestDSAIndexerPoolDecl(CustomTestCase):
    """The declaration is the single source of indexer host bytes. The mirror
    must not re-derive them."""

    def _stub(self):
        return SimpleNamespace(
            layer_num=5,
            layer_shard_enabled=False,
            store_dtype=torch.bfloat16,
            size=64 * 8,
            start_layer=0,
            end_layer=5,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
            quant_block_size=128,
            page_size=64,
            index_page_size=64,
            skip_topk_layers=[False] * 5,
        )

    def test_host_bytes_match_observed_allocation(self):
        # GLM-5.2 DSA, page 64, 5 layers, host 18192320 tokens: the server
        # allocated 12006973440 bytes (12.01 GB) for the indexer mirror.
        storage_info = make_dsa_indexer_pool_decl(self._stub()).storage_info
        self.assertEqual(storage_info.bytes_per_token_per_layer, 132)
        self.assertEqual(storage_info.page_bytes(64), 8448)
        self.assertEqual(
            storage_info.host_bytes(page_num=284256, layer_num=5, page_size=64),
            12006973440,
        )

    def test_mirror_consumes_decl(self):
        stub = self._stub()
        decl = make_dsa_indexer_pool_decl(stub)
        storage_info = decl.storage_info
        anchor = MLATokenToKVPoolHost(
            stub,
            host_to_device_ratio=2,
            host_size=0,
            page_size=64,
            layout="page_first",
            pin_memory=False,
            is_dummy=True,
        )
        mirror = DSAIndexerPoolHost(
            decl=decl,
            anchor_host=anchor,
            pin_memory=False,
            is_dummy=True,
        )
        self.assertEqual(mirror.layout, anchor.layout)
        self.assertEqual(mirror.indexer_page_stride_size, storage_info.page_bytes(64))
        self.assertEqual(
            mirror.get_size_per_token(), storage_info.bytes_per_token_per_layer * 5
        )
        self.assertEqual(
            storage_info.host_bytes(
                page_num=anchor.page_num, layer_num=5, page_size=64
            ),
            anchor.page_num * mirror.indexer_layout_dim,
        )

    def test_mirror_is_compact_over_layers_that_own_index_buffers(self):
        """Shared-topk layers have 0-row device buffers. Mirroring or
        transferring them dereferences a null pointer. The mirror must cover
        only the declared layers and translate packed-draft layer ids
        relative to that compact count."""
        stub = self._stub()
        stub.skip_topk_layers = [False, True, True, False, True]
        decl = make_dsa_indexer_pool_decl(stub)
        self.assertEqual(decl.owned_device_layers, (0, 3))
        anchor = MLATokenToKVPoolHost(
            stub,
            host_to_device_ratio=2,
            host_size=0,
            page_size=64,
            layout="page_first",
            pin_memory=False,
            is_dummy=True,
        )
        mirror = DSAIndexerPoolHost(
            decl=decl,
            anchor_host=anchor,
            pin_memory=False,
            is_dummy=True,
        )
        self.assertEqual(mirror.layer_num, 2)
        self.assertEqual(
            mirror.get_size_per_token(), decl.storage_info.bytes_per_token_per_layer * 2
        )
        self.assertEqual(mirror._owned_device_layer_ids(stub), [0, 3])
        self.assertEqual(mirror._host_layer_index(3), 1)
        self.assertFalse(mirror._is_device_layer_owned(stub, 1))
        # packed draft depth 0 arrives as device layer_num + 0 and lands after the live layers
        self.assertEqual(mirror._draft_host_layer(stub.layer_num), 2)


if __name__ == "__main__":
    unittest.main()
