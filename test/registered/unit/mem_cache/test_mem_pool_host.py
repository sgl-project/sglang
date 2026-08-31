"""Unit tests for host-pool allocation and free-list bookkeeping."""

import threading
import unittest
import unittest.mock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    LogicalHostPool,
)
from sglang.srt.mem_cache.pool_host import HostPoolGroup, PoolEntry, base
from sglang.srt.mem_cache.pool_host.mamba import MambaPoolHost
from sglang.srt.mem_cache.pool_host.mha import MHATokenToKVPoolHost
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


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
        fake_mem = unittest.mock.Mock(available=self._AVAILABLE)
        with unittest.mock.patch.object(
            base, "ranks_per_host", return_value=ranks
        ), unittest.mock.patch.object(
            base.psutil, "virtual_memory", return_value=fake_mem
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
        # The launcher slices ranks uniformly across nodes, so the co-located
        # rank count is world_size // nnodes — no hostname collective.
        fake_group = unittest.mock.Mock(world_size=16)
        with get_context().override_server_args(nnodes=2), unittest.mock.patch.object(
            torch.distributed, "is_initialized", return_value=True
        ), unittest.mock.patch.object(base, "get_world_group", return_value=fake_group):
            self.assertEqual(base.ranks_per_host(), 8)


class TestHostPoolGroup(CustomTestCase):
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
        group = self._group(kv=4, swa=2, mamba=1)
        transfers = [
            PoolTransfer(name=PoolName.SWA, device_indices=torch.arange(2)),
            PoolTransfer(name=PoolName.MAMBA, device_indices=torch.arange(2)),
        ]

        self.assertIsNone(group.resolve_host_transfers(transfers))
        self.assertIsNone(transfers[0].host_indices)
        self.assertEqual(group.available_size(PoolName.SWA), 2)


if __name__ == "__main__":
    unittest.main()
