# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Mamba overflow ring on the UnifiedRadixCache path.

Covers the three behaviours added with the overflow ring:
  1. BACKUP_HOST commit stashes overflow ring slots in ComponentData.metadata
     instead of marking the node mamba-backuped (ring rows recycle on archive
     ack and must never serve host->device restores).
  2. BACKUP_STORAGE build re-attaches overflow_slot_ids so the controller's
     archive-completion drain releases the ring slots.
  3. BACKUP_STORAGE commit clears the stash so a later backup is not treated
     as overflow-pending.

Runs CPU-only: no pool, no torch.cuda. The component under test only reads
node/component_data fields, so a minimal stub tree-core is enough.
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.hicache_storage import PoolHitPolicy, PoolName, PoolTransfer
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    CacheTransferPhase,
    ComponentType,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_component():
    from sglang.srt.mem_cache.unified_cache.components.mamba_component import (
        MambaComponent,
    )

    comp = MambaComponent.__new__(MambaComponent)
    comp._mamba_pool_host = None
    return comp


def _make_node(host_value=None, value=None):
    node = MagicMock()
    node.hash_value = ["h0"]
    cd = MagicMock()
    cd.host_value = host_value
    cd.value = value
    cd.metadata = {}
    node.component_data = {ComponentType.MAMBA: cd}
    return node, cd


class TestMambaOverflowBackupHostCommit(unittest.TestCase):
    def test_overflow_slots_stashed_not_marked_backuped(self):
        comp = _make_component()
        node, cd = _make_node()
        tr = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=torch.tensor([9]),
            overflow_slot_ids=[9],
        )
        comp.commit_hicache_transfer(
            node, CacheTransferPhase.BACKUP_HOST, [tr], cache_actions=[]
        )
        self.assertIsNone(cd.host_value)  # must NOT be marked backuped
        self.assertEqual(cd.metadata["_mamba_overflow_slot_ids"], [9])
        self.assertTrue(
            torch.equal(cd.metadata["_mamba_overflow_indices"], torch.tensor([9]))
        )

    def test_normal_commit_sets_host_value(self):
        comp = _make_component()
        node, cd = _make_node()
        tr = PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([3]))
        comp.commit_hicache_transfer(
            node, CacheTransferPhase.BACKUP_HOST, [tr], cache_actions=[]
        )
        self.assertTrue(torch.equal(cd.host_value, torch.tensor([3])))
        self.assertEqual(cd.metadata, {})


class TestMambaOverflowBackupStorage(unittest.TestCase):
    def test_build_reattaches_overflow_slot_ids(self):
        comp = _make_component()
        node, cd = _make_node()
        cd.metadata["_mamba_overflow_indices"] = torch.tensor([7])
        cd.metadata["_mamba_overflow_slot_ids"] = [7]
        transfers = comp.build_hicache_transfers(
            node, CacheTransferPhase.BACKUP_STORAGE
        )
        self.assertEqual(len(transfers), 1)
        self.assertEqual(transfers[0].overflow_slot_ids, [7])
        self.assertEqual(transfers[0].hit_policy, PoolHitPolicy.TRAILING_PAGES)
        self.assertEqual(transfers[0].keys, ["h0"])

    def test_build_normal_host_value_path(self):
        comp = _make_component()
        node, cd = _make_node(host_value=torch.tensor([4]))
        transfers = comp.build_hicache_transfers(
            node, CacheTransferPhase.BACKUP_STORAGE
        )
        self.assertEqual(len(transfers), 1)
        self.assertIsNone(transfers[0].overflow_slot_ids)

    def test_build_none_without_host_or_overflow(self):
        comp = _make_component()
        node, _ = _make_node()
        self.assertIsNone(
            comp.build_hicache_transfers(node, CacheTransferPhase.BACKUP_STORAGE)
        )

    def test_commit_clears_stash(self):
        comp = _make_component()
        # The BACKUP_STORAGE commit now drains stashed overflow slots via the
        # host pool, so the component needs a cache with a mamba pool.
        host_pool = MagicMock()
        host_pool_group = MagicMock()
        host_pool_group.get_pool = MagicMock(return_value=host_pool)
        comp.cache = MagicMock(host_pool_group=host_pool_group)
        node, cd = _make_node()
        cd.metadata["_mamba_overflow_indices"] = torch.tensor([7])
        cd.metadata["_mamba_overflow_slot_ids"] = [7]
        comp.commit_hicache_transfer(
            node, CacheTransferPhase.BACKUP_STORAGE, [], cache_actions=[]
        )
        self.assertEqual(cd.metadata, {})
        host_pool.overflow_release.assert_called_once_with(7)


class TestMambaOverflowBackupStorageReleasesRing(unittest.TestCase):
    """Regression test for the overflow-ring slot leak.

    Before the fix, a successful BACKUP_STORAGE commit dropped the stashed
    ``_mamba_overflow_slot_ids`` on the floor without ever calling
    ``overflow_release`` — so the 64-slot ring filled permanently after 64
    overflow writes and every later companion write was skipped. The commit
    must drain each stashed slot back to the ring.
    """

    def _make_component_with_overflow_ring(self, base_idx=100, size=64):
        from sglang.srt.mem_cache._mamba_overflow_buffer import (
            _MambaOverflowAllocator,
        )

        comp = _make_component()
        ring = _MambaOverflowAllocator(base_idx, size)

        host_pool = MagicMock()
        host_pool.overflow_release = ring.release
        host_pool_group = MagicMock()
        host_pool_group.get_pool = MagicMock(return_value=host_pool)
        cache = MagicMock()
        cache.host_pool_group = host_pool_group
        comp.cache = cache
        return comp, ring, host_pool_group

    def test_backup_storage_commit_releases_stashed_slots(self):
        comp, ring, host_pool_group = self._make_component_with_overflow_ring()
        # Simulate two slots acquired earlier via overflow_alloc: refcount 1.
        slot_a = ring.acquire()
        slot_b = ring.acquire()
        self.assertEqual(ring.stats()["in_use_now"], 2)

        node, cd = _make_node()
        cd.metadata["_mamba_overflow_indices"] = torch.tensor([slot_a, slot_b])
        cd.metadata["_mamba_overflow_slot_ids"] = [slot_a, slot_b]

        comp.commit_hicache_transfer(
            node, CacheTransferPhase.BACKUP_STORAGE, [], cache_actions=[]
        )

        # Both slots drained: refcounts back to 0, acquirable again.
        self.assertEqual(ring.stats()["in_use_now"], 0)
        self.assertEqual(cd.metadata, {})
        host_pool_group.get_pool.assert_called_once_with(PoolName.MAMBA)
        # Ring actually reusable: a fresh acquire succeeds where it would have
        # returned None on a saturated (leaked) ring of size 2... verify via
        # refcount instead: every stashed slot is releasable.
        self.assertEqual(ring._refcounts, [0] * ring._size)

    def test_backup_storage_commit_no_stash_is_noop(self):
        comp, ring, host_pool_group = self._make_component_with_overflow_ring()
        node, cd = _make_node()
        cd.metadata["_mamba_overflow_indices"] = torch.tensor([5])
        # No _mamba_overflow_slot_ids key at all -> release loop must not run.
        comp.commit_hicache_transfer(
            node, CacheTransferPhase.BACKUP_STORAGE, [], cache_actions=[]
        )
        self.assertEqual(cd.metadata, {})
        host_pool_group.get_pool.assert_not_called()
        self.assertEqual(ring.stats()["in_use_now"], 0)

    def test_backup_storage_commit_no_overflow_stash_untouched(self):
        # Node with no overflow metadata at all: commit is a pure no-op,
        # host pool never consulted.
        comp, _, host_pool_group = self._make_component_with_overflow_ring()
        node, cd = _make_node()
        comp.commit_hicache_transfer(
            node, CacheTransferPhase.BACKUP_STORAGE, [], cache_actions=[]
        )
        self.assertEqual(cd.metadata, {})
        host_pool_group.get_pool.assert_not_called()

    def test_backup_storage_commit_pool_without_overflow_support(self):
        # A pool lacking overflow_release (getattr -> None) must not crash.
        comp = _make_component()
        host_pool = object()  # plain object: no overflow_release attribute
        host_pool_group = MagicMock()
        host_pool_group.get_pool = MagicMock(return_value=host_pool)
        cache = MagicMock()
        cache.host_pool_group = host_pool_group
        comp.cache = cache

        node, cd = _make_node()
        cd.metadata["_mamba_overflow_indices"] = torch.tensor([7])
        cd.metadata["_mamba_overflow_slot_ids"] = [7]
        comp.commit_hicache_transfer(
            node, CacheTransferPhase.BACKUP_STORAGE, [], cache_actions=[]
        )
        self.assertEqual(cd.metadata, {})


if __name__ == "__main__":
    unittest.main()
