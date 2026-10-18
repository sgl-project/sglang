"""Opt-in tail-only SWA backups: admission keys and staged rows agree."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.buffer_mode.pipeline import (
    BufferModePipeline,
    _UnifiedBackupIntent,
)
from sglang.srt.mem_cache.buffer_mode.storage_existence_cache import (
    StorageExistenceCache,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    BufferBackupSnapshot,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

SIDECAR = PoolName.DEEPSEEK_V4_C4_INDEXER_STATE


def _pipeline(*, tail_only: bool, window: int = 2, swa_page_size: int = 2):
    cache = MagicMock()
    cache.page_size = 2
    cache.enable_storage = True
    cache.cache_controller.mem_pool_host.entry_map = {
        pool: SimpleNamespace(host_pool=SimpleNamespace(page_size=swa_page_size))
        for pool in (PoolName.SWA, PoolName.AUXILIARY_SWA)
    }
    cache.token_to_kv_pool_allocator.get_kvcache.return_value = SimpleNamespace(
        size=128
    )
    cache.sidecar_pool_specs = [
        SidecarPoolSpec(
            pool_name=SIDECAR,
            indices_from_pool=PoolName.SWA,
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
    ]
    cache.storage_existence_cache = StorageExistenceCache()
    with envs.SGLANG_ENABLE_HICACHE_SWA_TAIL_ONLY_BACKUP.override(tail_only):
        pipeline = BufferModePipeline(
            cache,
            swa_component=SimpleNamespace(
                pool_name=PoolName.SWA, full_window_pages=window
            ),
            auxiliary_swa_component=SimpleNamespace(
                pool_name=PoolName.AUXILIARY_SWA, full_window_pages=window + 1
            ),
            write_backlog_cap=128,
        )
    return pipeline, cache


def _snapshot(hashes):
    return BufferBackupSnapshot(
        node_id=7,
        parent_node_id=3,
        parent_is_root=False,
        parent_last_hash="page-p",
        hash_values=hashes,
        key=RadixKey(array("q", list(range(2 * len(hashes))))),
        prefix_keys=None,
    )


class TestSWATailOnlyBackup(unittest.TestCase):
    def _write(self, pipeline, cache, hashes, rows, pool=PoolName.SWA):
        """Admit a node whose SWA pages are ``hashes`` and launch its SWA
        intent off ``rows``; return the admitted SWA keys and the transfers."""
        snapshot = _snapshot(hashes)
        cache.sidecar_pool_specs = [
            SidecarPoolSpec(
                pool_name=SIDECAR,
                indices_from_pool=pool,
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            )
        ]
        cache.tree_core.buffer_backup_pool_keys.return_value = {
            ComponentType.SWA: {pool: hashes}
        }
        intents = dict(pipeline._write_intents(snapshot))
        self.assertEqual(intents[PoolName.KV], hashes)
        sidecar = PoolTransfer(name=SIDECAR, indices_from_pool=pool)
        cache._build_backup_sidecar.return_value = [sidecar]
        swa = PoolTransfer(name=pool, device_indices=rows)
        pipeline._build_backup_transfers = lambda node_id, *, kv_only=False: (
            torch.arange(6),
            {ComponentType.SWA: [swa]},
            None,
        )
        intent = _UnifiedBackupIntent(snapshot=snapshot, pool=pool, keys=intents[pool])
        device_value, keys_by_pool, transfers = pipeline._launch_transfers(intent)
        self.assertEqual(device_value.numel(), 0)
        self.assertEqual(transfers, [swa, sidecar])
        self.assertEqual(keys_by_pool, {pool: swa.keys, SIDECAR: sidecar.keys})
        self.assertEqual(swa.hit_policy, PoolHitPolicy.TRAILING_PAGES)
        return intents[pool], swa, sidecar

    def test_admission_and_launch_keep_the_node_tail(self):
        for pool, window in ((PoolName.SWA, 2), (PoolName.AUXILIARY_SWA, 3)):
            with self.subTest(pool=pool):
                pipeline, cache = _pipeline(tail_only=True)
                hashes = [f"page-{i}" for i in range(6)]
                admitted, swa, sidecar = self._write(
                    pipeline, cache, hashes, torch.arange(100, 112), pool
                )
                tail = hashes[-window:]
                self.assertEqual((admitted, swa.keys, sidecar.keys), (tail, tail, tail))
                self.assertTrue(
                    torch.equal(swa.device_indices, torch.arange(112 - 2 * window, 112))
                )

    def test_short_node_and_disabled_keep_every_page(self):
        cases = (
            (True, ["page-0"]),
            (True, ["page-0", "page-1"]),
            (False, [f"page-{i}" for i in range(6)]),
        )
        for tail_only, hashes in cases:
            with self.subTest(tail_only=tail_only, pages=len(hashes)):
                pipeline, cache = _pipeline(tail_only=tail_only)
                rows = torch.arange(100, 100 + 2 * len(hashes))
                admitted, swa, sidecar = self._write(pipeline, cache, hashes, rows)
                self.assertEqual((admitted, swa.keys, sidecar.keys), (hashes,) * 3)
                self.assertTrue(torch.equal(swa.device_indices, rows))

    def test_default_off(self):
        self.assertFalse(envs.SGLANG_ENABLE_HICACHE_SWA_TAIL_ONLY_BACKUP.get())
        # Pipelines built around __init__ (test fixtures) keep every page.
        self.assertEqual(
            BufferModePipeline.__new__(BufferModePipeline)._swa_tail_pages(
                PoolName.SWA
            ),
            0,
        )
        pipeline, _ = _pipeline(tail_only=True, window=0)
        self.assertEqual(pipeline._swa_tail_pages(PoolName.SWA), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
