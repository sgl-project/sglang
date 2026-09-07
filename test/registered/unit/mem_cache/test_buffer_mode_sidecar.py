"""Unit coverage for sidecar pools in HiCache buffer-only mode."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.buffer_mode.pipeline import (
    BufferModePipeline,
    _UnifiedBackupIntent,
    validate_buffer_only_stack,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    BufferBackupSnapshot,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestBufferModeSidecar(unittest.TestCase):
    @staticmethod
    def _swa_component():
        return SimpleNamespace(
            full_window_pages=2,
            _swa_kv_pool_host=SimpleNamespace(page_size=2, size=8),
        )

    @staticmethod
    def _dsv4_specs():
        return [
            SidecarPoolSpec(
                pool_name=PoolName.DEEPSEEK_V4_C4,
                indices_from_pool=PoolName.KV,
            ),
            SidecarPoolSpec(
                pool_name=PoolName.DEEPSEEK_V4_C4_INDEXER,
                indices_from_pool=PoolName.KV,
            ),
            SidecarPoolSpec(
                pool_name=PoolName.DEEPSEEK_V4_C128,
                indices_from_pool=PoolName.KV,
            ),
            SidecarPoolSpec(
                pool_name=PoolName.DEEPSEEK_V4_C4_STATE,
                indices_from_pool=PoolName.SWA,
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
            SidecarPoolSpec(
                pool_name=PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
                indices_from_pool=PoolName.SWA,
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
            SidecarPoolSpec(
                pool_name=PoolName.DEEPSEEK_V4_C128_STATE,
                indices_from_pool=PoolName.SWA,
                hit_policy=PoolHitPolicy.TRAILING_PAGES,
            ),
        ]

    @classmethod
    def _pool_group(
        cls,
        kv_size: int,
        swa_size: int,
        *,
        override_size: dict[PoolName, int] | None = None,
    ):
        sizes = {PoolName.KV: kv_size, PoolName.SWA: swa_size}
        for spec in cls._dsv4_specs():
            sizes[spec.pool_name] = sizes[spec.indices_from_pool]
        sizes.update(override_size or {})
        return SimpleNamespace(
            entry_map={
                name: SimpleNamespace(host_pool=SimpleNamespace(logical_size=size))
                for name, size in sizes.items()
            }
        )

    def test_stack_accepts_dsv4_full_and_swa_sidecars(self):
        validate_buffer_only_stack(
            sidecar_pool_specs=self._dsv4_specs(),
            host_pool_group=self._pool_group(kv_size=16, swa_size=8),
            swa_component=self._swa_component(),
        )

    def test_stack_rejects_sidecar_smaller_than_source(self):
        with self.assertRaisesRegex(ValueError, "smaller than its index source"):
            validate_buffer_only_stack(
                sidecar_pool_specs=[
                    SidecarPoolSpec(
                        pool_name=PoolName.DEEPSEEK_V4_C4_INDEXER_STATE,
                        indices_from_pool=PoolName.SWA,
                        hit_policy=PoolHitPolicy.TRAILING_PAGES,
                    )
                ],
                host_pool_group=self._pool_group(
                    kv_size=16,
                    swa_size=8,
                    override_size={PoolName.DEEPSEEK_V4_C4_INDEXER_STATE: 4},
                ),
                swa_component=self._swa_component(),
            )

    def test_write_stages_and_persists_dsv4_full_and_swa_sidecars(self):
        page_size = 2
        device_indices = torch.arange(4, dtype=torch.int64)
        host_indices = torch.arange(10, 14, dtype=torch.int64)
        swa_device_indices = torch.arange(30, 32, dtype=torch.int64)
        swa_host_indices = torch.arange(20, 22, dtype=torch.int64)
        swa = PoolTransfer(
            name=PoolName.SWA,
            device_indices=swa_device_indices,
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        sidecars = [
            PoolTransfer(
                name=spec.pool_name,
                hit_policy=spec.hit_policy,
                indices_from_pool=spec.indices_from_pool,
            )
            for spec in self._dsv4_specs()
        ]

        controller = MagicMock()
        controller.mem_pool_host.entry_map = {
            PoolName.SWA: SimpleNamespace(
                host_pool=SimpleNamespace(page_size=page_size)
            )
        }

        def _write(device_value, *, node_id, extra_pools):
            self.assertEqual(node_id, 7)
            self.assertEqual(
                [transfer.name for transfer in extra_pools],
                [PoolName.SWA, *[transfer.name for transfer in sidecars]],
            )
            # HostPoolGroup.resolve_host_transfers gives derived pools their
            # source pool's indices without allocating another staging span.
            swa.host_indices = swa_host_indices
            for sidecar in sidecars:
                if sidecar.indices_from_pool == PoolName.KV:
                    sidecar.host_indices = host_indices
                    sidecar.device_indices = device_value
                else:
                    sidecar.host_indices = swa_host_indices
                    sidecar.device_indices = swa_device_indices
            return host_indices

        controller.write.side_effect = _write
        controller.write_storage.return_value = 99

        cache = MagicMock()
        cache.cache_controller = controller
        cache.page_size = page_size
        cache._build_backup_sidecar.return_value = sidecars

        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        pipeline._cache = cache
        pipeline.ongoing_write_through = {}
        pipeline.ongoing_backup = {}
        pipeline.inflight_backup_hashes = {}
        pipeline.write_staged_tokens_ = 0
        pipeline.write_backlog_tokens_ = len(device_indices)

        hashes = ["page-0", "page-1"]
        snapshot = BufferBackupSnapshot(
            node_id=7,
            parent_node_id=0,
            parent_is_root=True,
            parent_last_hash=None,
            hash_values=hashes,
            key=RadixKey(array("q", [1, 2, 3, 4])),
            prefix_keys=None,
        )
        intent = _UnifiedBackupIntent(snapshot=snapshot)

        self.assertTrue(
            pipeline._launch_backup_intent(
                intent,
                device_indices,
                comp_xfers={ComponentType.SWA: [swa]},
            )
        )
        self.assertEqual(pipeline.ongoing_write_through[7].aux_xfers, [swa, *sidecars])

        pipeline.finish_backup_ack(7)

        storage_transfers = {
            transfer.name: transfer
            for transfer in controller.write_storage.call_args.kwargs["extra_pools"]
        }
        self.assertEqual(storage_transfers[PoolName.SWA].keys, [hashes[-1]])
        self.assertTrue(
            torch.equal(storage_transfers[PoolName.SWA].host_indices, swa_host_indices)
        )
        for spec in self._dsv4_specs():
            storage_sidecar = storage_transfers[spec.pool_name]
            expected_keys = (
                hashes if spec.indices_from_pool == PoolName.KV else [hashes[-1]]
            )
            self.assertEqual(storage_sidecar.keys, expected_keys)
            self.assertEqual(storage_sidecar.hit_policy, spec.hit_policy)
            self.assertEqual(storage_sidecar.indices_from_pool, spec.indices_from_pool)
            self.assertIsNone(storage_sidecar.host_indices)
        self.assertIn(99, pipeline.ongoing_backup)

    def test_completed_prefetch_keeps_dsv4_full_and_swa_sidecars_for_h2d(self):
        swa = PoolTransfer(
            name=PoolName.SWA,
            host_indices=torch.arange(20, 22, dtype=torch.int64),
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        sidecars = [
            PoolTransfer(
                name=spec.pool_name,
                hit_policy=spec.hit_policy,
                indices_from_pool=spec.indices_from_pool,
            )
            for spec in self._dsv4_specs()
        ]
        operation = SimpleNamespace(
            id=23,
            pool_transfers=sidecars,
            storage_start=0,
        )
        host_indices = torch.arange(4, dtype=torch.int64)
        req_id = "sidecar-prefetch"

        cache = MagicMock()
        cache.page_size = 2
        cache.cache_controller.prefetch_tokens_occupied = len(host_indices)
        cache.ongoing_prefetch = {
            req_id: (
                0,
                RadixKey(array("q", [1, 2, 3, 4])),
                host_indices,
                operation,
                None,
                {ComponentType.SWA: [swa]},
            )
        }
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.prefetch_loaded_storage_start_by_reqid = {}
        cache.storage_existence_cache = MagicMock()

        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        pipeline._cache = cache
        pipeline._prefetch_prefix_ctx = {req_id: ([], None, None)}
        pipeline.staged_prefetches = {}

        self.assertTrue(
            pipeline.stage_completed_prefetch(
                req_id=req_id,
                num_tokens=len(host_indices),
                hash_value=["page-0", "page-1"],
            )
        )

        staged = pipeline.staged_prefetches[req_id]
        self.assertEqual(staged.aux_xfers, [swa, *sidecars])
        self.assertEqual(staged.num_tokens, len(host_indices))
        self.assertEqual(
            cache.prefetch_loaded_tokens_by_reqid[req_id], len(host_indices)
        )
        self.assertEqual(cache.prefetch_loaded_storage_start_by_reqid[req_id], 0)


if __name__ == "__main__":
    unittest.main()
