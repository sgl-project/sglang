"""Unit coverage for sidecar pools in HiCache buffer-only mode."""

import unittest
from array import array
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import torch

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.buffer_mode.pipeline import (
    BufferModePipeline,
    validate_buffer_only_stack,
)
from sglang.srt.mem_cache.buffer_mode.storage_existence_cache import (
    StorageExistenceCache,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheFile,
    HiCacheStorageConfig,
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
    SidecarPoolSpec,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import (
    ComponentType,
)
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    BufferBackupSnapshot,
    BufferBackupState,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _OngoingPrefetch,
)
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestBufferModeSidecar(unittest.TestCase):
    @staticmethod
    def _swa_component():
        return SimpleNamespace(
            full_window_pages=2,
            _swa_kv_pool_host=SimpleNamespace(
                page_size=2,
                size=8,
                shared_allocation_domain=None,
            ),
        )

    def test_file_query_missing_rank_preserves_swa_repair(self):
        snapshot = self._backup_snapshot()
        token_ids = list(snapshot.key.token_ids)
        chain = get_storage_hash_str(token_ids, page_size=2)
        snapshot = msgspec.structs.replace(snapshot, hash_values=chain)
        swa = PoolTransfer(
            name=PoolName.SWA,
            keys=chain[-1:],
            hit_policy=PoolHitPolicy.TRAILING_PAGES,
        )
        for peer_swa_pages in (0, 1):
            with (
                TemporaryDirectory() as directory,
                patch.dict(
                    "os.environ", {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": directory}
                ),
            ):
                ranks, packed = [], []

                def capture(values, reduce_op, groups):
                    self.assertEqual(reduce_op, torch.distributed.ReduceOp.MIN)
                    packed.append(values.clone())

                for rank, swa_pages in enumerate((2, peer_swa_pages)):
                    backend = HiCacheFile(
                        HiCacheStorageConfig(
                            tp_rank=rank,
                            tp_size=2,
                            pp_rank=0,
                            pp_size=1,
                            attn_cp_rank=0,
                            attn_cp_size=1,
                            is_mla_model=False,
                            enable_storage_metrics=False,
                            is_page_first_layout=False,
                            model_name="swa-repair",
                            extra_config={"enable_metadata_cache": False},
                        )
                    )
                    for pool, keys in (
                        (PoolName.KV, chain),
                        (PoolName.SWA, chain[:swa_pages]),
                    ):
                        for key in keys:
                            Path(
                                directory,
                                backend._get_component_key(key, pool) + ".bin",
                            ).touch()
                    operation = PrefetchOperation(
                        CacheRequestHandle("repair", 0), token_ids, pool_transfers=[swa]
                    )
                    controller = SimpleNamespace(
                        storage_backend=backend,
                        page_size=2,
                        prefetch_hits_sync_groups=[],
                        _all_reduce=capture,
                    )
                    _, hit_tokens = HybridCacheController._storage_hit_query(
                        controller, operation
                    )
                    self.assertEqual(operation.all_hash_values, chain)
                    self.assertEqual(
                        operation.query_pool_hit_pages,
                        {
                            PoolName.KV: 2,
                            **({PoolName.SWA: swa_pages} if swa_pages else {}),
                        },
                    )
                    HybridCacheController._sync_prefetch_hit_query(
                        controller, operation, hit_tokens
                    )
                    ranks.append((controller, operation, hit_tokens))

                # Only transport is simulated: reduce the tensors built by the
                # actual controller, then deliver the same MIN to both ranks.
                agreed = torch.stack(packed).amin(dim=0)
                for rank, (controller, operation, hit_tokens) in enumerate(ranks):
                    with self.subTest(peer_swa_pages=peer_swa_pages, rank=rank):
                        controller._all_reduce = lambda values, *_: values.copy_(agreed)
                        operation.storage_hit_count = (
                            HybridCacheController._sync_prefetch_hit_query(
                                controller, operation, hit_tokens
                            )
                        )
                        tree = MagicMock()
                        tree.build_hicache_transfers.return_value = [swa]
                        tree.buffer_backup_pool_keys.return_value = {}
                        pipeline = self._launch_pipeline(tree, [])
                        cache = pipeline._cache
                        cache.host_memory_mode, cache.page_size = "buffer_only", 2
                        cache.components = {
                            ct: SimpleNamespace(buffer_belief_policy=lambda _: None)
                            for ct in (ComponentType.FULL, ComponentType.SWA)
                        }
                        cache._belief_policy = UnifiedRadixCache._belief_policy.__get__(
                            cache
                        )
                        for pool in (PoolName.KV, PoolName.SWA):
                            cache.storage_existence_cache.pool(pool).add(chain)
                        UnifiedRadixCache._invalidate_absent_from_hit_query(
                            cache, operation
                        )
                        intents = pipeline._write_intents(snapshot)
                        # Retain FULL while making only the missing SWA tail
                        # eligible, including when one rank omitted its verdict.
                        self.assertEqual(intents, [(PoolName.SWA, chain[-1:])])
                        self.assertTrue(
                            cache.storage_existence_cache.pool(
                                PoolName.KV
                            ).contains_all(chain)
                        )
                        self.assertEqual(
                            operation.storage_hit_count, peer_swa_pages * 2
                        )
                        self.assertEqual(
                            [
                                cache.storage_existence_cache.pool(
                                    PoolName.SWA
                                ).contains(h)
                                for h in chain
                            ],
                            [i < peer_swa_pages for i in range(2)],
                        )

        # An assumed-stored retry verifies nothing now, so it carries no pool
        # verdict and re-learns nothing a shortfall heal dropped.
        operation = PrefetchOperation(
            CacheRequestHandle("assumed", 0),
            token_ids,
            pool_transfers=[swa],
            assume_stored=True,
        )
        controller = SimpleNamespace(
            storage_backend=None,
            page_size=2,
            prefetch_hits_sync_groups=[],
            _all_reduce=lambda values, *_: None,
        )
        _, hit_tokens = HybridCacheController._storage_hit_query(controller, operation)
        operation.storage_hit_count = HybridCacheController._sync_prefetch_hit_query(
            controller, operation, hit_tokens
        )
        self.assertEqual(
            (operation.storage_hit_count, operation.query_pool_hit_pages), (4, {})
        )
        pipeline = self._launch_pipeline(MagicMock(), [])
        cache = pipeline._cache
        cache.host_memory_mode, cache.page_size = "buffer_only", 2
        cache.components = {
            ct: SimpleNamespace(buffer_belief_policy=lambda _: None)
            for ct in (ComponentType.FULL, ComponentType.SWA)
        }
        cache._belief_policy = UnifiedRadixCache._belief_policy.__get__(cache)
        cache.storage_existence_cache.pool(PoolName.KV).add(chain)
        UnifiedRadixCache._invalidate_absent_from_hit_query(cache, operation)
        self.assertTrue(
            cache.storage_existence_cache.pool(PoolName.KV).contains_all(chain)
        )
        self.assertFalse(
            any(
                cache.storage_existence_cache.pool(PoolName.SWA).contains(h)
                for h in chain
            )
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

    @staticmethod
    def _backup_snapshot():
        return BufferBackupSnapshot(
            node_id=7,
            parent_node_id=0,
            parent_is_root=True,
            parent_last_hash=None,
            hash_values=["page-0", "page-1"],
            key=RadixKey(array("q", [1, 2, 3, 4])),
            prefix_keys=["page-p"],
        )

    @staticmethod
    def _launch_pipeline(tree, sidecar_pool_specs):
        cache = SimpleNamespace(
            tree_core=tree,
            components={ComponentType.FULL: SimpleNamespace()},
            sidecar_pool_specs=sidecar_pool_specs,
            storage_existence_cache=StorageExistenceCache(),
        )
        cache._build_sidecar_transfers = (
            UnifiedRadixCache._build_sidecar_transfers.__get__(cache)
        )
        cache._build_backup_sidecar = UnifiedRadixCache._build_backup_sidecar.__get__(
            cache
        )
        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        pipeline._cache = cache
        return pipeline

    def test_write_stages_and_persists_dsv4_full_and_swa_sidecars(self):
        """Each pool of a node is a write of its own; every sidecar rides its
        source pool's write under the source's keys."""
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
        kv_sidecars = [s for s in sidecars if s.indices_from_pool == PoolName.KV]
        swa_sidecars = [s for s in sidecars if s.indices_from_pool == PoolName.SWA]

        controller = MagicMock()
        controller.mem_pool_host.anchor_entry.host_pool.shared_allocation_domain = None
        controller.mem_pool_host.entry_map = {
            PoolName.SWA: SimpleNamespace(
                host_pool=SimpleNamespace(
                    page_size=page_size,
                    size=8,
                    available_size=lambda: 8,
                    free=MagicMock(),
                )
            )
        }
        controller.mem_pool_host.size = 32
        controller.mem_pool_host.anchor_entry.host_pool = SimpleNamespace()
        controller.prefetch_tokens_occupied = 0
        ack_ids = []

        def _write(device_value, node_id, extra_pools, flush):
            self.assertFalse(flush)
            ack_ids.append(node_id)
            # HostPoolGroup.resolve_host_transfers gives derived pools their
            # source pool's indices without allocating another staging span.
            if device_value.numel():
                self.assertEqual(extra_pools, kv_sidecars)
                for sidecar in extra_pools:
                    sidecar.host_indices = host_indices
                    sidecar.device_indices = device_value
                return host_indices
            self.assertEqual(extra_pools, [swa, *swa_sidecars])
            swa.host_indices = swa_host_indices
            for sidecar in swa_sidecars:
                sidecar.host_indices = swa_host_indices
                sidecar.device_indices = swa_device_indices
            return device_value

        controller.write.side_effect = _write
        controller.write_storage.side_effect = [98, 99]

        cache = MagicMock()
        cache.cache_controller = controller
        cache.page_size = page_size
        cache.enable_storage = True
        cache.enable_storage_metrics = False
        cache.hicache_storage_pass_prefix_keys = True
        cache.components = {ComponentType.FULL: MagicMock()}
        cache.sidecar_pool_specs = self._dsv4_specs()
        cache._build_backup_sidecar.return_value = sidecars
        cache.storage_existence_cache = StorageExistenceCache()

        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        pipeline._cache = cache
        pipeline._swa_window_pages = 1
        pipeline.write_backlog_cap = 8
        pipeline.reset()
        pipeline._build_backup_transfers = lambda node_id, *, kv_only=False: (
            device_indices,
            {ComponentType.SWA: [swa]},
            None,
        )

        hashes = ["page-0", "page-1"]
        snapshot = BufferBackupSnapshot(
            node_id=7,
            parent_node_id=0,
            parent_is_root=True,
            parent_last_hash=None,
            hash_values=hashes,
            key=RadixKey(array("q", [1, 2, 3, 4])),
            prefix_keys=["page-p"],
        )
        tree = cache.tree_core
        tree.snapshot_buffer_backup.return_value = snapshot
        tree.validate_buffer_backup.return_value = BufferBackupState(0, True, None)
        tree.buffer_backup_pool_keys.return_value = {
            ComponentType.SWA: {PoolName.SWA: hashes[-1:]}
        }
        pipeline.enqueue_backup_intent(7)
        self.assertEqual(
            [intent.pool for intent in pipeline.pending_write_queue],
            [PoolName.KV, PoolName.SWA],
        )
        pipeline.flush_pending_writes()
        self.assertFalse(pipeline.pending_write_queue)
        kv_ack, swa_ack = ack_ids
        self.assertEqual(pipeline.ongoing_write_through[kv_ack].aux_xfers, kv_sidecars)
        self.assertEqual(
            pipeline.ongoing_write_through[swa_ack].aux_xfers, [swa, *swa_sidecars]
        )
        expected_keys = {PoolName.KV: hashes, PoolName.SWA: hashes[-1:]}
        for spec in self._dsv4_specs():
            expected_keys[spec.pool_name] = expected_keys[spec.indices_from_pool]
        for pool, keys in expected_keys.items():
            # In flight (covered for admission) but not yet believed stored.
            pool_beliefs = cache.storage_existence_cache.pool(pool)
            self.assertTrue(pool_beliefs.covered(keys))
            self.assertFalse(pool_beliefs.contains_all(keys))
        self.assertEqual(pipeline.write_backlog_tokens_, 0)

        pipeline.finish_backup_ack(kv_ack)
        pipeline.finish_backup_ack(swa_ack)

        kv_call, swa_call = controller.write_storage.call_args_list
        self.assertTrue(torch.equal(kv_call.args[0], host_indices))
        self.assertEqual(kv_call.args[2], hashes)
        self.assertEqual(kv_call.args[3], ["page-p"])
        # No KV page in the SWA write; the window's keys ride the transfer.
        self.assertEqual(swa_call.args[0].numel(), 0)
        self.assertEqual(swa_call.args[2], [])
        self.assertEqual(swa_call.args[3], ["page-p"])
        storage_transfers = {
            transfer.name: transfer
            for call in (kv_call, swa_call)
            for transfer in call.kwargs["extra_pools"]
        }
        self.assertEqual(storage_transfers[PoolName.SWA].keys, [hashes[-1]])
        self.assertTrue(
            torch.equal(storage_transfers[PoolName.SWA].host_indices, swa_host_indices)
        )
        for spec in self._dsv4_specs():
            storage_sidecar = storage_transfers[spec.pool_name]
            self.assertEqual(storage_sidecar.keys, expected_keys[spec.pool_name])
            self.assertEqual(storage_sidecar.hit_policy, spec.hit_policy)
            self.assertEqual(storage_sidecar.indices_from_pool, spec.indices_from_pool)
            self.assertIsNone(storage_sidecar.host_indices)
        self.assertEqual(set(pipeline.ongoing_backup), {98, 99})
        pipeline.finish_storage_write_ack(98)
        self.assertTrue(pipeline.backup_pending(7))
        pipeline.finish_storage_write_ack(99)
        self.assertFalse(pipeline.backup_pending(7))
        self.assertEqual(pipeline.write_staged_tokens_, 0)

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
        host_indices = torch.arange(4, dtype=torch.int64)
        operation = SimpleNamespace(
            id=23,
            pool_transfers=sidecars,
            storage_start=0,
            buffer_host_occupied_units=len(host_indices),
        )
        req_id = CacheRequestHandle("sidecar-prefetch", 0)

        cache = MagicMock()
        cache.page_size = 2
        cache.cache_controller.prefetch_tokens_occupied = len(host_indices)
        cache.ongoing_prefetch = {
            req_id: _OngoingPrefetch(
                anchor_node_id=0,
                prefetch_key=RadixKey(array("q", [1, 2, 3, 4])),
                host_indices=host_indices,
                operation=operation,
                anchor_lock_params=None,
                comp_xfers={ComponentType.SWA: [swa]},
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
                request=req_id,
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
