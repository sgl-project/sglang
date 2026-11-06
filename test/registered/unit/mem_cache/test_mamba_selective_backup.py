"""Sparse Mamba writes keep KV coverage and restore only complete checkpoints."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    InsertResult,
)
from sglang.srt.mem_cache.buffer_mode.pipeline import BufferModePipeline
from sglang.srt.mem_cache.buffer_mode.storage_existence_cache import (
    StorageExistenceCache,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
)
from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import slot_of, tree_slots_of

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def real_tree_core(backend):
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=MagicMock(spec=HybridReqToTokenPool),
        token_to_kv_pool_allocator=None,
        page_size=1,
        tree_slots=tree_slots_of(
            (ComponentType.FULL, ComponentType.MAMBA),
            checkpoint_interval=256,
        ),
    )
    with get_context().override_server_args(
        _mamba_cache_chunk_size=256, mamba_max_states_per_path=-1
    ):
        if backend == "rust":
            from sglang.srt.mem_cache.rust_tree_core.adapter import RustUnifiedTreeCore

            return RustUnifiedTreeCore(params)
        cache = SimpleNamespace(enable_session_radix_cache=False)
        components = {
            ComponentType.FULL: FullComponent(
                cache, params, slot_of(params, ComponentType.FULL)
            )
        }
        components[ComponentType.MAMBA] = MambaComponent(
            cache, params, slot_of(params, ComponentType.MAMBA)
        )
        core = UnifiedTreeCore(params, components)
        cache.tree_core = core
        return core


def pump_insert(core, key, *, mamba_slot=None):
    step = core.begin_insert(
        InsertParams(
            key=key,
            value=torch.arange(len(key), dtype=torch.int64),
            mamba_value=(
                torch.tensor([mamba_slot], dtype=torch.int64)
                if mamba_slot is not None
                else None
            ),
        )
    )
    while step.result is None:
        step = core.resume_insert()
    core.end_insert()
    return step.result


class TestCacheModeMambaSelectiveBackup(CustomTestCase):
    tree_backend = "python"

    def make_cache(self, selective=True, storage=True):
        core = real_tree_core(self.tree_backend)
        nodes = [
            pump_insert(
                core, RadixKey(array("q", range(length))), mamba_slot=length
            ).last_device_node
            for length in (2, 4)
        ]
        core.backfill_missing_hash_values()
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = core
        cache.buffer_pipeline = None
        cache._mamba_request_end_only = selective
        cache.enable_storage = storage
        cache.hicache_storage_pass_prefix_keys = False
        cache.cache_controller = MagicMock()
        cache.ongoing_write_through = {}
        cache.ongoing_backup = {}
        cache.is_write_back = False
        cache.write_through_threshold = 1
        cache.inc_lock_ref = MagicMock()
        cache.dec_lock_ref = MagicMock()
        cache.inc_host_lock_ref = MagicMock()
        cache._build_sidecar_transfers = lambda *args: []
        cache.writes = []
        cache.storage_writes = []

        def copy_to_host(node_id, device, transfers, sidecars):
            cache.writes.append((node_id, device.numel(), set(transfers)))
            for xfers in transfers.values():
                for xfer in xfers:
                    xfer.host_indices = xfer.device_indices.clone()
            return device.clone()

        def write_storage(host, tokens, hashes, prefixes, extra_pools=None):
            cache.storage_writes.append([x.name for x in extra_pools or ()])
            return len(cache.storage_writes)

        def drain(**kwargs):
            for ack in list(cache.ongoing_write_through):
                cache._finish_write_through_ack(ack)

        cache._execute_kv_backup = copy_to_host
        cache.cache_controller.write_storage = write_storage
        cache.writing_check = drain
        return cache, nodes

    def test_buffer_mode_uses_the_same_tree_endpoint_and_retry_policy(self):
        cache, nodes = self.make_cache()
        cache.page_size = 1
        cache.components = {ComponentType.MAMBA: MagicMock()}
        cache.sidecar_pool_specs = []
        cache.storage_existence_cache = StorageExistenceCache()
        cache.token_to_kv_pool_allocator = MagicMock()
        cache.token_to_kv_pool_allocator.get_kvcache.return_value = SimpleNamespace(
            size=128, layer_router=SimpleNamespace(sub_pool=lambda kind: None)
        )
        cache.cache_controller.mem_pool_host.entry_map = {
            PoolName.MAMBA: SimpleNamespace(host_pool=SimpleNamespace(page_size=1))
        }
        p = BufferModePipeline(cache, swa_component=None, write_backlog_cap=128)
        cache.buffer_pipeline = p
        p._backup_oversize = lambda *args: False
        cache._insert_params = InsertParams(
            key=RadixKey(array("q", range(4))), is_request_end=False
        )
        p.begin_insert_walk()
        cache._execute_and_commit_kv_backup(
            cache.tree_core.build_backup_kv_action(nodes[-1])
        )
        p.end_insert_walk()
        self.assertEqual([i.pool for i in p.pending_write_queue], [PoolName.KV] * 2)

        cache._insert_params.is_request_end = True
        for _ in range(2):
            p.begin_insert_walk()
            cache._retry_request_end_mamba_backup(
                InsertResult(prefix_len=4, last_device_node=nodes[-1])
            )
            p.end_insert_walk()
        states = [i for i in p.pending_write_queue if i.pool == PoolName.MAMBA]
        self.assertEqual([i.snapshot.node_id for i in states], [nodes[-1]])
        self.assertEqual(
            states[0].keys, cache.tree_core.get_hash_values(nodes[-1])[-1:]
        )
        self.assertEqual(len(p.pending_write_queue), 3)

    def test_intermediate_then_endpoint_adds_only_missing_mamba(self):
        """An existing KV backup must not hide a newly eligible endpoint state."""
        for storage in (False, True):
            with self.subTest(storage=storage):
                cache, nodes = self.make_cache(storage=storage)
                cache._insert_params = InsertParams(
                    key=RadixKey(array("q", range(4))), is_request_end=False
                )
                cache._execute_and_commit_kv_backup(
                    cache.tree_core.build_backup_kv_action(nodes[-1])
                )
                for node in nodes:
                    spec = cache.tree_core.build_storage_backup_spec(node, False)
                    self.assertIsNotNone(spec)
                    self.assertNotIn(ComponentType.MAMBA, spec.comp_xfers)
                self.assertEqual(sum(w[1] for w in cache.writes), 4)

                cache._insert_params.is_request_end = True
                cache._retry_request_end_mamba_backup(
                    InsertResult(prefix_len=4, last_device_node=nodes[-1])
                )
                self.assertEqual(
                    cache.writes[-1], (nodes[-1], 0, {ComponentType.MAMBA})
                )
                cache.writing_check()
                self.assertEqual(cache.inc_lock_ref.call_count, 3)
                self.assertEqual(cache.dec_lock_ref.call_count, 3)
                self.assertEqual(
                    cache.storage_writes.count([PoolName.MAMBA]), int(storage)
                )
                before = len(cache.writes)
                cache._retry_request_end_mamba_backup(
                    InsertResult(prefix_len=4, last_device_node=nodes[-1])
                )
                self.assertEqual(len(cache.writes), before)
                ancestor = cache.tree_core.build_storage_backup_spec(nodes[0], False)
                self.assertNotIn(ComponentType.MAMBA, ancestor.comp_xfers)

    def test_endpoint_retry_obeys_threshold_write_back_and_declined_tail(self):
        for write_back, threshold, declined in (
            (True, 1, False),
            (False, 100, False),
            (False, 1, True),
        ):
            with self.subTest(
                write_back=write_back, threshold=threshold, declined=declined
            ):
                cache, nodes = self.make_cache()
                cache.is_write_back = write_back
                cache.write_through_threshold = threshold
                cache._insert_params = InsertParams(
                    key=RadixKey(array("q", range(4))), is_request_end=True
                )
                cache._retry_request_end_mamba_backup(
                    InsertResult(
                        prefix_len=4,
                        last_device_node=nodes[-1],
                        rotation_tail_declined=declined,
                    )
                )
                self.assertEqual(cache.writes, [])
                self.assertEqual(cache.ongoing_write_through, {})

    def test_endpoint_chain_filters_ancestors_and_preserves_legacy_sources(self):
        for selective, source, expected in (
            (True, True, [False, True]),
            (True, False, [False, False]),
            (True, None, [True, True]),
            (False, False, [True, True]),
        ):
            with self.subTest(selective=selective, source=source):
                cache, nodes = self.make_cache(selective=selective)
                cache._insert_params = InsertParams(
                    key=RadixKey(array("q", range(4))), is_request_end=source
                )
                cache._execute_and_commit_kv_backup(
                    cache.tree_core.build_backup_kv_action(nodes[-1])
                )
                cache.writing_check()
                self.assertEqual(
                    [ComponentType.MAMBA in w[2] for w in cache.writes], expected
                )
                self.assertEqual(
                    [PoolName.MAMBA in w for w in cache.storage_writes], expected
                )


class TestRustCacheModeMambaSelectiveBackup(TestCacheModeMambaSelectiveBackup):
    tree_backend = "rust"


if __name__ == "__main__":
    unittest.main()
