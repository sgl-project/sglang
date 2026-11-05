"""Sparse Mamba writes keep KV coverage and restore only complete checkpoints."""

import unittest
from array import array
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.environ import envs
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
from sglang.srt.mem_cache.unified_cache.cache_action import BackupKV
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.components.mamba import MambaComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    BufferBackupSnapshot,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import slot_of, tree_slots_of

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def snapshot(node=7, hashes=None):
    hashes = hashes or ["h1", "h2"]
    return BufferBackupSnapshot(
        node_id=node,
        parent_node_id=0,
        parent_is_root=True,
        parent_last_hash=None,
        hash_values=hashes,
        key=RadixKey(array("q", range(2 * len(hashes)))),
        prefix_keys=None,
    )


def pipeline(selective):
    cache = MagicMock()
    cache._insert_params = None
    for name in (
        "_should_backup_mamba",
        "_retry_request_end_mamba_backup",
        "_execute_and_commit_kv_backup",
    ):
        setattr(cache, name, MethodType(getattr(UnifiedRadixCache, name), cache))
    cache.page_size = 2
    cache.enable_storage = True
    cache.components = {}
    cache.sidecar_pool_specs = []
    cache.storage_existence_cache = StorageExistenceCache()
    cache.token_to_kv_pool_allocator.get_kvcache.return_value = SimpleNamespace(
        size=128, layer_router=SimpleNamespace(sub_pool=lambda kind: None)
    )
    cache.tree_core.snapshot_buffer_backup.return_value = snapshot()
    cache.tree_core.buffer_backup_pool_keys.return_value = {
        ComponentType.MAMBA: {PoolName.MAMBA: ["h2"]}
    }
    cache.tree_core.build_backup_kv_action.return_value = BackupKV(node_ids=[7])
    cache.tree_core.collect_path_info.return_value = ((SimpleNamespace(path_end=4),),)
    with envs.SGLANG_HICACHE_MAMBA_REQUEST_END_ONLY.override(selective):
        cache._mamba_request_end_only = envs.SGLANG_HICACHE_MAMBA_REQUEST_END_ONLY.get()
        result = BufferModePipeline(cache, swa_component=None, write_backlog_cap=128)
    cache.buffer_pipeline = result
    result._backup_oversize = lambda *args: False
    return result, cache


def real_tree_core(backend, *, mamba=False):
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=MagicMock(spec=HybridReqToTokenPool) if mamba else None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        tree_slots=tree_slots_of(
            (ComponentType.FULL,) + ((ComponentType.MAMBA,) if mamba else ()),
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
        if mamba:
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


def insert(p, is_request_end, length=4, node=7):
    p._cache._insert_params = InsertParams(
        key=RadixKey(array("q", range(length))), is_request_end=is_request_end
    )
    p.begin_insert_walk()
    p.enqueue_backup_intent(node)
    p.end_insert_walk()
    p._cache._insert_params = None


class TestMambaSelectiveBackup(unittest.TestCase):
    def test_default_off_preserves_intermediate_and_endpoint(self):
        for source in (False, True, None):
            p, _ = pipeline(False)
            insert(p, source)
            self.assertEqual(
                [i.pool for i in p.pending_write_queue], [PoolName.KV, PoolName.MAMBA]
            )

    def test_intermediate_skip_does_not_publish_state_and_endpoint_can_follow(self):
        p, cache = pipeline(True)
        insert(p, False)
        self.assertEqual([i.pool for i in p.pending_write_queue], [PoolName.KV])
        self.assertFalse(
            cache.storage_existence_cache.pool(PoolName.MAMBA).covered(["h2"])
        )
        insert(p, True)
        self.assertEqual(
            [i.pool for i in p.pending_write_queue], [PoolName.KV, PoolName.MAMBA]
        )
        insert(p, True)
        self.assertEqual(len(p.pending_write_queue), 2)  # queued-state dedupe

    def test_endpoint_retry_respects_write_policy(self):
        for write_back, threshold, hits, prior_kv, declined, expected in (
            (False, 2, 1, False, False, []),
            (False, 2, 2, False, False, [PoolName.KV, PoolName.MAMBA]),
            (False, 1, 1, True, False, [PoolName.KV, PoolName.MAMBA]),
            (True, 2, 2, False, False, []),
            (False, 2, 2, False, True, []),
        ):
            with self.subTest(
                write_back=write_back, threshold=threshold, hits=hits, declined=declined
            ):
                p, cache = pipeline(True)
                if prior_kv:
                    insert(p, False)
                cache.disable = False
                cache.buffer_pipeline = p
                cache.is_write_back = write_back
                cache.write_through_threshold = threshold
                cache.tree_core.node_hit_count.return_value = hits
                cache.tree_core.has_ongoing_insert.return_value = False
                cache.tree_core.begin_insert.return_value = SimpleNamespace(
                    actions=[],
                    result=InsertResult(
                        prefix_len=4,
                        last_device_node=7,
                        rotation_tail_declined=declined,
                    ),
                )
                cache.tree_core.end_insert.return_value = []
                params = InsertParams(
                    key=RadixKey(array("q", range(4))),
                    is_request_end=True,
                )
                UnifiedRadixCache.insert(cache, params)
                self.assertEqual([i.pool for i in p.pending_write_queue], expected)
                self.assertIsNone(cache._insert_params)

    def test_endpoint_retry_reads_hit_count_from_each_tree_core_backend(self):
        for backend in ("python", "rust"):
            for threshold, expected in ((2, [7]), (3, [])):
                with self.subTest(backend=backend, threshold=threshold):
                    core = real_tree_core(backend)
                    key = RadixKey(array("q", range(4)))
                    first = pump_insert(core, key)
                    second = pump_insert(core, key)
                    self.assertEqual(first.last_device_node, second.last_device_node)
                    self.assertEqual(core.node_hit_count(second.last_device_node), 2)

                    p, cache = pipeline(True)
                    cache.tree_core = core
                    cache.is_write_back = False
                    cache.write_through_threshold = threshold
                    retried = []
                    p.enqueue_backup_intent = retried.append
                    cache._insert_params = InsertParams(key=key, is_request_end=True)
                    p.begin_insert_walk()
                    cache._retry_request_end_mamba_backup(
                        InsertResult(
                            prefix_len=4, last_device_node=second.last_device_node
                        )
                    )
                    p.end_insert_walk()
                    self.assertEqual(
                        retried, [second.last_device_node] if expected else []
                    )

    def test_ancestor_is_not_promoted_to_request_end(self):
        p, _ = pipeline(True)
        insert(p, True, length=8)
        self.assertEqual([i.pool for i in p.pending_write_queue], [PoolName.KV])

    def test_unknown_manual_source_remains_eligible(self):
        p, _ = pipeline(True)
        insert(p, None)
        self.assertIn(PoolName.MAMBA, [i.pool for i in p.pending_write_queue])

    def test_split_keeps_original_state_key_only(self):
        p, cache = pipeline(True)
        insert(p, True)
        p._validate_backup_intent = lambda intent: None
        pieces = [snapshot(6, ["h1"]), snapshot(7, ["h2"])]
        p._split_pieces = lambda original: pieces
        cache.tree_core.buffer_backup_pool_keys.side_effect = lambda node, hashes: {
            ComponentType.MAMBA: {PoolName.MAMBA: hashes[-1:]}
        }
        p._refresh_pending_backup_intents()
        states = [i for i in p.pending_write_queue if i.pool == PoolName.MAMBA]
        self.assertEqual(
            [(i.snapshot.node_id, i.keys) for i in states],
            [(7, ["h2"])],
        )
        self.assertEqual(
            len([i for i in p.pending_write_queue if i.pool == PoolName.KV]), 2
        )

    def test_request_provenance_including_prefill_completion(self):
        class Captured(Exception):
            pass

        for finished, mode, length, expected in (
            (False, "prefill", 2, False),
            (False, "prefill", 4, True),
            (False, "null", 4, False),
            (True, "null", 4, True),
        ):
            with self.subTest(finished=finished, mode=mode, length=length):
                cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
                cache.disable = False
                cache.tree_core = SimpleNamespace(is_eagle=False, page_size=2)
                cache._components_tuple = ()
                cache.components = {}
                cache.linker = None
                cache.req_to_token_pool = SimpleNamespace(
                    req_to_token=torch.arange(4).reshape(1, 4)
                )
                req = SimpleNamespace(
                    finished=lambda: finished,
                    full_untruncated_fill_ids=array("q", range(4)),
                    origin_input_ids=array("q", range(4)),
                    priority=0,
                    session_id=None,
                    kv_rotation_base=None,
                    extra_key=None,
                    cache_salt=None,
                    kv=SimpleNamespace(
                        req_pool_idx=0, cache_protected_len=0, cache_inserted_len=0
                    ),
                )

                def capture(params):
                    self.assertEqual(params.is_request_end, expected)
                    raise Captured

                cache.insert = capture
                with patch(
                    "sglang.srt.mem_cache.unified_radix_cache.get_disagg",
                    return_value=SimpleNamespace(disaggregation_mode=mode),
                ):
                    with self.assertRaises(Captured):
                        cache.checkpoint_into_tree(req, up_to=length)


class TestCacheModeMambaSelectiveBackup(CustomTestCase):
    tree_backend = "python"

    def make_cache(self, selective=True, storage=True):
        core = real_tree_core(self.tree_backend, mamba=True)
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
        p, buffer_cache = pipeline(True)
        buffer_cache.tree_core = cache.tree_core
        buffer_cache.page_size = 1
        buffer_cache.components = {ComponentType.MAMBA: MagicMock()}
        buffer_cache.cache_controller.mem_pool_host.entry_map = {
            PoolName.MAMBA: SimpleNamespace(host_pool=SimpleNamespace(page_size=1))
        }
        buffer_cache.hicache_storage_pass_prefix_keys = False
        buffer_cache.is_write_back = False
        buffer_cache.write_through_threshold = 1
        buffer_cache._insert_params = InsertParams(
            key=RadixKey(array("q", range(4))), is_request_end=False
        )
        p.begin_insert_walk()
        buffer_cache._execute_and_commit_kv_backup(
            cache.tree_core.build_backup_kv_action(nodes[-1])
        )
        p.end_insert_walk()
        self.assertEqual([i.pool for i in p.pending_write_queue], [PoolName.KV] * 2)

        buffer_cache._insert_params.is_request_end = True
        for _ in range(2):
            p.begin_insert_walk()
            buffer_cache._retry_request_end_mamba_backup(
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
