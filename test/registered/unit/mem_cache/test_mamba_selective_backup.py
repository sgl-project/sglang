"""Sparse Mamba writes keep KV coverage and restore only complete checkpoints."""

import unittest
from array import array
from types import SimpleNamespace
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
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    BufferBackupSnapshot,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci
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
    cache.tree_core.collect_path_info.return_value = ((SimpleNamespace(path_end=4),),)
    with envs.SGLANG_HICACHE_MAMBA_REQUEST_END_ONLY.override(selective):
        result = BufferModePipeline(cache, swa_component=None, write_backlog_cap=128)
    result._backup_oversize = lambda *args: False
    return result, cache


def real_tree_core(backend):
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
        page_size=1,
        tree_slots=tree_slots_of((ComponentType.FULL,)),
    )
    if backend == "rust":
        from sglang.srt.mem_cache.rust_tree_core.adapter import RustUnifiedTreeCore

        return RustUnifiedTreeCore(params)
    cache = SimpleNamespace(enable_session_radix_cache=False)
    return UnifiedTreeCore(
        params,
        {
            ComponentType.FULL: FullComponent(
                cache, params, slot_of(params, ComponentType.FULL)
            )
        },
    )


def pump_insert(core, key):
    step = core.begin_insert(
        InsertParams(key=key, value=torch.arange(len(key), dtype=torch.int64))
    )
    while step.result is None:
        step = core.resume_insert()
    core.end_insert()
    return step.result


def insert(p, is_request_end, length=4, node=7):
    p.begin_insert_walk(is_request_end=is_request_end, key_len=length)
    p.enqueue_backup_intent(node)
    p.end_insert_walk()


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
                self.assertIsNone(p._insert_is_request_end)

    def test_endpoint_retry_reads_hit_count_from_each_tree_core_backend(self):
        # The retry runs on the tree core the server selected; the Rust core has
        # no Python node objects, so the hit count must come through the interface.
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
                    p.begin_insert_walk(is_request_end=True, key_len=4)
                    p.on_insert_complete(
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
                        cache._checkpoint_into_tree(req, up_to=length)


if __name__ == "__main__":
    unittest.main()
