"""KV cache event tests for UnifiedRadixCache."""

import unittest
from array import array
from unittest import mock

import torch

from sglang.srt.disaggregation.kv_events import (
    StorageMedium,
    kv_event_namespace_seed,
    namespaced_block_hash,
)
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
)
from sglang.srt.mem_cache.events import KvEventLoraNames
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.unified_radix_cache.config import (
    CacheConfig,
)
from sglang.test.kits.unified_radix_cache.fixtures import (
    build_fixture,
    node_children,
    node_token_ids,
    write_backup,
)
from sglang.test.kits.unified_radix_cache.kv_events import (
    LoraReq,
    UnifiedRadixCacheKVEventsFixtures,
    make_lora_names,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")


class TestUnifiedRadixCacheKVEvents(UnifiedRadixCacheKVEventsFixtures, CustomTestCase):
    cfg = CacheConfig(page_size=2, kv_size=64, max_context_len=64)

    def test_kv_events_store_and_remove_full_blocks(self):
        cache, allocator, _ = build_fixture(self.cfg, enable_kv_cache_events=True)
        cache.take_events()  # Clear the reset event.

        seq = [1, 2, 3, 4]
        self._insert(cache, allocator, seq)
        stored = self._stored_events(cache, StorageMedium.GPU)
        self.assertEqual(len(stored), 1)
        self.assertEqual(list(stored[0].token_ids), [1, 2, 3, 4])
        stored_hashes = self._event_hashes(stored)

        result = cache.evict(EvictParams(num_tokens=len(seq)))
        self.assertGreaterEqual(result.num_tokens_evicted, len(seq))
        removed = self._removed_events(cache, StorageMedium.GPU)
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0].block_hashes, stored_hashes)

    def test_cache_salt_is_included_in_store_and_remove_events(self):
        cache, allocator, _ = build_fixture(self.cfg, enable_kv_cache_events=True)
        cache.take_events()

        seq = [1, 2, 3, 4]
        self._insert(cache, allocator, seq, cache_salt="tenant-a")
        stored = self._stored_events(cache, StorageMedium.GPU)
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0].cache_salt, "tenant-a")
        salted_hashes = self._event_hashes(stored)

        cache.evict(EvictParams(num_tokens=len(seq)))
        removed = self._removed_events(cache, StorageMedium.GPU)
        self.assertEqual(len(removed), 1)
        self.assertEqual(removed[0].block_hashes, salted_hashes)

        unsalted, unsalted_allocator, _ = build_fixture(
            self.cfg, enable_kv_cache_events=True
        )
        unsalted.take_events()
        self._insert(unsalted, unsalted_allocator, seq)
        self.assertNotEqual(
            self._event_hashes(self._stored_events(unsalted, StorageMedium.GPU)),
            salted_hashes,
        )

    def test_session_id_is_attributed_without_changing_block_hash(self):
        cache_a, allocator_a, _ = build_fixture(self.cfg, enable_kv_cache_events=True)
        cache_a.take_events()
        self._insert(cache_a, allocator_a, [1, 2, 3, 4], session_id="session-a")
        stored_a = self._stored_events(cache_a, StorageMedium.GPU)
        self.assertEqual(len(stored_a), 1)
        self.assertEqual(stored_a[0].session_id, "session-a")

        cache_b, allocator_b, _ = build_fixture(self.cfg, enable_kv_cache_events=True)
        cache_b.take_events()
        self._insert(cache_b, allocator_b, [1, 2, 3, 4], session_id="session-b")
        stored_b = self._stored_events(cache_b, StorageMedium.GPU)
        self.assertEqual(stored_b[0].session_id, "session-b")
        self.assertEqual(self._event_hashes(stored_a), self._event_hashes(stored_b))

    def test_shared_prefix_hit_is_quiet_and_divergent_tails_are_attributed(self):
        cache, allocator, _ = build_fixture(self.cfg, enable_kv_cache_events=True)
        cache.take_events()

        shared_prefix = [1, 2, 3, 4]
        self._insert(cache, allocator, shared_prefix, session_id="session-a")
        initial = self._stored_events(cache, StorageMedium.GPU)
        self.assertEqual(len(initial), 1)
        shared_parent = initial[0].block_hashes[-1]

        self._insert(cache, allocator, shared_prefix, session_id="session-b")
        self.assertEqual(self._stored_events(cache, StorageMedium.GPU), [])

        self._insert(
            cache,
            allocator,
            shared_prefix + [5, 6],
            session_id="session-a",
        )
        session_a_tail = self._stored_events(cache, StorageMedium.GPU)
        self.assertEqual(len(session_a_tail), 1)
        self.assertEqual(session_a_tail[0].parent_block_hash, shared_parent)
        self.assertEqual(list(session_a_tail[0].token_ids), [5, 6])
        self.assertEqual(session_a_tail[0].session_id, "session-a")

        self._insert(
            cache,
            allocator,
            shared_prefix + [7, 8],
            session_id="session-b",
        )
        session_b_tail = self._stored_events(cache, StorageMedium.GPU)
        self.assertEqual(len(session_b_tail), 1)
        self.assertEqual(session_b_tail[0].parent_block_hash, shared_parent)
        self.assertEqual(list(session_b_tail[0].token_ids), [7, 8])
        self.assertEqual(session_b_tail[0].session_id, "session-b")

    def test_cache_salt_event_parentage_survives_node_split(self):
        cache, allocator, _ = build_fixture(self.cfg, enable_kv_cache_events=True)
        cache.take_events()

        self._insert(cache, allocator, [1, 2, 3, 4], cache_salt="tenant-a")
        original = self._stored_events(cache, StorageMedium.GPU)
        self.assertEqual(len(original), 1)
        self.assertEqual(len(original[0].block_hashes), 2)

        self._insert(cache, allocator, [1, 2, 5, 6], cache_salt="tenant-a")
        branch = self._stored_events(cache, StorageMedium.GPU)
        self.assertEqual(len(branch), 1)
        self.assertEqual(branch[0].cache_salt, "tenant-a")
        self.assertEqual(branch[0].parent_block_hash, original[0].block_hashes[0])
        self.assertEqual(list(branch[0].token_ids), [5, 6])

    def test_event_hashes_depend_on_cache_salt_but_not_extra_key(self):
        def stored_hashes(*, extra_key=None, cache_salt=None):
            cache, allocator, _ = build_fixture(self.cfg, enable_kv_cache_events=True)
            cache.take_events()
            self._insert(
                cache,
                allocator,
                [1, 2, 3, 4],
                extra_key=extra_key,
                cache_salt=cache_salt,
            )
            return self._event_hashes(self._stored_events(cache, StorageMedium.GPU))

        self.assertNotEqual(
            stored_hashes(cache_salt="tenant-a"),
            stored_hashes(cache_salt="tenant-b"),
        )
        self.assertEqual(
            stored_hashes(extra_key="adapter-a", cache_salt="tenant-a"),
            stored_hashes(extra_key="adapter-b", cache_salt="tenant-a"),
        )

    def test_lora_external_ids_separate_adapter_loads(self):
        """External IDs distinguish loads; routers match tokens and public names."""
        lora_names, _reqs = make_lora_names(
            **{
                "uuid-a": "adapter-a",
                "user|uuid-a2": "adapter-a",
                "uuid-b": "adapter-b",
            }
        )

        def store_then_evict(extra_key):
            cache, allocator, _ = build_fixture(
                self.cfg, enable_kv_cache_events=True, kv_event_lora_names=lora_names
            )
            cache.take_events()
            self._insert(cache, allocator, [1, 2, 3, 4], extra_key=extra_key)
            stored = self._stored_events(cache, StorageMedium.GPU)
            cache.evict(EvictParams(num_tokens=4))
            removed = self._removed_events(cache, StorageMedium.GPU)
            self.assertEqual(self._event_hashes(removed), self._event_hashes(stored))
            return stored

        base = store_then_evict(None)
        lora_a = store_then_evict("uuid-a")
        self.assertIsNone(base[0].lora_name)
        self.assertEqual(lora_a[0].lora_name, "adapter-a")
        seed = kv_event_namespace_seed(extra_key="uuid-a", cache_salt=None)
        self.assertEqual(
            self._event_hashes(lora_a),
            [
                namespaced_block_hash(h, namespace_seed=seed)
                for h in self._event_hashes(base)
            ],
        )
        # Another load of the same public name has distinct external IDs.
        self.assertNotEqual(
            self._event_hashes(store_then_evict("user|uuid-a2")),
            self._event_hashes(lora_a),
        )
        self.assertNotEqual(
            self._event_hashes(store_then_evict("uuid-b")),
            self._event_hashes(lora_a),
        )

    def test_dynamo_lora_hashes_keep_the_dynamo_namespace(self):
        """The default LoRA mix must not be applied a second time to Dynamo hashes."""
        extra_key = "0123456789abcdef0123456789abcdef"
        names, reqs = make_lora_names(**{extra_key: "adapter-a"})

        def publish(*, dynamo_format, namespace):
            cache, allocator, _ = build_fixture(
                self.cfg,
                enable_kv_cache_events=True,
                dynamo_kv_event_format=dynamo_format,
                kv_event_lora_names=names,
            )
            cache.take_events()
            self._insert(
                cache,
                allocator,
                [1, 2, 3, 4],
                extra_key=namespace,
                cache_salt="tenant-a",
            )
            stored = self._stored_events(cache, StorageMedium.GPU)
            cache.evict(EvictParams(num_tokens=4))
            removed = self._removed_events(cache, StorageMedium.GPU)
            self.assertEqual(self._event_hashes(removed), self._event_hashes(stored))
            return stored

        base = publish(dynamo_format=False, namespace=None)
        lora = publish(dynamo_format=True, namespace=extra_key)
        seed = kv_event_namespace_seed(extra_key=extra_key, cache_salt="tenant-a")
        self.assertEqual(
            self._event_hashes(lora),
            [
                namespaced_block_hash(h, namespace_seed=seed)
                for h in self._event_hashes(base)
            ],
        )
        self.assertEqual(lora[0].lora_name, "adapter-a")

    def test_lora_request_without_optional_ipc_name_uses_loaded_adapter(self):
        names = KvEventLoraNames()
        names.register(lora_id="loaded-id", lora_name="adapter-a")
        req = LoraReq("user|loaded-id", lora_id="loaded-id")
        names.register(req)
        cache, allocator, _ = build_fixture(
            self.cfg, enable_kv_cache_events=True, kv_event_lora_names=names
        )
        cache.take_events()
        self._insert(cache, allocator, [1, 2, 3, 4], extra_key=req.extra_key)
        stored = self._stored_events(cache, StorageMedium.GPU)
        self.assertEqual(stored[0].lora_name, "adapter-a")
        cache.evict(EvictParams(num_tokens=4))
        self.assertEqual(
            self._event_hashes(self._removed_events(cache, StorageMedium.GPU)),
            self._event_hashes(stored),
        )

    def test_lora_name_survives_a_gap_before_a_delayed_store(self):
        """Write-back can publish a store after the request and GPU blocks are gone."""
        lora_names, reqs = make_lora_names(**{"uuid-a": "adapter-a"})
        cache, allocator, _ = build_fixture(
            self.cfg, enable_kv_cache_events=True, kv_event_lora_names=lora_names
        )
        cache.take_events()
        self._insert(cache, allocator, [1, 2, 3, 4], extra_key="uuid-a")
        del reqs
        stored = self._stored_events(cache, StorageMedium.GPU)
        self.assertEqual(lora_names.get("uuid-a"), "adapter-a")

        cache.evict(EvictParams(num_tokens=4))
        removed = self._removed_events(cache, StorageMedium.GPU)
        self.assertEqual(self._event_hashes(removed), self._event_hashes(stored))
        self.assertEqual(lora_names.get("uuid-a"), "adapter-a")
        # A late background store needs no new request/name registration.
        self._insert(cache, allocator, [1, 2, 3, 4], extra_key="uuid-a")
        delayed = self._stored_events(cache, StorageMedium.GPU)
        self.assertEqual(delayed[0].lora_name, "adapter-a")
        self.assertEqual(self._event_hashes(delayed), self._event_hashes(stored))

    def test_lora_name_survives_until_queued_stores_are_taken(self):
        """A finished request's blocks still publish under its adapter when
        other requests register before the events are taken."""
        lora_names, reqs = make_lora_names(**{"uuid-a": "adapter-a"})
        cache, allocator, _ = build_fixture(
            self.cfg, enable_kv_cache_events=True, kv_event_lora_names=lora_names
        )
        cache.take_events()
        self._insert(cache, allocator, [1, 2, 3, 4], extra_key="uuid-a")
        del reqs
        for i in range(200):
            lora_names.register(LoraReq(f"other-{i}"), "adapter-b")

        stored = self._stored_events(cache, StorageMedium.GPU)
        self.assertEqual([e.lora_name for e in stored], ["adapter-a"])

    def test_lora_event_parentage_survives_node_split(self):
        lora_names, _reqs = make_lora_names(**{"uuid-a": "adapter-a"})
        cache, allocator, _ = build_fixture(
            self.cfg, enable_kv_cache_events=True, kv_event_lora_names=lora_names
        )
        cache.take_events()

        self._insert(cache, allocator, [1, 2, 3, 4], extra_key="uuid-a")
        original = self._stored_events(cache, StorageMedium.GPU)
        self._insert(cache, allocator, [1, 2, 5, 6], extra_key="uuid-a")
        branch = self._stored_events(cache, StorageMedium.GPU)
        self.assertEqual(len(branch), 1)
        self.assertEqual(branch[0].lora_name, "adapter-a")
        self.assertEqual(branch[0].parent_block_hash, original[0].block_hashes[0])

    def test_kv_events_split_preserves_block_hash_parentage(self):
        cache, allocator, _ = build_fixture(self.cfg, enable_kv_cache_events=True)
        cache.take_events()  # Clear the reset event.

        self._insert(cache, allocator, [1, 2, 3, 4])
        first_insert = self._stored_events(cache, StorageMedium.GPU)
        self.assertEqual(len(first_insert), 1)
        split_parent_hash = first_insert[0].block_hashes[0]

        self._insert(cache, allocator, [1, 2, 5, 6])
        second_insert = self._stored_events(cache, StorageMedium.GPU)
        self.assertEqual(len(second_insert), 1)
        self.assertEqual(list(second_insert[0].token_ids), [5, 6])
        self.assertEqual(second_insert[0].parent_block_hash, split_parent_hash)

        split_parent = node_children(cache, cache.root_node_handle())[0]
        split_child = next(
            (
                child
                for child in node_children(cache, split_parent)
                if node_token_ids(cache, child)[:2] == [3, 4]
            ),
            None,
        )
        self.assertIsNotNone(split_child)
        self.assertEqual(len(cache.tree_core.get_hash_values(split_parent)), 1)
        self.assertEqual(len(cache.tree_core.get_hash_values(split_child)), 1)

    def test_hicache_kv_events_track_gpu_cpu_transitions(self):
        cache, allocator, _ = build_fixture(self.cfg, enable_kv_cache_events=True)
        self._init_hicache(cache)
        cache.take_events()  # Clear reset / init events.

        seq = [1, 2, 3, 4]
        self._insert(cache, allocator, seq)
        stored_gpu = self._stored_events(cache, StorageMedium.GPU)
        self.assertEqual(len(stored_gpu), 1)
        stored_hashes = self._event_hashes(stored_gpu)

        node = self._leaf_for(cache, seq)
        self._backup_node(cache, node)
        stored_cpu = self._stored_events(cache, StorageMedium.CPU)
        self.assertCountEqual(self._event_hashes(stored_cpu), stored_hashes)

        cache.evict(EvictParams(num_tokens=len(seq)))
        removed_gpu = self._removed_events(cache, StorageMedium.GPU)
        self.assertCountEqual(self._event_hashes(removed_gpu), stored_hashes)

        self._load_back_node(cache, node)
        restored_gpu = self._stored_events(cache, StorageMedium.GPU)
        self.assertCountEqual(self._event_hashes(restored_gpu), stored_hashes)

        cache.evict(EvictParams(num_tokens=len(seq)))
        self._removed_events(cache, StorageMedium.GPU)
        cache.evict_host(len(seq))
        removed_cpu = self._removed_events(cache, StorageMedium.CPU)
        self.assertCountEqual(self._event_hashes(removed_cpu), stored_hashes)

    def test_hicache_storage_prefetch_publishes_host_only_suffix(self):
        """A storage-prefetch refill has no write-through ack to publish it."""
        from sglang.srt.mem_cache.utils import get_hash_str, hash_str_to_int64

        cache, _, _ = build_fixture(self.cfg, enable_kv_cache_events=True)
        self._init_hicache(cache)
        cache.take_events()  # Clear reset / init events.

        tokens = [1, 2, 7, 8]
        hash_values = get_hash_str(array("q", tokens), None, page_size=cache.page_size)
        result = cache.tree_core.insert_host(
            cache.root_node_handle(),
            RadixKey(array("q", tokens)),
            torch.tensor([100, 101, 102, 103], dtype=torch.int64),
            hash_values,
        )
        self.assertFalse(result.host_insert_dropped)
        self.assertIsNotNone(result.inserted_host_node)

        # The recorder coalesces the parent-linked pages into one event.
        stored_cpu = self._stored_events(cache, StorageMedium.CPU)
        self.assertEqual(len(stored_cpu), 1)
        self.assertEqual(list(stored_cpu[0].token_ids), tokens)
        self.assertEqual(stored_cpu[0].block_size, cache.page_size)
        self.assertEqual(
            self._event_hashes(stored_cpu),
            [hash_str_to_int64(value) for value in hash_values],
        )
        self.assertIsNone(stored_cpu[0].parent_block_hash)

    def test_hicache_split_pending_write_through_publishes_fragments(self):
        cache, allocator, _ = build_fixture(self.cfg, enable_kv_cache_events=True)
        self._init_hicache(cache)
        cache.take_events()

        self._insert(cache, allocator, [1, 2, 3, 4])
        node = self._leaf_for(cache, [1, 2, 3, 4])
        backed_up = write_backup(cache, node, write_back=True)
        self.assertGreater(backed_up, 0)

        # Split the node while its write-through DMA is still pending.
        self._insert(cache, allocator, [1, 2, 5, 6])
        self.assertEqual(self._stored_events(cache, StorageMedium.CPU), [])

        # Each fragment must also be persisted to L3 on ack: lock_node only
        # holds the suffix after the split.
        cache.enable_storage = True
        with mock.patch.object(cache, "write_backup_storage") as backup_storage:
            cache.writing_check(write_back=True)
        self.assertEqual(
            [
                node_token_ids(cache, call.args[0])
                for call in backup_storage.call_args_list
            ],
            [[1, 2], [3, 4]],
        )

        # Both split fragments must be published as one parent-linked batch.
        stored_cpu = self._stored_events(cache, StorageMedium.CPU)
        self.assertEqual(len(stored_cpu), 1)
        self.assertEqual(list(stored_cpu[0].token_ids), [1, 2, 3, 4])
        self.assertIsNone(stored_cpu[0].parent_block_hash)
        self.assertEqual(len(stored_cpu[0].block_hashes), 2)

    def test_hicache_reinsert_evicted_node_emits_gpu_store(self):
        cache, allocator, _ = build_fixture(self.cfg, enable_kv_cache_events=True)
        self._init_hicache(cache)
        cache.take_events()  # Clear reset / init events.

        seq = [1, 2, 3, 4]
        self._insert(cache, allocator, seq)
        stored_gpu = self._stored_events(cache, StorageMedium.GPU)
        self.assertEqual(len(stored_gpu), 1)
        stored_hashes = self._event_hashes(stored_gpu)

        node = self._leaf_for(cache, seq)
        self._backup_node(cache, node)
        self._stored_events(cache, StorageMedium.CPU)

        cache.evict(EvictParams(num_tokens=len(seq)))
        self._removed_events(cache, StorageMedium.GPU)
        self.assertTrue(cache.tree_core.is_full_device_evicted(node))
        self.assertTrue(cache.tree_core.is_backuped(node))

        self._insert(cache, allocator, seq, session_id="session-a")
        restored_gpu = self._stored_events(cache, StorageMedium.GPU)
        self.assertFalse(cache.tree_core.is_full_device_evicted(node))
        self.assertCountEqual(self._event_hashes(restored_gpu), stored_hashes)
        self.assertEqual(restored_gpu[0].session_id, "session-a")


if __name__ == "__main__":
    unittest.main()
