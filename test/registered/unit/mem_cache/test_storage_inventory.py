"""Backend residency is independent of framework CPU/GPU residency."""

import sys
import unittest

from sglang.srt.disaggregation.kv_events import BlockRemoved, BlockStored, StorageMedium
from sglang.srt.mem_cache.events import KVCacheEventRecorder, _StorageEventSpan
from sglang.srt.mem_cache.storage_events import StorageEventReducer
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def stored(block=17, parent=None):
    return BlockStored(
        block_hashes=[block],
        parent_block_hash=parent,
        token_ids=[1, 2],
        block_size=2,
        lora_id=None,
        medium=StorageMedium.CPU,
        ownership="storage",
    )


class TestStorageInventory(CustomTestCase):
    @unittest.skipUnless(
        sys.platform == "linux", "HiCache native hasher requires Linux"
    )
    def test_prefetch_metadata_preserves_salted_storage_and_router_identities(self):
        class Tokens(list):
            def __init__(
                self, tokens, extra_key=None, cache_salt=None, is_bigram=False
            ):
                super().__init__(tokens)
                self.token_ids = self
                self.extra_key, self.cache_salt = extra_key, cache_salt
                self.is_bigram = is_bigram

            def raw_token_ids(self):
                return self

        # A non-publishing TP rank must still describe its physical shard.
        recorder = KVCacheEventRecorder(enabled=False, page_size=2, dynamo_format=True)
        for salt in (None, "tenant-a"):
            key = Tokens([1, 2, 3, 4], extra_key="adapter", cache_salt=salt)
            node = _StorageEventSpan(key)
            complete = recorder.storage_metadata(node)
            hashes = get_storage_hash_str(key, page_size=2)
            tail = Tokens([3, 4], extra_key="adapter", cache_salt=salt)
            suffix = recorder.prefetch_storage_metadata(tail, [1, 2], hashes[0])
            self.assertEqual(suffix, {hashes[1]: complete[hashes[1]]})
            self.assertEqual(recorder.take(), [])
            self.assertFalse(recorder.enabled)
            self.assertEqual(
                recorder.prefetch_storage_metadata(tail, [1, 2], "wrong"), {}
            )

    def test_tp_all_shards_store_any_shard_removes(self):
        reducer = StorageEventReducer(2)
        self.assertEqual(reducer.update({0: [stored()], 1: []}), [])
        self.assertEqual(reducer.update({0: [], 1: [stored()]}), [stored()])
        removal = BlockRemoved(
            block_hashes=[17], medium=StorageMedium.CPU, ownership="storage"
        )
        self.assertEqual(reducer.update({0: [removal], 1: []}), [removal])
        self.assertEqual(reducer.update({0: [], 1: [stored()]}), [])
        self.assertEqual(reducer.update({0: [stored()], 1: []}), [stored()])
        with self.assertRaisesRegex(ValueError, "every attention shard"):
            reducer.update({0: []})

    def test_stores_are_ordered_parent_before_child(self):
        reducer = StorageEventReducer(1)
        parent, child = stored(99), stored(1, parent=99)
        self.assertEqual(reducer.update({0: [child, parent]}), [parent, child])

    def test_cpu_events_from_different_owners_never_coalesce(self):
        recorder = KVCacheEventRecorder(enabled=True, page_size=2)
        framework = stored(17)
        framework.ownership = None
        recorder.enqueue(framework)
        recorder.enqueue(stored(18, parent=17))
        self.assertEqual(len(recorder.take()), 2)

    def test_wire_coalescing_cannot_change_inventory_removal_identity(self):
        reducer = StorageEventReducer(1)
        recorder = KVCacheEventRecorder(enabled=True, page_size=2)
        for event in reducer.update({0: [stored(17), stored(18, 17)]}):
            recorder.enqueue(event)
        self.assertEqual(recorder.take()[0].block_hashes, [17, 18])
        removals = reducer.update(
            {
                0: [
                    BlockRemoved(
                        block_hashes=[17], medium=StorageMedium.CPU, ownership="storage"
                    )
                ]
            }
        )
        self.assertEqual([event.block_hashes for event in removals], [[17]])


if __name__ == "__main__":
    unittest.main()
