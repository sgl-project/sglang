"""CPU coverage for per-block namespace metadata in KV placement events."""

import unittest
from array import array

import msgspec

from sglang.srt.disaggregation.kv_events import BlockStored, StorageMedium
from sglang.srt.mem_cache.events import KVCacheEventRecorder, cache_salt_extra_keys
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestKVCacheEventMetadata(CustomTestCase):
    @staticmethod
    def _node(tokens, *, salt=None, parent=None):
        if parent is None:
            parent = TreeNode()
            parent.key = RadixKey(array("q"))
            parent.hash_value = []
        node = TreeNode()
        node.parent = parent
        node.key = RadixKey(array("q", tokens), cache_salt=salt)
        return node

    @staticmethod
    def _store(hashes, *, parent=None, extras=None, salt="tenant-a", lora_name=None):
        return BlockStored(
            block_hashes=hashes,
            parent_block_hash=parent,
            token_ids=[1, 2] * len(hashes),
            block_size=2,
            lora_id=None,
            medium=StorageMedium.GPU,
            lora_name=lora_name,
            extra_keys=extras,
            cache_salt=salt,
        )

    def test_salt_only_appears_on_the_first_block_of_the_prefix(self):
        for medium in (StorageMedium.GPU, StorageMedium.CPU):
            with self.subTest(medium=medium):
                recorder = KVCacheEventRecorder(enabled=True, page_size=2)
                first = self._node([1, 2, 3, 4], salt="tenant-a")
                recorder.record_store(first, medium=medium)
                event = recorder.take()[0]
                self.assertEqual(event.extra_keys, [("tenant-a",), None])
                self.assertEqual(len(event.extra_keys), len(event.block_hashes))
                self.assertEqual(event.cache_salt, "tenant-a")
                self.assertIsNone(event.lora_name)

                child = self._node([5, 6, 7, 8], salt="tenant-a", parent=first)
                recorder.record_store(child, medium=medium)
                continuation = recorder.take()[0]
                self.assertEqual(continuation.parent_block_hash, event.block_hashes[-1])
                self.assertIsNone(continuation.extra_keys)
                self.assertEqual(continuation.cache_salt, "tenant-a")

    def test_partial_blocks_keep_extras_aligned(self):
        recorder = KVCacheEventRecorder(enabled=True, page_size=2)
        recorder.record_store(self._node([1, 2, 3], salt="tenant-a"))
        events = recorder.take()
        self.assertEqual([e.block_size for e in events], [2, 1])
        self.assertEqual(events[0].extra_keys, [("tenant-a",)])
        self.assertIsNone(events[1].extra_keys)
        self.assertEqual(events[1].parent_block_hash, events[0].block_hashes[-1])

    def test_unsalted_events_omit_extra_keys(self):
        recorder = KVCacheEventRecorder(enabled=True, page_size=2)
        recorder.record_store(self._node([1, 2, 3, 4]))
        event = recorder.take()[0]
        self.assertIsNone(event.extra_keys)
        wire = msgspec.msgpack.decode(msgspec.msgpack.encode(event))
        self.assertNotIn("extra_keys", wire)
        self.assertNotIn("cache_salt", wire)

    def test_merge_extends_extras_with_a_placeholder_for_each_block(self):
        recorder = KVCacheEventRecorder(enabled=True, page_size=2)
        recorder.enqueue(self._store([1], extras=[("tenant-a",)]))
        recorder.enqueue(self._store([2, 3], parent=1))
        event = recorder.take()[0]
        self.assertEqual(event.block_hashes, [1, 2, 3])
        self.assertEqual(event.extra_keys, [("tenant-a",), None, None])

    def test_merge_preserves_incoming_extras_and_backfills_missing_entries(self):
        recorder = KVCacheEventRecorder(enabled=True, page_size=2)
        recorder.enqueue(self._store([1, 2]))
        recorder.enqueue(self._store([3], parent=2, extras=[("metadata",)]))
        event = recorder.take()[0]
        self.assertEqual(event.block_hashes, [1, 2, 3])
        self.assertEqual(event.extra_keys, [None, None, ("metadata",)])

    def test_different_salts_and_lora_names_are_not_merged(self):
        for incoming in (
            self._store([2], parent=1, salt="tenant-b"),
            self._store([2], parent=1, lora_name="adapter-a"),
        ):
            recorder = KVCacheEventRecorder(enabled=True, page_size=2)
            recorder.enqueue(self._store([1], extras=[("tenant-a",)]))
            recorder.enqueue(incoming)
            self.assertEqual(len(recorder.take()), 2)

    def test_root_metadata_handles_empty_and_unicode_salts(self):
        cases = [
            (None, None, 2, None),
            (None, "", 2, None),
            (None, "tenant-a", 0, None),
            (0, "tenant-a", 2, None),
            (None, "tenant-\u03b1", 2, [("tenant-\u03b1",), None]),
        ]
        for parent, salt, count, expected in cases:
            with self.subTest(parent=parent, salt=salt, count=count):
                self.assertEqual(
                    cache_salt_extra_keys(
                        parent_block_hash=parent, cache_salt=salt, num_blocks=count
                    ),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
