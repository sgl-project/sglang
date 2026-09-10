"""Unit tests for session_ids propagation in KV cache events.

Covers:
1. BlockStored / BlockRemoved wire-format round-trip with session_ids.
2. KVCacheEventRecorder._collect_session_ids helper.
3. record_store threads session_id from the insert path into emitted events.
4. record_remove falls back to _collect_session_ids(node).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec

from sglang.srt.disaggregation.kv_events import (
    BlockRemoved,
    BlockStored,
    KVEventBatch,
    StorageMedium,
)
from sglang.srt.mem_cache.events import KVCacheEventRecorder
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _MockKey:
    """Minimal stand-in for RadixKey with __len__ support."""

    def __init__(self, token_ids, cache_salt=None):
        self.token_ids = list(token_ids)
        self.is_bigram = False
        self.cache_salt = cache_salt

    def __len__(self):
        return len(self.token_ids)


class _MockNode:
    """Minimal stand-in for one radix tree node."""

    def __init__(self, token_ids, component_data=None, cache_salt=None):
        self.key = _MockKey(token_ids, cache_salt)
        self.parent = None
        self.hash_value = None
        self.event_hash_value = None
        self.component_data = component_data


def _make_node(token_ids, component_data=None, cache_salt=None):
    """Create a mock node with a root-like parent (parent.parent is None)."""
    node = _MockNode(token_ids, component_data, cache_salt)
    # Root-like parent: has empty hash_value and no grandparent.
    node.parent = _MockNode([], cache_salt=None)
    node.parent.hash_value = []
    node.parent.event_hash_value = None
    return node


class TestSessionIdsWireFormat(unittest.TestCase):
    """session_ids survives msgpack round-trip for BlockStored and BlockRemoved."""

    def test_block_stored_round_trip_with_session_ids(self):
        event = BlockStored(
            block_hashes=[123],
            parent_block_hash=None,
            token_ids=[1, 2],
            block_size=2,
            lora_id=None,
            medium=StorageMedium.GPU,
            session_ids=["sess-a", "sess-b"],
        )
        encoded = msgspec.msgpack.encode(event)
        decoded = msgspec.msgpack.decode(encoded, type=BlockStored)
        self.assertEqual(decoded.session_ids, ["sess-a", "sess-b"])

    def test_block_stored_default_session_ids_is_none(self):
        event = BlockStored(
            block_hashes=[123],
            parent_block_hash=None,
            token_ids=[1, 2],
            block_size=2,
            lora_id=None,
        )
        self.assertIsNone(event.session_ids)

    def test_block_removed_round_trip_with_session_ids(self):
        event = BlockRemoved(
            block_hashes=[456],
            medium=StorageMedium.GPU,
            session_ids=["sess-c"],
        )
        encoded = msgspec.msgpack.encode(event)
        decoded = msgspec.msgpack.decode(encoded, type=BlockRemoved)
        self.assertEqual(decoded.session_ids, ["sess-c"])

    def test_block_removed_default_session_ids_is_none(self):
        event = BlockRemoved(block_hashes=[789])
        self.assertIsNone(event.session_ids)

    def test_kv_event_batch_preserves_session_ids(self):
        event = BlockStored(
            block_hashes=[123],
            parent_block_hash=None,
            token_ids=[1, 2],
            block_size=2,
            lora_id=None,
            session_ids=["sess-a"],
        )
        batch = KVEventBatch(ts=1.0, events=[event])
        round_tripped = msgspec.msgpack.decode(
            msgspec.msgpack.encode(batch), type=KVEventBatch
        )
        self.assertEqual(round_tripped.events[0].session_ids, ["sess-a"])


class TestCollectSessionIds(unittest.TestCase):
    """_collect_session_ids gathers the union of session_ids from component_data."""

    def test_no_component_data_returns_none(self):
        node = _make_node([1, 2], component_data=None)
        self.assertIsNone(KVCacheEventRecorder._collect_session_ids(node))

    def test_empty_component_data_list_returns_none(self):
        node = _make_node([1, 2], component_data=[])
        self.assertIsNone(KVCacheEventRecorder._collect_session_ids(node))

    def test_collects_from_single_component(self):
        cd = SimpleNamespace(session_ids={"sess-a", "sess-b"})
        node = _make_node([1, 2], component_data=[cd])
        result = KVCacheEventRecorder._collect_session_ids(node)
        self.assertEqual(set(result), {"sess-a", "sess-b"})

    def test_collects_union_from_multiple_components(self):
        cd1 = SimpleNamespace(session_ids={"sess-a"})
        cd2 = SimpleNamespace(session_ids={"sess-b", "sess-c"})
        cd3 = SimpleNamespace(session_ids=None)
        node = _make_node([1, 2], component_data=[cd1, cd2, cd3])
        result = KVCacheEventRecorder._collect_session_ids(node)
        self.assertEqual(set(result), {"sess-a", "sess-b", "sess-c"})

    def test_all_none_returns_none(self):
        cd1 = SimpleNamespace(session_ids=None)
        cd2 = SimpleNamespace(session_ids=None)
        node = _make_node([1, 2], component_data=[cd1, cd2])
        self.assertIsNone(KVCacheEventRecorder._collect_session_ids(node))


class TestRecordStoreSessionIds(unittest.TestCase):
    """record_store threads session_id from the insert path into events.

    The hash-computation internals (_node_event_hash_values /
    _parent_block_hash) are patched out so the tests run without native
    hash extensions (which require little-endian Linux).
    """

    def setUp(self):
        self.recorder = KVCacheEventRecorder(enabled=True, page_size=2)
        # Stub hash helpers to avoid native hash dependency.
        self._hash_patch = patch.object(
            self.recorder, "_node_event_hash_values", return_value=["a" * 64, "b" * 64]
        )
        self._parent_patch = patch.object(
            self.recorder, "_parent_block_hash", return_value=None
        )
        self._hash_patch.start()
        self._parent_patch.start()

    def tearDown(self):
        self._hash_patch.stop()
        self._parent_patch.stop()

    def test_record_store_with_explicit_session_id(self):
        node = _make_node([1, 2, 3, 4])
        self.recorder.record_store(node, session_id="sess-1")
        events = self.recorder.take()
        self.assertTrue(len(events) > 0)
        for event in events:
            self.assertIsInstance(event, BlockStored)
            self.assertEqual(event.session_ids, ["sess-1"])

    def test_record_store_falls_back_to_collect(self):
        cd = SimpleNamespace(session_ids={"sess-from-component"})
        node = _make_node([1, 2, 3, 4], component_data=[cd])
        self.recorder.record_store(node)
        events = self.recorder.take()
        self.assertTrue(len(events) > 0)
        for event in events:
            self.assertEqual(set(event.session_ids), {"sess-from-component"})

    def test_record_store_no_session_id_no_component_data(self):
        node = _make_node([1, 2], component_data=None)
        self.recorder.record_store(node)
        events = self.recorder.take()
        self.assertTrue(len(events) > 0)
        for event in events:
            self.assertIsNone(event.session_ids)

    def test_record_store_explicit_session_id_takes_precedence(self):
        cd = SimpleNamespace(session_ids={"from-component"})
        node = _make_node([1, 2], component_data=[cd])
        self.recorder.record_store(node, session_id="from-insert-path")
        events = self.recorder.take()
        for event in events:
            self.assertEqual(event.session_ids, ["from-insert-path"])


class TestRecordRemoveSessionIds(unittest.TestCase):
    """record_remove uses _collect_session_ids from node."""

    def setUp(self):
        self.recorder = KVCacheEventRecorder(enabled=True, page_size=2)
        self._hash_patch = patch.object(
            self.recorder, "_node_event_hash_values", return_value=["a" * 64, "b" * 64]
        )
        self._parent_patch = patch.object(
            self.recorder, "_parent_block_hash", return_value=None
        )
        self._hash_patch.start()
        self._parent_patch.start()

    def tearDown(self):
        self._hash_patch.stop()
        self._parent_patch.stop()

    def test_record_remove_collects_session_ids(self):
        cd = SimpleNamespace(session_ids={"sess-remove"})
        node = _make_node([1, 2], component_data=[cd])
        self.recorder.record_remove(node)
        events = self.recorder.take()
        self.assertTrue(len(events) > 0)
        for event in events:
            self.assertIsInstance(event, BlockRemoved)
            self.assertEqual(set(event.session_ids), {"sess-remove"})

    def test_record_remove_no_session_ids(self):
        node = _make_node([1, 2], component_data=None)
        self.recorder.record_remove(node)
        events = self.recorder.take()
        self.assertTrue(len(events) > 0)
        for event in events:
            self.assertIsNone(event.session_ids)


class TestDisabledRecorder(unittest.TestCase):
    """Disabled recorder is a no-op for all record_* methods."""

    def test_disabled_record_store(self):
        recorder = KVCacheEventRecorder(enabled=False, page_size=2)
        node = _make_node([1, 2])
        recorder.record_store(node, session_id="sess-1")
        self.assertEqual(recorder.take(), [])

    def test_disabled_record_remove(self):
        recorder = KVCacheEventRecorder(enabled=False, page_size=2)
        node = _make_node([1, 2])
        recorder.record_remove(node)
        self.assertEqual(recorder.take(), [])


if __name__ == "__main__":
    unittest.main()
