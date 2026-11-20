"""Backend-owned residency events are kept apart from framework residency."""

import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec
import torch

from sglang.srt.disaggregation.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    StorageMedium,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.events import KVCacheEventRecorder, _StorageEventSpan
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageFailureInjector,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
    PrefetchOperation,
    StorageOperation,
    _storage_read_extra,
    _storage_write_extra,
)
from sglang.srt.mem_cache.storage_events import StorageEventReducer
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def stored(block=17, parent=None, ownership="kvcr"):
    return BlockStored(
        block_hashes=[block],
        parent_block_hash=parent,
        token_ids=[1, 2],
        block_size=2,
        lora_id=None,
        medium=StorageMedium.CPU,
        ownership=ownership,
    )


def removed(block=17):
    return BlockRemoved(
        block_hashes=[block], medium=StorageMedium.CPU, ownership="kvcr"
    )


class _Tokens(list):
    """The RadixKey surface record_store and the span rebuild read."""

    def __init__(self, tokens, extra_key=None, cache_salt=None, is_bigram=False):
        super().__init__(tokens)
        self.token_ids = self
        self.extra_key, self.cache_salt = extra_key, cache_salt
        self.is_bigram = is_bigram

    def raw_token_ids(self):
        return self


class TestStorageEventReducer(CustomTestCase):
    def test_all_shards_store_any_shard_removes(self):
        reducer = StorageEventReducer(2)
        self.assertEqual(reducer.update({0: [stored()], 1: []}), [])
        self.assertEqual(reducer.update({0: [], 1: [stored()]}), [stored()])
        self.assertEqual(reducer.update({0: [removed()], 1: []}), [removed()])
        self.assertEqual(reducer.update({0: [], 1: [stored()]}), [])
        self.assertEqual(reducer.update({0: [stored()], 1: []}), [stored()])
        with self.assertRaisesRegex(ValueError, "every attention shard"):
            reducer.update({0: []})

    def test_stores_are_ordered_parent_before_child(self):
        reducer = StorageEventReducer(1)
        parent, child = stored(99), stored(1, parent=99)
        self.assertEqual(reducer.update({0: [child, parent]}), [parent, child])

    def test_disagreeing_shards_withhold_the_page_until_dropped(self):
        """Fails closed on the scheduler thread instead of raising."""
        reducer = StorageEventReducer(2)
        other = stored()
        other.token_ids = [3, 4]
        with self.assertLogs("sglang.srt.mem_cache.storage_events", "WARNING"):
            self.assertEqual(reducer.update({0: [stored()], 1: [other]}), [])
        # Withheld while any shard still holds it, even once both agree.
        self.assertEqual(reducer.update({0: [], 1: [removed(), stored()]}), [])
        self.assertEqual(reducer.update({0: [removed()], 1: [removed()]}), [])
        # Dropped everywhere: a fresh identity publishes normally.
        self.assertEqual(reducer.update({0: [stored()], 1: [stored()]}), [stored()])
        # A disagreement on a published page withdraws it.
        with self.assertLogs("sglang.srt.mem_cache.storage_events", "WARNING"):
            reducer._warned = False
            self.assertEqual(reducer.update({0: [other], 1: []}), [removed()])

    def test_drops_multi_page_stores_and_other_events(self):
        wide = stored()
        wide.block_hashes = [17, 18]
        for event in (wide, AllBlocksCleared()):
            with (
                self.subTest(event=type(event).__name__),
                self.assertLogs("sglang.srt.mem_cache.storage_events", "WARNING"),
            ):
                reducer = StorageEventReducer(1)
                self.assertEqual(reducer.update({0: [event, stored(5)]}), [stored(5)])

    def test_wire_coalescing_cannot_change_removal_identity(self):
        reducer = StorageEventReducer(1)
        recorder = KVCacheEventRecorder(enabled=True, page_size=2)
        for event in reducer.update({0: [stored(17), stored(18, parent=17)]}):
            recorder.enqueue(event)
        self.assertEqual(recorder.take()[0].block_hashes, [17, 18])
        removals = reducer.update({0: [removed(17)]})
        self.assertEqual([event.block_hashes for event in removals], [[17]])


class TestOwnershipOnTheWire(CustomTestCase):
    def test_events_from_different_owners_never_coalesce(self):
        recorder = KVCacheEventRecorder(enabled=True, page_size=2)
        recorder.enqueue(stored(17, ownership=None))
        recorder.enqueue(stored(18, parent=17))
        recorder.enqueue(removed(18))
        recorder.enqueue(BlockRemoved(block_hashes=[17], medium=StorageMedium.CPU))
        self.assertEqual(
            [event.ownership for event in recorder.take()],
            [None, "kvcr", "kvcr", None],
        )

    def test_framework_events_omit_ownership(self):
        """omit_defaults keeps the framework's encoding unchanged."""
        for event in (
            stored(ownership=None),
            BlockRemoved(block_hashes=[1]),
            AllBlocksCleared(),
        ):
            with self.subTest(event=type(event).__name__):
                self.assertNotIn(b"ownership", msgspec.msgpack.encode(event))
        owned = msgspec.msgpack.decode(msgspec.msgpack.encode(stored()))
        self.assertEqual(owned["ownership"], "kvcr")


class TestStorageEventMetadata(CustomTestCase):
    def test_metadata_splits_pages_and_drops_attribution(self):
        coalesced = BlockStored(
            block_hashes=[5, 6],
            parent_block_hash=4,
            token_ids=[1, 2, 3, 4],
            block_size=2,
            lora_id=None,
            medium=StorageMedium.CPU,
            cache_salt="salt",
            session_id="session",
            lora_name="adapter",
            mm_routing_spans=[[(0, 1, "x")], []],
        )
        metadata = KVCacheEventRecorder.metadata_for_events(["a", "b"], [coalesced])
        self.assertEqual(list(metadata), ["a", "b"])
        first, second = metadata["a"], metadata["b"]
        self.assertEqual((first.block_hashes, first.parent_block_hash), ([5], 4))
        self.assertEqual((second.block_hashes, second.parent_block_hash), ([6], 5))
        self.assertEqual((first.token_ids, second.token_ids), ([1, 2], [3, 4]))
        for page in (first, second):
            self.assertIsNone(page.session_id)
            self.assertIsNone(page.mm_routing_spans)
            self.assertIsNone(page.ownership)
            self.assertEqual((page.cache_salt, page.lora_name), ("salt", "adapter"))

    def test_disabled_recorder_describes_pages_without_publishing(self):
        """A non-publishing attention shard still describes its own pages."""
        recorder = KVCacheEventRecorder(enabled=False, page_size=2)
        pages = ["0a" * 32, "0b" * 32]
        node = _StorageEventSpan(_Tokens([1, 2, 3, 4]), hash_value=pages)
        metadata = recorder.storage_metadata(node)
        self.assertEqual(list(metadata), pages)
        first, second = (metadata[page] for page in pages)
        self.assertEqual(first.block_hashes, [int("0a" * 8, 16)])
        self.assertEqual(second.parent_block_hash, first.block_hashes[0])
        self.assertEqual((first.token_ids, second.token_ids), ([1, 2], [3, 4]))
        self.assertEqual(first.medium, StorageMedium.CPU)
        self.assertEqual(recorder.take(), [])
        self.assertFalse(recorder.enabled)

    def test_snapshot_never_enables_multimodal_label_registry(self):
        """Only a publishing dynamo-format recorder may fill the registry."""
        from sglang.srt.mem_cache import events

        enabled = events.MM_ROUTING_LABELS.enabled
        self.addCleanup(setattr, events.MM_ROUTING_LABELS, "enabled", enabled)
        events.MM_ROUTING_LABELS.enabled = False
        recorder = KVCacheEventRecorder(enabled=False, page_size=2, dynamo_format=True)
        node = _StorageEventSpan(_Tokens([1, 2]), hash_value=["0a" * 32])
        self.assertEqual(len(recorder.storage_metadata(node)), 1)
        self.assertFalse(events.MM_ROUTING_LABELS.enabled)

    def test_unverified_prefetch_prefix_describes_nothing(self):
        recorder = KVCacheEventRecorder(enabled=True, page_size=2)
        tail = _Tokens([3, 4])
        # A partial page, or a last hash without its prefix, cannot be verified.
        self.assertEqual(recorder.prefetch_storage_metadata(tail, [1], None), {})
        self.assertEqual(recorder.prefetch_storage_metadata(tail, [], "h"), {})
        self.assertEqual(recorder.take(), [])

    @unittest.skipUnless(
        sys.platform == "linux", "HiCache native hasher requires Linux"
    )
    def test_prefetch_metadata_matches_backup_metadata(self):
        recorder = KVCacheEventRecorder(enabled=False, page_size=2, dynamo_format=True)
        for salt in (None, "tenant-a"):
            with self.subTest(salt=salt):
                key = _Tokens([1, 2, 3, 4], extra_key="adapter", cache_salt=salt)
                complete = recorder.storage_metadata(_StorageEventSpan(key))
                hashes = get_storage_hash_str(key, page_size=2)
                self.assertEqual(list(complete), hashes)
                tail = _Tokens([3, 4], extra_key="adapter", cache_salt=salt)
                suffix = recorder.prefetch_storage_metadata(tail, [1, 2], hashes[0])
                self.assertEqual(suffix, {hashes[1]: complete[hashes[1]]})
                self.assertEqual(
                    recorder.prefetch_storage_metadata(tail, [1, 2], "wrong"), {}
                )
                self.assertEqual(recorder.take(), [])


class _RecordingBackend(HiCacheStorage):
    """Records the extra_info of KV writes; publishes per ``publishes``."""

    def __init__(self, publishes):
        self.publishes_kv_events = publishes
        self.writes = []

    def batch_set_v1(self, keys, host_indices, extra_info=None):
        self.writes.append(extra_info)
        return [True] * len(keys)

    # The abstract single-key API is unused here.
    def get(self, *args, **kwargs):
        raise NotImplementedError

    def batch_get(self, *args, **kwargs):
        raise NotImplementedError

    def set(self, *args, **kwargs):
        raise NotImplementedError

    def batch_set(self, *args, **kwargs):
        raise NotImplementedError

    def exists(self, *args, **kwargs):
        raise NotImplementedError


class TestKvEventMetadataPlumbing(CustomTestCase):
    def test_write_and_read_extra_carry_metadata_only_when_given(self):
        metadata = {"p0": stored()}
        write = StorageOperation(torch.arange(2), [1, 2], hash_value=["p0"])
        self.assertNotIn("kv_event_metadata", _storage_write_extra(write) or {})
        write = StorageOperation(
            torch.arange(2), [1, 2], hash_value=["p0"], kv_event_metadata=metadata
        )
        self.assertIs(_storage_write_extra(write)["kv_event_metadata"], metadata)

        handle = CacheRequestHandle("r", 0)
        read = PrefetchOperation(handle, [1, 2])
        self.assertNotIn("kv_event_metadata", _storage_read_extra(read) or {})
        read = PrefetchOperation(
            handle,
            [1, 2],
            storage_read_metadata={"kv_hints": "hints"},
            kv_event_metadata=metadata,
        )
        extra = _storage_read_extra(read)
        # A separate field: hint gating never touches it.
        self.assertIs(extra["kv_event_metadata"], metadata)
        self.assertEqual(extra["kv_hints"], "hints")

    def test_backup_write_hands_metadata_to_the_backend(self):
        for publishes in (False, True):
            with self.subTest(publishes=publishes):
                backend = _RecordingBackend(publishes)
                controller = object.__new__(HybridCacheController)
                controller.storage_backend = backend
                controller.storage_stop_event = threading.Event()
                controller.page_size = 2
                controller.page_set_func = lambda keys, indices, extra: all(
                    backend.batch_set_v1(keys, indices, extra)
                )
                metadata = {"p0": stored(17), "p1": stored(18, parent=17)}
                operation = StorageOperation(
                    torch.arange(4),
                    [1, 2, 3, 4],
                    hash_value=["p0", "p1"],
                    kv_event_metadata=metadata if publishes else None,
                )
                controller._page_backup_kv_with_trace(operation)
                self.assertEqual(operation.completed_tokens, 4)
                (extra,) = backend.writes
                carried = (extra.extra_info or {}).get("kv_event_metadata")
                self.assertIs(carried, metadata if publishes else None)

    def test_failure_injector_delegates_every_event_hook(self):
        backend = MagicMock(spec=HiCacheStorage)
        backend.publishes_kv_events = True
        backend.kv_event_shard = 3
        backend.kv_event_shard_count = 4
        backend.pending_kv_events.return_value = 2
        backend.take_kv_events.return_value = [stored()]
        backend.get_kv_hint_source.return_value = {"dp_rank": 0}
        wrapped = HiCacheStorageFailureInjector(backend, 0.0, 0.0)
        self.assertIs(wrapped.publishes_kv_events, True)
        self.assertEqual((wrapped.kv_event_shard, wrapped.kv_event_shard_count), (3, 4))
        self.assertEqual(wrapped.pending_kv_events(), 2)
        self.assertEqual(wrapped.take_kv_events(), [stored()])
        self.assertEqual(wrapped.get_kv_hint_source(), {"dp_rank": 0})

    def test_default_backend_publishes_nothing(self):
        backend = _RecordingBackend(False)
        del backend.publishes_kv_events
        self.assertFalse(backend.publishes_kv_events)
        self.assertEqual((backend.kv_event_shard, backend.kv_event_shard_count), (0, 1))
        self.assertEqual(backend.pending_kv_events(), 0)
        self.assertEqual(backend.take_kv_events(), [])
        self.assertIsNone(backend.get_kv_hint_source())


class TestPrefixCacheHintSources(CustomTestCase):
    def test_reports_the_backend_source_or_nothing(self):
        from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache

        cache = SimpleNamespace(cache_controller=None)
        sources = BasePrefixCache.storage_kv_hint_sources
        self.assertEqual(sources(cache), [])
        source = {"dp_rank": 1, "capabilities": ["router_hint"]}
        for reported, expected in ((None, []), (source, [source])):
            cache.cache_controller = SimpleNamespace(
                storage_backend=SimpleNamespace(get_kv_hint_source=lambda r=reported: r)
            )
            self.assertEqual(sources(cache), expected)


class TestUnifiedRadixCacheEventPlumbing(CustomTestCase):
    def _cache(self, backend):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(
            enable_storage=True,
            build_storage_backup_spec=lambda node_id, pass_prefix: SimpleNamespace(
                host_value=torch.arange(2),
                hash_value=["p0"],
                token_ids=[1, 2],
                prefix_keys=None,
                comp_xfers={},
            ),
            storage_event_metadata=lambda node_id: {"p0": node_id},
        )
        cache.cache_controller = MagicMock(
            storage_backend=backend, storage_config=SimpleNamespace(dcp_size=1)
        )
        cache.cache_controller.write_storage.return_value = 11
        # init_hicache builds the reducer for a launch-time publisher only.
        cache._storage_event_reducer = (
            StorageEventReducer(1) if backend.publishes_kv_events else None
        )
        cache.hicache_storage_pass_prefix_keys = False
        cache._build_sidecar_transfers = lambda *args: []
        cache.inc_host_lock_ref = MagicMock()
        cache.ongoing_backup = {}
        cache._backup_requested_tokens = {}
        cache.page_size = 2
        return cache

    def test_backup_carries_metadata_only_for_a_publishing_backend(self):
        for publishes in (False, True):
            with self.subTest(publishes=publishes):
                cache = self._cache(_RecordingBackend(publishes))
                self.assertIs(cache.storage_kv_events_enabled, publishes)
                cache.write_backup_storage(5)
                kwargs = cache.cache_controller.write_storage.call_args.kwargs
                if publishes:
                    self.assertEqual(kwargs["kv_event_metadata"], {"p0": 5})
                else:
                    self.assertNotIn("kv_event_metadata", kwargs)

    def test_publishing_is_fixed_at_launch_and_needs_pp1_without_dcp(self):
        for publishes in (False, True):
            with self.subTest(publishes=publishes):
                cache = self._cache(_RecordingBackend(publishes))
                del cache._storage_event_reducer
                cache.pp_size = 1
                cache._init_storage_kv_events()
                self.assertIs(cache.storage_kv_events_enabled, publishes)
                # Gone with the backend, e.g. after the shutdown detach.
                cache.cache_controller.storage_backend = None
                self.assertFalse(cache.storage_kv_events_enabled)
        for pp_size, dcp_size in ((2, 1), (1, 2)):
            with (
                self.subTest(pp_size=pp_size, dcp_size=dcp_size),
                self.assertRaisesRegex(ValueError, "requires --pp-size 1"),
            ):
                cache = self._cache(_RecordingBackend(True))
                cache.pp_size = pp_size
                cache.cache_controller.storage_config.dcp_size = dcp_size
                cache._init_storage_kv_events()

    def test_publishing_backend_rejects_runtime_attach_and_detach(self):
        from sglang.srt.mem_cache.unified_cache import storage_attachment as module

        publishing = _RecordingBackend(True)
        controller = MagicMock(storage_backend=None, write_policy="write_through")

        def attach(**kwargs):
            controller.storage_backend = publishing

        controller.attach_storage_backend.side_effect = attach
        cache = SimpleNamespace(
            cache_controller=controller,
            host_memory_mode="cache",
            token_to_kv_pool_allocator=None,
            is_write_back=False,
            enable_storage=False,
            prefetch_stop_policy="best_effort",
            write_through_threshold=1,
            storage_kv_events_enabled=False,
        )
        attachment = module.StorageAttachment(cache)
        attachment.apply_runtime_config = MagicMock()
        with patch.object(module, "page_interleave_shard_size", return_value=1):
            ok, message = attachment.attach("dynamic", "{}")
        self.assertFalse(ok)
        self.assertIn("publishes KV events", message)
        controller.detach_storage_backend.assert_called_once_with()
        attachment.apply_runtime_config.assert_not_called()
        # A failed rollback is reported, not swallowed.
        controller.detach_storage_backend.side_effect = RuntimeError("busy")
        with (
            patch.object(module, "page_interleave_shard_size", return_value=1),
            self.assertLogs(module.logger, "ERROR"),
        ):
            ok, message = attachment.attach("dynamic", "{}")
        self.assertFalse(ok)
        self.assertIn("still attached", message)
        controller.detach_storage_backend.side_effect = None

        # A launch-time publisher is detached only at shutdown.
        controller.detach_storage_backend.reset_mock()
        controller.storage_backend = publishing
        cache.storage_kv_events_enabled = True
        cache.drain_storage_control_queues_local = MagicMock()
        ok, message = attachment.detach()
        self.assertFalse(ok)
        controller.detach_storage_backend.assert_not_called()
        attachment._release_pending_storage_ops = MagicMock()
        ok, _ = attachment.detach(shutdown=True)
        self.assertTrue(ok)
        controller.detach_storage_backend.assert_called_once_with()


@unittest.skipUnless(sys.platform == "linux", "Rust TreeCore extension requires Linux")
class TestRustRecorderQueue(CustomTestCase):
    def test_take_appends_enqueued_storage_events_after_tree_events(self):
        from sglang.srt.mem_cache.events import KvEventLoraNames
        from sglang.srt.mem_cache.rust_tree_core.adapter import (
            _RustKVCacheEventRecorder,
        )

        binding = SimpleNamespace(take_events=lambda: [("all_blocks_cleared",)])
        recorder = _RustKVCacheEventRecorder(
            binding, enabled=True, page_size=2, lora_names=KvEventLoraNames()
        )
        self.assertIsInstance(recorder, KVCacheEventRecorder)
        recorder.enqueue(stored())
        self.assertEqual(recorder.take(), [AllBlocksCleared(), stored()])


if __name__ == "__main__":
    unittest.main()
