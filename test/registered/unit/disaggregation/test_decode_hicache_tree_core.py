"""Unit tests for decode HiCache TreeCore interactions."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
)
from sglang.srt.managers.kv_hints import KvHintAction, KvHintsEnvelope
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    CacheRequestHandle,
)
from sglang.srt.mem_cache.events import KVCacheEventRecorder, KvEventLoraNames
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorage,
    HiCacheStorageFailureInjector,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=9, suite="base-a-test-cpu")


_MASK = (1 << 64) - 1
_LORA_ID = "a" * 32
_HINT = KvHintsEnvelope(
    protocol_version="0.1",
    message_id="decode-probe",
    actions=[
        KvHintAction(action_id="fetch", action_type="kv.fetch", action_version="1.0")
    ],
)
_EMPTY_HINT = KvHintsEnvelope(protocol_version="0.1", message_id="no-actions")


def _lora_names(extra_key):
    names = KvEventLoraNames()
    if extra_key is not None:
        names.register(SimpleNamespace(extra_key=extra_key, lora_id=_LORA_ID), "lora")
    return names


class _Backend(HiCacheStorage):
    """A storage backend whose L3 lookups go to ``batch_exists``."""

    def __init__(self, batch_exists, consumes_kv_hints=None):
        self.batch_exists = batch_exists
        self._consumes_kv_hints = consumes_kv_hints

    def consumes_kv_hints(self) -> bool:
        if self._consumes_kv_hints is None:
            return super().consumes_kv_hints()
        return self._consumes_kv_hints

    def _unused(self, *args, **kwargs):
        raise NotImplementedError

    get = batch_get = set = batch_set = exists = _unused


def _decode_probe(page_size, kv_events, batch_exists, consumes_kv_hints=True):
    """A decode-side cache whose L3 lookups go to ``batch_exists``."""
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.tree_core = SimpleNamespace(
        enable_storage=True, is_eagle=False, kv_events=kv_events
    )
    cache.page_size = cache.prefetch_threshold = page_size
    cache._all_reduce_attn_groups = Mock()
    cache.prefix_device_indices = lambda req: torch.arange(page_size)
    cache.hicache_storage_pass_prefix_keys = False
    cache.is_backuped = lambda node: True

    controller = HybridCacheController.__new__(HybridCacheController)
    controller.page_size = page_size
    controller.prefetch_rate_limited = lambda: False
    controller.storage_backend = _Backend(batch_exists, consumes_kv_hints)
    cache.cache_controller = controller
    harness = SimpleNamespace(
        tree_cache=cache, scheduler=SimpleNamespace(enable_decode_hicache=True)
    )
    match = SimpleNamespace(host_hit_length=0, last_device_node=1, last_host_node=1)
    return cache, lambda req: DecodeHiCachePreallocMixin._build_decode_prefix_match(
        harness, req, match
    )


class TestDecodeHiCacheTreeCore(CustomTestCase):
    def test_decode_probe_binds_published_ids_on_a_nonpublishing_rank(self):
        """A rank that publishes no events binds hinted keys to the published IDs."""
        page_size = 4
        tokens = array("q", range(3 * page_size))
        cases = [
            (dynamo_format, salt, extra_key)
            for dynamo_format in (True, False)
            for salt, extra_key in (
                ("tenant", None),
                (None, "user-" + _LORA_ID),
                ("tenant", "user-" + _LORA_ID),
            )
        ]
        for dynamo_format, salt, extra_key in cases:
            with self.subTest(dynamo_format=dynamo_format, salt=salt, lora=extra_key):
                key = RadixKey(tokens, extra_key=extra_key, cache_salt=salt)
                keys = get_storage_hash_str(key, page_size=page_size)
                publisher = KVCacheEventRecorder(
                    enabled=True,
                    page_size=page_size,
                    dynamo_format=dynamo_format,
                    lora_names=_lora_names(extra_key),
                )
                publisher.record_store(
                    SimpleNamespace(
                        key=key, parent=None, hash_value=keys, event_hash_value=None
                    )
                )
                event_ids = publisher.take()[0].block_hashes[1:]
                expected = {k: v & _MASK for k, v in zip(keys[1:], event_ids)}

                def batch_exists(page_keys, info):
                    fields = info.extra_info
                    if fields.get("kv_hints") != _HINT:
                        return 0
                    binding = fields.get("storage_key_to_event_hash", {})
                    return sum(
                        page in binding and binding[page] == expected[page]
                        for page in page_keys
                    )

                follower = KVCacheEventRecorder(
                    enabled=False,
                    page_size=page_size,
                    dynamo_format=dynamo_format,
                    lora_names=_lora_names(extra_key),
                )
                cache, probe = _decode_probe(page_size, follower, batch_exists)
                req = SimpleNamespace(
                    origin_input_ids=tokens,
                    extra_key=extra_key,
                    cache_salt=salt,
                    kv_hints=_HINT,
                )
                cache.get_last_hash_value = lambda node: keys[0]
                self.assertEqual(probe(req).l3_storage_hit_length, 2 * page_size)

                cache.get_last_hash_value = lambda node: "00" * 32
                self.assertEqual(probe(req).l3_storage_hit_length, 0)

    def test_default_format_binding_needs_the_lora_name_on_every_rank(self):
        """Without the registered name, a follower's IDs miss a LoRA hint."""
        page_size = 4
        extra_key = "user-" + _LORA_ID
        key = RadixKey(array("q", range(2 * page_size)), extra_key=extra_key)
        keys = get_storage_hash_str(key, page_size=page_size)
        publisher = KVCacheEventRecorder(
            enabled=True, page_size=page_size, lora_names=_lora_names(extra_key)
        )
        publisher.record_store(
            SimpleNamespace(
                key=key, parent=None, hash_value=keys, event_hash_value=None
            )
        )
        published = publisher.take()[0].block_hashes[1] & _MASK
        for registered in (True, False):
            with self.subTest(registered=registered):
                follower = KVCacheEventRecorder(
                    enabled=False,
                    page_size=page_size,
                    lora_names=_lora_names(extra_key if registered else None),
                )
                bound = []
                cache, probe = _decode_probe(
                    page_size,
                    follower,
                    lambda page_keys, info: (
                        bound.append(info.extra_info["storage_key_to_event_hash"]) or 0
                    ),
                )
                cache.get_last_hash_value = lambda node: keys[0]
                probe(
                    SimpleNamespace(
                        origin_input_ids=key.raw_token_ids(),
                        extra_key=extra_key,
                        cache_salt=None,
                        kv_hints=_HINT,
                    )
                )
                self.assertEqual(bound[0][keys[1]] == published, registered)

    def test_decode_probe_skips_hint_metadata_unless_a_backend_reads_it(self):
        """Hints reach only opted-in backends, and only when they request work."""
        page_size = 4
        tokens = array("q", range(2 * page_size))
        cases = (
            ("default backend", None, _HINT),
            ("backend opted out", False, _HINT),
            ("hint has no actions", True, _EMPTY_HINT),
        )
        for name, consumes_kv_hints, hint in cases:
            with self.subTest(name):
                seen = []
                cache, probe = _decode_probe(
                    page_size,
                    KVCacheEventRecorder(enabled=False, page_size=page_size),
                    lambda page_keys, info: seen.append(info.extra_info) or 0,
                    consumes_kv_hints=consumes_kv_hints,
                )
                cache.get_last_hash_value = lambda node: "00" * 32
                with patch(
                    "sglang.srt.mem_cache.unified_radix_cache.storage_key_to_event_hashes"
                ) as bind:
                    probe(
                        SimpleNamespace(
                            origin_input_ids=tokens,
                            extra_key=None,
                            cache_salt="tenant",
                            kv_hints=hint,
                        )
                    )
                bind.assert_not_called()
                self.assertEqual(len(seen), 1)
                self.assertNotIn("kv_hints", seen[0] or {})

    def test_fault_injector_keeps_the_backend_opt_in(self):
        """Test-only fault injection must not hide a backend's hint opt-in."""
        for consumes_kv_hints in (None, True):
            with self.subTest(consumes_kv_hints=consumes_kv_hints):
                backend = _Backend(Mock(), consumes_kv_hints)
                wrapped = HiCacheStorageFailureInjector(backend, 0.0, 0.0)
                self.assertEqual(wrapped.consumes_kv_hints(), bool(consumes_kv_hints))

    def test_decode_probe_copies_the_prefix_only_for_a_namespaced_binding(self):
        """The matched prefix is sliced only when a backend will bind it."""
        page_size = 4
        tokens = array("q", range(3 * page_size))
        cases = (
            ("salted and bound", True, _HINT, "tenant", True),
            ("no namespace", True, _HINT, None, False),
            ("backend opted out", False, _HINT, "tenant", False),
            ("hint has no actions", True, _EMPTY_HINT, "tenant", False),
            ("no hint", True, None, "tenant", False),
        )
        for name, consumes_kv_hints, hint, salt, copies in cases:
            with self.subTest(name):
                cache, probe = _decode_probe(
                    page_size,
                    KVCacheEventRecorder(enabled=False, page_size=page_size),
                    Mock(),
                    consumes_kv_hints=consumes_kv_hints,
                )
                cache.get_last_hash_value = lambda node: "00" * 32
                cache.query_storage_hit_length = Mock(return_value=0)
                probe(
                    SimpleNamespace(
                        origin_input_ids=tokens,
                        extra_key=None,
                        cache_salt=salt,
                        kv_hints=hint,
                    )
                )
                prefix = cache.query_storage_hit_length.call_args.kwargs[
                    "matched_prefix_tokens"
                ]
                if copies:
                    self.assertEqual(list(prefix), list(tokens[:page_size]))
                else:
                    self.assertIsNone(prefix)

    def test_scheduler_names_hinted_lora_keys_only_where_hints_are_bound(self):
        """Non-publishing ranks keep LoRA names only for hints a backend reads."""
        for publishes, hint, consumes, expected in (
            (True, None, False, True),
            (False, _HINT, True, True),
            (False, _HINT, False, False),
            (False, _EMPTY_HINT, True, False),
            (False, None, True, False),
        ):
            with self.subTest(publishes=publishes, hint=hint, consumes=consumes):
                names = Mock()
                tree_cache = SimpleNamespace(
                    forwards_kv_hints=lambda: consumes,
                    supports_prefix_sharing=lambda: True,
                )
                tree_cache.forwards_kv_hint_actions = lambda kv_hints: (
                    BasePrefixCache.forwards_kv_hint_actions(tree_cache, kv_hints)
                )
                scheduler = SimpleNamespace(
                    kv_events_publisher=SimpleNamespace(
                        enable_kv_cache_events=publishes
                    ),
                    tree_cache=tree_cache,
                    kv_event_lora_names=names,
                )
                Scheduler._record_kv_event_lora_name(
                    scheduler, SimpleNamespace(kv_hints=hint), "lora"
                )
                self.assertEqual(names.register.called, expected)

    def test_stale_prefetch_anchor_degrades_to_l2(self):
        tree_cache = SimpleNamespace(
            tree_core=SimpleNamespace(is_eagle=False),
            hicache_storage_pass_prefix_keys=True,
            ongoing_prefetch={},
            get_last_hash_value=Mock(side_effect=KeyError(22)),
            get_prefix_hash_values=Mock(),
            prefetch_from_storage=Mock(),
        )
        harness = SimpleNamespace(tree_cache=tree_cache)
        req = SimpleNamespace(
            rid="req-0",
            cache_request_handle=CacheRequestHandle("req-0", 0),
            origin_input_ids=[0, 1, 2, 3, 4, 5],
            extra_key=None,
            cache_salt=None,
        )
        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.tensor([10, 11]),
            l2_host_hit_length=2,
            l3_storage_hit_length=2,
            last_device_node=11,
            last_host_node=22,
        )

        DecodeHiCachePreallocMixin._start_hicache_prefetch(harness, req, prefix_match)

        self.assertEqual(prefix_match.l3_storage_hit_length, 0)
        self.assertFalse(prefix_match.prefetch_registered)
        tree_cache.get_prefix_hash_values.assert_not_called()
        tree_cache.prefetch_from_storage.assert_not_called()

    def test_restored_indices_read_off_the_restored_path(self):
        restored_path = torch.tensor([10, 11, 12, 13, 14])
        rematch = SimpleNamespace(
            device_prefix_len=2,
            last_device_node="rematched",
            best_match_node="restored",
            host_hit_length=2,
        )
        last_node_during_load = []

        def init_load_back(params):
            last_node_during_load.append(req.last_node)
            return 2, "restored"

        tree_cache = SimpleNamespace(
            tree_core=SimpleNamespace(is_eagle=False),
            match_prefix=Mock(return_value=rematch),
            init_load_back=Mock(side_effect=init_load_back),
            path_device_indices=Mock(return_value=restored_path),
            lock=Mock(return_value="restore-lock"),
        )
        harness = SimpleNamespace(tree_cache=tree_cache)
        req = SimpleNamespace(
            rid="req-0",
            cache_request_handle=CacheRequestHandle("req-0", 0),
            origin_input_ids=[0, 1, 2, 3, 4, 5],
            last_node="prealloc",
        )
        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.tensor([10, 11]),
            l2_host_hit_length=2,
            l3_storage_hit_length=0,
            last_device_node="prealloc",
        )
        decode_req = SimpleNamespace(req=req, prefix_match=prefix_match)

        with patch(
            "sglang.srt.disaggregation.decode_hicache_mixin._req_radix_key",
            return_value="key",
        ):
            queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
                harness, decode_req
            )

        self.assertTrue(queued)
        # init_load_back reads the rematched device boundary off req.last_node;
        # the request keeps its own node afterwards.
        self.assertEqual(last_node_during_load, ["rematched"])
        self.assertEqual(req.last_node, "prealloc")
        tree_cache.path_device_indices.assert_called_once_with("restored")
        self.assertEqual(decode_req.hicache_restored_kv_indices.tolist(), [12, 13])
        tree_cache.lock.assert_called_once_with("restored")


if __name__ == "__main__":
    unittest.main()
