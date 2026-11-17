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
from sglang.srt.managers.kv_hints import KvHintsEnvelope
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.events import KVCacheEventRecorder
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.utils import get_storage_hash_str
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestDecodeHiCacheTreeCore(CustomTestCase):
    def test_decode_probe_binds_salted_and_adapter_ids_on_a_nonpublishing_rank(self):
        """A lookup before the first local storage operation can use published IDs."""
        page_size = 4
        hint = KvHintsEnvelope(protocol_version="0.1", message_id="decode-probe")
        publisher = KVCacheEventRecorder(
            enabled=True, page_size=page_size, dynamo_format=True
        )
        follower = KVCacheEventRecorder(
            enabled=False, page_size=page_size, dynamo_format=True
        )
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(
            enable_storage=True, is_eagle=False, kv_events=follower
        )
        cache.page_size = cache.prefetch_threshold = page_size
        cache._all_reduce_attn_groups = Mock()
        cache.prefix_device_indices = lambda req: torch.arange(page_size)
        cache.hicache_storage_pass_prefix_keys = False
        cache.is_backuped = lambda node: True

        controller = HybridCacheController.__new__(HybridCacheController)
        controller.page_size = page_size
        controller.prefetch_rate_limited = lambda: False
        cache.cache_controller = controller
        harness = SimpleNamespace(
            tree_cache=cache, scheduler=SimpleNamespace(enable_decode_hicache=True)
        )
        match = SimpleNamespace(host_hit_length=0, last_device_node=1, last_host_node=1)

        for salt, adapter in (("tenant", None), (None, "adapter")):
            with self.subTest(salt=salt, adapter=adapter):
                tokens = array("q", range(3 * page_size))
                key = RadixKey(tokens, extra_key=adapter, cache_salt=salt)
                keys = get_storage_hash_str(key, page_size=page_size)
                publisher.record_store(
                    SimpleNamespace(
                        key=key,
                        parent=None,
                        hash_value=keys,
                        event_hash_value=None,
                    )
                )
                event_ids = publisher.take()[0].block_hashes[1:]
                expected = dict(
                    zip(keys[1:], (value & ((1 << 64) - 1) for value in event_ids))
                )

                def batch_exists(page_keys, info):
                    fields = info.extra_info
                    if fields.get("kv_hints") != hint:
                        return 0
                    binding = fields.get("storage_key_to_event_hash", {})
                    return sum(
                        page in binding and binding[page] == expected[page]
                        for page in page_keys
                    )

                controller.storage_backend = SimpleNamespace(batch_exists=batch_exists)
                cache.get_last_hash_value = lambda node: keys[0]
                req = SimpleNamespace(
                    origin_input_ids=tokens,
                    extra_key=adapter,
                    cache_salt=salt,
                    kv_hints=hint,
                )
                result = DecodeHiCachePreallocMixin._build_decode_prefix_match(
                    harness, req, match
                )
                self.assertEqual(result.l3_storage_hit_length, 2 * page_size)
                cache.get_last_hash_value = lambda node: "00" * 32
                invalid_prefix = DecodeHiCachePreallocMixin._build_decode_prefix_match(
                    harness, req, match
                )
                self.assertEqual(invalid_prefix.l3_storage_hit_length, 0)

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
