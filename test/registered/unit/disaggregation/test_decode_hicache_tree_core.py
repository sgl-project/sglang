"""Unit tests for decode HiCache TreeCore interactions."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")


class TestDecodeHiCacheTreeCore(CustomTestCase):
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
        tree_cache = SimpleNamespace(
            tree_core=SimpleNamespace(is_eagle=False),
            init_load_back=Mock(return_value=(2, "restored")),
            path_device_indices=Mock(return_value=restored_path),
            lock=Mock(return_value="restore-lock"),
        )
        harness = SimpleNamespace(tree_cache=tree_cache)
        req = SimpleNamespace(
            rid="req-0",
            cache_request_handle=CacheRequestHandle("req-0", 0),
            origin_input_ids=[0, 1, 2, 3, 4, 5],
            last_node="rematched",
        )
        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.tensor([10, 11]),
            l2_host_hit_length=2,
            l3_storage_hit_length=0,
            last_device_node="prealloc",
        )
        decode_req = SimpleNamespace(req=req, prefix_match=prefix_match)
        rematch = SimpleNamespace(
            device_prefix_len=2, best_match_node="restored", host_hit_length=2
        )

        with patch(
            "sglang.srt.disaggregation.decode_hicache_mixin.match_kv_cache",
            return_value=rematch,
        ):
            queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
                harness, decode_req
            )

        self.assertTrue(queued)
        tree_cache.path_device_indices.assert_called_once_with("restored")
        self.assertEqual(decode_req.hicache_restored_kv_indices.tolist(), [12, 13])
        tree_cache.lock.assert_called_once_with("restored")
        self.assertEqual(req.last_node, "prealloc")


if __name__ == "__main__":
    unittest.main()
