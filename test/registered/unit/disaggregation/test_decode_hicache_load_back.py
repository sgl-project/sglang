"""Decode-side HiCache load-back restores base KV only."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCacheTransferMixin,
    DecodePrefixMatch,
    HiCacheRestoreResult,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDecodeLoadBackIsKvOnly(CustomTestCase):
    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_init_load_back_requests_kv_only(self, match_prefix):
        # L1 = 2 device tokens, L2 = 2 host tokens to restore.
        match_prefix.return_value = SimpleNamespace(
            best_match_node=5,
            host_hit_length=2,
            device_indices=torch.tensor([10, 11]),
        )
        tree_cache = Mock(
            init_load_back=Mock(return_value=(torch.tensor([20, 21]), 99)),
            inc_lock_ref=Mock(return_value=Mock(to_dec_params=Mock())),
        )
        harness = SimpleNamespace(tree_cache=tree_cache)
        dr = SimpleNamespace(
            req=SimpleNamespace(
                rid="req-0", origin_input_ids=list(range(8)), last_node=None
            ),
            prefix_match=DecodePrefixMatch(
                prefix_indices=torch.tensor([10, 11]),
                l2_host_hit_length=2,
                l3_storage_hit_length=0,
                last_device_node=11,
                last_host_node=None,
            ),
            hicache_restore_status=HiCacheRestoreResult.PENDING,
            hicache_restored_node=None,
        )

        queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(harness, dr)

        self.assertTrue(queued)
        params = tree_cache.init_load_back.call_args.args[0]
        # Component state (SWA window / Mamba) comes from the P/D transfer;
        # the decode restore must never write it.
        self.assertTrue(params.kv_only)
        self.assertEqual(params.host_hit_length, 2)
        self.assertEqual(dr.hicache_restored_node, 99)


if __name__ == "__main__":
    unittest.main()
