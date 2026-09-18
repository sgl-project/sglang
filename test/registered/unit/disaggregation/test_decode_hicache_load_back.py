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


def _tree_cache(*, full_match, new_indices, ongoing_load_back: bool) -> Mock:
    return Mock(
        check_prefetch_progress=Mock(return_value=True),
        match_full_prefix=Mock(return_value=full_match),
        init_load_back=Mock(return_value=(new_indices, 99)),
        inc_lock_ref=Mock(return_value=Mock(to_dec_params=Mock())),
        has_ongoing_load_back=Mock(return_value=ongoing_load_back),
    )


def _decode_req(*, prefix_indices, l2: int, l3: int) -> SimpleNamespace:
    return SimpleNamespace(
        req=SimpleNamespace(
            rid="req-0",
            cache_request_handle=object(),
            origin_input_ids=list(range(8)),
            extra_key=None,
            cache_salt=None,
            last_node=None,
        ),
        prefix_match=DecodePrefixMatch(
            prefix_indices=prefix_indices,
            l2_host_hit_length=l2,
            l3_storage_hit_length=l3,
            last_device_node=11,
            last_host_node=None,
        ),
        hicache_restore_status=HiCacheRestoreResult.PENDING,
        hicache_restored_node=None,
    )


def _rematch(device_indices, *, host_hit_length: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        best_match_node=5,
        last_device_node=11,
        host_hit_length=host_hit_length,
        device_indices=device_indices,
    )


class TestDecodeLoadBackIsKvOnly(CustomTestCase):
    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_init_load_back_requests_kv_only(self, match_prefix):
        # L1 = 2 device tokens, L2 = 2 host tokens to restore.
        match_prefix.return_value = _rematch(torch.tensor([10, 11]), host_hit_length=2)
        tree_cache = _tree_cache(
            full_match=(4, 5),
            new_indices=torch.tensor([20, 21]),
            ongoing_load_back=True,
        )
        dr = _decode_req(prefix_indices=torch.tensor([10, 11]), l2=2, l3=0)

        queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
            SimpleNamespace(tree_cache=tree_cache), dr
        )

        self.assertTrue(queued)
        params = tree_cache.init_load_back.call_args.args[0]
        # Component state (SWA window / Mamba) comes from the P/D transfer;
        # the decode restore must never write it.
        self.assertTrue(params.kv_only)
        self.assertEqual(params.best_match_node, 5)
        self.assertEqual(params.host_hit_length, 2)
        self.assertEqual(dr.hicache_restored_node, 99)
        self.assertEqual(dr.hicache_restored_kv_indices.tolist(), [20, 21])

    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_resident_full_kv_is_ready_without_dma(self, match_prefix):
        # Host hit made of component state only: init_load_back hands back the
        # resident FULL indices and issues no DMA, so the restore is READY now.
        match_prefix.return_value = _rematch(torch.tensor([10, 11]), host_hit_length=2)
        tree_cache = _tree_cache(
            full_match=(4, 5),
            new_indices=torch.tensor([20, 21]),
            ongoing_load_back=False,
        )
        dr = _decode_req(prefix_indices=torch.tensor([10, 11]), l2=2, l3=0)

        queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
            SimpleNamespace(tree_cache=tree_cache), dr
        )

        self.assertFalse(queued)
        self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.READY)
        self.assertEqual(dr.hicache_restored_kv_indices.tolist(), [20, 21])

    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_full_kv_behind_tombstoned_component_state_is_restored(self, match_prefix):
        # A prefix shared across requests (GSM8K few-shot header on an SWA
        # model): its FULL KV is in the tree but the SWA state is tombstoned,
        # so the all-component rematch reports nothing at any tier while the
        # KV-only L3 promise covers it. The restore must locate the KV the
        # KV-only way instead of failing the coverage check with a 500.
        match_prefix.return_value = _rematch(torch.tensor([], dtype=torch.int64))
        tree_cache = _tree_cache(
            full_match=(4, 7),
            new_indices=torch.tensor([20, 21, 22, 23]),
            ongoing_load_back=False,
        )
        dr = _decode_req(prefix_indices=torch.tensor([], dtype=torch.int64), l2=0, l3=4)

        queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
            SimpleNamespace(tree_cache=tree_cache), dr
        )

        self.assertFalse(queued)
        self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.READY)
        key = tree_cache.match_full_prefix.call_args.args[0]
        self.assertEqual(list(key.token_ids), [0, 1, 2, 3])
        params = tree_cache.init_load_back.call_args.args[0]
        self.assertEqual(params.best_match_node, 7)
        self.assertEqual(params.host_hit_length, 4)
        self.assertTrue(params.kv_only)
        self.assertEqual(dr.hicache_restored_kv_indices.tolist(), [20, 21, 22, 23])

    @patch("sglang.srt.disaggregation.decode_hicache_mixin.match_prefix_for_req")
    def test_device_match_covering_the_promise_needs_no_load_back(self, match_prefix):
        # The rematch already covers decode_prefix_len (and more): nothing to
        # restore, and the commit is bounded to [l1, decode_prefix_len).
        match_prefix.return_value = _rematch(torch.tensor([10, 11, 12, 13]))
        tree_cache = _tree_cache(
            full_match=(2, 5),
            new_indices=torch.tensor([], dtype=torch.int64),
            ongoing_load_back=False,
        )
        dr = _decode_req(prefix_indices=torch.tensor([10, 11]), l2=0, l3=0)

        queued = DecodeHiCacheTransferMixin._try_hicache_queue_load_back(
            SimpleNamespace(tree_cache=tree_cache), dr
        )

        self.assertFalse(queued)
        self.assertEqual(dr.hicache_restore_status, HiCacheRestoreResult.READY)
        tree_cache.init_load_back.assert_not_called()
        self.assertEqual(dr.hicache_restored_node, 11)
        self.assertEqual(dr.hicache_restored_kv_indices.numel(), 0)


if __name__ == "__main__":
    unittest.main()
