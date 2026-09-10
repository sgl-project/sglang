"""Unit tests for decode HiCache prefix-match shaping (decode_hicache_mixin)."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.disaggregation.decode_hicache_mixin import (
    DecodeHiCachePreallocMixin,
    DecodePrefixMatch,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 128


def _harness(
    *,
    l3_hit: int,
    fill_len: int,
    swa_tail_len: int,
    uses_swa_tail: bool = True,
    all_or_nothing: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        scheduler=SimpleNamespace(enable_decode_hicache=True),
        tree_cache=SimpleNamespace(
            storage_prefetch_is_all_or_nothing=all_or_nothing,
            hicache_storage_pass_prefix_keys=False,
            is_backuped=Mock(return_value=True),
            is_root=Mock(return_value=False),
            get_last_hash_value=Mock(return_value="hash"),
            query_storage_hit_length=Mock(return_value=l3_hit),
        ),
        token_to_kv_pool_allocator=SimpleNamespace(page_size=PAGE_SIZE),
        _uses_swa_tail_prealloc=lambda: uses_swa_tail,
        _pre_alloc_fill_len=lambda req: fill_len,
        _swa_tail_len=lambda seq_len: swa_tail_len,
    )


def _match(harness: SimpleNamespace, *, l1: int, l2: int) -> DecodePrefixMatch:
    req = SimpleNamespace(origin_input_ids=list(range(4096)))
    result = SimpleNamespace(
        device_indices=torch.arange(l1),
        host_hit_length=l2,
        last_host_node=22,
        last_device_node=11,
    )
    return DecodeHiCachePreallocMixin._build_decode_prefix_match(harness, req, result)


class TestDecodeHiCachePrefixMatch(CustomTestCase):
    def test_hybrid_all_or_nothing_skips_storage_query(self):
        harness = _harness(
            l3_hit=1024, fill_len=2048, swa_tail_len=512, all_or_nothing=True
        )

        match = _match(harness, l1=0, l2=0)

        self.assertEqual(match.l3_storage_hit_length, 0)
        self.assertIsNone(match.last_host_node)
        harness.tree_cache.query_storage_hit_length.assert_not_called()

    def test_swa_tail_cap_on_restored_range(self):
        # (l1, l2, l3_hit, fill_len, tail_len) -> (expected_l2, expected_l3)
        cases = [
            # Under the cap (cap = 2048 - 512 = 1536): unchanged.
            ((0, 0, 512, 2048, 512), (0, 512)),
            ((128, 256, 512, 2048, 512), (256, 512)),
            # L3 crosses the cap (cap = 512): trimmed to the cap.
            ((0, 0, 1024, 1025, 513), (0, 512)),
            # Trimmed L3 is page-aligned down (cap = 500 -> 384).
            ((0, 0, 1024, 1025, 525), (0, 384)),
            # L1 + L2 crosses the cap (cap = 512): L2 nodes cannot split
            # mid-node, so the whole restore degrades to none.
            ((128, 1024, 512, 1025, 513), (0, 0)),
            ((1024, 128, 0, 1025, 513), (0, 0)),
        ]
        for (l1, l2, l3_hit, fill_len, tail_len), (exp_l2, exp_l3) in cases:
            with self.subTest(l1=l1, l2=l2, l3_hit=l3_hit, fill_len=fill_len):
                harness = _harness(
                    l3_hit=l3_hit, fill_len=fill_len, swa_tail_len=tail_len
                )

                match = _match(harness, l1=l1, l2=l2)

                self.assertEqual(match.l1_prefix_len, l1)
                self.assertEqual(match.l2_host_hit_length, exp_l2)
                self.assertEqual(match.l3_storage_hit_length, exp_l3)
                self.assertEqual(match.last_host_node is not None, exp_l3 > 0)

    def test_no_cap_without_swa_tail_prealloc(self):
        harness = _harness(
            l3_hit=1024, fill_len=1025, swa_tail_len=513, uses_swa_tail=False
        )

        match = _match(harness, l1=0, l2=256)

        self.assertEqual(match.l2_host_hit_length, 256)
        self.assertEqual(match.l3_storage_hit_length, 1024)


class TestDecodeHiCachePrefetchDecline(CustomTestCase):
    def _prefetch(self, *, registers: bool) -> DecodePrefixMatch:
        ongoing_prefetch = {}

        def prefetch_from_storage(req_id, *_args, **_kwargs):
            if registers:
                ongoing_prefetch[req_id] = object()

        harness = SimpleNamespace(
            tree_cache=SimpleNamespace(
                hicache_storage_pass_prefix_keys=False,
                ongoing_prefetch=ongoing_prefetch,
                get_last_hash_value=Mock(return_value="hash"),
                prefetch_from_storage=Mock(side_effect=prefetch_from_storage),
            ),
        )
        req = SimpleNamespace(
            rid="req-0",
            origin_input_ids=list(range(2048)),
            extra_key=None,
            cache_salt=None,
        )
        prefix_match = DecodePrefixMatch(
            prefix_indices=torch.arange(256),
            l2_host_hit_length=0,
            l3_storage_hit_length=512,
            last_device_node=11,
            last_host_node=22,
        )
        DecodeHiCachePreallocMixin._start_hicache_prefetch(harness, req, prefix_match)
        return prefix_match

    def test_registered_prefetch_keeps_l3_promise(self):
        prefix_match = self._prefetch(registers=True)

        self.assertTrue(prefix_match.prefetch_registered)
        self.assertEqual(prefix_match.l3_storage_hit_length, 512)

    def test_declined_prefetch_degrades_to_l2_only(self):
        # A silently declined prefetch (rate limit, host buffer alloc failure)
        # would leave the promised L3 range unrestorable after the transfer
        # was already trimmed by decode_prefix_len.
        prefix_match = self._prefetch(registers=False)

        self.assertFalse(prefix_match.prefetch_registered)
        self.assertEqual(prefix_match.l3_storage_hit_length, 0)
        self.assertEqual(prefix_match.decode_prefix_len, 256)


if __name__ == "__main__":
    unittest.main()
