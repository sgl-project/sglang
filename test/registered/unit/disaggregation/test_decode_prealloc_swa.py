"""Decode prealloc on SWA models: resume-path tail reclaim and prefix lengths."""

import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    alloc_for_decode_prealloc,
)
from sglang.srt.mem_cache.base_prefix_cache import EvictParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 64
SWA_TAIL_LEN = 192


def _resume_harness(*, swa_available: list[int]) -> tuple[SimpleNamespace, Mock]:
    calls = Mock()
    req = SimpleNamespace(rid="req-0", is_retracted=True)
    harness = SimpleNamespace(
        retracted_queue=[req],
        req_to_token_pool=SimpleNamespace(available_size=lambda: 8),
        token_to_kv_pool_allocator=SimpleNamespace(
            page_size=PAGE_SIZE,
            swa_available_size=Mock(side_effect=swa_available),
        ),
        tree_cache=calls.tree_cache,
        _uses_swa_tail_prealloc=lambda: True,
        _swa_aware_allocatable_token_budgets=lambda count_retracted: (10**6, 10**6),
        _swa_tail_allocatable_token_budget=Mock(return_value=10**6),
        _prealloc_required_tokens=lambda req: (4096, SWA_TAIL_LEN),
        _prealloc_kv_lens=lambda req: (4096, SWA_TAIL_LEN),
        _pre_alloc=calls._pre_alloc,
    )
    harness._reclaim_swa_tail_capacity = types.MethodType(
        DecodePreallocQueue._reclaim_swa_tail_capacity, harness
    )
    return harness, calls


@patch("sglang.srt.disaggregation.decode.retraction_restore")
@patch("sglang.srt.disaggregation.decode.get_disagg")
class TestResumeRetractedReclaimsSwaTail(CustomTestCase):
    def test_evicts_tail_shortfall_before_prealloc(self, _get_disagg, _restore):
        # Budget admitted the request on evictable SWA pages: 64 free now,
        # 256 after eviction.
        harness, calls = _resume_harness(swa_available=[64, 256])

        resumed = DecodePreallocQueue.resume_retracted_reqs(harness)

        self.assertEqual(len(resumed), 1)
        evict, prealloc = calls.mock_calls[0], calls.mock_calls[-1]
        self.assertEqual(evict[0], "tree_cache.evict_for_alloc")
        self.assertEqual(evict[1][0], EvictParams(swa_num_tokens=SWA_TAIL_LEN - 64))
        self.assertEqual(prealloc[0], "_pre_alloc")

    def test_shortfall_after_eviction_keeps_request_retracted(
        self, _get_disagg, _restore
    ):
        harness, calls = _resume_harness(swa_available=[64, 64])

        resumed = DecodePreallocQueue.resume_retracted_reqs(harness)

        self.assertEqual(resumed, [])
        self.assertEqual(len(harness.retracted_queue), 1)
        calls._pre_alloc.assert_not_called()


class TestAllocForDecodePreallocSwa(CustomTestCase):
    def test_swa_branch_uses_total_prefix_len(self):
        # L1 = 1024 on device, L2 = 2048 restored by load_back, fill = 4096.
        allocator = SimpleNamespace(
            page_size=PAGE_SIZE,
            device="cpu",
            alloc_extend_swa_tail=Mock(return_value=torch.arange(1024)),
        )
        req = SimpleNamespace(kv=SimpleNamespace())

        alloc_for_decode_prealloc(
            allocator,
            req=req,
            fill_len=4096,
            delta_len=1024,
            prefix_len=1024,
            total_prefix_len=3072,
            prefix_indices=torch.arange(1024),
            uses_swa_tail=True,
            swa_tail_len=SWA_TAIL_LEN,
        )

        kwargs = allocator.alloc_extend_swa_tail.call_args.kwargs
        # The allocator must see the same prefix the extend size was derived
        # from; otherwise it fills fill - l1 slots into a fill - total buffer.
        self.assertEqual(kwargs["prefix_lens_cpu"].item(), 3072)
        self.assertEqual(kwargs["seq_lens_cpu"].item(), 4096)
        self.assertEqual(kwargs["extend_num_tokens"], 1024)
        self.assertEqual(kwargs["swa_tail_len"], SWA_TAIL_LEN)
        self.assertEqual(req.kv.swa_evicted_seqlen, 4096 - SWA_TAIL_LEN)


if __name__ == "__main__":
    unittest.main()
