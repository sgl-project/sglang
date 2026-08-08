"""Regression tests for KV reclaim in RadixCache.cache_finished_req.

Bug: ``cache_finished_req`` sliced the request's ``req_to_token`` row by
``len(origin_input_ids + output_ids)``. A dLLM FDFO request aborted with an
unresolved denoise block holds committed KV beyond that length (the block's
tokens live only in ``dllm_incomplete_ids``), so the tail free never saw those
slots and every such abort leaked one block of KV -- the idle invariant check
then fails with ``pool memory leak detected``. The row must be sliced by
``kv_len_to_handle`` (the KV length) instead.
"""

import unittest
from array import array
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_KV_BASE = 100  # req_to_token row fingerprint: kv slot for position i is 100+i


class _RecordingAllocator:
    """Records every KV slot freed through the segment APIs."""

    def __init__(self):
        self.device = "cpu"
        self.freed = []

    def free_segments(self, segments):
        for free_index, _start_pos in segments:
            self.freed.extend(free_index.tolist())

    def free_segment(self, free_index, *, start_pos):
        self.freed.extend(free_index.tolist())


def _make_cache(allocator, max_context: int = 64):
    pool = SimpleNamespace(
        req_to_token=(
            torch.arange(_KV_BASE, _KV_BASE + max_context, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(4, 1)
        )
    )
    return RadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
        )
    )


def _make_req(*, num_prompt: int, num_output: int):
    return SimpleNamespace(
        rid="test-req",
        origin_input_ids=array("q", range(num_prompt)),
        output_ids=array("q", range(1000, 1000 + num_output)),
        req_pool_idx=1,
        cache_protected_len=0,
        extra_key=None,
        priority=0,
        last_node=None,
        session=None,
    )


class TestCacheFinishedReqFreesByKvLen(CustomTestCase):
    def test_aborted_fdfo_req_frees_kv_beyond_token_ids(self):
        # Abort with an unresolved denoise block: 8 tokens in origin+output,
        # but 12 KV slots committed (the last 4 belong to the unresolved
        # block). Every committed slot must end up either freed or held by the
        # radix tree -- the pre-fix slicing left the 4 tail slots unaccounted.
        allocator = _RecordingAllocator()
        cache = _make_cache(allocator)
        req = _make_req(num_prompt=6, num_output=2)
        kv_len = 12

        cache.cache_finished_req(req, kv_len_to_handle=kv_len)

        self.assertEqual(
            sorted(allocator.freed),
            [_KV_BASE + 8, _KV_BASE + 9, _KV_BASE + 10, _KV_BASE + 11],
        )
        self.assertEqual(cache.evictable_size(), 8)
        self.assertEqual(len(allocator.freed) + cache.evictable_size(), kv_len)

    def test_ar_finished_req_reclaim_is_unchanged(self):
        # AR invariance: an autoregressive request finishes with
        # kv_len_to_handle < len(origin+output) (the last output token's KV is
        # never committed), where the kv-length slice and the old
        # len(token_ids) slice are identical -- nothing extra may be freed.
        allocator = _RecordingAllocator()
        cache = _make_cache(allocator)
        req = _make_req(num_prompt=6, num_output=2)
        kv_len = 7

        cache.cache_finished_req(req, kv_len_to_handle=kv_len)

        self.assertEqual(allocator.freed, [])
        self.assertEqual(cache.evictable_size(), kv_len)


if __name__ == "__main__":
    unittest.main()
