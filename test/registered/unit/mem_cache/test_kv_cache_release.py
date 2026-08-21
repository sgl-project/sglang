"""Regression test for finished-request KV cache release."""

import unittest
from array import array

import torch

from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestKVCacheRelease(CustomTestCase):
    def test_release_frees_committed_kv_without_token_id(self):
        """Finishing a request returns KV slots beyond its token sequence."""
        req = Req(
            rid="overlap-abort",
            origin_input_text="",
            origin_input_ids=array("q", [1, 2]),
            sampling_params=SamplingParams(max_new_tokens=2),
        )
        req.output_ids = array("q", [3])
        allocated_len = req.seqlen + 1
        req.kv_committed_len = allocated_len
        req.kv = ReqKvInfo(kv_allocated_len=allocated_len, swa_evicted_seqlen=0)
        req.cache_protected_len = 0

        allocator = TokenToKVPoolAllocator(
            size=16,
            dtype=torch.float16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        request_indices = allocator.alloc(allocated_len)
        self.assertIsNotNone(request_indices)

        req_to_token_pool = ReqToTokenPool(
            size=1,
            max_context_len=allocated_len,
            device="cpu",
            enable_memory_saver=False,
        )
        self.assertIsNotNone(req_to_token_pool.alloc([req]))
        req_to_token_pool.write(
            (req.req_pool_idx, slice(0, allocated_len)), request_indices
        )

        cache = RadixCache.create_simulated(mock_allocator=allocator)
        cache.req_to_token_pool = req_to_token_pool
        req.last_node = cache.root_node

        available_before = allocator.available_size()
        with get_context().override_server_args(
            speculative_algorithm=None,
            strip_thinking_cache=False,
        ):
            allocator.free_group_begin()
            release_kv_cache(req, cache)
            allocator.free_group_end()

        self.assertEqual(cache.total_size(), req.seqlen)
        self.assertEqual(allocator.available_size(), available_before + 1)


if __name__ == "__main__":
    unittest.main()
