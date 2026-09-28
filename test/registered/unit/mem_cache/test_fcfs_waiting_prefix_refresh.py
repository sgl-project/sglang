import time
import unittest
from array import array

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

OLDER_PREFIX = [1, 2, 3, 4]
NEWER_PREFIX = [5, 6, 7, 8]


class TestFcfsWaitingPrefixRefresh(CustomTestCase):
    """Under FCFS an LRU eviction must not take the cached prefix of a request that
    is waiting in the queue while a prefix no pending request needs is available."""

    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(ServerArgs(model_path="dummy"), role="test")
        torch.set_default_device(None)
        kv_cache = MHATokenToKVPool(
            size=16,
            page_size=1,
            dtype=torch.float16,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        allocator = TokenToKVPoolAllocator(
            size=16,
            dtype=torch.float16,
            device="cpu",
            kvcache=kv_cache,
            need_sort=False,
        )
        req_to_token_pool = ReqToTokenPool(
            size=8, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        self.cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                eviction_policy="lru",
                enable_kv_cache_events=False,
            )
        )
        self.policy = SchedulePolicy(
            policy="fcfs",
            tree_cache=self.cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )
        self._insert(OLDER_PREFIX, [10, 11, 12, 13])
        time.sleep(0.005)
        self._insert(NEWER_PREFIX, [20, 21, 22, 23])
        time.sleep(0.005)
        self.waiting = Req(1, "", array("q", OLDER_PREFIX + [9]), SamplingParams())

    def _insert(self, tokens, slots):
        self.cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=torch.tensor(slots, dtype=torch.int64),
            )
        )

    def _matched_len(self, tokens):
        result = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        )
        return len(result.device_indices)

    def test_refresh_keeps_the_waiting_request_prefix_resident(self):
        with envs.SGLANG_ENABLE_WAITING_PREFIX_REFRESH.override(True):
            self.policy.calc_priority([self.waiting])
        self.assertEqual(len(self.waiting.prefix_indices), len(OLDER_PREFIX))
        self.cache.evict(EvictParams(num_tokens=len(NEWER_PREFIX)))
        self.assertEqual(self._matched_len(OLDER_PREFIX), len(OLDER_PREFIX))
        self.assertEqual(self._matched_len(NEWER_PREFIX), 0)

    def test_refresh_disabled_restores_plain_lru(self):
        with envs.SGLANG_ENABLE_WAITING_PREFIX_REFRESH.override(False):
            self.policy.calc_priority([self.waiting])
        self.assertEqual(len(self.waiting.prefix_indices), 0)
        self.cache.evict(EvictParams(num_tokens=len(OLDER_PREFIX)))
        self.assertEqual(self._matched_len(OLDER_PREFIX), 0)
        self.assertEqual(self._matched_len(NEWER_PREFIX), len(NEWER_PREFIX))


if __name__ == "__main__":
    unittest.main()
