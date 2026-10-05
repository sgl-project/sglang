import time
import unittest
import unittest.mock
from array import array
from dataclasses import replace

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import (
    WAITING_PREFIX_REFRESH_MAX_QUEUE,
    SchedulePolicy,
)
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
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

    def _make_cache(self, params):
        return RadixCache(params)

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
        self.cache = self._make_cache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                eviction_policy="lru",
                enable_kv_cache_events=False,
            )
        )
        self.policy = self._make_policy(self.cache)
        self._insert(OLDER_PREFIX, [10, 11, 12, 13])
        time.sleep(0.005)
        self._insert(NEWER_PREFIX, [20, 21, 22, 23])
        time.sleep(0.005)
        self.waiting = self._make_req(1, OLDER_PREFIX + [9])

    def _make_policy(self, cache):
        return SchedulePolicy(
            policy="fcfs",
            tree_cache=cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )

    def _make_req(self, rid, tokens):
        return Req(rid, "", array("q", tokens), SamplingParams())

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

    def _assert_older_prefix_survives_one_eviction(self):
        self.cache.evict(EvictParams(num_tokens=len(NEWER_PREFIX)))
        self.assertEqual(self._matched_len(OLDER_PREFIX), len(OLDER_PREFIX))
        self.assertEqual(self._matched_len(NEWER_PREFIX), 0)

    def test_refresh_keeps_the_waiting_request_prefix_resident(self):
        with envs.SGLANG_ENABLE_WAITING_PREFIX_REFRESH.override(True):
            self.policy.calc_priority([self.waiting])
        self._assert_older_prefix_survives_one_eviction()

    def test_refresh_disabled_restores_plain_lru(self):
        with envs.SGLANG_ENABLE_WAITING_PREFIX_REFRESH.override(False):
            self.policy.calc_priority([self.waiting])
        self.cache.evict(EvictParams(num_tokens=len(OLDER_PREFIX)))
        self.assertEqual(self._matched_len(OLDER_PREFIX), 0)
        self.assertEqual(self._matched_len(NEWER_PREFIX), len(NEWER_PREFIX))

    def test_head_of_queue_is_refreshed_last(self):
        # FCFS admits the head next, so when eviction must take a waiting prefix
        # it has to be the one furthest from admission.
        second = self._make_req(2, NEWER_PREFIX + [9])
        with envs.SGLANG_ENABLE_WAITING_PREFIX_REFRESH.override(True):
            self.policy.calc_priority([self.waiting, second])
        self._assert_older_prefix_survives_one_eviction()

    def test_deep_queue_still_refreshes_the_head(self):
        fillers = [
            self._make_req(2 + i, [1000 + i])
            for i in range(WAITING_PREFIX_REFRESH_MAX_QUEUE)
        ]
        with envs.SGLANG_ENABLE_WAITING_PREFIX_REFRESH.override(True):
            self.policy.calc_priority([self.waiting] + fillers)
        self._assert_older_prefix_survives_one_eviction()

    def test_refresh_never_calls_the_general_matcher(self):
        # External-cache implementations (FlexKV, LMCache, hierarchical tiers) give
        # match_prefix side effects: lookups enqueued, KV allocated or loaded per call.
        calls = []
        original = type(self.cache).match_prefix

        def spy(cache, params):
            calls.append(params)
            return original(cache, params)

        with (
            unittest.mock.patch.object(type(self.cache), "match_prefix", spy),
            envs.SGLANG_ENABLE_WAITING_PREFIX_REFRESH.override(True),
        ):
            self.policy.calc_priority([self.waiting])
        self.assertEqual(calls, [])
        self._assert_older_prefix_survives_one_eviction()

    def test_chunk_cache_skips_the_refresh(self):
        # --disable-radix-cache deployments have no tree or LRU state to refresh; the
        # scheduler hot path must not build keys or match for every waiting request.
        chunk_cache = ChunkCache(
            CacheInitParams(
                disable=True,
                req_to_token_pool=self.cache.req_to_token_pool,
                token_to_kv_pool_allocator=self.cache.token_to_kv_pool_allocator,
                page_size=1,
                eviction_policy="lru",
                enable_kv_cache_events=False,
            )
        )
        policy = self._make_policy(chunk_cache)
        with (
            unittest.mock.patch.object(ChunkCache, "match_prefix") as match_prefix,
            unittest.mock.patch.object(ChunkCache, "refresh_device_prefix") as refresh,
            envs.SGLANG_ENABLE_WAITING_PREFIX_REFRESH.override(True),
        ):
            policy.calc_priority([self.waiting])
        match_prefix.assert_not_called()
        refresh.assert_not_called()


class TestFcfsWaitingPrefixRefreshUnified(TestFcfsWaitingPrefixRefresh):
    """The same contract on UnifiedRadixCache, the default prefix cache."""

    def _make_cache(self, params):
        return UnifiedRadixCache(replace(params, tree_components=(ComponentType.FULL,)))


if __name__ == "__main__":
    unittest.main()
