import time
import unittest
from array import array
from unittest import mock

import torch

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

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

OLDER_PREFIX = [1, 2, 3, 4]
NEWER_PREFIX = [5, 6, 7, 8]


class TestFcfsWaitingPrefixRefresh(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        torch.set_default_device(None)

    def _run_and_evict_one_prefix(self, eviction_policy):
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
        cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=ReqToTokenPool(
                    size=8, max_context_len=64, device="cpu", enable_memory_saver=False
                ),
                token_to_kv_pool_allocator=allocator,
                page_size=1,
                eviction_policy=eviction_policy,
                enable_kv_cache_events=False,
            )
        )
        for tokens, slots in (
            (OLDER_PREFIX, [10, 11, 12, 13]),
            (NEWER_PREFIX, [20, 21, 22, 23]),
        ):
            cache.insert(
                InsertParams(
                    key=RadixKey(array("q", tokens)),
                    value=torch.tensor(slots, dtype=torch.int64),
                )
            )
            time.sleep(0.005)
        queue = [
            Req(rid, "", array("q", tokens + [9]), SamplingParams())
            for rid, tokens in ((1, OLDER_PREFIX), (2, NEWER_PREFIX))
        ]
        publish(
            ServerArgs(model_path="dummy", radix_eviction_policy=eviction_policy),
            role="test",
        )
        SchedulePolicy(
            policy="fcfs",
            tree_cache=cache,
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        ).calc_priority(waiting_queue=queue, forward_ct=0)
        cache.evict(EvictParams(num_tokens=len(OLDER_PREFIX)))
        return [
            cache.match_prefix(
                MatchPrefixParams(key=RadixKey(array("q", tokens)))
            ).device_prefix_len
            for tokens in (OLDER_PREFIX, NEWER_PREFIX)
        ]

    def test_lru_evicts_the_prefix_furthest_from_admission(self):
        self.assertEqual(self._run_and_evict_one_prefix("lru"), [4, 0])

    def test_mru_keeps_plain_order(self):
        # MRU evicts the most recent first, so a refresh would invert its intent.
        self.assertEqual(self._run_and_evict_one_prefix("mru"), [4, 0])

    def test_refresh_decisions_ignore_wall_clock_spacing(self):
        """TP/PP ranks fed the same calls must refresh on the same calls,
        however far apart in time each rank makes them."""
        publish(
            server_args=ServerArgs(model_path="dummy", radix_eviction_policy="lru"),
            role="test",
        )
        queue = [
            Req(
                rid=1,
                origin_input_text="",
                origin_input_ids=array("q", OLDER_PREFIX),
                sampling_params=SamplingParams(),
            )
        ]

        def refreshes(gap_s):
            policy = SchedulePolicy(
                policy="fcfs",
                tree_cache=RadixCache.create_simulated(),
                enable_hierarchical_cache=False,
                enable_priority_scheduling=False,
                schedule_low_priority_values_first=False,
            )
            touched = []
            for forward_ct in range(3):
                if forward_ct:
                    time.sleep(gap_s)
                with mock.patch(
                    "sglang.srt.managers.schedule_policy.touch_waiting_prefix"
                ) as touch:
                    policy.calc_priority(waiting_queue=queue, forward_ct=forward_ct)
                touched.append(touch.called)
            return touched

        self.assertEqual(refreshes(gap_s=0), [True, False, False])
        self.assertEqual(refreshes(gap_s=0.6), [True, False, False])


if __name__ == "__main__":
    unittest.main()
