"""Regression: requests queued behind a full running batch keep their cached state.

calc_priority, where the waiting queue gets touched, does not run while admission is
blocked. A session that waited behind a full batch lost its mamba checkpoint to LRU
churn before admission and re-prefilled its whole prompt, and its next turn queued
behind the same full batch again.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")

import unittest
from array import array
from types import SimpleNamespace
from unittest import mock

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.managers.schedule_policy import SchedulePolicy
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture
from sglang.test.test_utils import CustomTestCase

PAGE = 4
SESSION = list(range(1, 1 + 3 * PAGE))


class TestWaitingPrefixRefreshBlocked(CustomTestCase):
    tree_core_backend = "python"
    cfg = CacheConfig(
        page_size=PAGE,
        components=(ComponentType.FULL, ComponentType.MAMBA),
        enable_mamba_extra_buffer=True,
        mamba_cache_size=6,
        kv_size=256,
        max_context_len=64,
    )

    def _insert(self, cache, allocator, pool, tokens):
        slot = pool.mamba_allocator.alloc(1)
        if slot is None:
            cache.evict(EvictParams(num_tokens=0, mamba_num=1))
            slot = pool.mamba_allocator.alloc(1)
        cache.insert(
            InsertParams(
                key=RadixKey(array("q", tokens)),
                value=allocator.alloc(len(tokens)),
                mamba_value=slot,
            )
        )

    def _queued_session_hit_after_churn(self, refresh_while_blocked):
        cache, allocator, pool = build_fixture(
            self.cfg, tree_core_backend=self.tree_core_backend
        )
        self._insert(cache, allocator, pool, SESSION)
        queued = Req(1, "", array("q", SESSION + [99]), SamplingParams())
        policy = SchedulePolicy("fcfs", cache, False, False, False)
        with mock.patch(
            "sglang.srt.managers.schedule_policy.WAITING_PREFIX_REFRESH_INTERVAL", 1
        ):
            for k in range(12):  # other requests' checkpoints churn the mamba pool
                if refresh_while_blocked:
                    policy.refresh_waiting_prefixes([queued])
                self._insert(cache, allocator, pool, [1000 + 10 * k] * PAGE)
        key = RadixKey(array("q", SESSION))
        return cache.match_prefix(MatchPrefixParams(key=key)).device_prefix_len

    def test_queued_session_keeps_its_checkpoint(self):
        self.assertEqual(self._queued_session_hit_after_churn(True), len(SESSION))
        self.assertEqual(self._queued_session_hit_after_churn(False), 0)

    def test_blocked_admission_refreshes_the_queue(self):
        queue = [object()]
        for full, allocatable in ((True, 1), (False, 0)):
            sched = SimpleNamespace(
                grammar_manager=SimpleNamespace(has_waiting_grammars=lambda: False),
                enable_priority_preemption=False,
                is_hybrid_swa=False,
                chunked_req=None,
                waiting_queue=queue,
                min_free_slots_delayer=None,
                get_num_allocatable_reqs=lambda *a, **k: allocatable,
                policy=mock.MagicMock(),
            )
            batch = SimpleNamespace(
                batch_is_full=full, is_empty=lambda: False, reqs=[object()]
            )
            with self.subTest(full=full):
                self.assertEqual(
                    Scheduler._get_new_batch_prefill_raw(sched, None, batch),
                    (None, batch),
                )
                sched.policy.refresh_waiting_prefixes.assert_called_once_with(queue)


class TestWaitingPrefixRefreshBlockedRust(TestWaitingPrefixRefreshBlocked):
    tree_core_backend = "rust"


if __name__ == "__main__":
    unittest.main()
