"""Unit tests for HiCache PP synchronization."""

import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.storage_prefetch import StoragePrefetchRetries
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _OngoingPrefetch,
    _StoragePrefetchWait,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestUnifiedPPSyncBatching(unittest.TestCase):
    def _make_cache(self, pp_rank, write_ready, load_ready):
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(
            enable_storage=False,
            write_back_duplicate_reclaim_digest=0,
        )
        cache.pp_rank = pp_rank
        cache.pp_size = 2
        cache.host_memory_mode = "cache"
        # The per-tick deadline vote runs under storage; no live prefetch
        # means it issues no collective, which the asserts below pin.
        cache.prefetch_stop_policy = "timeout"
        cache.ongoing_prefetch = {}
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.buffer_pipeline = None
        cache.linker = None
        cache._drain_async_work = MagicMock()
        cache._all_reduce = MagicMock()
        cache.writing_check = MagicMock()
        cache.loading_check = MagicMock()
        cache.cache_controller = SimpleNamespace(
            start_writing=MagicMock(),
            ack_write_queue=[
                SimpleNamespace(
                    finish_event=SimpleNamespace(query=MagicMock(return_value=ready))
                )
                for ready in write_ready
            ],
            ack_load_queue=[
                SimpleNamespace(
                    finish_event=SimpleNamespace(query=MagicMock(return_value=ready))
                )
                for ready in load_ready
            ],
        )
        return cache

    def _arm_live_prefetch(self, cache):
        """Cache-mode storage with one live prefetch on this PP stage."""
        cache.tree_core.enable_storage = True
        cache._drain_storage_control_queues_impl = MagicMock()
        cache.storage_prefetch_retries = StoragePrefetchRetries()
        cache._storage_prefetch_wait = {}
        cc = cache.cache_controller
        for name in (
            "prefetch_hit_queue",
            "ack_prefetch_queue",
            "ack_backup_queue",
            "host_mem_release_queue",
        ):
            setattr(cc, name, Queue())
        cc.extra_host_mem_release_queues = {}
        cc.terminate_prefetch = MagicMock()
        handle = CacheRequestHandle("r", 0)
        operation = SimpleNamespace(
            handle=handle,
            deadline_expired=False,
            host_indices=None,
            hash_value=[],
            completed_tokens=0,
        )
        cache.ongoing_prefetch[handle] = _OngoingPrefetch(
            0, None, None, operation, None, {}
        )
        return handle, operation

    def test_pp_relays_the_deadline_vote_after_the_counts(self):
        """PP0 evaluates the clocks and votes; a follower contributes zeros,
        receives the verdict through the same tag after the counts, and marks
        and terminates the operation without reading its own clock."""
        leader = self._make_cache(0, [True], [True])
        handle, operation = self._arm_live_prefetch(leader)
        wait = _StoragePrefetchWait(timeout_s=1.0)
        with patch("time.perf_counter", return_value=100.0):
            wait.timer.observe(ready=False)
        leader._storage_prefetch_wait[handle] = wait
        with patch("time.perf_counter", return_value=103.0):
            leader.check_hicache_events()

        counts, vote = leader._all_reduce.call_args_list
        self.assertIs(counts.args[1], torch.distributed.ReduceOp.MIN)
        self.assertEqual(vote.args[0].tolist(), [1])
        self.assertIs(vote.args[1], torch.distributed.ReduceOp.MAX)
        self.assertTrue(wait.expired)
        self.assertTrue(operation.deadline_expired)
        leader.cache_controller.terminate_prefetch.assert_called_once_with(operation)

        follower = self._make_cache(1, [True], [True])
        handle, operation = self._arm_live_prefetch(follower)
        follower._prefetch_deadline_expired = MagicMock()
        relayed = []

        def relay_from_pp0(data, _):
            relayed.append(data.tolist())
            if len(relayed) == 1:
                data[0], data[1] = 1, 1  # counts; digest slots stay local
            else:
                data.fill_(1)  # PP0's verdict

        follower._all_reduce.side_effect = relay_from_pp0
        follower.check_hicache_events()

        self.assertEqual(relayed[1], [0])
        self.assertEqual(follower._all_reduce.call_count, 2)
        follower._prefetch_deadline_expired.assert_not_called()
        self.assertTrue(follower._storage_prefetch_wait[handle].expired)
        self.assertTrue(operation.deadline_expired)
        follower.cache_controller.terminate_prefetch.assert_called_once_with(operation)
        follower.writing_check.assert_called_once_with(finish_count=1)

    def test_pp_batches_write_and_load_counts_once(self):
        leader = self._make_cache(0, [True, False], [True, True])
        leader.check_hicache_events()

        leader._all_reduce.assert_called_once()
        self.assertEqual(leader._all_reduce.call_args.args[0].tolist(), [1, 2, 0, 0])
        leader.writing_check.assert_called_once_with(finish_count=1)
        leader.loading_check.assert_called_once_with(finish_count=2)

        follower = self._make_cache(1, [True], [True])

        def reduce_to_min(counts, _):
            counts.copy_(torch.tensor([1, 1, 0, 0], dtype=torch.int64))

        follower._all_reduce.side_effect = reduce_to_min
        follower.check_hicache_events()

        for queue in (
            follower.cache_controller.ack_write_queue,
            follower.cache_controller.ack_load_queue,
        ):
            queue[0].finish_event.query.assert_not_called()
        follower._all_reduce.assert_called_once()
        follower.writing_check.assert_called_once_with(finish_count=1)
        follower.loading_check.assert_called_once_with(finish_count=1)

    def test_buffer_mode_follower_drains_rank_local_completions(self):
        cache = self._make_cache(1, [True, False], [True, True])
        cache.host_memory_mode = "buffer_only"
        cache.tree_core.enable_storage = True
        cache._all_reduce_attn_groups = MagicMock()
        cache._drain_storage_control_queues_impl = MagicMock()
        cc = cache.cache_controller
        for size, name in enumerate(
            (
                "prefetch_hit_queue",
                "ack_prefetch_queue",
                "ack_backup_queue",
                "host_mem_release_queue",
            ),
            start=1,
        ):
            queue = Queue()
            for _ in range(size):
                queue.put(object())
            setattr(cc, name, queue)

        cache.check_hicache_events()

        cache._all_reduce.assert_not_called()
        cache._all_reduce_attn_groups.assert_called_once()
        cache.writing_check.assert_called_once_with(finish_count=1)
        cache.loading_check.assert_called_once_with(finish_count=2)
        cache._drain_storage_control_queues_impl.assert_called_once_with(
            n_storage_hit=1,
            n_ack_prefetch=2,
            n_backup=3,
            n_release=4,
            extra_release_counts={},
            log_metrics=True,
        )


if __name__ == "__main__":
    unittest.main()
