"""Unit tests for HiCache PP synchronization."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _FakeWork:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


class _Holder:
    """Minimal carrier exposing only what _drain_async_work touches."""


class TestPPSyncDrain(unittest.TestCase):
    def _drain_fns(self):
        return (HiRadixCache._drain_async_work, UnifiedRadixCache._drain_async_work)

    def test_drain_waits_all_and_clears(self):
        for drain in self._drain_fns():
            holder = _Holder()
            works = [_FakeWork(), _FakeWork(), _FakeWork()]
            holder.work_list = list(works)

            drain(holder)

            self.assertTrue(all(w.waited for w in works))
            self.assertEqual(holder.work_list, [])

    def test_drain_empty_is_noop(self):
        for drain in self._drain_fns():
            holder = _Holder()
            holder.work_list = []

            drain(holder)

            self.assertEqual(holder.work_list, [])


class TestUnifiedPPSyncBatching(unittest.TestCase):
    def _make_cache(self, pp_rank, write_ready, load_ready):
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(enable_storage=False)
        cache.pp_rank = pp_rank
        cache.pp_size = 2
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.buffer_pipeline = None
        cache._drain_async_work = MagicMock()
        cache._all_reduce = MagicMock()
        cache.writing_check = MagicMock()
        cache.loading_check = MagicMock()
        cache.drain_storage_control_queues = MagicMock()
        cache.cache_controller = SimpleNamespace(
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

    def test_pp_batches_write_and_load_counts_once(self):
        leader = self._make_cache(0, [True, False], [True, True])
        leader.check_hicache_events()

        leader._all_reduce.assert_called_once()
        self.assertEqual(leader._all_reduce.call_args.args[0].tolist(), [1, 2])
        leader.writing_check.assert_called_once_with(finish_count=1)
        leader.loading_check.assert_called_once_with(finish_count=2)

        follower = self._make_cache(1, [True], [True])
        follower._all_reduce.side_effect = lambda counts, _: counts.fill_(1)
        follower.check_hicache_events()

        for queue in (
            follower.cache_controller.ack_write_queue,
            follower.cache_controller.ack_load_queue,
        ):
            queue[0].finish_event.query.assert_not_called()
        follower._all_reduce.assert_called_once()
        follower.writing_check.assert_called_once_with(finish_count=1)
        follower.loading_check.assert_called_once_with(finish_count=1)


if __name__ == "__main__":
    unittest.main()
