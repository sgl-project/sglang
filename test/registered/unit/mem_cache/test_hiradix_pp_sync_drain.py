"""Unit tests for HiCache PP synchronization."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

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
        cache.tree_core = SimpleNamespace(
            enable_storage=False,
            write_back_duplicate_reclaim_digest=0,
        )
        cache.pp_rank = pp_rank
        cache.pp_size = 2
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.buffer_pipeline = None
        cache.linker = None
        cache._hicache_async_ack_sync = False
        cache._pending_ready_counts = None
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


if __name__ == "__main__":
    unittest.main()


class TestUnifiedAsyncAckSync(unittest.TestCase):
    """The pipelined ack-count reduce: nothing is popped on the first step, the
    previous step's reduced counts are popped on the next one, and the reduce
    issued after popping counts only the acks still queued."""

    def _make_cache(self, write_ready):
        cache = object.__new__(UnifiedRadixCache)
        cache.tree_core = SimpleNamespace(
            enable_storage=False,
            write_back_duplicate_reclaim_digest=0,
        )
        cache.pp_rank = 0
        cache.pp_size = 1
        cache.enable_storage = False
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.buffer_pipeline = None
        cache.linker = None
        cache._hicache_async_ack_sync = True
        cache._pending_ready_counts = None
        cache._ready_counts_group = object()
        cache._drain_async_work = MagicMock()
        cache.loading_check = MagicMock()
        cache.cache_controller = SimpleNamespace(
            start_writing=MagicMock(),
            ack_write_queue=[
                SimpleNamespace(
                    finish_event=SimpleNamespace(query=MagicMock(return_value=ready))
                )
                for ready in write_ready
            ],
            ack_load_queue=[],
        )
        popped = []

        def writing_check(finish_count):
            popped.append(finish_count)
            del cache.cache_controller.ack_write_queue[:finish_count]

        cache.writing_check = writing_check
        return cache, popped

    def test_pops_previous_counts_then_reduces_the_remainder(self):
        cache, popped = self._make_cache([True, True, False])
        issued = []

        def fake_all_reduce(tensor, op, group, async_op):
            issued.append(tensor.clone())
            return SimpleNamespace(wait=MagicMock())

        with patch.object(torch.distributed, "all_reduce", fake_all_reduce):
            cache.check_hicache_events()
            # First step: nothing to consume, one reduce issued with 2 ready acks.
            self.assertEqual(popped, [])
            self.assertEqual(issued[0][0].item(), 2)

            cache.check_hicache_events()
            # Second step: pop the 2 acks from step one, then reduce the rest.
            self.assertEqual(popped, [2])
            self.assertEqual(len(cache.cache_controller.ack_write_queue), 1)
            self.assertEqual(issued[1][0].item(), 0)

    def test_reduce_result_is_waited_before_use(self):
        cache, popped = self._make_cache([True])
        waited = MagicMock()

        def fake_all_reduce(tensor, op, group, async_op):
            return SimpleNamespace(wait=waited)

        with patch.object(torch.distributed, "all_reduce", fake_all_reduce):
            cache.check_hicache_events()
            waited.assert_not_called()
            cache.check_hicache_events()
            waited.assert_called_once()
            self.assertEqual(popped, [1])
