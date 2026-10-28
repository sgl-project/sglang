"""Unit tests for HiCache PP synchronization."""

import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

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
    def test_drain_waits_all_and_clears(self):
        holder = _Holder()
        works = [_FakeWork(), _FakeWork(), _FakeWork()]
        holder.work_list = list(works)

        UnifiedRadixCache._drain_async_work(holder)

        self.assertTrue(all(w.waited for w in works))
        self.assertEqual(holder.work_list, [])

    def test_drain_empty_is_noop(self):
        holder = _Holder()
        holder.work_list = []

        UnifiedRadixCache._drain_async_work(holder)

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
        cache.host_memory_mode = "cache"
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

    def test_writing_and_loading_check_drain_prefix_fifo(self):
        cache = object.__new__(UnifiedRadixCache)
        cache.pp_rank = 0
        cache.metrics_collector = None
        cache.buffer_pipeline = None
        cache.components = {}
        finished_writes = []
        finished_loads = []
        cache._finish_write_through_ack = finished_writes.append
        cache._log_write_ack_metrics = MagicMock()
        cache.dec_lock_ref = MagicMock()
        cache.dec_host_lock_ref = MagicMock()
        cache.tree_core = SimpleNamespace(finish_load_back=finished_loads.append)
        cache.ongoing_load_back = {
            10: ("node10", "l10", "hl10"),
            20: ("node20", "l20", "hl20"),
            30: ("node30", "l30", "hl30"),
        }
        w_acks = [
            SimpleNamespace(finish_event=MagicMock(), node_ids=[1]),
            SimpleNamespace(finish_event=MagicMock(), node_ids=[2]),
            SimpleNamespace(finish_event=MagicMock(), node_ids=[3]),
        ]
        l_acks = [
            SimpleNamespace(
                finish_event=MagicMock(),
                node_ids=[10],
                num_tokens_by_pool=None,
                num_bytes=0,
                timing_enabled=False,
            ),
            SimpleNamespace(
                finish_event=MagicMock(),
                node_ids=[20],
                num_tokens_by_pool=None,
                num_bytes=0,
                timing_enabled=False,
            ),
            SimpleNamespace(
                finish_event=MagicMock(),
                node_ids=[30],
                num_tokens_by_pool=None,
                num_bytes=0,
                timing_enabled=False,
            ),
        ]
        cache.cache_controller = SimpleNamespace(
            ack_write_queue=list(w_acks),
            ack_load_queue=list(l_acks),
        )

        cache.writing_check(finish_count=2)
        self.assertEqual(finished_writes, [1, 2])
        self.assertEqual(cache.cache_controller.ack_write_queue, [w_acks[2]])

        cache.loading_check(finish_count=2)
        self.assertEqual(finished_loads, ["node10", "node20"])
        self.assertEqual(cache.cache_controller.ack_load_queue, [l_acks[2]])
        self.assertEqual(list(cache.ongoing_load_back.keys()), [30])

    def test_decode_offload_progress_drains_prefix_fifo(self):
        from sglang.srt.disaggregation.decode_kvcache_offload_manager import (
            DecodeKVCacheOffloadManager,
            OffloadedState,
        )

        manager = object.__new__(DecodeKVCacheOffloadManager)
        req1 = MagicMock()
        req1.finished.return_value = False
        req2 = MagicMock()
        req2.finished.return_value = False
        manager.offloaded_state = {req1: OffloadedState(inc_len=4), req2: OffloadedState(inc_len=4)}
        manager.offload_inflight = {req1: 2, req2: 1}
        manager.ongoing_offload = {
            1: (req1, torch.arange(0, 4), [10, 11], 0.0, []),
            2: (req1, torch.arange(4, 8), [12, 13], 0.0, []),
            3: (req2, torch.arange(8, 12), [20, 21], 0.0, []),
        }
        acks = [
            SimpleNamespace(finish_event=MagicMock(), node_ids=[1]),
            SimpleNamespace(finish_event=MagicMock(), node_ids=[2]),
            SimpleNamespace(finish_event=MagicMock(), node_ids=[3]),
        ]
        manager.cache_controller = SimpleNamespace(ack_write_queue=list(acks))
        backed_up = []
        manager._trigger_backup = lambda req, indices, tokens, *args: (
            backed_up.append(tuple(tokens)) or f"hash_{tokens[0]}"
        )

        manager._check_offload_progress(2)
        self.assertEqual(backed_up, [(10, 11), (12, 13)])
        self.assertEqual(manager.cache_controller.ack_write_queue, [acks[2]])
        self.assertNotIn(req1, manager.offload_inflight)
        self.assertEqual(manager.offload_inflight[req2], 1)
        self.assertEqual(list(manager.ongoing_offload.keys()), [3])


if __name__ == "__main__":
    unittest.main()
