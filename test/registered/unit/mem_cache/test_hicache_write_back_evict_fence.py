"""Write-back eviction fences on a PD prefill server: which backups may skip the
in-flight forward, and the scheduler events they wait on. Streams are mocks."""

import logging
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import sglang.srt.mem_cache.unified_radix_cache as urc_mod
from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import CacheOperation, HiCacheController
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _controller(write_policy="write_back", forward_start_event="F"):
    cc = object.__new__(HiCacheController)
    cc.write_policy = write_policy
    cc.write_queue = []
    cc.ack_write_queue = []
    cc.write_fence_stream = "FWD_STREAM"
    cc.forward_start_event = None
    cc.war_barrier_event = None
    cc.war_barrier_seq = 0
    cc.war_barrier_after_forward = False
    if forward_start_event is not None:
        cc.record_forward_start(forward_start_event)
    cc.start_writing = MagicMock()
    return cc


def _cache(
    *,
    pd_prefill=True,
    write_policy="write_back",
    forward_start_event="F",
    in_scope=True,
    paths=None,
):
    """A cache with the write-back eviction state UnifiedRadixCache.__init__ and
    init_hicache set; ``paths`` maps node id -> (node_id, parent_id) path."""
    paths = paths or {}

    def collect_path_info(node_ids):
        return tuple(
            tuple(SimpleNamespace(node_id=n, parent_id=p) for n, p in paths.get(i, ()))
            for i in node_ids
        )

    cache = object.__new__(UnifiedRadixCache)
    cache.cache_controller = _controller(write_policy, forward_start_event)
    cache.evicts_by_joint_lru = False
    cache.slot_id = MagicMock(return_value=ComponentType.MAMBA)
    cache._window_sub_pools_by_slot = MagicMock(return_value={})
    cache.ongoing_write_through = {}
    cache.tree_components = (ComponentType.FULL, ComponentType.MAMBA)
    cache.tree_core = MagicMock()
    cache.tree_core.collect_path_info.side_effect = collect_path_info
    cache.is_write_back = write_policy == "write_back"
    cache._wb_evict_pd_prefill = pd_prefill
    cache._wb_evict_relaxed_fence_logged = False
    cache._wb_overlap_in_scope = in_scope
    cache._wb_overlap_stream = None
    cache._wb_overlap_max_seen_id = -1
    cache._wb_overlap_watermark = -1
    cache._wb_overlap_barrier_seq = 0
    cache._wb_overlap_counts = {"overlap": 0, "fresh": 0, "scope": 0}
    cache._wb_batch_on_overlap = False
    cache._wb_batch_on_current = False
    return cache


def _after_barrier(paths=None, **kwargs):
    """Leaf 4 (parent 0) was inserted before a WAR barrier that follows the
    in-flight forward's launch."""
    cache = _cache(paths={4: ((4, 0),), **(paths or {})}, **kwargs)
    cache._wb_overlap_observe(4)
    cache.cache_controller.record_war_barrier("W")
    return cache


class TestRelaxedFencePredicate(CustomTestCase):
    def test_requires_pd_prefill_write_back_forward_event_and_env(self):
        self.assertTrue(envs.SGLANG_HICACHE_WB_EVICT_RELAXED_FENCE.get())
        self.assertTrue(_cache()._wb_evict_relaxed_fence())
        self.assertFalse(_cache(pd_prefill=False)._wb_evict_relaxed_fence())
        self.assertFalse(_cache(write_policy="write_through")._wb_evict_relaxed_fence())
        self.assertFalse(_cache(forward_start_event=None)._wb_evict_relaxed_fence())
        with envs.SGLANG_HICACHE_WB_EVICT_RELAXED_FENCE.override(False):
            self.assertFalse(_cache()._wb_evict_relaxed_fence())
        cache = _cache()
        cache.cache_controller = None
        self.assertFalse(cache._wb_evict_relaxed_fence())


class TestWritingCheckFence(CustomTestCase):
    def test_victim_opt_in_fences_on_forward_start_and_logs_once(self):
        cache = _cache()
        with self.assertLogs(urc_mod.logger, level=logging.INFO) as logs:
            cache.writing_check(write_back=True, relaxed_fence=True)
            cache.writing_check(write_back=True, relaxed_fence=True)
        self.assertEqual(
            cache.cache_controller.start_writing.call_args_list,
            [call(fence_event="F"), call(fence_event="F")],
        )
        self.assertEqual(sum("relaxed D2H fence" in m for m in logs.output), 1)

    def test_full_fence_without_opt_in_or_outside_scope(self):
        for cache, relaxed_fence in (
            (_cache(), False),
            (_cache(pd_prefill=False), True),
            (_cache(forward_start_event=None), True),
        ):
            cache.writing_check(write_back=True, relaxed_fence=relaxed_fence)
            cache.cache_controller.start_writing.assert_called_once_with(
                fence_event=None
            )
        cache = _cache()
        with envs.SGLANG_HICACHE_WB_EVICT_RELAXED_FENCE.override(False):
            cache.writing_check(write_back=True, relaxed_fence=True)
        cache.cache_controller.start_writing.assert_called_once_with(fence_event=None)


class TestEvictionCallSites(CustomTestCase):
    def test_internal_state_backup_keeps_full_fence(self):
        # An older SWA window of the in-flight checkpoint is evictable here.
        cache = _after_barrier()
        cache.writing_check = MagicMock()
        cache._execute_and_commit_kv_backup = MagicMock(return_value=4)
        cache._wb_evict_overlap_stream = MagicMock()
        cache._flush_queued_writes_fenced = MagicMock()
        self.assertTrue(cache.backup_node_for_write_back(7))
        self.assertEqual(cache.writing_check.call_args_list, [call(write_back=True)])
        cache._wb_evict_overlap_stream.assert_not_called()
        cache._flush_queued_writes_fenced.assert_not_called()

    def test_device_eviction_victim_opts_in(self):
        cache = _cache()
        cache.tree_components = (ComponentType.FULL,)
        cache.tree_core.full_evictable_size.return_value = 0
        cache.tree_core.mark_write_back_queued.return_value = 3
        cache.ongoing_write_through = {5: SimpleNamespace(publish_node_ids=[5])}
        cache._evict_device_next_node = MagicMock(
            side_effect=[(5, True), (None, False)]
        )
        cache._evict_device_leaf = MagicMock(return_value=SimpleNamespace(node_ids=[5]))
        cache._execute_and_commit_kv_backup = MagicMock(return_value=3)
        cache._demote = MagicMock()
        cache.writing_check = MagicMock()
        cache._evict_components({ComponentType.FULL: 1}, {ComponentType.FULL: 0})
        cache.writing_check.assert_called_once_with(write_back=True, relaxed_fence=True)
        cache._demote.assert_called_once()

    def _evict(self, cache):
        order = MagicMock()
        cache.disable = False
        cache.tree_components = (ComponentType.FULL,)
        cache.cache_controller.write_queue = ["insert-time backup"]
        cache.cache_controller.start_writing = order.start_writing
        cache._evict_request_by_type = MagicMock(return_value={ComponentType.FULL: 1})
        cache._evict_components = order.evict_components
        cache.writing_check = order.writing_check
        cache.update_eviction_metrics = MagicMock()
        cache._evict(SimpleNamespace(num_tokens=1, swa_num_tokens=0, mamba_num=0))
        return order.mock_calls

    def test_evict_submits_earlier_writes_with_full_fence_first(self):
        # Insert-time backups may read pages the in-flight forward still writes.
        calls = self._evict(_cache())
        self.assertEqual(calls[0], call.start_writing())
        self.assertEqual(calls[1][0], "evict_components")
        self.assertEqual(calls[-1], call.writing_check(write_back=True))

    def test_evict_without_relaxed_fence_does_not_presubmit(self):
        calls = self._evict(_cache(pd_prefill=False))
        self.assertNotIn("start_writing", [name for name, _, _ in calls])


class TestStartWritingFence(CustomTestCase):
    def _start_writing(self, **kwargs):
        cc = object.__new__(HiCacheController)
        cc.write_queue = ["op"]
        cc.ack_write_queue = []
        cc._move_write_operation = MagicMock(return_value=("H", "D", None))
        cc._l2_transfers = MagicMock(return_value=["T"])
        cc._num_tokens_by_pool = MagicMock(return_value={})
        cc._transfer_num_bytes = MagicMock(return_value=0)
        cc.l2_transfer_engine = MagicMock()
        cc.mem_pool_device_allocator = MagicMock()
        cc.write_fence_stream = "FWD_STREAM"
        cc.kv_storage = None
        merged = SimpleNamespace(node_ids=[1], device_indices=[0, 1])
        with patch.object(CacheOperation, "merge_ops", return_value=merged):
            cc.start_writing(**kwargs)
        self.assertEqual(len(cc.ack_write_queue), 1)
        return cc.l2_transfer_engine.device_to_host_stream

    def test_fence_event_replaces_forward_stream_fence(self):
        d2h = self._start_writing(fence_event="F")
        d2h.wait_event.assert_called_once_with("F")
        d2h.wait_stream.assert_not_called()
        d2h = self._start_writing()
        d2h.wait_stream.assert_called_once_with("FWD_STREAM")
        d2h.wait_event.assert_not_called()


class TestFreshnessGuard(CustomTestCase):
    def test_watermark_is_the_largest_id_seen_before_the_barrier(self):
        cache = _cache()
        cc = cache.cache_controller
        cache._wb_overlap_observe(5)
        cache._wb_overlap_observe(9)
        cache._wb_overlap_observe(None)
        self.assertEqual(cache._wb_overlap_watermark, -1)
        cc.record_war_barrier("W1")
        # First look after the barrier: snapshot, then fold in the new id.
        cache._wb_overlap_observe(12)
        self.assertEqual(cache._wb_overlap_watermark, 9)
        cache._wb_overlap_observe(15)
        self.assertEqual(cache._wb_overlap_watermark, 9)
        cc.record_war_barrier("W2")
        cache._wb_overlap_sync_watermark()
        self.assertEqual(cache._wb_overlap_watermark, 15)

    def test_chain_is_fresh_if_created_or_split_after_the_barrier(self):
        paths = {
            4: ((4, 0),),
            6: ((4, 0), (6, 4)),
            11: ((11, 0),),  # created after the barrier
            7: ((11, 0), (7, 11)),  # split tail: old id, new parent
        }
        cache = _cache(paths=paths)
        cache._wb_overlap_observe(10)
        cache.cache_controller.record_war_barrier("W")
        self.assertFalse(cache._wb_overlap_chain_is_fresh([4, 6]))
        self.assertTrue(cache._wb_overlap_chain_is_fresh([11]))
        self.assertTrue(cache._wb_overlap_chain_is_fresh([4, 7]))
        self.assertTrue(cache._wb_overlap_chain_is_fresh([8]))  # unknown path
        self.assertTrue(cache._wb_overlap_chain_is_fresh([]))


class TestOverlapStream(CustomTestCase):
    def _decide(self, cache, node_ids=(4,)):
        stream = MagicMock()
        with patch.object(cache, "_wb_overlap_get_stream", return_value=stream):
            got = cache._wb_evict_overlap_stream(SimpleNamespace(node_ids=node_ids))
        return got, stream

    def test_waits_only_on_forward_start_and_barrier_events(self):
        got, stream = self._decide(_after_barrier())
        self.assertIs(got, stream)
        self.assertEqual(stream.wait_event.call_args_list, [call("F"), call("W")])

    def test_stays_on_current_stream(self):
        cache = _after_barrier()
        cache.cache_controller.record_forward_start("F2")  # no barrier since
        self.assertIsNone(self._decide(cache)[0])
        with envs.SGLANG_HICACHE_WB_EVICT_RELAXED_FENCE.override(False):
            self.assertIsNone(self._decide(_after_barrier())[0])
        self.assertIsNone(self._decide(_after_barrier(pd_prefill=False))[0])
        # SWA tree or page-relocating allocator.
        cache = _after_barrier(in_scope=False)
        self.assertIsNone(self._decide(cache)[0])
        self.assertEqual(cache._wb_overlap_counts["scope"], 1)
        # Leaf 9 was not seen before the barrier.
        cache = _after_barrier(paths={9: ((9, 0),)})
        self.assertIsNone(self._decide(cache, node_ids=(9,))[0])
        self.assertEqual(cache._wb_overlap_counts["fresh"], 1)

    def _stream_ctx(self, events):
        def stream(s):
            ctx = MagicMock()
            ctx.__enter__.side_effect = lambda *a: events.append(("enter", s))
            ctx.__exit__.side_effect = lambda *a: events.append(("exit", s))
            return ctx

        return patch.object(urc_mod.device_module, "stream", side_effect=stream)

    def _batch(self, streams):
        """Queue one victim per entry of ``streams`` (None: current stream),
        then submit; returns the recorded order."""
        cache = _after_barrier()
        cache._wb_overlap_stream = "S"
        events = []
        cache._execute_and_commit_kv_backup = MagicMock(
            side_effect=lambda *a, **k: events.append("backup") or 3
        )
        cache.writing_check = MagicMock(side_effect=lambda **k: events.append(k))
        with (
            patch.object(cache, "_wb_evict_overlap_stream", side_effect=streams),
            patch.object(urc_mod.device_module, "current_stream") as current,
            self._stream_ctx(events),
        ):
            for _ in streams:
                self.assertEqual(cache._queue_write_back_victim("BKV"), 3)
            cache._wait_write_back_victims()
        self.assertFalse(cache._wb_batch_on_overlap or cache._wb_batch_on_current)
        return events, current.return_value.wait_stream

    def test_overlap_batch_is_built_and_submitted_on_the_overlap_stream(self):
        events, wait_stream = self._batch(["S", "S"])
        backup = ["backup", ("exit", "S")]
        submit = [{"write_back": True, "relaxed_fence": True}, ("exit", "S")]
        self.assertEqual(
            events,
            [("enter", "S"), *backup, ("enter", "S"), *backup, ("enter", "S"), *submit],
        )
        self.assertEqual(wait_stream.call_args_list, [call("S"), call("S")])

    def test_mixed_batch_submits_on_the_current_stream_after_the_overlap_stream(self):
        # The copy must see index tensors built on both streams: the current
        # stream waits for the overlap stream before the submit.
        events, wait_stream = self._batch(["S", None])
        self.assertEqual(
            events,
            [
                ("enter", "S"),
                "backup",
                ("exit", "S"),
                "backup",
                {"write_back": True, "relaxed_fence": True},
            ],
        )
        wait_stream.assert_called_once_with("S")


class TestSchedulerEvents(CustomTestCase):
    def _scheduler(self, controller, shared_read_done_event=None):
        order = MagicMock()
        order.schedule_stream.record_event.return_value = "W"
        runner = SimpleNamespace(shared_read_done_event=shared_read_done_event)
        scheduler = SimpleNamespace(
            _war_barrier_enabled=True,
            model_worker=SimpleNamespace(last_shared_read_runner=runner),
            _hicache_fence_controller=controller,
            schedule_stream=order.schedule_stream,
            forward_stream="FWD_STREAM",
        )
        return scheduler, order

    def test_war_barrier_event_is_recorded_before_the_wait(self):
        for shared_read_done_event, wait in (
            (None, "schedule_stream.wait_stream"),
            ("SR", "schedule_stream.wait_event"),
        ):
            cc = _controller()
            scheduler, order = self._scheduler(cc, shared_read_done_event)
            Scheduler._apply_war_barrier(scheduler)
            self.assertEqual(
                [name for name, _, _ in order.mock_calls],
                ["schedule_stream.record_event", wait],
            )
            self.assertEqual(
                (
                    cc.war_barrier_event,
                    cc.war_barrier_seq,
                    cc.war_barrier_after_forward,
                ),
                ("W", 1, True),
            )

    def test_war_barrier_without_controller(self):
        scheduler, order = self._scheduler(controller=None)
        Scheduler._apply_war_barrier(scheduler)
        self.assertEqual(
            [name for name, _, _ in order.mock_calls], ["schedule_stream.wait_stream"]
        )

    def test_barrier_follows_forward_only_until_the_next_launch(self):
        cc = _controller(forward_start_event=None)
        cc.record_war_barrier("W0")  # no forward launched yet
        self.assertFalse(cc.war_barrier_after_forward)
        cc.record_forward_start("F1")
        cc.record_war_barrier("W1")
        self.assertTrue(cc.war_barrier_after_forward)
        cc.record_forward_start("F2")
        self.assertFalse(cc.war_barrier_after_forward)
        self.assertEqual(cc.war_barrier_seq, 2)


if __name__ == "__main__":
    unittest.main()
