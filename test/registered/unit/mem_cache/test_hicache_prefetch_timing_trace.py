"""CPU coverage for the opt-in HiCache prefetch lifecycle timing trace."""

import json
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.managers.cache_controller import (
    HiCacheController,
    PrefetchOperation,
)
from sglang.srt.mem_cache.base_prefix_cache import CacheRequestHandle
from sglang.srt.mem_cache.hicache_storage import (
    HICACHE_PREFETCH_TIMING_MARKER,
    hicache_prefetch_timing_enabled,
    log_hicache_prefetch_consume,
    log_hicache_prefetch_forward_entries,
    mark_prefetch_timing,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    PrefetchOperation as HybridPrefetchOperation,
)
from sglang.srt.mem_cache.unified_radix_cache import (
    UnifiedRadixCache,
    _OngoingPrefetch,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _records(lines):
    records = []
    for line in lines:
        marker_index = line.index(HICACHE_PREFETCH_TIMING_MARKER)
        payload = line[marker_index + len(HICACHE_PREFETCH_TIMING_MARKER) + 1 :]
        records.append(json.loads(payload))
    return records


def _timed_operation(rid="rid", operation_cls=PrefetchOperation):
    operation = operation_cls(rid, [1] * 16)
    operation.host_indices = torch.arange(16)
    operation.completed_tokens = 16
    operation.enqueue_ts = 1_000
    operation.lookup_start_ts = 2_000
    operation.lookup_end_ts = 3_000
    operation.read_start_ts = 4_000
    operation.read_end_ts = 5_000
    return operation


class TestTimingTraceFormat(CustomTestCase):
    def tearDown(self):
        envs.SGLANG_HICACHE_PREFETCH_TIMING_TRACE.clear()
        envs.SGLANG_HICACHE_TRACE_RUN_ID.clear()

    def test_disabled_by_default_is_a_noop(self):
        envs.SGLANG_HICACHE_PREFETCH_TIMING_TRACE.clear()
        self.assertFalse(hicache_prefetch_timing_enabled())
        operation = PrefetchOperation("disabled", [1, 2, 3, 4])
        with self.assertNoLogs("sglang.srt.mem_cache.hicache_storage", level="INFO"):
            mark_prefetch_timing(operation, "enqueue_ts")
            log_hicache_prefetch_consume(operation)
            log_hicache_prefetch_forward_entries([SimpleNamespace(rid="r1")])
        self.assertIsNone(operation.enqueue_ts)

    def test_consume_record_contains_boundaries_and_stage_durations(self):
        operation = _timed_operation("summary")
        with (
            envs.SGLANG_HICACHE_PREFETCH_TIMING_TRACE.override(True),
            envs.SGLANG_HICACHE_TRACE_RUN_ID.override("run-abc"),
            mock.patch("time.perf_counter_ns", return_value=9_000),
            self.assertLogs(
                "sglang.srt.mem_cache.hicache_storage", level="INFO"
            ) as logs,
        ):
            log_hicache_prefetch_consume(
                operation,
                requested_tokens=16,
                completed_tokens=16,
                loaded_tokens=12,
            )

        record = _records(logs.output)[0]
        self.assertEqual(record["event"], "consume")
        self.assertEqual(record["run_id"], "run-abc")
        self.assertEqual(record["rid"], "summary")
        self.assertEqual(record["operation_id"], operation.id)
        self.assertEqual(record["lookup_queue_ns"], 1_000)
        self.assertEqual(record["lookup_ns"], 1_000)
        self.assertEqual(record["read_queue_ns"], 1_000)
        self.assertEqual(record["read_ns"], 1_000)
        self.assertEqual(record["done_wait_consume_ns"], 4_000)
        self.assertEqual(record["loaded_tokens"], 12)

    def test_missing_or_out_of_order_boundaries_are_omitted(self):
        operation = SimpleNamespace(
            request_id="partial",
            enqueue_ts=5_000,
            lookup_start_ts=1_000,
            lookup_end_ts=2_000,
        )
        with (
            envs.SGLANG_HICACHE_PREFETCH_TIMING_TRACE.override(True),
            mock.patch("time.perf_counter_ns", return_value=3_000),
            self.assertLogs(
                "sglang.srt.mem_cache.hicache_storage", level="INFO"
            ) as logs,
        ):
            log_hicache_prefetch_consume(operation)

        record = _records(logs.output)[0]
        self.assertNotIn("lookup_queue_ns", record)
        self.assertEqual(record["lookup_ns"], 1_000)
        self.assertNotIn("read_ns", record)
        self.assertNotIn("done_wait_consume_ns", record)

    def test_forward_entries_share_one_timestamp(self):
        requests = [SimpleNamespace(rid="r1"), SimpleNamespace(rid="r2")]
        with (
            envs.SGLANG_HICACHE_PREFETCH_TIMING_TRACE.override(True),
            self.assertLogs(
                "sglang.srt.mem_cache.hicache_storage", level="INFO"
            ) as logs,
        ):
            log_hicache_prefetch_forward_entries(
                requests + [SimpleNamespace()], timestamp_monotonic_ns=777
            )

        records = _records(logs.output)
        self.assertEqual([record["rid"] for record in records], ["r1", "r2"])
        self.assertEqual(
            [record["timestamp_monotonic_ns"] for record in records], [777, 777]
        )


class TestTraceAttemptsAndPartialProgress(CustomTestCase):
    def test_attempts_and_terminal_deduplication(self):
        first = HybridPrefetchOperation(CacheRequestHandle("same", 3), [1, 2])
        second = HybridPrefetchOperation(CacheRequestHandle("same", 4), [1, 2])
        with (
            envs.SGLANG_HICACHE_PREFETCH_TIMING_TRACE.override(True),
            self.assertLogs(
                "sglang.srt.mem_cache.hicache_storage", level="INFO"
            ) as logs,
        ):
            log_hicache_prefetch_consume(first)
            log_hicache_prefetch_consume(first)
            log_hicache_prefetch_consume(second)
        records = _records(logs.output)
        self.assertEqual([row["attempt_id"] for row in records], [3, 4])
        self.assertNotEqual(records[0]["operation_id"], records[1]["operation_id"])

    def test_incomplete_read_has_a_visible_start(self):
        operation = PrefetchOperation("blocked", [1])
        with (
            envs.SGLANG_HICACHE_PREFETCH_TIMING_TRACE.override(True),
            mock.patch("time.perf_counter_ns", return_value=42),
            self.assertLogs(
                "sglang.srt.mem_cache.hicache_storage", level="INFO"
            ) as logs,
        ):
            mark_prefetch_timing(operation, "read_start_ts")
        record = _records(logs.output)[0]
        self.assertEqual(record["event"], "read_start")
        self.assertEqual(record["rid"], "blocked")
        self.assertEqual(record["timestamp_monotonic_ns"], 42)
        self.assertIsNone(operation.read_end_ts)

    def test_pp_handoff_is_traced_once(self):
        operation = HybridPrefetchOperation(CacheRequestHandle("pp", 2), [1, 2])
        operation.completed_tokens = 2
        operation.host_indices = torch.arange(2)
        state = SimpleNamespace(
            operation=operation, consumed=False, ready_event=threading.Event()
        )
        state.ready_event.set()
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.pp_prefetch_state_lock = threading.Lock()
        controller.pp_prefetch_states = {"pp": state}
        controller.pp_prefetch_decisions = {}
        with (
            envs.SGLANG_HICACHE_PREFETCH_TIMING_TRACE.override(True),
            self.assertLogs(
                "sglang.srt.mem_cache.hicache_storage", level="INFO"
            ) as logs,
        ):
            self.assertIs(controller.take_ready_pp_prefetch("pp"), state)
            self.assertIsNone(controller.take_ready_pp_prefetch("pp"))
        records = _records(logs.output)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["path"], "pp_handoff")
        self.assertEqual(records[0]["attempt_id"], 2)


class TestControllerTimingBoundaries(CustomTestCase):
    def test_base_and_hybrid_enqueue_are_marked_before_queue_put(self):
        # main-0923: the hybrid controller takes a CacheRequestHandle.
        for controller_cls, expected_cls, request in (
            (HiCacheController, PrefetchOperation, "enqueue"),
            (
                HybridCacheController,
                HybridPrefetchOperation,
                CacheRequestHandle("enqueue", 0),
            ),
        ):
            controller = controller_cls.__new__(controller_cls)
            controller.prefetch_queue = mock.MagicMock()
            with (
                envs.SGLANG_HICACHE_PREFETCH_TIMING_TRACE.override(True),
                mock.patch("time.perf_counter_ns", return_value=42),
            ):
                operation = controller.prefetch(request, [1, 2, 3, 4])
            self.assertIsInstance(operation, expected_cls)
            self.assertEqual(operation.enqueue_ts, 42)
            controller.prefetch_queue.put.assert_called_once_with(operation)

    def test_lookup_and_read_boundaries_wrap_controller_work(self):
        controller = HiCacheController.__new__(HiCacheController)
        controller.storage_stop_event = threading.Event()
        controller.prefetch_threshold = 16
        controller.page_size = 4
        operation = PrefetchOperation("lifecycle", [1] * 16)
        operation.host_indices = torch.arange(16)

        controller.prefetch_queue = mock.MagicMock()

        def dequeue_prefetch(*args, **kwargs):
            controller.storage_stop_event.set()
            return operation

        controller.prefetch_queue.get.side_effect = dequeue_prefetch
        controller.prefetch_queue.empty.return_value = True
        controller._storage_hit_query = mock.MagicMock(return_value=(["h"] * 4, 16))
        # v0.5.19: the hit count is all-reduced through the generic _all_reduce
        # helper, and the revoke decision moved to the scheduler thread.
        controller.prefetch_hits_sync_groups = None
        controller._all_reduce = mock.MagicMock()
        controller.prefetch_hit_queue = mock.MagicMock()

        with (
            envs.SGLANG_HICACHE_PREFETCH_TIMING_TRACE.override(True),
            mock.patch("time.perf_counter_ns", side_effect=[100, 150, 200]),
            mock.patch("sglang.srt.managers.cache_controller.threading.Thread"),
        ):
            controller.prefetch_thread_func()

        self.assertEqual(operation.lookup_start_ts, 100)
        self.assertEqual(operation.lookup_end_ts, 200)
        controller.prefetch_hit_queue.put.assert_called_once_with(operation)

        controller.storage_stop_event.clear()
        controller.prefetch_buffer = mock.MagicMock()

        def dequeue_read(*args, **kwargs):
            controller.storage_stop_event.set()
            return operation

        controller.prefetch_buffer.get.side_effect = dequeue_read
        controller._page_transfer = mock.MagicMock()
        controller.prefetch_sync_queue = mock.MagicMock()
        with (
            envs.SGLANG_HICACHE_PREFETCH_TIMING_TRACE.override(True),
            mock.patch("time.perf_counter_ns", side_effect=[300, 400]),
        ):
            controller.prefetch_io_aux_func()

        self.assertEqual(operation.read_start_ts, 300)
        self.assertEqual(operation.read_end_ts, 400)
        controller._page_transfer.assert_called_once_with(operation)


class TestUnifiedRadixConsumeHook(CustomTestCase):
    """main-0923 port: HiRadixCache is gone; the consume boundary is emitted
    by UnifiedRadixCache._handle_prefetch_result after the host insert."""

    def _cache_with_ongoing_prefetch(self, handle, operation, prefix_len):
        cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
        # page_size is a property backed by the tree core.
        cache.tree_core = mock.MagicMock()
        cache.tree_core.page_size = 4
        cache.enable_storage_metrics = False
        cache.storage_metrics_collector = None
        cache.buffer_pipeline = None
        cache.prefetch_loaded_tokens_by_reqid = {}
        cache.prefetch_loaded_storage_start_by_reqid = {}
        cache.cache_controller = mock.MagicMock()
        cache.cache_controller.prefetch_tokens_occupied = 16
        cache._check_hybrid_prefetch_result = mock.MagicMock(return_value=True)
        cache._resolve_storage_prefetch_tokens = mock.MagicMock()
        cache._apply_cache_actions = mock.MagicMock()
        cache.dec_host_lock_ref = mock.MagicMock()
        cache.tree_core.insert_host.return_value = SimpleNamespace(
            cache_actions=[], host_insert_dropped=False, prefix_len=prefix_len
        )
        cache.ongoing_prefetch = {
            handle: _OngoingPrefetch(
                anchor_node_id=1,
                prefetch_key=[1] * 16,
                host_indices=torch.arange(16),
                operation=operation,
                anchor_lock_params=None,
                comp_xfers={},
            )
        }
        return cache

    def test_completed_prefetch_emits_consume(self):
        handle = CacheRequestHandle("unified", 0)
        operation = _timed_operation(handle, operation_cls=HybridPrefetchOperation)
        operation.hash_value = ["h0", "h1", "h2", "h3"]
        operation.storage_start = 0
        cache = self._cache_with_ongoing_prefetch(handle, operation, prefix_len=4)

        with (
            envs.SGLANG_HICACHE_PREFETCH_TIMING_TRACE.override(True),
            mock.patch("time.perf_counter_ns", return_value=9_000),
            self.assertLogs(
                "sglang.srt.mem_cache.hicache_storage", level="INFO"
            ) as logs,
        ):
            cache._handle_prefetch_result(operation)

        record = _records(logs.output)[0]
        self.assertEqual(record["event"], "consume")
        self.assertEqual(record["rid"], "unified")
        self.assertEqual(record["done_wait_consume_ns"], 4_000)
        self.assertEqual(record["requested_tokens"], 16)
        self.assertEqual(record["completed_tokens"], 16)
        self.assertEqual(record["loaded_tokens"], 12)
        self.assertEqual(cache.prefetch_loaded_tokens_by_reqid[handle], 12)

    def test_revocation_emits_terminal_record(self):
        handle = CacheRequestHandle("revoked", 2)
        operation = _timed_operation(handle, operation_cls=HybridPrefetchOperation)
        cache = self._cache_with_ongoing_prefetch(handle, operation, prefix_len=0)
        cache.host_memory_mode = "cache"
        cache._finish_storage_prefetch = mock.Mock()
        cache._invalidate_absent_from_hit_query = mock.Mock()
        cache._prefetch_occupied_span = mock.Mock(return_value=16)
        with (
            envs.SGLANG_HICACHE_PREFETCH_TIMING_TRACE.override(True),
            self.assertLogs(
                "sglang.srt.mem_cache.hicache_storage", level="INFO"
            ) as logs,
        ):
            cache.revoke_pending_prefetch(handle)
        record = _records(logs.output)[0]
        self.assertEqual(record["path"], "retire")
        self.assertEqual(record["attempt_id"], 2)
        self.assertNotIn(handle, cache.ongoing_prefetch)

    def test_buffer_handoff_and_hybrid_discard_are_traced(self):
        for path in ("buffer_staging", "hybrid_discard"):
            with self.subTest(path=path):
                handle = CacheRequestHandle(path, 1)
                operation = _timed_operation(
                    handle, operation_cls=HybridPrefetchOperation
                )
                operation.hash_value = ["h0", "h1", "h2", "h3"]
                cache = self._cache_with_ongoing_prefetch(
                    handle, operation, prefix_len=0
                )
                cache.buffer_pipeline = mock.Mock()
                cache._check_hybrid_prefetch_result.return_value = (
                    path != "hybrid_discard"
                )
                with (
                    envs.SGLANG_HICACHE_PREFETCH_TIMING_TRACE.override(True),
                    self.assertLogs(
                        "sglang.srt.mem_cache.hicache_storage", level="INFO"
                    ) as logs,
                ):
                    cache._handle_prefetch_result(operation)
                self.assertEqual(_records(logs.output)[0]["path"], path)
                cache.tree_core.insert_host.assert_not_called()

    def test_disabled_trace_emits_nothing(self):
        handle = CacheRequestHandle("quiet", 0)
        operation = _timed_operation(handle, operation_cls=HybridPrefetchOperation)
        operation.hash_value = ["h0", "h1", "h2", "h3"]
        operation.storage_start = 0
        cache = self._cache_with_ongoing_prefetch(handle, operation, prefix_len=4)
        envs.SGLANG_HICACHE_PREFETCH_TIMING_TRACE.clear()
        with self.assertNoLogs("sglang.srt.mem_cache.hicache_storage", level="INFO"):
            cache._handle_prefetch_result(operation)
        self.assertEqual(cache.prefetch_loaded_tokens_by_reqid[handle], 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
