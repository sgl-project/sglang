"""Buffer-only backup drops: every drop site labels the metric with its reason
and logs a rate-limited line; the write window is published every tick.

Separate from test_buffer_mode_sidecar.py because these cases need a metrics
collector double (storage metrics on) and a recording collector class, which
the sidecar fixtures run without.
"""

import logging
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache.buffer_mode import pipeline as pipeline_module
from sglang.srt.mem_cache.buffer_mode.pipeline import (
    BACKUP_DROP_REASONS,
    BufferModePipeline,
)
from sglang.srt.mem_cache.buffer_mode.storage_existence_cache import (
    StorageExistenceCache,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components.base import ComponentType
from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import (
    BufferBackupSnapshot,
    BufferBackupState,
)
from sglang.srt.observability.metrics_collector import StorageMetricsCollector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE_SIZE = 2
HOST_POOL_TOKENS = 32
# max(0.2 * pool, pool - 0 prefetch - pool // 10) for HOST_POOL_TOKENS.
LIVE_CAP = 29


class _RecordingMetric:
    def __init__(self, *args, name=None, labelnames=(), **kwargs):
        self.name = name if name is not None else args[0]
        self.labelnames = tuple(labelnames)
        self.seeded_reasons = set()
        self.increments = []
        self.sets = []

    def labels(self, **labels):
        self.seeded_reasons.add(labels.get("reason"))
        return SimpleNamespace(
            inc=lambda value: self.increments.append((labels, value)),
            set=lambda value: self.sets.append((labels, value)),
        )


class _RecordingStorageMetricsCollector(StorageMetricsCollector):
    _counter_cls = _RecordingMetric
    _gauge_cls = _RecordingMetric
    _histogram_cls = _RecordingMetric


class TestBufferModeBackupDrops(CustomTestCase):
    def setUp(self):
        self.snapshots: dict[int, BufferBackupSnapshot] = {}
        self.states: dict[int, BufferBackupState] = {}
        self.device_values: dict[int, torch.Tensor] = {}
        self.pipeline = self._make_pipeline()
        self.collector = self.pipeline._cache.storage_metrics_collector

    def _make_pipeline(self) -> BufferModePipeline:
        controller = MagicMock()
        controller.mem_pool_host.size = HOST_POOL_TOKENS
        # Not a HostKVCache: the pipeline takes the per-pool (non-shared) path.
        controller.mem_pool_host.anchor_entry.host_pool = SimpleNamespace()
        controller.mem_pool_host.entry_map = {}
        controller.prefetch_tokens_occupied = 0
        controller.write.side_effect = (
            lambda device_value, node_id, extra_pools, flush: torch.arange(
                len(device_value), dtype=torch.int64
            )
        )
        controller.write_storage.side_effect = iter(range(1000, 2000))

        cache = MagicMock()
        cache.cache_controller = controller
        cache.page_size = PAGE_SIZE
        cache.enable_storage = True
        cache.enable_storage_metrics = True
        cache.storage_metrics_collector = MagicMock()
        cache.hicache_storage_pass_prefix_keys = False
        cache.components = {
            ComponentType.FULL: SimpleNamespace(
                buffer_backup_parent_covered=lambda parent_id, node_id: False
            )
        }
        cache.sidecar_pool_specs = []
        cache._build_backup_sidecar.return_value = []
        cache.storage_existence_cache = StorageExistenceCache()
        tree = cache.tree_core
        tree.snapshot_buffer_backup.side_effect = lambda node_id, _: self.snapshots.get(
            node_id
        )
        tree.validate_buffer_backup.side_effect = lambda node_id, _: self.states.get(
            node_id
        )
        tree.buffer_backup_pool_keys.return_value = {}

        pipeline = BufferModePipeline.__new__(BufferModePipeline)
        pipeline._cache = cache
        pipeline._swa_window_pages = 0
        pipeline.write_backlog_cap = 1 << 20
        pipeline.reset()
        pipeline._build_backup_transfers = lambda node_id, *, kv_only=False: (
            self.device_values[node_id],
            {},
            None,
        )
        return pipeline

    def _add_node(
        self, node_id: int, *, parent: int = 0, pages: int = 2, device_rows=None
    ) -> int:
        """Register a device-resident node of ``pages`` pages; returns its
        admission-view token count."""
        parent_is_root = parent == 0
        self.snapshots[node_id] = BufferBackupSnapshot(
            node_id=node_id,
            parent_node_id=parent,
            parent_is_root=parent_is_root,
            parent_last_hash=None,
            hash_values=[f"n{node_id}-p{i}" for i in range(pages)],
            key=RadixKey(
                array("q", range(node_id * 1000, node_id * 1000 + pages * PAGE_SIZE))
            ),
            prefix_keys=None,
        )
        self.states[node_id] = BufferBackupState(parent, parent_is_root, None)
        rows = pages * PAGE_SIZE if device_rows is None else device_rows
        self.device_values[node_id] = torch.arange(rows, dtype=torch.int64)
        return pages * PAGE_SIZE

    def _evict(self, node_id: int) -> None:
        """The device LRU destroyed the node: validation and re-snapshot fail."""
        del self.states[node_id]
        del self.snapshots[node_id]

    def _drops(self) -> list[tuple[int, str]]:
        return [
            (call.args[0], call.args[1])
            for call in self.collector.log_backup_dropped_tokens.call_args_list
        ]

    def test_source_evicted_before_launch_is_labeled_and_warned(self):
        """The pack-2 mechanism: a queued intent whose node the device LRU
        evicted before its D2H launched used to vanish into an unlabeled
        counter with no log line."""
        tokens = self._add_node(1)
        self.pipeline.enqueue_backup_intent(1)
        self.assertEqual(self.pipeline.write_backlog_tokens_, tokens)
        self._evict(1)
        with self.assertLogs(pipeline_module.logger, level=logging.WARNING) as logs:
            self.pipeline.flush_pending_writes()
        self.assertEqual(self._drops(), [(tokens, "source_evicted")])
        self.assertEqual(len(logs.records), 1)
        record = logs.records[0]
        self.assertEqual(record.levelno, logging.WARNING)
        message = record.getMessage()
        self.assertIn("reason=source_evicted", message)
        self.assertIn("node_ids=[1]", message)
        self.assertIn(f"{tokens} tokens", message)
        self.assertIn(f"live cap {LIVE_CAP}", message)
        self.assertFalse(self.pipeline.pending_write_queue)
        self.assertFalse(self.pipeline.inflight_backup_node_ids)
        self.assertEqual(self.pipeline.write_backlog_tokens_, 0)
        self.assertEqual(self.pipeline.dropped_backup_tokens("source_evicted"), tokens)

    def test_admission_rejections_carry_their_reason(self):
        with self.subTest(reason="parent_uncovered"):
            # Parent 1 is neither root, in flight nor believed stored.
            tokens = self._add_node(2, parent=1)
            self.pipeline.enqueue_backup_intent(2)
            self.assertEqual(self._drops(), [(tokens, "parent_uncovered")])
        with self.subTest(reason="oversize"):
            tokens = self._add_node(3, pages=HOST_POOL_TOKENS)
            self.pipeline.enqueue_backup_intent(3)
            self.assertEqual(self._drops()[-1], (tokens, "oversize"))
        with self.subTest(reason="backlog_cap"):
            self.pipeline.write_backlog_cap = 0
            tokens = self._add_node(4)
            self.pipeline.enqueue_backup_intent(4)
            self.assertEqual(self._drops()[-1], (tokens, "backlog_cap"))
        # Admission-time drops are reported by the next tick, the cap at ERROR.
        with self.assertLogs(pipeline_module.logger, level=logging.WARNING) as logs:
            self.pipeline.flush_pending_writes()
        by_reason = {
            record.getMessage().split("reason=")[1].split(")")[0]: record
            for record in logs.records
        }
        self.assertEqual(
            set(by_reason), {"parent_uncovered", "oversize", "backlog_cap"}
        )
        self.assertEqual(by_reason["backlog_cap"].levelno, logging.ERROR)
        self.assertIn("(cap 0)", by_reason["backlog_cap"].getMessage())
        self.assertEqual(by_reason["oversize"].levelno, logging.WARNING)
        self.assertFalse(self.pipeline.pending_write_queue)
        self.assertFalse(self.pipeline.inflight_backup_node_ids)

    def test_flush_time_drops_carry_their_reason(self):
        with self.subTest(reason="parent_cascade"):
            parent_tokens = self._add_node(5)
            child_tokens = self._add_node(6, parent=5)
            self.pipeline.enqueue_backup_intent(5)
            # Admitted on the strength of the parent's queued write.
            self.pipeline.enqueue_backup_intent(6)
            self.assertEqual(len(self.pipeline.pending_write_queue), 2)
            self._evict(5)
            self.pipeline.flush_pending_writes()
            self.assertEqual(
                self._drops(),
                [(parent_tokens, "source_evicted"), (child_tokens, "parent_cascade")],
            )
        with self.subTest(reason="oversize"):
            # Admission sized two pages; the launch reads a span that cannot
            # stage, so the head is retired rather than wedging the queue.
            tokens = self._add_node(7, device_rows=2 * HOST_POOL_TOKENS)
            self.pipeline.enqueue_backup_intent(7)
            self.pipeline.flush_pending_writes()
            self.assertEqual(self._drops()[-1], (tokens, "oversize"))
        self.assertFalse(self.pipeline.pending_write_queue)
        self.assertFalse(self.pipeline.inflight_backup_node_ids)
        self.assertEqual(self.pipeline.write_backlog_tokens_, 0)

    def test_drop_log_is_rate_limited_per_reason_while_the_metric_counts_spans(self):
        tokens = sum(self._add_node(node_id) for node_id in (8, 9))
        for node_id in (8, 9):
            self.pipeline.enqueue_backup_intent(node_id)
            self._evict(node_id)
        with self.assertLogs(pipeline_module.logger, level=logging.WARNING) as logs:
            self.pipeline.flush_pending_writes()
        # Two swept spans: two metric increments, one line for the tick,
        # reporting the post-sweep queue.
        self.assertEqual(len(self._drops()), 2)
        self.assertEqual(len(logs.records), 1)
        message = logs.records[0].getMessage()
        self.assertIn("2 span(s)", message)
        self.assertIn("Queue 0 intents / 0 tokens", message)
        # Inside the interval: a later drop is tallied, not reported.
        tokens += self._add_node(10)
        self.pipeline.enqueue_backup_intent(10)
        self._evict(10)
        with self.assertNoLogs(pipeline_module.logger, level=logging.WARNING):
            self.pipeline.flush_pending_writes()
        self.assertEqual(len(self._drops()), 3)

        # Interval elapsed: the next tick reports the tail of the burst even
        # without a new drop, so the log converges on the metric.
        self.pipeline.backup_drop_log_interval_s = 0
        with self.assertLogs(pipeline_module.logger, level=logging.WARNING) as logs:
            self.pipeline.flush_pending_writes()
        message = logs.records[0].getMessage()
        self.assertIn("1 span(s)", message)
        self.assertIn(f"({tokens} total)", message)
        with self.assertNoLogs(pipeline_module.logger, level=logging.WARNING):
            self.pipeline.flush_pending_writes()
        self.assertEqual(self.pipeline.dropped_backup_tokens("source_evicted"), tokens)

    def test_write_window_gauges_are_published_every_flush(self):
        tokens = self._add_node(11)
        self.pipeline.enqueue_backup_intent(11)
        self.pipeline.flush_pending_writes()
        (ack_id,) = self.pipeline.ongoing_write_through
        self.pipeline.finish_backup_ack(ack_id)
        # An empty queue still publishes the draining staging.
        self.pipeline.flush_pending_writes()
        (operation_id,) = self.pipeline.ongoing_backup
        self.pipeline.finish_storage_write_ack(operation_id)
        self.pipeline.flush_pending_writes()
        self.assertEqual(
            [
                (
                    call.kwargs["backlog_tokens"],
                    call.kwargs["staged_tokens"],
                    call.kwargs["staging_cap_tokens"],
                )
                for call in self.collector.log_backup_write_window.call_args_list
            ],
            [(tokens, 0, LIVE_CAP), (0, tokens, LIVE_CAP), (0, 0, LIVE_CAP)],
        )
        self.assertEqual(self._drops(), [])

    def test_collector_preseeds_every_pipeline_reason(self):
        labels = {"model_name": "test"}
        collector = _RecordingStorageMetricsCollector(labels=labels)
        counter = collector.backup_dropped_tokens_total
        self.assertEqual(counter.seeded_reasons - {None}, set(BACKUP_DROP_REASONS))
        self.assertEqual(counter.labelnames, tuple(labels.keys()) + ("reason",))
        collector.log_backup_dropped_tokens(4, "source_evicted")
        self.assertEqual(
            counter.increments, [({**labels, "reason": "source_evicted"}, 4)]
        )
        collector.log_backup_write_window(
            backlog_tokens=6, staged_tokens=8, staging_cap_tokens=29
        )
        self.assertEqual(collector.backup_backlog_tokens.sets, [(labels, 6)])
        self.assertEqual(collector.backup_staged_tokens.sets, [(labels, 8)])
        self.assertEqual(collector.backup_staging_cap_tokens.sets, [(labels, 29)])


if __name__ == "__main__":
    unittest.main()
