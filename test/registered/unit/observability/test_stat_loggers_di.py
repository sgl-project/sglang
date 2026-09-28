"""Pure-CPU unit tests for ``ServerArgs.stat_loggers`` DI plumbing.

These tests cover the small, in-process pieces of the ``stat_loggers``
dependency injection feature:

* ``resolve_collector_class()`` returns the registered subclass when a role
  is present in ``stat_loggers`` and falls back to the default otherwise.
* Without any subclass override, collectors instantiate the real
  prometheus_client classes.

The full Engine-level integration test (which boots ``sgl.Engine`` and
verifies that emissions land on a FakeRayMetric-style recording double in
the scheduler subprocess) lives in
``test/registered/observability/test_metrics.py`` alongside the other
GPU-backed metrics tests.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest import mock

import prometheus_client

from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_EXPERT_DISPATCH,
    STAT_LOGGER_ROLE_RADIX_CACHE,
    STAT_LOGGER_ROLE_SCHEDULER,
    STAT_LOGGER_ROLE_STORAGE,
    STAT_LOGGER_ROLE_TOKENIZER,
    RadixCacheMetricsCollector,
    SchedulerMetricsCollector,
    StorageMetrics,
    StorageMetricsCollector,
    TokenizerMetricsCollector,
    radix_cache_metric_labels,
    resolve_collector_class,
)
from sglang.srt.runtime_context import get_context, reset_context


class _BoundRecordingMetric:
    def __init__(self, metric, labels):
        self.metric = metric
        self.labels = labels

    def inc(self, value=1):
        self.metric.increments.append((self.labels, value))

    def observe(self, value):
        self.metric.observations.append((self.labels, value))

    def set(self, value):
        self.metric.sets.append((self.labels, value))


class _RecordingMetric:
    """Small prometheus_client-compatible metric that preserves labels."""

    def __init__(self, *args, name=None, labelnames=(), **kwargs):
        self.name = name if name is not None else args[0]
        self.labelnames = tuple(labelnames)
        self.increments = []
        self.observations = []
        self.sets = []

    def labels(self, *values, **labels):
        if values:
            labels = dict(zip(self.labelnames, values, strict=True))
        return _BoundRecordingMetric(self, labels)


class _RecordingTokenizerMetricsCollector(TokenizerMetricsCollector):
    _counter_cls = _RecordingMetric
    _gauge_cls = _RecordingMetric
    _histogram_cls = _RecordingMetric


class _RecordingStorageMetricsCollector(StorageMetricsCollector):
    _counter_cls = _RecordingMetric
    _histogram_cls = _RecordingMetric


class TestResolveCollectorClass(unittest.TestCase):
    """The role table is read from the published `observability` bag."""

    def _resolve(self, role, default_cls, **fields):
        if not fields:
            return resolve_collector_class(role, default_cls)
        with get_context().override_server_args(**fields):
            return resolve_collector_class(role, default_cls)

    def test_returns_default_when_nothing_is_published(self):
        reset_context()
        self.assertIs(
            resolve_collector_class("scheduler", SchedulerMetricsCollector),
            SchedulerMetricsCollector,
        )

    def test_returns_default_when_stat_loggers_none(self):
        cls = self._resolve("scheduler", SchedulerMetricsCollector, stat_loggers=None)
        self.assertIs(cls, SchedulerMetricsCollector)

    def test_returns_default_when_stat_loggers_empty(self):
        cls = self._resolve("scheduler", SchedulerMetricsCollector, stat_loggers={})
        self.assertIs(cls, SchedulerMetricsCollector)

    def test_returns_default_when_role_missing(self):
        class MyTokenizer(TokenizerMetricsCollector):
            pass

        cls = self._resolve(
            "scheduler",
            SchedulerMetricsCollector,
            stat_loggers={"tokenizer": MyTokenizer},
        )
        self.assertIs(cls, SchedulerMetricsCollector)

    def test_returns_subclass_when_role_registered(self):
        class MyScheduler(SchedulerMetricsCollector):
            pass

        cls = self._resolve(
            "scheduler",
            SchedulerMetricsCollector,
            stat_loggers={"scheduler": MyScheduler},
        )
        self.assertIs(cls, MyScheduler)

    def test_role_constants_match_collector_keys(self):
        """The exported role constants must be the exact strings the
        instantiation sites use to look up subclasses."""
        self.assertEqual(STAT_LOGGER_ROLE_SCHEDULER, "scheduler")
        self.assertEqual(STAT_LOGGER_ROLE_TOKENIZER, "tokenizer")
        self.assertEqual(STAT_LOGGER_ROLE_STORAGE, "storage")
        self.assertEqual(STAT_LOGGER_ROLE_RADIX_CACHE, "radix_cache")
        self.assertEqual(STAT_LOGGER_ROLE_EXPERT_DISPATCH, "expert_dispatch")


class TestDefaultBackend(unittest.TestCase):
    """Without any subclass override, collectors instantiate the real
    prometheus_client classes; the existing behavior is unchanged."""

    def test_default_path_uses_prometheus_client(self):
        labels = {"cache_type": "test_default"}
        collector = RadixCacheMetricsCollector(labels=labels)
        self.assertIsInstance(collector.eviction_num_tokens, prometheus_client.Counter)
        self.assertIsInstance(
            collector.eviction_duration_seconds, prometheus_client.Histogram
        )


class TestRadixCacheMetricLabels(unittest.TestCase):
    """Radix-cache series must stay distinct per scheduler rank: an unlabeled
    family is summed across local ranks by the multiprocess registry, which
    reported TP x the logical token count in production. The rank keys follow
    the storage collector's DP-aware convention so L2 and L3 series line up."""

    def test_labels_follow_the_storage_collector_rank_keys(self):
        parallel = SimpleNamespace(tp_rank=3, pp_rank=1, attn_tp_rank=1, attn_dp_rank=2)
        self.assertEqual(
            radix_cache_metric_labels("UnifiedRadixCache", parallel, True),
            {
                "cache_type": "UnifiedRadixCache",
                "tp_rank": 1,
                "pp_rank": 1,
                "dp_rank": 2,
            },
        )
        self.assertEqual(
            radix_cache_metric_labels("RadixCache", parallel, False),
            {"cache_type": "RadixCache", "tp_rank": 3, "pp_rank": 1, "dp_rank": 0},
        )


class TestHiCacheMetrics(unittest.TestCase):
    def test_cached_tokens_uses_literal_storage_source(self):
        labels = {"model_name": "test"}
        with get_context().override_server_args(
            prompt_tokens_buckets=None, generation_tokens_buckets=None
        ):
            collector = _RecordingTokenizerMetricsCollector(labels=labels)

        collector.observe_one_finished_request(
            labels=labels,
            prompt_tokens=20,
            generation_tokens=2,
            cached_tokens=12,
            e2e_latency=0.1,
            has_grammar=False,
            cached_tokens_details={
                "device": 3,
                "host": 4,
                "storage": 5,
                "storage_backend": "BackendShim",
            },
        )

        by_source = {
            metric_labels["cache_source"]: value
            for metric_labels, value in collector.cached_tokens_total.increments
        }
        self.assertEqual(by_source, {"device": 3, "host": 4, "storage": 5})

    def test_storage_prefetch_lifecycle_metrics(self):
        labels = {"model_name": "test"}
        collector = _RecordingStorageMetricsCollector(labels=labels)

        collector.log_storage_prefetch_hit_tokens(21)
        collector.log_storage_prefetch_unfulfilled_tokens(4, "storage_transfer")
        collector.log_storage_prefetch_deferred_tokens(7, "device_capacity")

        self.assertEqual(
            collector.storage_prefetch_hit_tokens_total.increments, [(labels, 21)]
        )
        self.assertEqual(
            collector.storage_prefetch_unfulfilled_tokens_total.increments,
            [({**labels, "reason": "storage_transfer"}, 4)],
        )
        self.assertEqual(
            collector.storage_prefetch_deferred_tokens_total.increments,
            [({**labels, "reason": "device_capacity"}, 7)],
        )


class _FakeMetricChild:
    def __init__(self, metric, label_values):
        self.metric = metric
        self.label_values = label_values

    def inc(self, value=1):
        self.metric.values[self.label_values] = (
            self.metric.values.get(self.label_values, 0) + value
        )

    def observe(self, value):
        self.inc(value)

    def set(self, value):
        self.metric.values[self.label_values] = value


class _FakeMetric:
    instances = []

    def __init__(self, name, documentation, labelnames=(), **kwargs):
        self.name = name
        self.documentation = documentation
        self.labelnames = tuple(labelnames)
        self.values = {}
        self.__class__.instances.append(self)

    def labels(self, **labels):
        return _FakeMetricChild(self, tuple(sorted(labels.items())))


class TestStoragePrefetchOutcomeMetrics(unittest.TestCase):
    """Delta accounting for the three cumulative outcome snapshots exported
    through ``StorageMetrics``:

    * ``prefetch_stats`` (L3->L2) is projected by ``log_prefetch_outcomes``
      onto ``tier=l3_to_l2_load`` with the raw outcome keys as ``reason``;
      a snapshot reset (current < last) starts a new epoch that re-exports
      the current value rather than a negative delta.
    * ``load_outcome_stats`` (L2->L1) and ``write_outcome_stats``
      (L1->L2 / L2->L3) are flushed by ``_flush_outcome_delta`` as strict
      per-(tier, reason) deltas with no epoch logic.

    Guards the invariant that cumulative dicts map to a cumulative Counter
    without double counting across repeated flushes of the same snapshot.
    """

    L3 = "l3_to_l2_load"
    L2_L1 = "l2_to_l1_load"
    L1_L2 = "l1_to_l2_writ"
    L2_L3 = "l2_to_l3_writ"

    def setUp(self):
        _FakeMetric.instances = []
        self.patches = [
            mock.patch.object(StorageMetricsCollector, "_counter_cls", _FakeMetric),
            mock.patch.object(StorageMetricsCollector, "_gauge_cls", _FakeMetric),
            mock.patch.object(StorageMetricsCollector, "_histogram_cls", _FakeMetric),
        ]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.collector = StorageMetricsCollector(labels={"model": "test"})
        self.counter = next(
            metric
            for metric in _FakeMetric.instances
            if metric.name == "sglang:hicache_transfer_outcomes_total"
        )

    def _value(self, tier, reason):
        # _FakeMetric.labels sorts label items, so the stored key is the
        # sorted (model, reason, tier) tuple.
        key = tuple(sorted({"model": "test", "tier": tier, "reason": reason}.items()))
        return self.counter.values.get(key, 0)

    # ---- prefetch_stats -> tier=l3_to_l2_load (log_prefetch_outcomes) ----

    def test_prefetch_deltas_without_double_counting(self):
        first = {"attempts": 2, "issued": 1, "revoked_full_miss": 1}
        self.collector.log_storage_metrics(StorageMetrics(prefetch_stats=first))
        # Re-flushing the same cumulative snapshot must not double count.
        self.collector.log_storage_metrics(StorageMetrics(prefetch_stats=first))
        self.assertEqual(self._value(self.L3, "attempts"), 2)
        self.assertEqual(self._value(self.L3, "issued"), 1)
        self.assertEqual(self._value(self.L3, "revoked_full_miss"), 1)

        self.collector.log_storage_metrics(
            StorageMetrics(
                prefetch_stats={
                    "attempts": 3,
                    "issued": 2,
                    "declined_rate_limited": 1,
                    "revoked_full_miss": 1,
                }
            )
        )
        self.assertEqual(self._value(self.L3, "attempts"), 3)
        self.assertEqual(self._value(self.L3, "issued"), 2)
        self.assertEqual(self._value(self.L3, "declined_rate_limited"), 1)

        self.collector.log_storage_metrics(
            StorageMetrics(prefetch_stats={"attempts": 1, "issued": 1})
        )
        self.assertEqual(self._value(self.L3, "attempts"), 4)
        self.assertEqual(self._value(self.L3, "issued"), 3)

    def test_prefetch_snapshot_reset_starts_new_epoch(self):
        # A cache-side reset makes current < last; the new epoch must export
        # the current value (not a negative delta), then resume delta mode.
        self.collector.log_storage_metrics(
            StorageMetrics(prefetch_stats={"attempts": 5, "issued": 5})
        )
        self.assertEqual(self._value(self.L3, "attempts"), 5)

        self.collector.log_storage_metrics(
            StorageMetrics(prefetch_stats={"attempts": 2, "issued": 2})
        )
        # 5 (epoch 1) + 2 (epoch 2 re-export) = 7, never 5 - 3 = 2.
        self.assertEqual(self._value(self.L3, "attempts"), 7)
        self.assertEqual(self._value(self.L3, "issued"), 7)

        # Delta resumes within the new epoch.
        self.collector.log_storage_metrics(
            StorageMetrics(prefetch_stats={"attempts": 4, "issued": 4})
        )
        self.assertEqual(self._value(self.L3, "attempts"), 9)
        self.assertEqual(self._value(self.L3, "issued"), 9)

    def test_prefetch_failure_reasons_exported(self):
        self.collector.log_storage_metrics(
            StorageMetrics(
                prefetch_stats={
                    "aux_alloc_failed": 1,
                    "host_alloc_failed": 2,
                    "read_failed": 3,
                }
            )
        )
        self.assertEqual(self._value(self.L3, "aux_alloc_failed"), 1)
        self.assertEqual(self._value(self.L3, "host_alloc_failed"), 2)
        self.assertEqual(self._value(self.L3, "read_failed"), 3)

    # ---- load_outcome_stats -> tier=l2_to_l1_load (_flush_outcome_delta) ----

    def test_load_outcome_deltas(self):
        snapshot = {
            self.L2_L1: {
                "attempts": 3,
                "device_alloc_failed": 1,
                "declined_too_short": 1,
            }
        }
        self.collector.log_storage_metrics(StorageMetrics(load_outcome_stats=snapshot))
        self.assertEqual(self._value(self.L2_L1, "attempts"), 3)
        self.assertEqual(self._value(self.L2_L1, "device_alloc_failed"), 1)
        self.assertEqual(self._value(self.L2_L1, "declined_too_short"), 1)

        # Same snapshot re-flushed: no double counting.
        self.collector.log_storage_metrics(StorageMetrics(load_outcome_stats=snapshot))
        self.assertEqual(self._value(self.L2_L1, "attempts"), 3)

        growth = {
            self.L2_L1: {
                "attempts": 5,
                "device_alloc_failed": 1,
                "declined_too_short": 2,
            }
        }
        self.collector.log_storage_metrics(StorageMetrics(load_outcome_stats=growth))
        self.assertEqual(self._value(self.L2_L1, "attempts"), 5)
        self.assertEqual(self._value(self.L2_L1, "declined_too_short"), 2)

    def test_load_outcome_new_reason_appears_lazily(self):
        # A reason absent from the first flush and present in the second must
        # export its full current value on first appearance (delta = current - 0).
        self.collector.log_storage_metrics(
            StorageMetrics(load_outcome_stats={self.L2_L1: {"attempts": 2}})
        )
        self.collector.log_storage_metrics(
            StorageMetrics(
                load_outcome_stats={
                    self.L2_L1: {"attempts": 2, "device_alloc_failed": 1}
                }
            )
        )
        self.assertEqual(self._value(self.L2_L1, "device_alloc_failed"), 1)
        self.assertEqual(self._value(self.L2_L1, "attempts"), 2)

    # ---- write_outcome_stats -> tier=l1_to_l2_writ / l2_to_l3_writ ----

    def test_write_outcome_deltas_across_two_tiers(self):
        snapshot = {
            self.L1_L2: {"attempts": 2, "host_alloc_failed": 1},
            self.L2_L3: {
                "attempts": 2,
                "l3_write_tokens": 64,
                "write_failed": 1,
                "l3_write_failed_tokens": 32,
            },
        }
        self.collector.log_storage_metrics(StorageMetrics(write_outcome_stats=snapshot))
        self.assertEqual(self._value(self.L1_L2, "attempts"), 2)
        self.assertEqual(self._value(self.L1_L2, "host_alloc_failed"), 1)
        self.assertEqual(self._value(self.L2_L3, "attempts"), 2)
        self.assertEqual(self._value(self.L2_L3, "l3_write_tokens"), 64)
        self.assertEqual(self._value(self.L2_L3, "write_failed"), 1)
        self.assertEqual(self._value(self.L2_L3, "l3_write_failed_tokens"), 32)

        # Re-flush same snapshot: no growth.
        self.collector.log_storage_metrics(StorageMetrics(write_outcome_stats=snapshot))
        self.assertEqual(self._value(self.L2_L3, "l3_write_tokens"), 64)

        growth = {
            self.L1_L2: {"attempts": 4, "host_alloc_failed": 1},
            self.L2_L3: {
                "attempts": 5,
                "l3_write_tokens": 128,
                "write_failed": 2,
                "l3_write_failed_tokens": 48,
            },
        }
        self.collector.log_storage_metrics(StorageMetrics(write_outcome_stats=growth))
        self.assertEqual(self._value(self.L1_L2, "attempts"), 4)
        self.assertEqual(self._value(self.L2_L3, "attempts"), 5)
        self.assertEqual(self._value(self.L2_L3, "l3_write_tokens"), 128)
        self.assertEqual(self._value(self.L2_L3, "write_failed"), 2)
        self.assertEqual(self._value(self.L2_L3, "l3_write_failed_tokens"), 48)

    def test_write_outcome_token_delta_is_per_reason(self):
        # l3_write_tokens is a cumulative token sum, not a req count: deltas
        # must add token amounts (current - last), not +1 per flush.
        self.collector.log_storage_metrics(
            StorageMetrics(write_outcome_stats={self.L2_L3: {"l3_write_tokens": 100}})
        )
        self.assertEqual(self._value(self.L2_L3, "l3_write_tokens"), 100)
        self.collector.log_storage_metrics(
            StorageMetrics(write_outcome_stats={self.L2_L3: {"l3_write_tokens": 130}})
        )
        self.assertEqual(self._value(self.L2_L3, "l3_write_tokens"), 130)

    # ---- independent delta streams ----

    def test_three_streams_have_independent_last_state(self):
        # Each snapshot dict carries its own _last_* bookkeeping; advancing one
        # must not bleed into another's delta on a later mixed flush.
        self.collector.log_storage_metrics(
            StorageMetrics(
                prefetch_stats={"attempts": 1},
                load_outcome_stats={self.L2_L1: {"attempts": 1}},
                write_outcome_stats={self.L1_L2: {"attempts": 1}},
            )
        )
        self.assertEqual(self._value(self.L3, "attempts"), 1)
        self.assertEqual(self._value(self.L2_L1, "attempts"), 1)
        self.assertEqual(self._value(self.L1_L2, "attempts"), 1)

        self.collector.log_storage_metrics(
            StorageMetrics(
                prefetch_stats={"attempts": 3},
                load_outcome_stats={self.L2_L1: {"attempts": 1}},
                write_outcome_stats={self.L1_L2: {"attempts": 2}},
            )
        )
        self.assertEqual(self._value(self.L3, "attempts"), 3)
        # load unchanged -> no new delta.
        self.assertEqual(self._value(self.L2_L1, "attempts"), 1)
        self.assertEqual(self._value(self.L1_L2, "attempts"), 2)

    def test_empty_snapshots_are_noops(self):
        # Empty dicts (no tier/reason yet) must not raise or emit; the
        # production flush always supplies non-None snapshots, so this only
        # guards the degenerate empty case.
        self.collector.log_storage_metrics(
            StorageMetrics(
                load_outcome_stats={}, write_outcome_stats={}, prefetch_stats={}
            )
        )
        self.assertEqual(self.counter.values, {})


if __name__ == "__main__":
    unittest.main()
