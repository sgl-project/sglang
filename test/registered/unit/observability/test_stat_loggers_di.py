"""Pure-CPU unit tests for ``ServerArgs.stat_loggers`` DI plumbing.

These tests cover the small, in-process pieces of the ``stat_loggers``
dependency injection feature:

* The four DI hook class attributes (``_counter_cls``/``_gauge_cls``/
  ``_histogram_cls``/``_summary_cls``) default to ``None`` on every
  collector, so the existing prometheus_client backend is used unchanged.
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

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from unittest import mock

import prometheus_client

from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_EXPERT_DISPATCH,
    STAT_LOGGER_ROLE_RADIX_CACHE,
    STAT_LOGGER_ROLE_SCHEDULER,
    STAT_LOGGER_ROLE_STORAGE,
    STAT_LOGGER_ROLE_TOKENIZER,
    ExpertDispatchCollector,
    RadixCacheMetricsCollector,
    SchedulerMetricsCollector,
    StorageMetrics,
    StorageMetricsCollector,
    TokenizerMetricsCollector,
    resolve_collector_class,
)


class _StubArgs:
    """Minimal ServerArgs stand-in.

    Avoids triggering the heavy real ServerArgs import chain for unit-level
    ``resolve_collector_class`` cases.
    """

    def __init__(self, stat_loggers=None):
        self.stat_loggers = stat_loggers


class TestCollectorClassAttrs(unittest.TestCase):
    """All five collectors expose four DI hook class attrs, all defaulting to
    None so the existing prometheus_client backend is used unchanged."""

    def test_scheduler_collector_attrs_default_none(self):
        self.assertIsNone(SchedulerMetricsCollector._counter_cls)
        self.assertIsNone(SchedulerMetricsCollector._gauge_cls)
        self.assertIsNone(SchedulerMetricsCollector._histogram_cls)
        self.assertIsNone(SchedulerMetricsCollector._summary_cls)

    def test_tokenizer_collector_attrs_default_none(self):
        self.assertIsNone(TokenizerMetricsCollector._counter_cls)
        self.assertIsNone(TokenizerMetricsCollector._histogram_cls)

    def test_storage_collector_attrs_default_none(self):
        self.assertIsNone(StorageMetricsCollector._counter_cls)
        self.assertIsNone(StorageMetricsCollector._histogram_cls)

    def test_expert_dispatch_collector_attrs_default_none(self):
        self.assertIsNone(ExpertDispatchCollector._histogram_cls)

    def test_radix_cache_collector_attrs_default_none(self):
        self.assertIsNone(RadixCacheMetricsCollector._counter_cls)
        self.assertIsNone(RadixCacheMetricsCollector._histogram_cls)


class TestResolveCollectorClass(unittest.TestCase):
    def test_returns_default_when_server_args_none(self):
        cls = resolve_collector_class(None, "scheduler", SchedulerMetricsCollector)
        self.assertIs(cls, SchedulerMetricsCollector)

    def test_returns_default_when_stat_loggers_none(self):
        cls = resolve_collector_class(
            _StubArgs(stat_loggers=None), "scheduler", SchedulerMetricsCollector
        )
        self.assertIs(cls, SchedulerMetricsCollector)

    def test_returns_default_when_stat_loggers_empty(self):
        cls = resolve_collector_class(
            _StubArgs(stat_loggers={}), "scheduler", SchedulerMetricsCollector
        )
        self.assertIs(cls, SchedulerMetricsCollector)

    def test_returns_default_when_role_missing(self):
        class MyTokenizer(TokenizerMetricsCollector):
            pass

        cls = resolve_collector_class(
            _StubArgs(stat_loggers={"tokenizer": MyTokenizer}),
            "scheduler",
            SchedulerMetricsCollector,
        )
        self.assertIs(cls, SchedulerMetricsCollector)

    def test_returns_subclass_when_role_registered(self):
        class MyScheduler(SchedulerMetricsCollector):
            pass

        cls = resolve_collector_class(
            _StubArgs(stat_loggers={"scheduler": MyScheduler}),
            "scheduler",
            SchedulerMetricsCollector,
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
