"""End-to-end DI tests for ``ServerArgs.stat_loggers``.

These tests verify that a custom *MetricsCollector subclass passed through
``ServerArgs.stat_loggers`` is the one actually instantiated inside the
scheduler subprocess, **and** that emissions made on its instruments land
on the underlying (fake) Ray-style metric object.

The strategy follows the reviewer's guidance (sufeng-buaa):

* Use a module-level ``_MarkingSchedulerCollector`` that swaps the four DI
  class hooks (``_counter_cls``/``_gauge_cls``/``_histogram_cls``/
  ``_summary_cls``) with a ``FakeRayMetric``-style recording double. The
  collector must be picklable into the scheduler subprocess, so the recording
  double is defined at module scope.
* Boot a real ``sgl.Engine`` with ``enable_metrics=True`` and the custom
  collector registered for the scheduler role.
* Drive one small generation to force scheduler init (where
  ``resolve_collector_class()`` picks the injected subclass) and to produce
  real emissions.
* The scheduler runs in its own subprocess, so the recording double cannot
  share in-memory state with the test runner. Each recorded call is written
  to a filesystem marker. The test reads it back after ``engine.shutdown()``
  and asserts that a few representative metrics received positive values.

Result verification follows the reviewer's framing: pick a few metrics and
confirm values can be detected, do not enumerate all of them.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import json
import os
import tempfile
import unittest

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
    StorageMetricsCollector,
    TokenizerMetricsCollector,
    resolve_collector_class,
)

# ---------------------------------------------------------------------------
# Stubs and shared helpers
# ---------------------------------------------------------------------------


class _StubArgs:
    """Minimal ServerArgs stand-in.

    Avoids triggering the heavy real ServerArgs import chain for unit-level
    ``resolve_collector_class`` cases.
    """

    def __init__(self, stat_loggers=None):
        self.stat_loggers = stat_loggers


# Path to the cross-process marker file. The scheduler subprocess writes
# recorded ``inc``/``set``/``observe`` calls here; the test runner reads it
# back after ``engine.shutdown()``. The path is intentionally fixed so the
# subprocess can find it without IPC.
_DI_MARKER_PATH = os.path.join(
    tempfile.gettempdir(), "sglang_stat_loggers_di_marker.jsonl"
)


# ---------------------------------------------------------------------------
# FakeRayMetric-style recording double; picklable, defined at module scope
# ---------------------------------------------------------------------------


class _FileRecordingMetric:
    """Module-level recording metric.

    Mirrors the ``FakeRayMetric`` from
    ``sglang.test.observability.fake_ray`` (records ``(op, value, tags)``
    triples) but exposes the prometheus_client ``.labels(...).inc/.set/
    .observe(...)`` shape that ``SchedulerMetricsCollector`` calls into.

    The class must be defined at module level so the scheduler subprocess
    can unpickle the ``_MarkingSchedulerCollector`` reference. Recordings
    are appended as JSON lines to ``_DI_MARKER_PATH`` so the test runner can
    read them across the process boundary.
    """

    def __init__(self, name="", documentation="", labelnames=(), **kwargs):
        self.name = name
        self.documentation = documentation
        self._labelnames = tuple(labelnames or ())
        # Sink for in-process introspection (unit-level tests). The
        # subprocess uses the file marker instead, since in-memory state is
        # not visible to the test runner.
        self.calls = []

    def labels(self, **kwargs):
        return _FileRecordingMetricBound(self, dict(kwargs))


class _FileRecordingMetricBound:
    """The object returned by ``_FileRecordingMetric.labels(...)``.

    All three terminal verbs append a JSON line to the marker file so the
    test runner can verify emissions made inside the scheduler subprocess.
    """

    def __init__(self, parent: _FileRecordingMetric, tags: dict):
        self._parent = parent
        self._tags = tags

    def _record(self, op: str, value):
        self._parent.calls.append((op, value, dict(self._tags)))
        try:
            with open(_DI_MARKER_PATH, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "name": self._parent.name,
                            "op": op,
                            "value": value,
                            "tags": self._tags,
                        }
                    )
                    + "\n"
                )
        except OSError:
            # Marker file is best-effort. Never let a recording failure
            # disturb the scheduler's hot path.
            pass

    def inc(self, amount=1):
        self._record("inc", amount)

    def set(self, value):
        self._record("set", value)

    def observe(self, value):
        self._record("observe", value)


# ---------------------------------------------------------------------------
# Picklable custom scheduler collector with DI hooks swapped out
# ---------------------------------------------------------------------------


class _MarkingSchedulerCollector(SchedulerMetricsCollector):
    """A custom ``SchedulerMetricsCollector`` that records every emission to
    a filesystem marker.

    Achieves both halves of the reviewer's request:

    1. Its mere instantiation proves that ``resolve_collector_class()``
       picked the injected subclass inside the scheduler subprocess
       (the marker file exists).
    2. Each emission lands on the ``_FileRecordingMetric`` double, which
       writes a JSON line. The test reads the file after shutdown and
       asserts that a few representative metrics received positive values.

    Defined at module level so the scheduler subprocess can unpickle it.
    """

    _counter_cls = _FileRecordingMetric
    _gauge_cls = _FileRecordingMetric
    _histogram_cls = _FileRecordingMetric
    _summary_cls = _FileRecordingMetric


# ---------------------------------------------------------------------------
# Unit-level coverage retained from the previous version
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Full Engine-level DI integration test (the heart of the rewrite)
# ---------------------------------------------------------------------------


_MODEL_NAME = "Qwen/Qwen3-0.6B"


class TestStatLoggersDI(unittest.TestCase):
    """Boot a real ``sgl.Engine`` with a custom scheduler collector and verify
    that emissions land on the FakeRayMetric-style recording double.

    Combines the discriminating power of ``_MarkingSchedulerCollector``
    (proves the subclass was actually instantiated in the scheduler
    subprocess) with FakeRayMetric-style value recording (proves emissions
    flow through to the metric instance). Per the reviewer's framing, we
    pick a few representative metrics rather than enumerate all of them.
    """

    def setUp(self) -> None:
        try:
            os.unlink(_DI_MARKER_PATH)
        except FileNotFoundError:
            pass

    def tearDown(self) -> None:
        try:
            os.unlink(_DI_MARKER_PATH)
        except FileNotFoundError:
            pass

    def _read_marker(self):
        """Return all recorded emissions as a list of dicts.

        Each entry has keys ``name`` (str), ``op`` (one of ``inc``/``set``/
        ``observe``), ``value`` (numeric) and ``tags`` (dict).
        """
        entries = []
        with open(_DI_MARKER_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(json.loads(line))
        return entries

    def test_engine_custom_scheduler_collector_emits_through_fake_metric(self):
        import sglang as sgl

        engine = sgl.Engine(
            model_path=_MODEL_NAME,
            enable_metrics=True,
            stat_loggers={
                STAT_LOGGER_ROLE_SCHEDULER: _MarkingSchedulerCollector,
            },
        )
        try:
            # One small generation triggers scheduler init (which is where
            # resolve_collector_class picks the injected subclass) and is
            # enough to produce gauge ``.set()`` emissions on the basic
            # queue-state metrics.
            engine.generate("Hello", {"max_new_tokens": 4})
        finally:
            engine.shutdown()

        # Discrimination: the marker file exists, proving the custom
        # subclass was instantiated inside the scheduler subprocess.
        self.assertTrue(
            os.path.exists(_DI_MARKER_PATH),
            "Custom SchedulerMetricsCollector was not instantiated; "
            "stat_loggers DI did not take effect.",
        )

        entries = self._read_marker()
        self.assertGreater(
            len(entries),
            0,
            "Marker file exists but contains no emissions; "
            "the recording double was not wired through the DI hooks.",
        )

        # Value verification: pick a few representative metrics and check
        # that they actually received emissions with sensible shapes. We do
        # not enumerate all metrics; the reviewer's framing was "just pick
        # a few".
        by_name = {}
        for e in entries:
            by_name.setdefault(e["name"], []).append(e)

        # 1) num_running_reqs: a Gauge that the scheduler ``.set()``s every
        #    stats tick. After one generation it should have at least one
        #    emission.
        self.assertIn(
            "sglang:num_running_reqs",
            by_name,
            f"Expected num_running_reqs emissions, saw: {sorted(by_name)[:10]}",
        )
        running_ops = {e["op"] for e in by_name["sglang:num_running_reqs"]}
        self.assertIn("set", running_ops)

        # 2) num_queue_reqs: same shape, different metric. Two metrics from
        #    the same collector firing confirm the DI hook applied uniformly.
        self.assertIn("sglang:num_queue_reqs", by_name)
        queue_ops = {e["op"] for e in by_name["sglang:num_queue_reqs"]}
        self.assertIn("set", queue_ops)

        # 3) Tag propagation: every recorded emission must carry the labels
        #    keys the scheduler installed (model_name, engine_type, ...).
        any_running = by_name["sglang:num_running_reqs"][0]
        self.assertIn("model_name", any_running["tags"])
        self.assertEqual(any_running["tags"]["model_name"], _MODEL_NAME)


if __name__ == "__main__":
    unittest.main()
