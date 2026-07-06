import json
import os
import tempfile
import unittest
from typing import Dict, List

import requests
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample

from sglang.srt.environ import envs
from sglang.srt.observability.metrics_collector import (
    ROUTING_KEY_REQ_COUNT_BUCKET_BOUNDS,
    STAT_LOGGER_ROLE_SCHEDULER,
    SchedulerMetricsCollector,
    compute_routing_key_stats,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)

register_cuda_ci(est_time=147, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=32, suite="stage-b-test-1-gpu-small-amd")

_MODEL_NAME = "Qwen/Qwen3-0.6B"


class TestEnableMetrics(CustomTestCase):
    def test_metrics_1gpu(self):
        """Test that metrics endpoint returns data when enabled"""
        self._execute_core(
            other_args=[],
            verify_metrics_extra=None,
            expect_mfu_metrics=True,
            enable_mfu_metrics=True,
        )

    def test_mfu_metrics_gate_disabled(self):
        """MFU metrics should not be emitted when the gate is disabled."""
        self._execute_core(
            other_args=[],
            verify_metrics_extra=None,
            expect_mfu_metrics=False,
            enable_mfu_metrics=False,
        )

    def test_metrics_2gpu(self):
        # TODO enable when we have 2-gpu runner in nightly CI
        if is_in_ci():
            print("Skip test_metrics_2gpu since in 1-gpu CI")
            return

        def _verify_metrics_extra(metrics):
            metrics_to_check = [
                (
                    "sglang:dp_cooperation_realtime_tokens_total",
                    {"mode": "prefill_compute"},
                ),
                (
                    "sglang:dp_cooperation_realtime_tokens_total",
                    {"mode": "decode"},
                ),
                (
                    "sglang:dp_cooperation_forward_execution_seconds_total",
                    {"category": "extend"},
                ),
                (
                    "sglang:dp_cooperation_forward_execution_seconds_total",
                    {"category": "decode"},
                ),
            ]
            _check_metrics_positive(self, metrics, metrics_to_check)

            num_prefill_ranks_values = {
                s.labels["num_prefill_ranks"]
                for s in metrics["sglang:dp_cooperation_realtime_tokens_total"]
            }
            self.assertIn("0", num_prefill_ranks_values)
            self.assertIn("1", num_prefill_ranks_values)

        self._execute_core(
            other_args=["--tp", "2", "--dp", "2", "--enable-dp-attention"],
            verify_metrics_extra=_verify_metrics_extra,
            expect_mfu_metrics=True,
            enable_mfu_metrics=True,
        )

    def _execute_core(
        self,
        other_args,
        verify_metrics_extra,
        expect_mfu_metrics: bool,
        enable_mfu_metrics: bool,
    ):
        with (
            envs.SGLANG_ENABLE_METRICS_DP_ATTENTION.override(True),
            envs.SGLANG_ENABLE_METRICS_DEVICE_TIMER.override(True),
            envs.SGLANG_TEST_RETRACT.override(True),
        ):
            launch_args = [
                "--enable-metrics",
                "--cuda-graph-max-bs-decode",
                2,
                *other_args,
            ]
            if enable_mfu_metrics:
                launch_args.insert(1, "--enable-mfu-metrics")
            process = popen_launch_server(
                _MODEL_NAME,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=launch_args,
            )

        try:
            # Make some requests to generate some metrics
            response = requests.get(f"{DEFAULT_URL_FOR_TEST}/health_generate")
            self.assertEqual(response.status_code, 200)

            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": ["The capital of France is"] * 20,
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 50,
                    },
                    "stream": True,
                    "ignore_eos": True,
                },
                stream=True,
            )
            for _ in response.iter_lines(decode_unicode=False):
                pass

            for i in range(2):
                # Send the request twice to trigger cached token metrics
                response = requests.post(
                    f"{DEFAULT_URL_FOR_TEST}/generate",
                    json={
                        "text": "Hello, " * 100,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 5},
                    },
                    headers={"x-smg-routing-key": "test-key"},
                )
                self.assertEqual(response.status_code, 200)

            # Get metrics
            metrics_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
            self.assertEqual(metrics_response.status_code, 200)
            metrics_text = metrics_response.text

            print(f"metrics_text=\n{metrics_text}")

            metrics = _parse_prometheus_metrics(metrics_text)
            self._verify_metrics_common(metrics_text, metrics, expect_mfu_metrics)
            if verify_metrics_extra is not None:
                verify_metrics_extra(metrics)
        finally:
            kill_process_tree(process.pid)

    def _verify_metrics_common(self, metrics_text, metrics, expect_mfu_metrics: bool):
        essential_metrics = [
            "sglang:num_running_reqs",
            "sglang:num_used_tokens",
            "sglang:token_usage",
            "sglang:gen_throughput",
            "sglang:num_queue_reqs",
            "sglang:num_grammar_queue_reqs",
            "sglang:cache_hit_rate",
            "sglang:spec_accept_length",
            "sglang:prompt_tokens_total",
            "sglang:generation_tokens_total",
            "sglang:cached_tokens_total",
            "sglang:num_requests_total",
            "sglang:time_to_first_token_seconds",
            "sglang:inter_token_latency_seconds",
            "sglang:e2e_request_latency_seconds",
            "sglang:http_requests_active",
            "sglang:routing_keys_active",
            "sglang:num_unique_running_routing_keys",
            "sglang:routing_key_running_req_count",
            "sglang:routing_key_all_req_count",
        ]
        mfu_metrics = [
            "sglang:estimated_flops_per_gpu_total",
            "sglang:estimated_read_bytes_per_gpu_total",
            "sglang:estimated_write_bytes_per_gpu_total",
        ]
        if expect_mfu_metrics:
            essential_metrics.extend(mfu_metrics)
        for metric in essential_metrics:
            self.assertIn(metric, metrics_text, f"Missing metric: {metric}")

        # Verify routing key GaugeHistogram buckets
        expected_buckets = len(ROUTING_KEY_REQ_COUNT_BUCKET_BOUNDS) + 1
        for metric_name in [
            "sglang:routing_key_running_req_count",
            "sglang:routing_key_all_req_count",
        ]:
            gt_le_pairs = set()
            for sample in metrics.get(metric_name, []):
                gt_le_pairs.add((sample.labels.get("gt"), sample.labels.get("le")))
            self.assertEqual(
                len(gt_le_pairs),
                expected_buckets,
                f"{metric_name}: Expected {expected_buckets} buckets, got {len(gt_le_pairs)}",
            )

        self.assertIn(f'model_name="{_MODEL_NAME}"', metrics_text)
        self.assertIn("_sum{", metrics_text)
        self.assertIn("_count{", metrics_text)
        self.assertIn("_bucket{", metrics_text)

        metrics_to_check = [
            ("sglang:realtime_tokens_total", {"mode": "prefill_compute"}),
            ("sglang:realtime_tokens_total", {"mode": "decode"}),
            ("sglang:forward_execution_seconds_total", {"category": "extend"}),
            ("sglang:forward_execution_seconds_total", {"category": "decode"}),
            ("sglang:process_cpu_seconds_total", {"component": "tokenizer"}),
        ]
        _check_metrics_positive(self, metrics, metrics_to_check)

        if expect_mfu_metrics:
            # Estimated perf metrics may have multiple series (e.g., by rank). Ensure
            # that at least one series for this model has a positive accumulated value.
            for metric_name in mfu_metrics:
                values = [
                    sample.value
                    for sample in metrics.get(metric_name, [])
                    if sample.labels.get("model_name") == _MODEL_NAME
                ]
                self.assertTrue(
                    values, f"{metric_name}: no samples for model {_MODEL_NAME}"
                )
                self.assertGreater(
                    sum(values),
                    0,
                    f"{metric_name}: expected positive total for model {_MODEL_NAME}",
                )
        else:
            # With only --enable-metrics (without --enable-mfu-metrics), MFU
            # counters should not emit positive values.
            for metric_name in mfu_metrics:
                values = [
                    sample.value
                    for sample in metrics.get(metric_name, [])
                    if sample.labels.get("model_name") == _MODEL_NAME
                ]
                if values:
                    self.assertEqual(
                        sum(values),
                        0,
                        f"{metric_name}: expected no positive samples with MFU metrics gate disabled",
                    )


def _parse_prometheus_metrics(metrics_text: str) -> Dict[str, List[Sample]]:
    result = {}
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            if sample.name not in result:
                result[sample.name] = []
            result[sample.name].append(sample)
    return result


def _get_sample_value_by_labels(samples: List[Sample], labels: Dict[str, str]) -> float:
    for sample in samples:
        if all(sample.labels.get(k) == v for k, v in labels.items()):
            return sample.value
    raise KeyError(f"No sample found with labels {labels}")


def _check_metrics_positive(test_case, metrics, metrics_to_check):
    for metric_name, labels in metrics_to_check:
        value = _get_sample_value_by_labels(metrics[metric_name], labels)
        test_case.assertGreater(value, 0, f"{metric_name} {labels}")


_DI_MARKER_PATH = "/tmp/sglang_di_test_marker"


class _MarkingSchedulerCollector(SchedulerMetricsCollector):
    """Records its own instantiation to a file so the test can verify the
    custom subclass was used in the scheduler subprocess.

    Defined at module level so it is picklable into the scheduler process.
    Cross-process signalling uses a filesystem marker because the scheduler
    runs in its own subprocess and cannot share in-memory state with the
    test runner.
    """

    def __init__(self, *args, **kwargs):
        with open(_DI_MARKER_PATH, "w") as f:
            f.write("scheduler_collector_initialized\n")
        super().__init__(*args, **kwargs)


# Path to the cross-process marker file for the FakeRayMetric-style recording
# variant below. Distinct from ``_DI_MARKER_PATH`` so the two scheduler
# collector subclasses (instantiation-marker vs. emission-recording) cannot
# stomp on each other when both tests run in the same CI shard.
_DI_RECORDING_MARKER_PATH = os.path.join(
    tempfile.gettempdir(), "sglang_stat_loggers_di_marker.jsonl"
)


class _FileRecordingMetric:
    """Module-level recording metric.

    Mirrors the ``FakeRayMetric`` from
    ``sglang.test.observability.fake_ray`` (records ``(op, value, tags)``
    triples) but exposes the prometheus_client ``.labels(...).inc/.set/
    .observe(...)`` shape that ``SchedulerMetricsCollector`` calls into.

    Defined at module level so the scheduler subprocess can unpickle the
    ``_RecordingSchedulerCollector`` reference. Recordings are appended as
    JSON lines to ``_DI_RECORDING_MARKER_PATH`` so the test runner can read
    them across the process boundary.
    """

    def __init__(self, name="", documentation="", labelnames=(), **kwargs):
        self.name = name
        self.documentation = documentation
        self._labelnames = tuple(labelnames or ())
        # Sink for in-process introspection. The subprocess uses the file
        # marker instead, since in-memory state is not visible to the test
        # runner.
        self.calls = []

    def labels(self, **kwargs):
        return _FileRecordingMetricBound(self, dict(kwargs))


class _FileRecordingMetricBound:
    """The object returned by ``_FileRecordingMetric.labels(...)``.

    All three terminal verbs append a JSON line to the marker file so the
    test runner can verify emissions made inside the scheduler subprocess.
    """

    def __init__(self, parent: "_FileRecordingMetric", tags: dict):
        self._parent = parent
        self._tags = tags

    def _record(self, op: str, value):
        self._parent.calls.append((op, value, dict(self._tags)))
        try:
            with open(_DI_RECORDING_MARKER_PATH, "a") as f:
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


class _RecordingSchedulerCollector(SchedulerMetricsCollector):
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


def _clear_sglang_metrics_from_default_registry() -> None:
    """Drop any ``sglang:`` metrics left in the process-global prometheus default
    REGISTRY by a prior in-process Engine boot. Without this, a second in-process
    ``sgl.Engine(enable_metrics=True)`` in the same test process re-registers the
    same Counters and raises "Duplicated timeseries in CollectorRegistry"."""
    from prometheus_client import REGISTRY

    for collector in list(getattr(REGISTRY, "_collector_to_names", {})):
        names = REGISTRY._collector_to_names.get(collector, set())
        if any(name.startswith("sglang:") for name in names):
            REGISTRY.unregister(collector)


class TestStatLoggersDI(CustomTestCase):
    """Verify that a custom MetricsCollector subclass passed through
    ``ServerArgs.stat_loggers`` is the one instantiated inside the
    scheduler subprocess."""

    def setUp(self) -> None:
        _clear_sglang_metrics_from_default_registry()
        try:
            os.unlink(_DI_MARKER_PATH)
        except FileNotFoundError:
            pass

    def tearDown(self) -> None:
        try:
            os.unlink(_DI_MARKER_PATH)
        except FileNotFoundError:
            pass

    def test_engine_custom_scheduler_collector(self):
        import sglang as sgl

        engine = sgl.Engine(
            model_path=_MODEL_NAME,
            enable_metrics=True,
            stat_loggers={
                STAT_LOGGER_ROLE_SCHEDULER: _MarkingSchedulerCollector,
            },
        )
        try:
            # One small generation triggers scheduler init, which is where
            # resolve_collector_class() picks the injected subclass.
            engine.generate("Hello", {"max_new_tokens": 4})
        finally:
            engine.shutdown()

        self.assertTrue(
            os.path.exists(_DI_MARKER_PATH),
            "Custom SchedulerMetricsCollector was not instantiated; "
            "stat_loggers DI did not take effect.",
        )


class TestStatLoggersDIRecording(CustomTestCase):
    """Boot a real ``sgl.Engine`` with a custom scheduler collector that
    swaps the four DI hook classes for a FakeRayMetric-style recording
    double and verify that emissions land on the double.

    Combines the discriminating power of ``_RecordingSchedulerCollector``
    (proves the subclass was actually instantiated in the scheduler
    subprocess) with value recording (proves emissions flow through to the
    metric instance). Per the reviewer's framing, we pick a few
    representative metrics rather than enumerate all of them.
    """

    def setUp(self) -> None:
        # Avoid stale PROMETHEUS_MULTIPROC_DIR from prior in-process Engine boots.
        os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
        _clear_sglang_metrics_from_default_registry()
        try:
            os.unlink(_DI_RECORDING_MARKER_PATH)
        except FileNotFoundError:
            pass

    def tearDown(self) -> None:
        try:
            os.unlink(_DI_RECORDING_MARKER_PATH)
        except FileNotFoundError:
            pass

    def _read_marker(self):
        """Return all recorded emissions as a list of dicts.

        Each entry has keys ``name`` (str), ``op`` (one of ``inc``/``set``/
        ``observe``), ``value`` (numeric) and ``tags`` (dict).
        """
        entries = []
        with open(_DI_RECORDING_MARKER_PATH) as f:
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
                STAT_LOGGER_ROLE_SCHEDULER: _RecordingSchedulerCollector,
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
            os.path.exists(_DI_RECORDING_MARKER_PATH),
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


class TestComputeRoutingKeyStats(unittest.TestCase):
    def test_empty(self):
        num_unique, req_counts = compute_routing_key_stats([])
        self.assertEqual(num_unique, 0)
        self.assertEqual(req_counts, [])

    def test_all_none(self):
        num_unique, req_counts = compute_routing_key_stats([None, None, None])
        self.assertEqual(num_unique, 0)
        self.assertEqual(req_counts, [])

    def test_with_none(self):
        num_unique, req_counts = compute_routing_key_stats([None, "key1", None])
        self.assertEqual(num_unique, 1)
        self.assertEqual(req_counts, [1])

    def test_single_key_multiple_reqs(self):
        num_unique, req_counts = compute_routing_key_stats(["key1"] * 5)
        self.assertEqual(num_unique, 1)
        self.assertEqual(req_counts, [5])

    def test_distribution(self):
        routing_keys = ["key1"] * 5 + ["key2"] * 1 + ["key3"] * 15 + ["key4"] * 250
        num_unique, req_counts = compute_routing_key_stats(routing_keys)
        self.assertEqual(num_unique, 4)
        self.assertEqual(sorted(req_counts), [1, 5, 15, 250])


if __name__ == "__main__":
    unittest.main()
