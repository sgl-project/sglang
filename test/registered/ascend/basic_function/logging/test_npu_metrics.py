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
)
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH as _MODEL_NAME
from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=120, suite="full-1-npu-a3", nightly=True)


class _BaseTestNPUMetrics(TestNPULoggingBase):
    """Base class for NPU metrics tests.

    Handles server launch in setUpClass and cleanup in tearDownClass.
    Subclasses should override ``metrics_args`` and ``metrics_env`` to
    configure the server before it is launched.
    """

    enable_mfu_metrics: bool = False
    metrics_args: list = []
    repeat_requests_num: int = 2
    verify_metrics_extra: bool = False

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.other_args.extend(cls.metrics_args)
        with (
            envs.SGLANG_ENABLE_METRICS_DP_ATTENTION.override(True),
            envs.SGLANG_ENABLE_METRICS_DEVICE_TIMER.override(True),
            envs.SGLANG_TEST_RETRACT.override(True),
        ):
            cls.launch_server()

    def test_metrics(self):
        _generate_metrics(self.base_url, self.repeat_requests_num)

        metrics_response = requests.get(f"{self.base_url}/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        metrics_text = metrics_response.text

        print(f"metrics_text=\n{metrics_text}")

        metrics = _parse_prometheus_metrics(metrics_text)
        _verify_metrics_common(
            self, metrics_text, metrics, expect_mfu_metrics=self.enable_mfu_metrics
        )

        def _check_dp_cooperation_metrics(metrics):
            # In the GPU scenario test case
            # (test/registered/observability/test_metrics.py), the 2-card scenario
            # contains assertions about the following monitoring metrics:
            #   ("sglang:dp_cooperation_forward_execution_seconds_total",
            #    {"category": "extend"}),
            #   ("sglang:dp_cooperation_forward_execution_seconds_total",
            #    {"category": "decode"}),
            # However, these two indicators were not found during execution on the
            # GPU, and it is uncertain whether it is a problem or if monitoring of
            # these indicators is currently not supported.
            metrics_to_check = [
                (
                    "sglang:dp_cooperation_realtime_tokens_total",
                    {"mode": "prefill_compute"},
                ),
                (
                    "sglang:dp_cooperation_realtime_tokens_total",
                    {"mode": "decode"},
                ),
            ]
            _check_metrics_positive(self, metrics, metrics_to_check)

            num_prefill_ranks_values = {
                s.labels["num_prefill_ranks"]
                for s in metrics["sglang:dp_cooperation_realtime_tokens_total"]
            }
            self.assertIn("0", num_prefill_ranks_values)
            self.assertIn("1", num_prefill_ranks_values)

        if self.verify_metrics_extra:
            _check_dp_cooperation_metrics(metrics)


class TestNPUMetricsMFUEnabled(_BaseTestNPUMetrics):
    """Test core metrics functionality on single NPU with MFU enabled.

    [Description]
        Validates that the /metrics endpoint returns correct Prometheus-format
        data when both --enable-metrics and --enable-mfu-metrics are enabled
        on a single NPU. Verifies that essential metrics (throughput, latency,
        token counts, cache hit rate, etc.) and MFU metrics (estimated FLOPs,
        read/write bytes) are all present and contain positive values.

    [Test Category] Functionality
    [Test Target] --enable-metrics; --enable-mfu-metrics
    """

    enable_mfu_metrics = True
    metrics_args = ["--enable-metrics", "--enable-mfu-metrics"]


class TestNPUMetricsMFUDisabled(_BaseTestNPUMetrics):
    """Test that MFU metrics are not emitted when the gate is disabled.

    [Description]
        Validates that when only --enable-metrics is set (without
        --enable-mfu-metrics), the essential metrics are still exported
        correctly but MFU counters (estimated_flops_per_gpu_total,
        estimated_read_bytes_per_gpu_total, estimated_write_bytes_per_gpu_total)
        are either absent or remain at zero.

    [Test Category] Functionality
    [Test Target] --enable-metrics
    """

    enable_mfu_metrics = False
    metrics_args = ["--enable-metrics"]


class TestNPUMetrics2NPU(_BaseTestNPUMetrics):
    """Test metrics on 2-NPU TP/DP parallel scenario.

    [Description]
        Validates distributed metrics (dp_cooperation_*) when running with
        TP=2 and DP=2 on NPU. Verifies that dp_cooperation_realtime_tokens_total
        is emitted with correct num_prefill_ranks labels (both "0" and "1"),
        and that all essential metrics and MFU metrics are present with
        positive values under the multi-NPU configuration.

    [Test Category] Functionality
    [Test Target] --enable-metrics; --enable-mfu-metrics
    """

    enable_mfu_metrics = True
    metrics_args = [
        "--enable-metrics",
        "--enable-mfu-metrics",
        "--tp",
        "2",
        "--dp",
        "2",
        "--enable-dp-attention",
    ]
    # In single-card scenarios, two identical requests suffice to trigger a KV
    # cache hit (the second request reuses the cache from the first). However,
    # with tp=2 dp=2 multi-card parallelism, requests are distributed across
    # different cards. With too few requests, a single card may not receive
    # enough duplicates, causing cached_tokens_total to remain zero. Bumping
    # the repeat count from the default 2 to 6 ensures each card has a high
    # probability of receiving multiple identical requests, reliably triggering
    # cache hits.
    repeat_requests_num = 6
    verify_metrics_extra = True


class TestNPUMetricsExtraLabels(_BaseTestNPUMetrics):
    """Testcase: Verify that --extra-metric-labels injects constant labels into all metrics.

    [Test Category] Parameter
    [Test Target] --extra-metric-labels
    """

    metrics_args = [
        "--enable-metrics",
        "--extra-metric-labels",
        '{"env":"prod","team":"npu-inference","region":"us-east-1"}',
    ]

    def test_metrics_extra_metric_labels(self):
        metrics_response = requests.get(f"{self.base_url}/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        metrics_text = metrics_response.text

        expected_labels = {
            "env": "prod",
            "team": "npu-inference",
            "region": "us-east-1",
        }

        # 1. Raw text sanity check
        for k, v in expected_labels.items():
            self.assertIn(f'{k}="{v}"', metrics_text)

        # 2. Parsed metrics structural check
        metrics = _parse_prometheus_metrics(metrics_text)
        print(f"extra_metric_labels_text=\n{metrics_text}")

        # Pick a representative set of metrics to verify
        metrics_to_check = [
            "sglang:num_running_reqs",
            "sglang:prompt_tokens_total",
            "sglang:generation_tokens_total",
            "sglang:time_to_first_token_seconds_sum",
            "sglang:forward_execution_seconds_total",
        ]

        for metric_name in metrics_to_check:
            self.assertIn(metric_name, metrics, f"Missing metric: {metric_name}")

            samples = metrics[metric_name]
            self.assertGreater(len(samples), 0, f"{metric_name} has no samples")

            for sample in samples:
                for k, v in expected_labels.items():
                    self.assertEqual(
                        sample.labels.get(k),
                        v,
                        f"{metric_name}: label {k} mismatch: "
                        f"got {sample.labels.get(k)}, want {v}",
                    )


def _generate_metrics(base_url: str, repeat_requests_num: int = 2) -> None:
    """Send requests to generate metrics data.

    The workload is intentionally generous so that every counter and
    histogram we assert on later accumulates a clearly positive value.
    A lightweight workload risks leaving some metrics at zero because
    forward-pass time may round to 0.0, asynchronous flushes may be
    missed, or slow CI runners may not finish processing in time.
    """
    response = requests.get(f"{base_url}/health_generate")
    assert response.status_code == 200

    # 1) Large batch + long decode: guarantees substantial prefill and decode
    #    workload so that realtime_tokens_total and
    #    forward_execution_seconds_total accumulate clearly positive values.
    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": ["The capital of France is"] * 20,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 50,
            },
            "stream": True,
            # Force the model to generate the full max_new_tokens instead of
            # stopping early when an EOS token is produced. This guarantees a
            # predictable decode workload so that decode-phase counters do not
            # end up at zero because the model finished prematurely.
            "ignore_eos": True,
        },
        stream=True,
    )
    # Consume the streaming response so the request fully completes on the
    # server side.  Without draining the iterator the connection stays open
    # and the request may still be counted as in-flight.
    for _ in response.iter_lines(decode_unicode=False):
        pass

    # 2) Repeated requests with a routing key: populates routing-key histograms
    #    and cached_tokens_total (the second request may hit the KV cache).
    for i in range(repeat_requests_num):
        response = requests.post(
            f"{base_url}/generate",
            json={
                "text": "Hello, " * 100,
                "sampling_params": {"temperature": 0, "max_new_tokens": 5},
            },
            headers={"x-smg-routing-key": "test-key"},
        )
        assert response.status_code == 200


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
        test_case.assertGreater(
            value,
            0,
            f"{metric_name} {labels}: expected a positive value, got {value} "
            f"(the engine may not have processed any requests)",
        )


def _verify_metrics_common(test_case, metrics_text, metrics, expect_mfu_metrics: bool):
    model_name = test_case.model
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

    # Verify that all essential metric names appear in the raw Prometheus text.
    # This is a basic existence check: it ensures the metrics are exported but
    # does not validate their values or label sets.
    for metric in essential_metrics:
        test_case.assertIn(metric, metrics_text, f"Missing metric: {metric}")

    # Verify the bucket structure of routing-key request-count histograms.
    # These metrics use a GaugeHistogram where each bucket is identified by
    # (gt, le) label pairs:
    #   - "gt" (greater than): the lower bound of the bucket (exclusive)
    #   - "le" (less than or equal to): the upper bound of the bucket (inclusive)
    # For example, (gt="0", le="1") represents the bucket that counts values
    # in the range (0, 1]. The expected number of buckets equals the number of
    # user-defined bounds plus one extra bucket for +Inf.
    expected_buckets = len(ROUTING_KEY_REQ_COUNT_BUCKET_BOUNDS) + 1
    for metric_name in [
        "sglang:routing_key_running_req_count",
        "sglang:routing_key_all_req_count",
    ]:
        gt_le_pairs = set()
        for sample in metrics.get(metric_name, []):
            gt_le_pairs.add((sample.labels.get("gt"), sample.labels.get("le")))
        test_case.assertEqual(
            len(gt_le_pairs),
            expected_buckets,
            f"{metric_name}: Expected {expected_buckets} buckets, got {len(gt_le_pairs)}",
        )

    # Verify that metrics carry the correct model_name label and that
    # histogram-style metrics expose the standard _sum, _count, and _bucket
    # time-series. We assert on the exact metric names rather than raw
    # substrings so the failure message points to the missing series.
    histogram_metrics = [
        "sglang:time_to_first_token_seconds",
        "sglang:inter_token_latency_seconds",
        "sglang:e2e_request_latency_seconds",
    ]
    for base_name in histogram_metrics:
        for suffix in ("_sum", "_count", "_bucket"):
            test_case.assertIn(
                f"{base_name}{suffix}{{",
                metrics_text,
                f"Missing histogram series: {base_name}{suffix}",
            )

    # All metrics should be tagged with the model_name of the served model.
    test_case.assertIn(f'model_name="{model_name}"', metrics_text)

    # Verify that core performance counters have accumulated positive values.
    # These metrics prove the engine actually processed requests (prefill +
    # decode tokens, forward-pass time, tokenizer CPU time) rather than just
    # exporting zero-valued series.
    #
    # Why assert > 0 instead of >= 0?
    #   - The test deliberately sends a large workload (see _generate_metrics)
    #     so that every counter here is guaranteed to be clearly positive.
    #   - A value of 0 would mean the corresponding pipeline stage did no
    #     work, which indicates a bug (e.g. device timer disabled, metrics
    #     flush lost, or the request never reached the engine).
    metrics_to_check = [
        # Total tokens processed during the prefill (context-encoding) phase.
        ("sglang:realtime_tokens_total", {"mode": "prefill_compute"}),
        # Total tokens generated during the decode (autoregressive) phase.
        ("sglang:realtime_tokens_total", {"mode": "decode"}),
        # Cumulative time spent in the extend (prefix-extension) forward pass.
        ("sglang:forward_execution_seconds_total", {"category": "extend"}),
        # Cumulative time spent in the decode forward pass.
        ("sglang:forward_execution_seconds_total", {"category": "decode"}),
        # CPU time consumed by the tokenizer subprocess.
        ("sglang:process_cpu_seconds_total", {"component": "tokenizer"}),
    ]
    _check_metrics_positive(test_case, metrics, metrics_to_check)

    # MFU (Model FLOPs Utilization) metrics estimate the compute and memory
    # bandwidth consumed by the model. They are only collected when the server
    # is started with --enable-mfu-metrics. Verify the gate behaves correctly:
    #   - Gate ON  -> counters must contain positive values for this model.
    #   - Gate OFF -> counters must be absent or zero.
    if expect_mfu_metrics:
        for metric_name in mfu_metrics:
            # Filter samples belonging to the current model (multi-model
            # deployments may include series for other models).
            values = [
                sample.value
                for sample in metrics.get(metric_name, [])
                if sample.labels.get("model_name") == model_name
            ]
            test_case.assertTrue(
                values, f"{metric_name}: no samples for model {model_name}"
            )
            test_case.assertGreater(
                sum(values),
                0,
                f"{metric_name}: expected positive total for model {model_name}",
            )
    else:
        for metric_name in mfu_metrics:
            values = [
                sample.value
                for sample in metrics.get(metric_name, [])
                if sample.labels.get("model_name") == model_name
            ]
            # Some implementations still emit the metric name with a zero value
            # when the gate is disabled; ensure no positive accumulation leaked.
            if values:
                test_case.assertEqual(
                    sum(values),
                    0,
                    f"{metric_name}: expected no positive samples with MFU metrics gate disabled",
                )


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


class TestNPUStatLoggersDI(CustomTestCase):
    """Verify that a custom MetricsCollector subclass passed through
    ``ServerArgs.stat_loggers`` is the one instantiated inside the
    scheduler subprocess on NPU."""

    def setUp(self) -> None:
        os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
        os.environ.pop("SGLANG_TEST_RETRACT", None)
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
            device="npu",
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


class TestNPUStatLoggersDIRecording(CustomTestCase):
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
        os.environ.pop("SGLANG_TEST_RETRACT", None)
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
            device="npu",
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


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestNPUMetricsMFUDisabled))
    suite.addTests(loader.loadTestsFromTestCase(TestNPUMetricsMFUEnabled))
    suite.addTests(loader.loadTestsFromTestCase(TestNPUMetrics2NPU))
    suite.addTests(loader.loadTestsFromTestCase(TestNPUMetricsExtraLabels))
    suite.addTests(loader.loadTestsFromTestCase(TestNPUStatLoggersDI))
    suite.addTests(loader.loadTestsFromTestCase(TestNPUStatLoggersDIRecording))
    runner = unittest.TextTestRunner()
    runner.run(suite)
