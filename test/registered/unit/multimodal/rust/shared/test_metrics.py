"""Exercise real multiprocess collection and the private exporter's lifetime."""

import asyncio
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import ClientSession, UnixConnector, web
from aiohttp.test_utils import TestClient, TestServer
from prometheus_client import multiprocess
from prometheus_client.openmetrics.parser import (
    text_string_to_metric_families as openmetrics_families,
)
from prometheus_client.parser import text_string_to_metric_families

from sglang.srt.rust_server import metrics, metrics_sources
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def python_frontend_schema():
    """Capture the actual Python registrations, before any observations."""
    import prometheus_client
    from fastapi import FastAPI

    maybe_stub_sgl_kernel()
    from sglang.srt.observability import metrics_collector
    from sglang.srt.utils.common import add_prometheus_track_response_middleware

    registry = prometheus_client.CollectorRegistry()
    schema = {}

    def factory(kind, constructor):
        def register(*args, **kwargs):
            metric = constructor(*args, **kwargs, registry=registry)
            record = {
                "type": kind,
                "help": metric._documentation,
                "labels": sorted(metric._labelnames),
                "zero_series": any(f.samples for f in metric.collect()),
            }
            if kind == "histogram":
                record["buckets"] = [
                    bound for bound in metric._upper_bounds if math.isfinite(bound)
                ]
            if kind == "gauge":
                record["gauge_mode"] = metric._multiprocess_mode
            schema[kwargs["name"]] = record
            return metric

        return register

    labels = {
        "model_name": "test",
        "engine_type": "unified",
        "cluster": "test",
        "tenant": "",
        "priority": "",
    }
    with (
        patch.multiple(
            prometheus_client,
            Counter=factory("counter", prometheus_client.Counter),
            Histogram=factory("histogram", prometheus_client.Histogram),
            Gauge=factory("gauge", prometheus_client.Gauge),
        ),
        patch.object(
            metrics_collector,
            "get_observability",
            return_value=SimpleNamespace(
                prompt_tokens_buckets=["default"],
                generation_tokens_buckets=["default"],
            ),
        ),
    ):
        metrics_collector.TokenizerMetricsCollector(labels=labels)
        add_prometheus_track_response_middleware(
            FastAPI(), extra_labels={"cluster": "test"}
        )
    return schema


def write_worker_metrics(directory, rank, gauge_value=1):
    # A fresh interpreter imports prometheus_client after its directory is set,
    # just as a scheduler process does. Its counters survive process exit.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from prometheus_client import Counter, Gauge, Histogram
rank = sys.argv[1]
Counter('sglang_test_tokens', 'Generated tokens', ['rank']).labels(rank).inc(3)
Gauge('sglang_test_active', 'Active workers', ['rank'],
      multiprocess_mode='livesum').labels(rank).set(1)
Histogram('sglang:test_latency_seconds', 'Stage latency', ['rank']).labels(rank).observe(0.5)
for mode in ('all', 'liveall', 'sum', 'livesum', 'min', 'livemin',
             'max', 'livemax', 'mostrecent', 'livemostrecent'):
    Gauge('sglang_test_gauge_' + mode, 'Gauge mode test',
          multiprocess_mode=mode).set(float(sys.argv[2]))
""",
            str(rank),
            str(gauge_value),
        ],
        env={**os.environ, "PROMETHEUS_MULTIPROC_DIR": directory},
        check=True,
        timeout=10,
    )


class TestMetricsCollection(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.collection = metrics_sources.MetricsCollection(
            self.directory.name, "local"
        )
        self.client = TestClient(
            TestServer(metrics.make_metrics_app(self.directory.name, self.collection))
        )
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def test_distinct_sources_merge_gauge_modes_and_histograms_once(self):
        from prometheus_client import CollectorRegistry, Histogram, generate_latest
        from prometheus_client.core import CounterMetricFamily, HistogramMetricFamily

        remote_dir = tempfile.TemporaryDirectory()
        self.addCleanup(remote_dir.cleanup)
        write_worker_metrics(self.directory.name, 0, 1)
        write_worker_metrics(remote_dir.name, 0, 2)
        remote_collection = metrics_sources.MetricsCollection(remote_dir.name, "remote")
        remote = TestServer(
            metrics.make_metrics_app(
                remote_dir.name, remote_collection, "/source/remote"
            )
        )
        await remote.start_server()
        self.addAsyncCleanup(remote.close)

        registry = CollectorRegistry()
        counter = CounterMetricFamily(
            "sglang_test_tokens", "Generated tokens", labels=["rank"]
        )
        counter.add_metric(["0"], 5)
        histogram = HistogramMetricFamily(
            "sglang:test_latency_seconds", "Stage latency", labels=["rank"]
        )
        histogram.add_metric(
            ["0"],
            [(str(bound), int(bound >= 1.5)) for bound in Histogram.DEFAULT_BUCKETS],
            1.5,
        )
        registry.register(metrics_sources.CollectedMetrics([counter, histogram]))
        native_app = web.Application()

        async def native_snapshot(request):
            return web.Response(body=generate_latest(registry))

        native_app.router.add_get("/metrics/native", native_snapshot)
        native = TestServer(native_app)
        await native.start_server()
        self.addAsyncCleanup(native.close)
        sources = [
            metrics_sources.MetricSource(
                "remote", str(remote.make_url("/source/remote"))
            ),
            metrics_sources.MetricSource(
                "native", str(native.make_url("/metrics/native")), "native"
            ),
        ]
        # Several DP leaders advertise the same host's Python collector.
        self.collection.configure(sources * 4)
        self.collection.set_startup_time(
            {
                "load_weight": 2,
                "kv_cache_allocation": 3,
                "scheduler_e2e": 7,
                "tokenizer_e2e": 9,
                "cuda_graph": {"target": 1.5},
            },
            {"model_name": "test", "engine_type": "unified"},
        )
        async with self.client.get(
            "/metrics",
            headers={"Accept": "application/openmetrics-text; version=1.0.0"},
        ) as response:
            self.assertEqual(response.status, 200, await response.text())
            families = list(openmetrics_families(await response.text()))
        samples = {family.name: family.samples for family in families}
        self.assertEqual(samples["sglang_test_tokens"][0].value, 11)
        histogram = samples["sglang:test_latency_seconds"]
        self.assertEqual(
            next(s.value for s in histogram if s.name.endswith("_count")), 3
        )
        self.assertEqual(
            next(s.value for s in histogram if s.name.endswith("_sum")), 2.5
        )
        self.assertEqual(
            next(s.value for s in histogram if s.labels.get("le") == "0.5"), 2
        )
        for mode, expected in (("sum", 3), ("min", 1), ("max", 2), ("mostrecent", 2)):
            for name in (mode, "live" + mode):
                self.assertEqual(
                    samples["sglang_test_gauge_" + name][0].value, expected
                )
        for name in ("all", "liveall"):
            series = samples["sglang_test_gauge_" + name]
            self.assertEqual(sorted(s.value for s in series), [1, 2])
            self.assertEqual(
                {s.labels["pid"].split("/")[0] for s in series}, {"local", "remote"}
            )
        startup = samples["sglang:startup_time_seconds"]
        self.assertEqual(
            {s.labels["phase"]: s.value for s in startup},
            {
                "load_weight": 2,
                "kv_cache_allocation": 3,
                "scheduler_e2e": 7,
                "tokenizer_e2e": 9,
            },
        )
        # Missing a remote collector fails the whole scrape; a partial sum
        # would look healthy while dropping a node's measurements.
        await remote.close()
        async with self.client.get("/metrics") as response:
            self.assertEqual(response.status, 500)

    async def test_source_identity_and_histogram_schema_conflicts_are_rejected(self):
        write_worker_metrics(self.directory.name, 0)
        original = metrics_sources.read_python_metrics(self.directory.name, "one")
        duplicate = copy.deepcopy(original)
        duplicate["source_id"] = "two"
        # The same OS pid on two nodes still produces two all/liveall series.
        merged = metrics_sources.merge_sources(
            [
                (metrics_sources.MetricSource("one", ""), original),
                (metrics_sources.MetricSource("two", ""), duplicate),
            ]
        )
        gauge = next(f for f in merged if f.name == "sglang_test_gauge_all")
        self.assertEqual(len(gauge.samples), 2)
        with self.assertRaisesRegex(ValueError, "identity"):
            metrics_sources.merge_sources(
                [(metrics_sources.MetricSource("wrong", ""), original)]
            )
        histogram = next(f for f in duplicate["metrics"] if f["type"] == "histogram")
        histogram["samples"] = [
            s for s in histogram["samples"] if s["labels"].get("le") != "0.5"
        ]
        with self.assertRaisesRegex(ValueError, "histogram buckets"):
            metrics_sources.merge_sources(
                [
                    (metrics_sources.MetricSource("one", ""), original),
                    (metrics_sources.MetricSource("two", ""), duplicate),
                ]
            )
        with self.assertRaisesRegex(ValueError, "summary quantiles"):
            metrics_sources.merge_sources(
                [
                    (
                        metrics_sources.MetricSource("quantiles", "", "native"),
                        "# HELP latency Request latency\n# TYPE latency summary\n"
                        'latency{quantile="0.5"} 1\nlatency_sum 1\nlatency_count 1\n',
                    ),
                ]
            )

    async def test_workers_are_collected_once_and_exposition_is_negotiated(self):
        for rank in range(2):
            write_worker_metrics(self.directory.name, rank)
        async with self.client.get("/metrics") as response:
            self.assertEqual(response.status, 200)
            self.assertTrue(response.headers["Content-Type"].startswith("text/plain"))
            body = await response.text()
        samples = [
            sample
            for family in text_string_to_metric_families(body)
            for sample in family.samples
            if sample.name == "sglang_test_tokens_total"
        ]
        self.assertEqual(len(samples), 2)
        self.assertEqual({sample.labels["rank"] for sample in samples}, {"0", "1"})
        self.assertEqual([sample.value for sample in samples], [3, 3])

        # The existing collector's live-gauge cleanup removes only live gauges;
        # counters from exited workers remain in the engine's cumulative totals.
        for path in Path(self.directory.name).glob("gauge_livesum_*.db"):
            multiprocess.mark_process_dead(
                int(path.stem.rsplit("_", 1)[1]), path=self.directory.name
            )
        async with self.client.get(
            "/metrics/",
            headers={"Accept": "application/openmetrics-text; version=1.0.0"},
        ) as response:
            self.assertEqual(response.status, 200)
            self.assertTrue(
                response.headers["Content-Type"].startswith(
                    "application/openmetrics-text"
                )
            )
            body = await response.text()
        self.assertEqual(body.count("# EOF\n"), 1)
        self.assertTrue(body.endswith("# EOF\n"))
        families = list(openmetrics_families(body))
        histogram = next(f for f in families if f.name == "sglang:test_latency_seconds")
        counts = [s for s in histogram.samples if s.name.endswith("_count")]
        self.assertEqual(len(counts), 2)
        self.assertEqual([sample.value for sample in counts], [1, 1])
        self.assertNotIn("sglang_test_active", body)
        self.assertIn('sglang_test_tokens_total{rank="0"} 3.0', body)

    async def test_overlapping_scrapes_share_bounded_collection_work(self):
        started = threading.Event()
        release = threading.Event()
        completed = threading.Event()
        calls = 0
        original = metrics_sources.read_python_metrics

        def slow_collect(*args):
            nonlocal calls
            calls += 1
            started.set()
            try:
                if not release.wait(timeout=5):
                    raise TimeoutError("test did not release collection")
                return original(*args)
            finally:
                completed.set()

        with (
            patch.object(metrics_sources, "read_python_metrics", slow_collect),
            patch.object(metrics, "COLLECTION_TIMEOUT_SECONDS", 0.1),
        ):
            try:
                responses = await asyncio.gather(
                    *(self.client.get("/metrics") for _ in range(12))
                )
                self.assertTrue(started.is_set())
                self.assertEqual(calls, 1)
                self.assertTrue(all(response.status == 503 for response in responses))
                for response in responses:
                    await response.read()
            finally:
                release.set()
                self.assertTrue(await asyncio.to_thread(completed.wait, 5))

    async def test_collection_errors_fail_the_scrape(self):
        with patch.object(
            metrics_sources, "read_python_metrics", side_effect=OSError("broken")
        ):
            async with self.client.get("/metrics") as response:
                self.assertEqual(response.status, 500)
                self.assertEqual(await response.text(), "Metrics collection failed")

    async def test_exporter_owns_a_private_socket_and_restores_environment(self):
        with patch.dict(
            os.environ,
            {
                "PROMETHEUS_MULTIPROC_DIR": self.directory.name,
                metrics.METRICS_SOCKET_ENV: "/previous/collector.sock",
            },
        ):
            with metrics.MetricsExporter() as exporter:
                socket = Path(exporter.socket_path)
                self.assertEqual(os.environ[metrics.METRICS_SOCKET_ENV], str(socket))
                self.assertEqual(socket.stat().st_mode & 0o777, 0o600)
                self.assertEqual(socket.parent.stat().st_mode & 0o777, 0o700)
                async with ClientSession(
                    connector=UnixConnector(path=str(socket))
                ) as client:
                    async with client.get("http://localhost/metrics") as response:
                        self.assertEqual(response.status, 200)
            self.assertFalse(socket.exists())
            self.assertFalse(exporter._thread.is_alive())
            self.assertEqual(
                os.environ[metrics.METRICS_SOCKET_ENV], "/previous/collector.sock"
            )


class TestMetricsDirectory(unittest.TestCase):
    def test_python_collector_schema_matches_the_native_contract(self):
        fixture = (
            Path(__file__).resolve().parents[6]
            / "rust/sglang-server/testdata/metrics_python_schema.json"
        )
        self.assertEqual(python_frontend_schema(), json.loads(fixture.read_text()))

    def test_repeated_setup_keeps_live_collector_files(self):
        maybe_stub_sgl_kernel()
        from sglang.srt.utils import common

        with (
            patch.dict(os.environ),
            patch.object(common, "prometheus_multiproc_dir", None),
        ):
            os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
            common.set_prometheus_multiproc_dir()
            owner = common.prometheus_multiproc_dir
            self.addCleanup(owner.cleanup)
            directory = os.environ["PROMETHEUS_MULTIPROC_DIR"]
            path = Path(directory) / "live-collector.db"
            path.write_bytes(b"live")

            common.set_prometheus_multiproc_dir()

            self.assertIs(common.prometheus_multiproc_dir, owner)
            self.assertEqual(os.environ["PROMETHEUS_MULTIPROC_DIR"], directory)
            self.assertEqual(path.read_bytes(), b"live")


if __name__ == "__main__":
    unittest.main()
