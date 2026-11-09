"""Fixtures pinning the Rust frontend's ``/metrics`` to Python's.

The Rust server (``rust/sglang-server``) reads prometheus_client's multiprocess
files and renders them itself (``src/utils/prometheus.rs``), and records the
``TokenizerMetricsCollector`` families itself (``src/api_server/core/metrics.rs``).
These tests fail if prometheus_client's file format, merge or text rendering, or
the collector's families, drift from what the committed fixtures record;
regenerate them as each builder's docstring says, then update the Rust side.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=11, suite="base-a-test-cpu")

import json
import os
import struct
import tempfile
import unittest
from unittest import mock

from prometheus_client import Counter, Gauge, Histogram, Summary, values
from prometheus_client.exposition import generate_latest
from prometheus_client.multiprocess import MultiProcessCollector

from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase

TESTDATA = os.path.join(
    os.path.dirname(__file__), *[os.pardir] * 4, "rust/sglang-server/testdata"
)
MULTIPROCESS_FIXTURE = os.path.join(TESTDATA, "prometheus_multiprocess.json")
TOKENIZER_FIXTURE = os.path.join(TESTDATA, "tokenizer_metrics.json")


def _write_process(directory: str, pid: int, now: float, observe) -> None:
    """Run ``observe`` as process ``pid`` writing into ``directory`` at
    time ``now`` (the timestamp ``mostrecent`` gauges record)."""
    previous = values.ValueClass
    values.ValueClass = values.MultiProcessValue(lambda: pid)
    try:
        with (
            mock.patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": directory}),
            mock.patch("time.time", return_value=now),
        ):
            observe()
    finally:
        values.ValueClass = previous


def _metrics():
    help_text = 'Help with "quotes", a back\\slash and a\nnewline.'
    return {
        "counter": Counter(
            "sglang:requests_total",
            help_text,
            ["model_name", "path"],
            registry=None,
        ),
        "bare_counter": Counter("bare_events", "Unlabelled counter.", registry=None),
        "histogram": Histogram(
            "sglang:latency_seconds",
            "Latency.",
            ["model_name"],
            buckets=(0.0001, 0.5, 1, 1000000, 1100000, 1e16),
            registry=None,
        ),
        "summary": Summary("sglang:size_bytes", "Sizes.", ["kind"], registry=None),
        **{
            f"gauge_{mode}": Gauge(
                f"sglang:gauge_{mode}",
                f"A {mode} gauge.",
                ["tp_rank"],
                multiprocess_mode=mode,
                registry=None,
            )
            for mode in ("all", "liveall", "min", "max", "sum", "livesum", "mostrecent")
        },
    }


def build_rust_multiprocess_fixture() -> dict:
    """prometheus_client files from two processes, and the text Python's
    ``/metrics`` renders from them. Regenerate the fixture with
    ``json.dump(build_rust_multiprocess_fixture(), f, indent=2)``."""
    with tempfile.TemporaryDirectory() as directory:

        def first():
            m = _metrics()
            m["counter"].labels("mé", 'a"b\\c\nd').inc(3)
            m["counter"].labels("m2", "/generate").inc(0.5)
            m["bare_counter"].inc()
            for value in (0.00005, 0.3, 2, 1050000, 5e20):
                m["histogram"].labels("m1").observe(value)
            m["summary"].labels("x").observe(10)
            for mode in ("all", "liveall", "min", "max", "sum", "livesum"):
                m[f"gauge_{mode}"].labels("0").set(4)
            m["gauge_mostrecent"].labels("0").set(1)

        def second():
            m = _metrics()
            m["counter"].labels("m2", "/generate").inc(2)
            for value in (0.5, 7):
                m["histogram"].labels("m1").observe(value)
            m["histogram"].labels("m2").observe(1e-9)
            m["summary"].labels("x").observe(2.5)
            for mode in ("all", "liveall", "min", "max", "sum", "livesum"):
                m[f"gauge_{mode}"].labels("0").set(-1.5)
                m[f"gauge_{mode}"].labels("1").set(1e-7)
            m["gauge_mostrecent"].labels("0").set(2)

        _write_process(directory, 101, 1000.0, first)
        _write_process(directory, 202, 2000.0, second)
        files = []
        for name in sorted(os.listdir(directory)):
            with open(os.path.join(directory, name), "rb") as f:
                data = f.read()
            # Drop the zero-filled preallocation past the used length.
            (used,) = struct.unpack_from("i", data)
            files.append({"name": name, "hex": data[:used].hex()})
    return {"files": files, "text": _render_python(files)}


def _render_python(files: list) -> str:
    """Python's ``/metrics`` text for ``files``, merged in the listed order."""
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for file in files:
            path = os.path.join(directory, file["name"])
            with open(path, "wb") as f:
                f.write(bytes.fromhex(file["hex"]))
            paths.append(path)

        class _Files:
            def collect(self):
                return MultiProcessCollector.merge(paths, accumulate=True)

        return generate_latest(_Files()).decode()


def build_rust_tokenizer_metrics_fixture() -> dict:
    """Every family ``TokenizerMetricsCollector`` registers under the default
    server args, as its constructor arguments. Regenerate the fixture with
    ``json.dump(build_rust_tokenizer_metrics_fixture(), f, indent=2)``."""
    families = []

    def recorder(kind):
        def register(name, documentation, labelnames=(), buckets=None, **_):
            families.append(
                {
                    "name": name,
                    "kind": kind,
                    "documentation": documentation,
                    "labelnames": list(labelnames),
                    "buckets": (
                        None if buckets is None else [float(b) for b in buckets]
                    ),
                }
            )
            return mock.MagicMock()

        return staticmethod(register)

    class _Recording(TokenizerMetricsCollector):
        _counter_cls = recorder("counter")
        _gauge_cls = recorder("gauge")
        _histogram_cls = recorder("histogram")

    labels = {"model_name": "model", "engine_type": "unified"}
    with get_context().override_server_args():
        _Recording(labels=labels)
    return {"labels": labels, "families": families}


class TestRustMetricsFixtures(CustomTestCase):
    def test_tokenizer_metrics_fixture_matches_python(self):
        with open(TOKENIZER_FIXTURE) as f:
            fixture = json.load(f)
        self.assertEqual(fixture, build_rust_tokenizer_metrics_fixture())

    def test_multiprocess_fixture_matches_python(self):
        with open(MULTIPROCESS_FIXTURE) as f:
            fixture = json.load(f)
        self.assertEqual(fixture, build_rust_multiprocess_fixture())


if __name__ == "__main__":
    unittest.main()
