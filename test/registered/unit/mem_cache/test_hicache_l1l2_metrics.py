"""Unit tests for HiCache L1/L2 transfer metrics.

All tests run on CPU; no GPU device or distributed process group is required.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import unittest
from unittest.mock import MagicMock

from sglang.srt.managers.cache_controller import HiCacheAck, HiCacheController
from sglang.srt.observability.metrics_collector import (
    HiCacheL1L2TransferMetricsCollector,
)

# ── Test doubles ──────────────────────────────────────────────────────────────


class _RecordingCounter:
    """prometheus_client.Counter stand-in that records every call."""

    def __init__(self, *args, **kwargs):
        self.inc_calls: list = []

    def labels(self, **kwargs):
        return self

    def inc(self, amount=1):
        self.inc_calls.append(amount)


class _RecordingHistogram(_RecordingCounter):
    """prometheus_client.Histogram stand-in."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observe_calls: list = []

    def observe(self, value):
        self.observe_calls.append(value)


# ── Controller stub ───────────────────────────────────────────────────────────


class _ControllerStub:
    """Minimal stand-in that lets HiCacheController unbound methods run on CPU."""

    def __init__(self):
        self.hicache_l1_l2_transfer_metrics_collector = None
        self.hicache_l1_l2_transfer_totals = {
            "offload": {"events": 0, "blocks": 0, "bytes": 0, "xfer_us": 0},
            "onboard": {"events": 0, "blocks": 0, "bytes": 0, "xfer_us": 0},
        }

    _transfer_elapsed_us = HiCacheController._transfer_elapsed_us
    record_l1_l2_transfer_complete = HiCacheController.record_l1_l2_transfer_complete


# ── Ack factory ───────────────────────────────────────────────────────────────


def _make_ack(*, block_count=4, byte_count=65536) -> HiCacheAck:
    return HiCacheAck(
        start_event=MagicMock(),
        finish_event=MagicMock(),
        node_ids=[1, 2],
        token_count=64,
        block_count=block_count,
        byte_count=byte_count,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HiCacheL1L2TransferMetricsCollector — call-chain (DI doubles)
# ══════════════════════════════════════════════════════════════════════════════


class TestHiCacheL1L2TransferCollectorRecordTransfer(unittest.TestCase):
    class _Collector(HiCacheL1L2TransferMetricsCollector):
        _counter_cls = _RecordingCounter
        _histogram_cls = _RecordingHistogram

    def setUp(self):
        HiCacheL1L2TransferMetricsCollector._metric_cache.clear()

    def tearDown(self):
        HiCacheL1L2TransferMetricsCollector._metric_cache.clear()

    def _make_collector(self):
        return self._Collector(labels={"tp_rank": "0", "io_backend": "test"})

    def test_record_offload_increments_blocks_counter(self):
        c = self._make_collector()
        c.record_transfer(
            direction="offload",
            src="sglang_hicache::L1",
            dst="sglang_hicache::L2",
            blocks=8,
            bytes_=131072,
            xfer_us=None,
        )
        self.assertEqual(c.transfer_blocks_total.inc_calls, [8])

    def test_record_with_xfer_us_observes_histogram(self):
        c = self._make_collector()
        c.record_transfer(
            direction="offload",
            src="sglang_hicache::L1",
            dst="sglang_hicache::L2",
            blocks=2,
            bytes_=32768,
            xfer_us=5000,
        )
        self.assertEqual(c.transfer_duration_us.observe_calls, [5000])


# ══════════════════════════════════════════════════════════════════════════════
# HiCacheController.record_l1_l2_transfer_complete
# ══════════════════════════════════════════════════════════════════════════════


class TestRecordL1L2TransferComplete(unittest.TestCase):
    def _make_ctrl(self) -> _ControllerStub:
        ctrl = _ControllerStub()
        ctrl.hicache_l1_l2_transfer_metrics_collector = MagicMock()
        ctrl._transfer_elapsed_us = MagicMock(return_value=2000)
        return ctrl

    def test_offload_totals_accumulate_over_two_calls(self):
        ctrl = self._make_ctrl()
        ack = _make_ack(block_count=4, byte_count=65536)
        ctrl.record_l1_l2_transfer_complete(direction="offload", ack=ack)
        ctrl.record_l1_l2_transfer_complete(direction="offload", ack=ack)

        totals = ctrl.hicache_l1_l2_transfer_totals["offload"]
        self.assertEqual(totals["events"], 2)
        self.assertEqual(totals["blocks"], 8)
        self.assertEqual(totals["bytes"], 131072)

    def test_collector_called_with_correct_args_offload(self):
        ctrl = self._make_ctrl()
        ctrl._transfer_elapsed_us = MagicMock(return_value=5000)
        ack = _make_ack(block_count=3, byte_count=49152)
        ctrl.record_l1_l2_transfer_complete(direction="offload", ack=ack)

        ctrl.hicache_l1_l2_transfer_metrics_collector.record_transfer.assert_called_once_with(
            direction="offload",
            src="sglang_hicache::L1",
            dst="sglang_hicache::L2",
            blocks=3,
            bytes_=49152,
            xfer_us=5000,
        )

    def test_collector_called_with_correct_args_onboard(self):
        ctrl = self._make_ctrl()
        ctrl._transfer_elapsed_us = MagicMock(return_value=3000)
        ack = _make_ack(block_count=6, byte_count=98304)
        ctrl.record_l1_l2_transfer_complete(direction="onboard", ack=ack)

        ctrl.hicache_l1_l2_transfer_metrics_collector.record_transfer.assert_called_once_with(
            direction="onboard",
            src="sglang_hicache::L2",
            dst="sglang_hicache::L1",
            blocks=6,
            bytes_=98304,
            xfer_us=3000,
        )


# ══════════════════════════════════════════════════════════════════════════════
# Prometheus text exposition — real prometheus_client, isolated registry
# ══════════════════════════════════════════════════════════════════════════════


class TestHiCacheL1L2TransferCollectorPrometheusOutput(unittest.TestCase):
    """Verifies metric names, label values, and sample values in real Prometheus
    text output using an isolated CollectorRegistry per test."""

    def setUp(self):
        import prometheus_client
        from prometheus_client import CollectorRegistry

        self.registry = CollectorRegistry()
        _reg = self.registry
        _PC = prometheus_client.Counter
        _PH = prometheus_client.Histogram

        # Each setUp() defines new class objects, giving a unique _metric_cache
        # key so every test gets fresh Counter/Histogram instances.
        class _BoundCounter:
            def __init__(self, *args, **kwargs):
                kwargs["registry"] = _reg
                self._inner = _PC(*args, **kwargs)

            def labels(self, **kwargs):
                return self._inner.labels(**kwargs)

        class _BoundHistogram:
            def __init__(self, *args, **kwargs):
                kwargs["registry"] = _reg
                self._inner = _PH(*args, **kwargs)

            def labels(self, **kwargs):
                return self._inner.labels(**kwargs)

        class _TestCollector(HiCacheL1L2TransferMetricsCollector):
            _counter_cls = _BoundCounter
            _histogram_cls = _BoundHistogram

        self._TestCollector = _TestCollector
        HiCacheL1L2TransferMetricsCollector._metric_cache.clear()

    def tearDown(self):
        HiCacheL1L2TransferMetricsCollector._metric_cache.clear()

    def _make_collector(self):
        return self._TestCollector(labels={"tp_rank": "0", "io_backend": "nixl"})

    def _exposition(self) -> str:
        from prometheus_client import generate_latest

        return generate_latest(self.registry).decode("utf-8")

    def _find_metric_line(
        self, exposition: str, name_fragment: str, **label_filters
    ) -> str:
        """Return the first non-comment line matching name_fragment and all
        label_filters, or fail the test."""
        for line in exposition.splitlines():
            if line.startswith("#"):
                continue
            if name_fragment not in line:
                continue
            if all(f'{k}="{v}"' in line for k, v in label_filters.items()):
                return line
        self.fail(
            f"No metric line for fragment={name_fragment!r} "
            f"labels={label_filters} in:\n{exposition}"
        )

    def test_blocks_counter_value(self):
        c = self._make_collector()
        c.record_transfer(
            direction="offload",
            src="sglang_hicache::L1",
            dst="sglang_hicache::L2",
            blocks=7,
            bytes_=114688,
            xfer_us=None,
        )
        line = self._find_metric_line(
            self._exposition(),
            "hicache_l1_l2_transfer_blocks_total",
            direction="offload",
        )
        self.assertTrue(line.endswith(" 7.0"), f"Expected value 7.0 in: {line!r}")

    def test_bytes_counter_value(self):
        c = self._make_collector()
        c.record_transfer(
            direction="onboard",
            src="sglang_hicache::L2",
            dst="sglang_hicache::L1",
            blocks=1,
            bytes_=32768,
            xfer_us=None,
        )
        line = self._find_metric_line(
            self._exposition(),
            "hicache_l1_l2_transfer_bytes_total",
            direction="onboard",
        )
        self.assertTrue(
            line.endswith(" 32768.0"), f"Expected value 32768.0 in: {line!r}"
        )

    def test_duration_histogram_sum_when_xfer_us_provided(self):
        c = self._make_collector()
        c.record_transfer(
            direction="offload",
            src="sglang_hicache::L1",
            dst="sglang_hicache::L2",
            blocks=1,
            bytes_=1024,
            xfer_us=2500,
        )
        line = self._find_metric_line(
            self._exposition(),
            "hicache_l1_l2_transfer_duration_us_sum",
            direction="offload",
        )
        self.assertTrue(line.endswith(" 2500.0"), f"Expected sum 2500.0 in: {line!r}")

    def test_duration_histogram_count_when_xfer_us_provided(self):
        c = self._make_collector()
        c.record_transfer(
            direction="offload",
            src="sglang_hicache::L1",
            dst="sglang_hicache::L2",
            blocks=1,
            bytes_=1024,
            xfer_us=1000,
        )
        line = self._find_metric_line(
            self._exposition(),
            "hicache_l1_l2_transfer_duration_us_count",
            direction="offload",
        )
        self.assertTrue(line.endswith(" 1.0"), f"Expected count 1.0 in: {line!r}")


if __name__ == "__main__":
    unittest.main()
