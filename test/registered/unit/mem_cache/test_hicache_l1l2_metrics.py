"""Unit tests for HiCache L1/L2 transfer metrics.

Covers:
- HiCacheL1L2TransferMetricsCollector.record_transfer
- HiCacheController helper methods: _transfer_index_count, _host_pool_size_per_token,
  _estimate_l1_l2_transfer_bytes, _make_hicache_ack, _transfer_elapsed_us
- HiCacheController.record_l1_l2_transfer_complete (totals, early-exit, direction labels)

All tests run on CPU; no GPU device or distributed process group is required.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import logging
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.cache_controller import HiCacheAck, HiCacheController
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.observability.metrics_collector import HiCacheL1L2TransferMetricsCollector


# ── Test doubles ──────────────────────────────────────────────────────────────


class _RecordingCounter:
    """prometheus_client.Counter stand-in.

    .labels() returns self so the inc() call chains directly, and every call
    is recorded for assertion.
    """

    def __init__(self, *args, **kwargs):
        self.inc_calls: list = []
        self.label_calls: list = []

    def labels(self, **kwargs):
        self.label_calls.append(dict(kwargs))
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


# ── Host-pool helpers ─────────────────────────────────────────────────────────


class _PlainHostPool:
    """Pool with a flat size_per_token attribute — the common HiCache case."""

    def __init__(self, size_per_token: int):
        self.size_per_token = size_per_token


class _UnknownPool:
    """Pool without size_per_token and without entry_map — triggers the warning path."""


class _HybridPool:
    """Pool with entry_map — models the hybrid/sidecar cache host pool."""

    def __init__(self, pool_map: dict):
        # pool_map: {PoolName: object with .host_pool.size_per_token}  or  None value
        self.entry_map = pool_map


def _hybrid_entry(size_per_token: int):
    """Build a minimal entry_map value (entry.host_pool.size_per_token)."""
    return types.SimpleNamespace(host_pool=types.SimpleNamespace(size_per_token=size_per_token))


# ── Controller stub ───────────────────────────────────────────────────────────


class _ControllerStub:
    """Minimal stand-in that lets HiCacheController unbound methods execute on CPU.

    Real methods are bound at class level; tests set instance attributes to
    configure the method's observable state.
    """

    def __init__(self):
        self.page_size = 16
        self.has_draft = False
        self.mem_pool_host = _PlainHostPool(2048)
        self.mem_pool_host_draft = None
        self._warned_unknown_host_pool_bytes = False
        self.hicache_l1_l2_transfer_metrics_collector = None
        self.hicache_l1_l2_transfer_totals = {
            "offload": {"events": 0, "blocks": 0, "bytes": 0, "xfer_us": 0},
            "onboard": {"events": 0, "blocks": 0, "bytes": 0, "xfer_us": 0},
        }

    # Bind real HiCacheController methods
    _warn_unknown_host_pool_bytes = HiCacheController._warn_unknown_host_pool_bytes
    _host_pool_size_per_token = HiCacheController._host_pool_size_per_token
    _transfer_index_count = HiCacheController._transfer_index_count
    _estimate_l1_l2_transfer_bytes = HiCacheController._estimate_l1_l2_transfer_bytes
    _make_hicache_ack = HiCacheController._make_hicache_ack
    _transfer_elapsed_us = HiCacheController._transfer_elapsed_us
    record_l1_l2_transfer_complete = HiCacheController.record_l1_l2_transfer_complete


# ── Ack factory ───────────────────────────────────────────────────────────────


def _make_ack(
    *,
    node_ids=None,
    token_count=64,
    block_count=4,
    byte_count=65536,
    start_event=None,
    finish_event=None,
) -> HiCacheAck:
    return HiCacheAck(
        start_event=start_event or MagicMock(),
        finish_event=finish_event or MagicMock(),
        node_ids=node_ids if node_ids is not None else [1, 2],
        token_count=token_count,
        block_count=block_count,
        byte_count=byte_count,
    )


# ══════════════════════════════════════════════════════════════════════════════
# HiCacheL1L2TransferMetricsCollector
# ══════════════════════════════════════════════════════════════════════════════


class TestHiCacheL1L2TransferCollectorDI(unittest.TestCase):
    """DI hook attrs must default to None so prometheus_client is used unchanged."""

    def test_di_attrs_default_none(self):
        self.assertIsNone(HiCacheL1L2TransferMetricsCollector._counter_cls)
        self.assertIsNone(HiCacheL1L2TransferMetricsCollector._histogram_cls)


class TestHiCacheL1L2TransferCollectorRecordTransfer(unittest.TestCase):
    """Verify record_transfer updates counters and histograms correctly."""

    class _Collector(HiCacheL1L2TransferMetricsCollector):
        _counter_cls = _RecordingCounter
        _histogram_cls = _RecordingHistogram

    def setUp(self):
        # Clear the metric cache so every test gets fresh Recording instances.
        HiCacheL1L2TransferMetricsCollector._metric_cache.clear()

    def tearDown(self):
        HiCacheL1L2TransferMetricsCollector._metric_cache.clear()

    def _make_collector(self, extra_labels=None):
        labels = {"tp_rank": "0", "io_backend": "test"}
        if extra_labels:
            labels.update(extra_labels)
        return self._Collector(labels=labels)

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

    def test_record_offload_increments_bytes_counter(self):
        c = self._make_collector()
        c.record_transfer(
            direction="offload",
            src="sglang_hicache::L1",
            dst="sglang_hicache::L2",
            blocks=8,
            bytes_=131072,
            xfer_us=None,
        )
        self.assertEqual(c.transfer_bytes_total.inc_calls, [131072])

    def test_record_onboard_increments_blocks_and_bytes(self):
        c = self._make_collector()
        c.record_transfer(
            direction="onboard",
            src="sglang_hicache::L2",
            dst="sglang_hicache::L1",
            blocks=4,
            bytes_=65536,
            xfer_us=None,
        )
        self.assertEqual(c.transfer_blocks_total.inc_calls, [4])
        self.assertEqual(c.transfer_bytes_total.inc_calls, [65536])

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

    def test_record_without_xfer_us_skips_histogram(self):
        c = self._make_collector()
        c.record_transfer(
            direction="offload",
            src="sglang_hicache::L1",
            dst="sglang_hicache::L2",
            blocks=2,
            bytes_=32768,
            xfer_us=None,
        )
        self.assertEqual(c.transfer_duration_us.observe_calls, [])

    def test_metric_labels_contain_direction_src_dst(self):
        c = self._make_collector()
        c.record_transfer(
            direction="offload",
            src="sglang_hicache::L1",
            dst="sglang_hicache::L2",
            blocks=1,
            bytes_=1024,
            xfer_us=None,
        )
        labels_used = c.transfer_blocks_total.label_calls[0]
        self.assertEqual(labels_used["direction"], "offload")
        self.assertEqual(labels_used["src"], "sglang_hicache::L1")
        self.assertEqual(labels_used["dst"], "sglang_hicache::L2")
        # Extra labels from constructor are also present
        self.assertEqual(labels_used["tp_rank"], "0")

    def test_multiple_transfers_accumulate(self):
        c = self._make_collector()
        for _ in range(3):
            c.record_transfer(
                direction="onboard",
                src="sglang_hicache::L2",
                dst="sglang_hicache::L1",
                blocks=2,
                bytes_=4096,
                xfer_us=100,
            )
        self.assertEqual(c.transfer_blocks_total.inc_calls, [2, 2, 2])
        self.assertEqual(c.transfer_duration_us.observe_calls, [100, 100, 100])


# ══════════════════════════════════════════════════════════════════════════════
# HiCacheController._transfer_index_count
# ══════════════════════════════════════════════════════════════════════════════


class TestHiCacheControllerTransferIndexCount(unittest.TestCase):
    def setUp(self):
        self.ctrl = _ControllerStub()

    def test_host_indices_preferred(self):
        xfer = PoolTransfer(name=PoolName.KV, host_indices=torch.arange(32))
        self.assertEqual(self.ctrl._transfer_index_count(xfer), 32)

    def test_device_indices_fallback_when_host_none(self):
        xfer = PoolTransfer(
            name=PoolName.MAMBA,
            host_indices=None,
            device_indices=torch.arange(16),
        )
        self.assertEqual(self.ctrl._transfer_index_count(xfer), 16)

    def test_both_none_returns_zero(self):
        xfer = PoolTransfer(name=PoolName.SWA, host_indices=None, device_indices=None)
        self.assertEqual(self.ctrl._transfer_index_count(xfer), 0)


# ══════════════════════════════════════════════════════════════════════════════
# HiCacheController._host_pool_size_per_token
# ══════════════════════════════════════════════════════════════════════════════


class TestHiCacheControllerHostPoolSizePerToken(unittest.TestCase):
    def setUp(self):
        self.ctrl = _ControllerStub()

    def test_plain_pool_returns_size_per_token(self):
        self.ctrl.mem_pool_host = _PlainHostPool(4096)
        result = self.ctrl._host_pool_size_per_token(PoolName.KV)
        self.assertEqual(result, 4096)

    def test_plain_pool_called_without_pool_name(self):
        self.ctrl.mem_pool_host = _PlainHostPool(1024)
        result = self.ctrl._host_pool_size_per_token()
        self.assertEqual(result, 1024)

    def test_entry_map_pool_found(self):
        entry = _hybrid_entry(size_per_token=512)
        self.ctrl.mem_pool_host = _HybridPool({PoolName.MAMBA: entry})
        result = self.ctrl._host_pool_size_per_token(PoolName.MAMBA)
        self.assertEqual(result, 512)

    def test_entry_map_pool_not_found_returns_zero(self):
        entry = _hybrid_entry(size_per_token=512)
        self.ctrl.mem_pool_host = _HybridPool({PoolName.MAMBA: entry})
        result = self.ctrl._host_pool_size_per_token(PoolName.SWA)
        self.assertEqual(result, 0)

    def test_unknown_pool_warns_and_returns_zero(self):
        self.ctrl.mem_pool_host = _UnknownPool()
        with self.assertLogs("sglang.srt.managers.cache_controller", level="WARNING"):
            result = self.ctrl._host_pool_size_per_token(PoolName.KV)
        self.assertEqual(result, 0)

    def test_unknown_pool_warns_only_once(self):
        self.ctrl.mem_pool_host = _UnknownPool()
        with self.assertLogs("sglang.srt.managers.cache_controller", level="WARNING") as cm:
            self.ctrl._host_pool_size_per_token(PoolName.KV)
            self.ctrl._host_pool_size_per_token(PoolName.KV)
        # The flag should have been set after the first call, suppressing the second.
        self.assertTrue(self.ctrl._warned_unknown_host_pool_bytes)
        self.assertEqual(len(cm.output), 1)


# ══════════════════════════════════════════════════════════════════════════════
# HiCacheController._estimate_l1_l2_transfer_bytes
# ══════════════════════════════════════════════════════════════════════════════


class TestHiCacheControllerEstimateBytes(unittest.TestCase):
    def setUp(self):
        self.ctrl = _ControllerStub()
        self.ctrl.mem_pool_host = _PlainHostPool(2048)  # 2 KiB per KV token

    def test_plain_kv_only(self):
        # 32 tokens × 2048 bytes = 65536
        result = self.ctrl._estimate_l1_l2_transfer_bytes(token_count=32)
        self.assertEqual(result, 32 * 2048)

    def test_zero_tokens_returns_zero(self):
        result = self.ctrl._estimate_l1_l2_transfer_bytes(token_count=0)
        self.assertEqual(result, 0)

    def test_with_sidecar_pool_transfer(self):
        mamba_size = 256
        entry = _hybrid_entry(size_per_token=mamba_size)
        self.ctrl.mem_pool_host = _HybridPool(
            {PoolName.KV: _hybrid_entry(2048), PoolName.MAMBA: entry}
        )
        xfer = PoolTransfer(name=PoolName.MAMBA, host_indices=torch.arange(8))
        # KV: 16 tokens × 2048 + Mamba sidecar: 8 indices × 256
        expected = 16 * 2048 + 8 * mamba_size
        result = self.ctrl._estimate_l1_l2_transfer_bytes(
            token_count=16, pool_transfers=[xfer]
        )
        self.assertEqual(result, expected)

    def test_with_draft_pool(self):
        self.ctrl.has_draft = True
        self.ctrl.mem_pool_host_draft = _PlainHostPool(512)
        # KV: 10 tokens × 2048 + draft: 10 tokens × 512
        expected = 10 * 2048 + 10 * 512
        result = self.ctrl._estimate_l1_l2_transfer_bytes(token_count=10)
        self.assertEqual(result, expected)

    def test_draft_pool_missing_size_per_token_warns_and_omits(self):
        self.ctrl.has_draft = True
        self.ctrl.mem_pool_host_draft = _UnknownPool()
        with self.assertLogs("sglang.srt.managers.cache_controller", level="WARNING"):
            result = self.ctrl._estimate_l1_l2_transfer_bytes(token_count=4)
        # Draft bytes omitted; only KV contribution.
        self.assertEqual(result, 4 * 2048)


# ══════════════════════════════════════════════════════════════════════════════
# HiCacheController._make_hicache_ack — block_count ceiling division
# ══════════════════════════════════════════════════════════════════════════════


class TestHiCacheControllerMakeHicacheAck(unittest.TestCase):
    def _make_ack_via_ctrl(self, token_count, page_size=16):
        ctrl = _ControllerStub()
        ctrl.page_size = page_size
        ctrl.mem_pool_host = _PlainHostPool(1024)
        return ctrl._make_hicache_ack(
            start_event=MagicMock(),
            finish_event=MagicMock(),
            node_ids=[1],
            token_count=token_count,
        )

    def test_block_count_exact_page(self):
        ack = self._make_ack_via_ctrl(token_count=32, page_size=16)
        self.assertEqual(ack.block_count, 2)

    def test_block_count_partial_page(self):
        ack = self._make_ack_via_ctrl(token_count=33, page_size=16)
        self.assertEqual(ack.block_count, 3)

    def test_block_count_single_token(self):
        ack = self._make_ack_via_ctrl(token_count=1, page_size=16)
        self.assertEqual(ack.block_count, 1)

    def test_block_count_full_single_page(self):
        ack = self._make_ack_via_ctrl(token_count=16, page_size=16)
        self.assertEqual(ack.block_count, 1)

    def test_byte_count_matches_estimate(self):
        ctrl = _ControllerStub()
        ctrl.page_size = 16
        ctrl.mem_pool_host = _PlainHostPool(2048)
        ack = ctrl._make_hicache_ack(
            start_event=MagicMock(),
            finish_event=MagicMock(),
            node_ids=[1],
            token_count=10,
        )
        # _estimate_l1_l2_transfer_bytes(10) = 10 * 2048 = 20480
        self.assertEqual(ack.byte_count, 10 * 2048)

    def test_token_count_stored_correctly(self):
        ack = self._make_ack_via_ctrl(token_count=48, page_size=16)
        self.assertEqual(ack.token_count, 48)


# ══════════════════════════════════════════════════════════════════════════════
# HiCacheController._transfer_elapsed_us
# ══════════════════════════════════════════════════════════════════════════════


class TestHiCacheControllerTransferElapsedUs(unittest.TestCase):
    def setUp(self):
        self.ctrl = _ControllerStub()

    def _ack_with_elapsed(self, elapsed_ms: float) -> HiCacheAck:
        start = MagicMock()
        finish = MagicMock()
        start.elapsed_time.return_value = elapsed_ms
        return _make_ack(start_event=start, finish_event=finish)

    def test_converts_ms_to_us(self):
        ack = self._ack_with_elapsed(1.5)
        result = self.ctrl._transfer_elapsed_us(ack)
        self.assertEqual(result, 1500)

    def test_exact_milliseconds(self):
        ack = self._ack_with_elapsed(10.0)
        result = self.ctrl._transfer_elapsed_us(ack)
        self.assertEqual(result, 10000)

    def test_clamps_negative_to_zero(self):
        ack = self._ack_with_elapsed(-5.0)
        result = self.ctrl._transfer_elapsed_us(ack)
        self.assertEqual(result, 0)

    def test_exception_returns_none(self):
        # Non-CUDA backends raise on elapsed_time() (timing not supported).
        start = MagicMock()
        start.elapsed_time.side_effect = RuntimeError("CUDA not available")
        ack = _make_ack(start_event=start)
        result = self.ctrl._transfer_elapsed_us(ack)
        self.assertIsNone(result)


# ══════════════════════════════════════════════════════════════════════════════
# HiCacheController.record_l1_l2_transfer_complete
# ══════════════════════════════════════════════════════════════════════════════


class TestRecordL1L2TransferComplete(unittest.TestCase):
    """Test totals accumulation, early-exit, direction labels, and collector call."""

    def _make_ctrl(self, *, with_collector: bool = True) -> _ControllerStub:
        ctrl = _ControllerStub()
        if with_collector:
            ctrl.hicache_l1_l2_transfer_metrics_collector = MagicMock()
        # Return a fixed xfer_us so totals tests are deterministic.
        ctrl._transfer_elapsed_us = MagicMock(return_value=2000)
        return ctrl

    # ── early exit ────────────────────────────────────────────────────────────

    def test_early_exit_leaves_totals_unchanged(self):
        ctrl = _ControllerStub()
        ctrl.hicache_l1_l2_transfer_metrics_collector = None
        ack = _make_ack(block_count=4, byte_count=65536)

        with patch("sglang.srt.managers.cache_controller.logger") as mock_log:
            mock_log.isEnabledFor.return_value = False
            ctrl.record_l1_l2_transfer_complete(direction="offload", ack=ack)

        totals = ctrl.hicache_l1_l2_transfer_totals["offload"]
        self.assertEqual(totals["events"], 0)
        self.assertEqual(totals["blocks"], 0)
        self.assertEqual(totals["bytes"], 0)

    # ── totals accumulation ───────────────────────────────────────────────────

    def test_offload_totals_accumulate_over_two_calls(self):
        ctrl = self._make_ctrl()
        ack = _make_ack(block_count=4, byte_count=65536)
        ctrl.record_l1_l2_transfer_complete(direction="offload", ack=ack)
        ctrl.record_l1_l2_transfer_complete(direction="offload", ack=ack)

        totals = ctrl.hicache_l1_l2_transfer_totals["offload"]
        self.assertEqual(totals["events"], 2)
        self.assertEqual(totals["blocks"], 8)
        self.assertEqual(totals["bytes"], 131072)

    def test_onboard_totals_accumulate_independently(self):
        ctrl = self._make_ctrl()
        ack = _make_ack(block_count=2, byte_count=32768)
        ctrl.record_l1_l2_transfer_complete(direction="onboard", ack=ack)

        on = ctrl.hicache_l1_l2_transfer_totals["onboard"]
        off = ctrl.hicache_l1_l2_transfer_totals["offload"]
        self.assertEqual(on["events"], 1)
        self.assertEqual(off["events"], 0)

    def test_xfer_us_accumulates_in_totals(self):
        ctrl = self._make_ctrl()
        ctrl._transfer_elapsed_us = MagicMock(return_value=1000)
        ack = _make_ack()
        ctrl.record_l1_l2_transfer_complete(direction="offload", ack=ack)
        ctrl.record_l1_l2_transfer_complete(direction="offload", ack=ack)

        self.assertEqual(
            ctrl.hicache_l1_l2_transfer_totals["offload"]["xfer_us"], 2000
        )

    def test_xfer_us_none_leaves_total_zero(self):
        ctrl = self._make_ctrl()
        ctrl._transfer_elapsed_us = MagicMock(return_value=None)
        ack = _make_ack()
        ctrl.record_l1_l2_transfer_complete(direction="offload", ack=ack)

        self.assertEqual(
            ctrl.hicache_l1_l2_transfer_totals["offload"]["xfer_us"], 0
        )

    # ── invalid direction ─────────────────────────────────────────────────────

    def test_invalid_direction_raises_value_error(self):
        ctrl = self._make_ctrl()
        ack = _make_ack()
        with self.assertRaises(ValueError):
            ctrl.record_l1_l2_transfer_complete(direction="sideways", ack=ack)

    # ── collector call-through ────────────────────────────────────────────────

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

    def test_offload_src_dst_labels(self):
        ctrl = self._make_ctrl()
        ack = _make_ack()
        ctrl.record_l1_l2_transfer_complete(direction="offload", ack=ack)
        call_kwargs = (
            ctrl.hicache_l1_l2_transfer_metrics_collector.record_transfer.call_args.kwargs
        )
        self.assertEqual(call_kwargs["src"], "sglang_hicache::L1")
        self.assertEqual(call_kwargs["dst"], "sglang_hicache::L2")

    def test_onboard_src_dst_labels(self):
        ctrl = self._make_ctrl()
        ack = _make_ack()
        ctrl.record_l1_l2_transfer_complete(direction="onboard", ack=ack)
        call_kwargs = (
            ctrl.hicache_l1_l2_transfer_metrics_collector.record_transfer.call_args.kwargs
        )
        self.assertEqual(call_kwargs["src"], "sglang_hicache::L2")
        self.assertEqual(call_kwargs["dst"], "sglang_hicache::L1")

    def test_no_collector_does_not_call_record_transfer(self):
        ctrl = _ControllerStub()
        ctrl.hicache_l1_l2_transfer_metrics_collector = None
        ctrl._transfer_elapsed_us = MagicMock(return_value=1000)
        ack = _make_ack()

        # Set DEBUG logging so we don't early-exit (should_log=True keeps it running).
        cache_ctrl_logger = logging.getLogger("sglang.srt.managers.cache_controller")
        old_level = cache_ctrl_logger.level
        try:
            cache_ctrl_logger.setLevel(logging.DEBUG)
            ctrl.record_l1_l2_transfer_complete(direction="offload", ack=ack)
        finally:
            cache_ctrl_logger.setLevel(old_level)

        # Totals should still be updated; no collector to assert on.
        self.assertEqual(ctrl.hicache_l1_l2_transfer_totals["offload"]["events"], 1)


# ══════════════════════════════════════════════════════════════════════════════
# Prometheus text exposition — real prometheus_client, isolated registry
# ══════════════════════════════════════════════════════════════════════════════


class TestHiCacheL1L2TransferCollectorPrometheusOutput(unittest.TestCase):
    """Verify that record_transfer produces correct Prometheus text exposition.

    Uses an isolated CollectorRegistry per test so metrics don't leak into
    the global registry and tests remain independent of each other.
    """

    def setUp(self):
        import prometheus_client
        from prometheus_client import CollectorRegistry

        self.registry = CollectorRegistry()
        _reg = self.registry
        _PC = prometheus_client.Counter
        _PH = prometheus_client.Histogram

        # Wrapper classes that forward construction to the real prometheus_client
        # types but bind them to our isolated registry.  Each setUp() call
        # produces a new class object, so the _metric_cache key is unique and
        # a fresh Counter/Histogram pair is created every test.
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

    # ── metric names ──────────────────────────────────────────────────────────

    def test_blocks_counter_name_in_exposition(self):
        c = self._make_collector()
        c.record_transfer(
            direction="offload",
            src="sglang_hicache::L1",
            dst="sglang_hicache::L2",
            blocks=5,
            bytes_=81920,
            xfer_us=None,
        )
        self.assertIn("sglang:hicache_l1_l2_transfer_blocks_total", self._exposition())

    def test_bytes_counter_name_in_exposition(self):
        c = self._make_collector()
        c.record_transfer(
            direction="offload",
            src="sglang_hicache::L1",
            dst="sglang_hicache::L2",
            blocks=1,
            bytes_=16384,
            xfer_us=None,
        )
        self.assertIn("sglang:hicache_l1_l2_transfer_bytes_total", self._exposition())

    def test_duration_histogram_name_in_exposition(self):
        c = self._make_collector()
        c.record_transfer(
            direction="onboard",
            src="sglang_hicache::L2",
            dst="sglang_hicache::L1",
            blocks=2,
            bytes_=32768,
            xfer_us=3000,
        )
        self.assertIn("sglang:hicache_l1_l2_transfer_duration_us", self._exposition())

    # ── label values ──────────────────────────────────────────────────────────

    def test_direction_label_in_exposition(self):
        c = self._make_collector()
        c.record_transfer(
            direction="onboard",
            src="sglang_hicache::L2",
            dst="sglang_hicache::L1",
            blocks=3,
            bytes_=49152,
            xfer_us=None,
        )
        self.assertIn('direction="onboard"', self._exposition())

    def test_src_and_dst_labels_in_exposition(self):
        c = self._make_collector()
        c.record_transfer(
            direction="offload",
            src="sglang_hicache::L1",
            dst="sglang_hicache::L2",
            blocks=1,
            bytes_=1024,
            xfer_us=None,
        )
        output = self._exposition()
        self.assertIn('src="sglang_hicache::L1"', output)
        self.assertIn('dst="sglang_hicache::L2"', output)

    def test_constructor_labels_in_exposition(self):
        c = self._make_collector()
        c.record_transfer(
            direction="offload",
            src="sglang_hicache::L1",
            dst="sglang_hicache::L2",
            blocks=1,
            bytes_=1024,
            xfer_us=None,
        )
        output = self._exposition()
        self.assertIn('tp_rank="0"', output)
        self.assertIn('io_backend="nixl"', output)

    # ── counter values ────────────────────────────────────────────────────────

    def _find_metric_line(self, exposition: str, name_fragment: str, **label_filters) -> str:
        """Return the first non-comment exposition line matching name_fragment
        and all label_filters, or raise AssertionError."""
        for line in exposition.splitlines():
            if line.startswith("#"):
                continue
            if name_fragment not in line:
                continue
            if all(f'{k}="{v}"' in line for k, v in label_filters.items()):
                return line
        self.fail(
            f"No metric line found for fragment={name_fragment!r} "
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
        self.assertTrue(line.endswith(" 32768.0"), f"Expected value 32768.0 in: {line!r}")

    # ── histogram ─────────────────────────────────────────────────────────────

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

    def test_duration_histogram_not_observed_when_xfer_us_none(self):
        c = self._make_collector()
        c.record_transfer(
            direction="offload",
            src="sglang_hicache::L1",
            dst="sglang_hicache::L2",
            blocks=1,
            bytes_=1024,
            xfer_us=None,
        )
        # Count must be 0 — no observation recorded.
        line = self._find_metric_line(
            self._exposition(),
            "hicache_l1_l2_transfer_duration_us_count",
            direction="offload",
        )
        self.assertTrue(line.endswith(" 0.0"), f"Expected count 0.0 in: {line!r}")


if __name__ == "__main__":
    unittest.main()
