"""Bounded refill, renewal fairness and wakeup races without CUDA."""

import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import msgspec
import torch
from sglang.srt.training_capture.catalog import HTTPCaptureCatalog
from sglang.srt.training_capture.config import CaptureConfig, StoreSetup
from sglang.srt.training_capture.coordinator import CaptureCoordinator
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_catalog import TestCaptureCatalog
from sglang.test.training_capture_utils import (
    BufferStore,
    FakeReplicateConfig,
    make_snapshot,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestCaptureReservationRefill(CustomTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.catalog = TestCaptureCatalog()
        self.addCleanup(self.catalog.close)
        manifest, _ = make_snapshot()
        config = CaptureConfig(
            dataset_id=manifest.dataset_id,
            model_id="test",
            producer_revision="test",
            selected_layer_ids=manifest.kv.selected_layer_ids,
            catalog_endpoint=self.catalog.endpoint,
            journal_directory=directory.name + "/journal",
            store=StoreSetup(
                local_hostname="localhost", master_server_addr="localhost:1"
            ),
            max_sample_tokens=8,
            max_inflight_samples=4,
            max_host_bytes=8 << 20,
            sample_ratio=1.0,
        )
        buffers = {
            f"target_{component}.{layer.layer_id}": torch.zeros(
                16, 2, 4, dtype=torch.bfloat16
            )
            for layer in manifest.kv.layers
            for component in ("k", "v")
        }
        self.coordinator = CaptureCoordinator(
            config=config,
            teacher=manifest.teacher,
            kv=manifest.kv,
            exporter=SelectedLayerKVExporter(manifest.kv, buffers),
            req_to_token=None,
            store=MooncakeSnapshotStore(BufferStore(), FakeReplicateConfig()),
            catalog=HTTPCaptureCatalog(self.catalog.endpoint),
            pin_memory=False,
            autostart=False,
        )
        self.addCleanup(self.coordinator.close)
        self.waits = queue.Queue()
        wait = self.coordinator.lease_wake.wait

        def observe_wait(timeout):
            with self.coordinator.lock:
                self.waits.put(tuple(self.coordinator.records))
            # Hold the maintenance poll: progress must come from an actual wake.
            return wait(10)

        self.coordinator.lease_wake.wait = observe_wait

    def next_wait(self):
        return self.waits.get(timeout=5)

    def test_fills_bounded_capacity_before_first_idle_wait(self):
        self.coordinator.activate()
        self.assertEqual(len(self.next_wait()), 4)
        self.assertEqual(len(self.catalog.captures), 4)
        self.assertEqual(self.coordinator.pool.stats()["free"], 0)
        self.assertIsNone(self.coordinator.pool.acquire())

    def test_writer_release_wakes_refill_without_poll_timeout(self):
        self.coordinator.activate()
        before = self.next_wait()
        record = self.coordinator.records[before[0]]
        record.invalid_reason = "test_retirement"
        self.coordinator._queue_record(record)
        after = self.next_wait()
        self.assertEqual(len(after), 4)
        self.assertNotIn(before[0], after)
        self.assertEqual(record.state, "done")
        self.assertEqual(self.catalog.captures[before[0]]["state"], "FAILED")
        self.assertEqual(len(self.catalog.captures), 5)

    def test_release_between_failed_acquire_and_wait_is_not_lost(self):
        exhausted, resume = threading.Event(), threading.Event()
        acquire = self.coordinator.pool.acquire

        def blocked_acquire():
            slot = acquire()
            if slot is None and not exhausted.is_set():
                exhausted.set()
                resume.wait(5)
            return slot

        with patch.object(self.coordinator.pool, "acquire", blocked_acquire):
            self.coordinator.activate()
            try:
                self.assertTrue(exhausted.wait(5))
                record = self.coordinator.available.popleft()
                self.coordinator.catalog.fail(record.lease, "test_retirement")
                self.coordinator._retire(record, complete=True)
            finally:
                resume.set()
            self.assertEqual(len(self.next_wait()), 3)
            self.assertEqual(len(self.next_wait()), 4)
        self.assertEqual(len(self.catalog.captures), 5)

    def test_renewal_is_checked_between_reservation_calls(self):
        calls = []
        begin = self.coordinator.catalog.begin
        heartbeat = self.coordinator.catalog.heartbeat

        def reserve(identity):
            calls.append("begin")
            lease = begin(identity)
            if len(calls) == 1:
                lease = msgspec.structs.replace(lease, renew_after_seconds=1e-9)
            return lease

        def renew(lease):
            calls.append("heartbeat")
            return heartbeat(lease)

        with (
            patch.object(self.coordinator.catalog, "begin", reserve),
            patch.object(self.coordinator.catalog, "heartbeat", renew),
        ):
            self.coordinator.activate()
            self.assertEqual(len(self.next_wait()), 4)
        self.assertEqual(calls, ["begin", "heartbeat", "begin", "begin", "begin"])

    def test_wake_does_not_bypass_catalog_failure_backoff(self):
        begin = self.coordinator.catalog.begin
        failures = []

        def fail_once(identity):
            if not failures:
                failures.append(time.monotonic())
                raise RuntimeError("Catalog temporarily unavailable")
            return begin(identity)

        with patch.object(
            self.coordinator.catalog, "begin", side_effect=fail_once
        ) as rpc:
            self.coordinator.activate()
            self.assertEqual(len(self.next_wait()), 0)
            self.assertGreaterEqual(
                self.coordinator.reservation_retry_at, failures[0] + 0.1
            )
            self.assertEqual(self.coordinator.pool.stats()["free"], 4)
            self.coordinator.reservation_retry_at = time.monotonic() + 60
            for _ in range(2):
                self.coordinator.lease_wake.set()
                self.assertEqual(len(self.next_wait()), 0)
                self.assertEqual(rpc.call_count, 1)
            self.coordinator.reservation_retry_at = 0
            self.coordinator.lease_wake.set()
            self.assertEqual(len(self.next_wait()), 4)
            self.assertEqual(rpc.call_count, 5)

    def test_pause_is_rechecked_between_successful_reservations(self):
        begin = self.coordinator.catalog.begin

        def pause_after_begin(identity):
            lease = begin(identity)
            self.coordinator.admission.pause_until = time.monotonic() + 60
            return lease

        with patch.object(self.coordinator.catalog, "begin", pause_after_begin):
            self.coordinator.activate()
            self.assertEqual(len(self.next_wait()), 1)
        self.coordinator.admission.pause_until = 0
        self.coordinator.lease_wake.set()
        self.assertEqual(len(self.next_wait()), 4)

    def test_disable_during_begin_prevents_further_reservations(self):
        begin = self.coordinator.catalog.begin

        def disable_after_begin(identity):
            lease = begin(identity)
            self.coordinator.disable("test_disable")
            return lease

        with patch.object(self.coordinator.catalog, "begin", disable_after_begin):
            self.coordinator.activate()
            self.assertEqual(len(self.next_wait()), 1)
            self.coordinator.lease_wake.set()
            self.assertEqual(len(self.next_wait()), 1)
        self.assertEqual(len(self.catalog.captures), 1)

    def test_quarantined_slot_is_not_refilled(self):
        self.coordinator.activate()
        before = self.next_wait()
        record = self.coordinator.available.popleft()
        self.coordinator.catalog.fail(record.lease, "test_uncertain_transfer")
        self.coordinator._retire(record, complete=False)
        self.assertEqual(len(self.next_wait()), 3)
        self.coordinator.lease_wake.set()
        self.assertEqual(len(self.next_wait()), 3)
        self.assertEqual(self.coordinator.pool.stats()["quarantined"], 1)
        self.assertEqual(len(self.catalog.captures), len(before))

    def test_shutdown_wakes_idle_lease_thread_and_drains_spares(self):
        self.coordinator.activate()
        self.assertEqual(len(self.next_wait()), 4)
        closer = threading.Thread(target=self.coordinator.close)
        closer.start()
        try:
            closer.join(5)
            self.assertFalse(closer.is_alive())
        finally:
            self.coordinator.lease_wake.set()
            closer.join(15)
        self.assertTrue(self.coordinator.closed)
        self.assertFalse(self.coordinator.lease_thread.is_alive())
        self.assertFalse(self.coordinator.writer_thread.is_alive())
        self.assertFalse(self.coordinator.records)
        self.assertTrue(
            all(
                record["state"] == "FAILED" for record in self.catalog.captures.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
