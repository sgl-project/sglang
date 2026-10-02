"""Writer ownership, uncertain publication and shutdown with real cohort control."""

import tempfile
import threading
import time
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import torch.distributed as dist
from sglang.srt.training_capture.catalog import CatalogUnavailable, HTTPCaptureCatalog
from sglang.srt.training_capture.cohort import CaptureCohortAllocator
from sglang.srt.training_capture.cohort_service import CaptureCohortService
from sglang.srt.training_capture.cohort_writer import CohortSnapshotWriter
from sglang.srt.training_capture.config import CaptureConfig, StoreSetup
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.training_capture.protocol import ContractError
from sglang.srt.training_capture.resources import CaptureResources
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_catalog import TestCaptureCatalog
from sglang.test.training_capture_partition import make_cohort_context
from sglang.test.training_capture_utils import (
    BufferStore,
    FakeReplicateConfig,
    make_snapshot,
    read_snapshot,
)

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


class CopyGate:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def synchronize(self):
        self.entered.set()
        if not self.release.wait(10):
            raise TimeoutError("test did not release the copy event")


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo required")
class TestCohortSnapshotWriter(CustomTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.catalog = TestCaptureCatalog()
        root = Path(self.directory.name)
        dist.init_process_group(
            "gloo",
            init_method=(root / "rendezvous").as_uri(),
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=10),
        )
        self.group = dist.new_group(backend="gloo", timeout=timedelta(seconds=10))
        base, _ = make_snapshot(response_length=4)
        self.layout = plan_capture_layout(base.kv, tp_size=1, pp_layer_ranges=[(0, 4)])
        self.config = CaptureConfig(
            dataset_id="writer-test",
            model_id="fixture",
            producer_revision="test",
            selected_layer_ids=base.kv.selected_layer_ids,
            catalog_endpoint=self.catalog.endpoint,
            journal_directory=str(root / "journal"),
            store=StoreSetup(local_hostname="rank-0", master_server_addr="localhost:1"),
            max_sample_tokens=8,
            max_inflight_samples=2,
            max_host_bytes=4 << 20,
        )
        self.client = BufferStore()
        self.resources = CaptureResources()
        self.resources.store = MooncakeSnapshotStore(
            self.client, FakeReplicateConfig(), max_receive_bytes=4 << 20
        )
        self.resources.catalog = HTTPCaptureCatalog(self.catalog.endpoint)
        self.resources._allocate(
            self.config, base.kv, self.layout.partitions[0], pin_memory=False
        )
        allocator = CaptureCohortAllocator(
            group=self.group,
            layout=self.layout,
            config=self.config,
            teacher=base.teacher,
            kv=base.kv,
            resources=self.resources,
            timeout_seconds=5,
        )
        self.service = CaptureCohortService(allocator, poll_seconds=0.01)
        self.worker = CohortSnapshotWriter(
            self.service, poll_seconds=0.01, retry_seconds=0.05
        )
        self.gates = []
        self.service.start()
        self.worker.start()
        self.wait_until(lambda: self.worker.stats()["ready"])

    def tearDown(self):
        for gate in self.gates:
            gate.release.set()
        self.assertTrue(self.worker.close(timeout=10), self.worker.stats())
        self.service.close(timeout=10)
        self.assertFalse(self.service.thread.is_alive())
        # These tests use CPU tensors and a synchronous Store double. Explicit
        # transport close ends the simulated uncertainty before fixture disposal.
        self.resources.close()
        self.service._retained.discard(self.service)
        self.worker._retained.discard(self.worker)
        dist.destroy_process_group(self.group)
        dist.destroy_process_group()
        self.catalog.close()
        self.directory.cleanup()

    def wait_until(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate():
            self.assertIsNone(self.worker.error, self.worker.stats())
            self.assertIsNone(self.service.error)
            if time.monotonic() >= deadline:
                self.fail(f"writer condition timed out: {self.worker.stats()}")
            time.sleep(0.01)

    def claim(self):
        tickets = []

        def ready():
            ticket = self.service.claim("ab" * 32)
            if ticket is not None:
                tickets.append(ticket)
            return bool(tickets)

        self.wait_until(ready)
        handle = self.service.bind(tickets[0], "ab" * 32)
        self.assertIsNotNone(handle)
        return handle

    def submit(self, handle, *, gate=None):
        context, metadata = make_cohort_context(handle.cohort, self.layout, 0)
        if gate is not None:
            context.last_event = gate
        self.worker.submit(
            handle, context=context, metadata=metadata, execution_sha256="cd" * 32
        )
        return context, metadata

    def test_copy_owner_survives_duplicate_submission_and_close(self):
        handle = self.claim()
        gate = CopyGate()
        self.gates.append(gate)
        context, metadata = self.submit(handle, gate=gate)
        self.assertTrue(gate.entered.wait(5))
        with self.assertRaisesRegex(ContractError, "duplicate"):
            self.worker.submit(
                handle, context=context, metadata=metadata, execution_sha256="cd" * 32
            )
        self.assertFalse(self.worker.close(timeout=0))
        self.assertFalse(self.service.close(timeout=0))
        self.assertFalse(handle.drained)
        self.assertEqual(handle.cohort.slot.state, "filling")
        gate.release.set()
        self.wait_until(lambda: self.worker.stats()["pending"] == 0)
        self.assertTrue(self.worker.close(timeout=5))
        self.assertTrue(self.service.close(timeout=5))
        self.assertEqual(handle.cohort.slot.state, "free")
        self.assertFalse(self.catalog.publications)

    def test_metadata_is_frozen_before_background_copy_wait(self):
        handle = self.claim()
        gate = CopyGate()
        self.gates.append(gate)
        _, metadata = self.submit(handle, gate=gate)
        self.assertTrue(gate.entered.wait(5))
        metadata.provenance.sampling_config["temperature"] = 99
        gate.release.set()
        self.wait_until(lambda: self.worker.stats()["counters"].get("published") == 1)
        stages = self.worker.stats()["stage_timings"]
        for stage in ("queue_wait", "copy_wait", "snapshot_build", "store_payload"):
            self.assertEqual(stages[stage]["calls"], 1, stage)
            self.assertEqual(stages[stage]["errors"], 0, stage)
        self.assertEqual(stages["catalog_publish"]["calls"], 1)
        manifest, tensors = read_snapshot(
            self.resources.store, self.catalog.wait_publications(1)[0]
        )
        self.assertEqual(manifest.provenance.sampling_config["temperature"], 0.8)
        self.assertEqual(tensors["loss_mask"].tolist(), [0, 0, 1, 1, 1, 1])
        self.assertFalse(list(self.resources.journal.pending()))

    def test_uncertain_copy_is_quarantined_without_store_writes(self):
        handle = self.claim()
        context, _ = make_cohort_context(handle.cohort, self.layout, 0)
        context.abort("copy_failed")
        context.transfer_uncertain = True
        self.worker.submit(handle, context=context, failure_reason="copy_failed")
        self.wait_until(lambda: self.worker.stats()["pending"] == 0)
        self.service.close(timeout=5)
        self.assertEqual(handle.cohort.slot.state, "quarantined")
        self.assertEqual(self.client.put_keys, [])
        self.assertFalse(self.catalog.publications)

    def test_lost_publish_response_recovers_even_after_cancellation(self):
        handle = self.claim()
        publish = self.resources.catalog.publish
        committed = threading.Event()
        retry_gate = CopyGate()
        self.gates.append(retry_gate)
        calls = []

        def lose_reply(payload):
            calls.append(payload)
            if len(calls) > 1:
                retry_gate.synchronize()
                return publish(payload)
            publish(payload)
            committed.set()
            raise CatalogUnavailable("committed publication response lost")

        with patch.object(self.resources.catalog, "publish", lose_reply):
            self.submit(handle)
            self.assertTrue(committed.wait(5))
            self.service.cancel(handle.ticket, "request_cancelled")
            self.assertTrue(retry_gate.entered.wait(5))
            self.assertFalse(self.worker.close(timeout=0))
            self.assertFalse(handle.drained)
            retry_gate.release.set()
            self.wait_until(lambda: self.worker.stats()["pending"] == 0)
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(
            self.catalog.captures[handle.cohort.lease.capture_id]["state"], "AVAILABLE"
        )
        self.assertTrue(handle.published)
        self.assertFalse(list(self.resources.journal.pending()))
        self.assertFalse(self.catalog.errors)

    def test_journal_unlink_error_does_not_turn_publication_into_failure(self):
        handle = self.claim()
        complete = self.resources.journal.complete
        calls = []

        def lose_cleanup(capture_id):
            complete(capture_id)
            calls.append(capture_id)
            if len(calls) == 1:
                raise OSError("directory sync failed after unlink")

        with patch.object(self.resources.journal, "complete", lose_cleanup):
            self.submit(handle)
            self.wait_until(
                lambda: self.worker.stats()["counters"].get("published") == 1
            )
        self.assertEqual(calls, [handle.cohort.lease.capture_id] * 2)
        self.assertEqual(len(self.catalog.publications), 1)
        self.assertFalse(self.catalog.errors)

    def test_manifest_write_uncertainty_survives_successful_recovery(self):
        handle = self.claim()
        put = self.client.put_from
        ambiguous = []

        def lose_completion(key, pointer, size, config):
            result = put(key, pointer, size, config)
            if key.endswith("/manifest") and not ambiguous:
                ambiguous.append(pointer)
                return -1
            return result

        with patch.object(self.client, "put_from", lose_completion):
            self.submit(handle)
            self.wait_until(
                lambda: self.worker.stats()["counters"].get("published") == 1
            )
        self.assertEqual(len(ambiguous), 1)
        self.assertFalse(self.service.close(timeout=5))
        self.assertEqual(handle.cohort.slot.state, "quarantined")
        self.assertIn(
            handle.cohort.slot.storage.data_ptr(), self.resources.store.quarantined
        )
        self.assertEqual(len(self.catalog.publications), 1)
        self.assertFalse(self.catalog.errors)
        read_snapshot(self.resources.store, self.catalog.wait_publications(1)[0])

    def test_publication_outcome_does_not_imply_source_completion(self):
        """A confirmed publish may coexist with an earlier uncertain source transfer."""
        handle = self.claim()
        try:
            self.service.finish(handle, outcome="published", transfer_complete=False)
        finally:
            if not handle.drained:
                self.service.finish(handle, outcome="failed", transfer_complete=True)
        self.assertTrue(handle.published)
        self.assertFalse(self.service.close(timeout=5))
        self.assertEqual(handle.cohort.slot.state, "quarantined")

    def test_startup_recovery_keeps_actor_ownership_with_caller(self):
        self.assertTrue(self.worker.close(timeout=5))
        self.worker = CohortSnapshotWriter(self.service, retry_seconds=0.05)
        gate = CopyGate()
        self.gates.append(gate)
        recover = self.worker.writer.recover

        def wait_for_recovery():
            gate.synchronize()
            return recover()

        with patch.object(self.worker.writer, "recover", wait_for_recovery):
            self.worker.start()
            self.assertTrue(gate.entered.wait(5))
            self.assertFalse(self.worker.stats()["ready"])
            handle = self.claim()
            with self.assertRaisesRegex(ContractError, "not accepting"):
                self.submit(handle)
            self.assertFalse(handle.drained)
            self.assertEqual(self.worker.stats()["pending"], 0)
            gate.release.set()
            self.wait_until(lambda: self.worker.stats()["ready"])
            self.submit(handle)
            self.wait_until(
                lambda: self.worker.stats()["counters"].get("published") == 1
            )

    def test_thread_initialization_failure_is_observable_before_admission(self):
        self.assertTrue(self.worker.close(timeout=5))
        self.worker = CohortSnapshotWriter(self.service)
        with patch(
            "sglang.srt.training_capture.cohort_writer.torch.set_num_threads",
            side_effect=RuntimeError("thread initialization failed"),
        ):
            self.worker.start()
            self.worker.thread.join(timeout=5)
        self.assertFalse(self.worker.thread.is_alive())
        self.assertIsInstance(self.worker.error, RuntimeError)
        self.assertFalse(self.worker.stats()["ready"])
        self.assertTrue(self.worker.close(timeout=0))


if __name__ == "__main__":
    unittest.main()
