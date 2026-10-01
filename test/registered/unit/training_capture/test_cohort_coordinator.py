"""Bound ownership must survive collector setup and writer handoff failures."""

import tempfile
import threading
import time
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch.distributed as dist
from sglang.srt.training_capture.catalog import HTTPCaptureCatalog
from sglang.srt.training_capture.cohort import CaptureCohortAllocator
from sglang.srt.training_capture.cohort_coordinator import CohortCaptureCoordinator
from sglang.srt.training_capture.config import CaptureConfig, StoreSetup
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.training_capture.protocol import ContractError
from sglang.srt.training_capture.resources import CaptureResources
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_catalog import TestCaptureCatalog
from sglang.test.training_capture_utils import (
    BufferStore,
    CaptureTestRequest,
    FakeReplicateConfig,
    make_snapshot,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo required")
class TestCohortCaptureCoordinator(CustomTestCase):
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
        layout = plan_capture_layout(base.kv, tp_size=1, pp_layer_ranges=[(0, 4)])
        config = CaptureConfig(
            dataset_id="coordinator-test",
            model_id="fixture",
            producer_revision="test",
            selected_layer_ids=base.kv.selected_layer_ids,
            catalog_endpoint=self.catalog.endpoint,
            journal_directory=str(root / "journal"),
            store=StoreSetup(local_hostname="rank-0", master_server_addr="localhost:1"),
            max_sample_tokens=8,
            max_inflight_samples=1,
            max_host_bytes=2 << 20,
            sample_ratio=1.0,
        )
        self.resources = CaptureResources()
        self.resources.store = MooncakeSnapshotStore(
            BufferStore(), FakeReplicateConfig()
        )
        self.resources.catalog = HTTPCaptureCatalog(self.catalog.endpoint)
        self.resources._allocate(
            config, base.kv, layout.partitions[0], pin_memory=False
        )
        self.coordinator = CohortCaptureCoordinator(
            allocator=CaptureCohortAllocator(
                group=self.group,
                layout=layout,
                config=config,
                teacher=base.teacher,
                kv=base.kv,
                resources=self.resources,
                timeout_seconds=5,
            ),
            req_to_token=None,
        )
        self.gates = []
        self.wait_until(
            lambda: self.coordinator.stats()["states"].get("available", 0) == 1
        )

    def tearDown(self):
        for gate in self.gates:
            gate.set()
        self.assertTrue(self.coordinator.close(), self.coordinator.stats())
        dist.destroy_process_group(self.group)
        dist.destroy_process_group()
        self.catalog.close()
        self.directory.cleanup()

    def wait_until(self, predicate):
        deadline = time.monotonic() + 10
        while not predicate():
            self.assertIsNone(self.coordinator.error)
            self.assertIsNone(self.coordinator.service.error)
            self.assertIsNone(self.coordinator.writer_actor.error)
            if time.monotonic() >= deadline:
                self.fail(str(self.coordinator.stats()))
            time.sleep(0.01)

    def bound_route(self):
        service = self.coordinator.service
        ticket = service.claim("ab" * 32)
        self.assertIsNotNone(ticket)
        handle = service.bind(ticket, ticket.request_sha256)
        self.assertIsNotNone(handle)
        return SimpleNamespace(handle=handle, execution_sha256="cd" * 32)

    def test_post_bind_status_failure_retires_the_unused_slot(self):
        coordinator = self.coordinator
        route = self.bound_route()
        req = CaptureTestRequest("failed-setup")
        with (
            patch.object(coordinator.request_router, "bind", return_value=route),
            patch.object(
                coordinator.service, "status", side_effect=RuntimeError("status")
            ),
        ):
            coordinator.before_forward([req])
        self.assertIsNone(req.training_capture_context)
        self.assertIsNone(req.training_capture_finalize)
        self.assertTrue(route.handle.drained)
        self.wait_until(
            lambda: (
                route.handle.cohort.lease.capture_id not in coordinator.service.records
            )
        )
        self.assertEqual(
            self.catalog.captures[route.handle.cohort.lease.capture_id]["state"],
            "FAILED",
        )
        self.assertEqual(coordinator.records, {})
        self.assertFalse(self.catalog.publications)

    def test_rejected_handoff_waits_for_copy_before_returning_ownership(self):
        coordinator = self.coordinator
        route = self.bound_route()
        req = CaptureTestRequest("copy-in-flight")
        with patch.object(coordinator.request_router, "bind", return_value=route):
            coordinator.before_forward([req])
        context = req.training_capture_context.context
        entered, release = threading.Event(), threading.Event()
        self.gates.append(release)

        def wait_for_copy():
            entered.set()
            if not release.wait(10):
                raise TimeoutError("copy gate was not released")

        context.last_event = SimpleNamespace(synchronize=wait_for_copy)
        with patch.object(
            coordinator.writer_actor, "submit", side_effect=ContractError("stopping")
        ):
            coordinator.disable("test-shutdown")
            self.assertTrue(entered.wait(5))
            self.assertIsNone(req.training_capture_context)
            self.assertFalse(route.handle.drained)
            self.assertEqual(self.resources.pool.stats()["filling"], 1)
            self.assertFalse(self.resources.closed)
            self.assertEqual(coordinator.writer_actor.stats()["pending"], 0)
            release.set()
            self.wait_until(lambda: not coordinator.records)
        self.wait_until(
            lambda: (
                route.handle.cohort.lease.capture_id not in coordinator.service.records
            )
        )
        self.assertTrue(route.handle.drained)
        self.assertTrue(route.handle.transfer_complete)
        self.assertEqual(self.resources.pool.stats()["quarantined"], 0)
        self.assertFalse(self.catalog.publications)


if __name__ == "__main__":
    unittest.main()
