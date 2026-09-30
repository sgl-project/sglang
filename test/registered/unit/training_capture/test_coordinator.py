"""Bounded admission, background publication and shutdown without CUDA."""

import tempfile
import time
import unittest

import msgspec
import torch
from sglang.srt.training_capture.catalog import HTTPCaptureCatalog
from sglang.srt.training_capture.config import CaptureConfig, StoreSetup
from sglang.srt.training_capture.coordinator import CaptureCoordinator
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.training_capture.teacher import capture_teacher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_catalog import TestCaptureCatalog
from sglang.test.training_capture_utils import (
    BufferStore,
    CaptureTestRequest,
    FakeReplicateConfig,
    VerifyCaptureFixture,
    make_snapshot,
    read_snapshot,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestCaptureCoordinator(CustomTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.catalog = TestCaptureCatalog()
        manifest, _ = make_snapshot()
        self.sdk = BufferStore()
        self.store = MooncakeSnapshotStore(self.sdk, FakeReplicateConfig())
        config = CaptureConfig(
            dataset_id=manifest.dataset_id,
            model_id="test",
            producer_revision="test",
            selected_layer_ids=manifest.kv.selected_layer_ids,
            catalog_endpoint=self.catalog.endpoint,
            journal_directory=self.directory.name + "/journal",
            store=StoreSetup(
                local_hostname="localhost", master_server_addr="localhost:1"
            ),
            max_sample_tokens=8,
            max_inflight_samples=1,
            max_host_bytes=2 << 20,
            sample_ratio=1.0,
        )
        buffers = {
            f"target_{c}.{g.layer_id}": torch.zeros(16, 2, 4, dtype=torch.bfloat16)
            for g in manifest.kv.layers
            for c in ("k", "v")
        }
        self.coordinator = CaptureCoordinator(
            config=config,
            teacher=manifest.teacher,
            kv=manifest.kv,
            exporter=SelectedLayerKVExporter(manifest.kv, buffers),
            req_to_token=None,
            store=self.store,
            catalog=HTTPCaptureCatalog(self.catalog.endpoint),
            pin_memory=False,
        )

    def tearDown(self):
        self.coordinator.close()
        self.catalog.close()
        self.directory.cleanup()

    def wait_until(self, predicate):
        deadline = time.monotonic() + 5
        while not predicate():
            if time.monotonic() > deadline:
                self.fail(str(self.coordinator.stats()))
            time.sleep(0.01)

    def request(self, rid):
        return CaptureTestRequest(rid)

    def test_verify_reject_owns_raw_teacher_and_only_committed_kv(self):
        self.wait_until(lambda: len(self.coordinator.available) == 1)
        fixture = VerifyCaptureFixture(self.coordinator, self.request("reject"))
        ticket = fixture.forward()
        raw = fixture.logits.clone()
        fixture.logits.fill_(-1000)
        fixture.forward_batch.input_ids.zero_()
        fixture.forward_batch.positions.zero_()
        fixture.forward_batch.out_cache_loc.zero_()
        result = fixture.accept(ticket, [[55, 0, 0, 0]], [1])
        for source in self.coordinator.exporter.buffers.values():
            source.zero_()
        fixture.finish([10, 55], 2)
        self.coordinator.after_result(result)
        publication = self.catalog.wait_publications(1)[0]
        manifest, tensors = read_snapshot(self.store, publication)
        self.assertEqual(
            manifest.provenance.capture_mode, "speculative_accepted_target_path"
        )
        self.assertEqual(tensors["token_ids"].tolist(), [3, 4, 10, 55])
        self.assertEqual(tensors["kv_valid"].tolist(), [1, 1, 1, 0])
        self.assertEqual(tensors["logits_positions"].tolist(), [2, 3])
        values, indices = raw[0].topk(128)
        torch.testing.assert_close(
            tensors["teacher_topk_logits"][1], values, rtol=0, atol=0
        )
        torch.testing.assert_close(tensors["teacher_topk_ids"][1], indices.int())
        torch.testing.assert_close(tensors["teacher_logsumexp"][1], raw[0].logsumexp(0))
        for name, source in fixture.sources.items():
            torch.testing.assert_close(tensors[name], source[[7, 3, 6]], rtol=0, atol=0)

    def test_verify_length_truncation_keeps_final_computed_kv(self):
        self.wait_until(lambda: len(self.coordinator.available) == 1)
        fixture = VerifyCaptureFixture(self.coordinator, self.request("length"))
        ticket = fixture.forward()
        fixture.accept(ticket, [[20, 30, 55, 0]], [3])
        fixture.finish([10, 20, 30, 55], 3)
        manifest, tensors = read_snapshot(
            self.store, self.catalog.wait_publications(1)[0]
        )
        self.assertEqual(manifest.sequence.response_length, 3)
        self.assertEqual(tensors["token_ids"].tolist(), [3, 4, 10, 20, 30])
        self.assertEqual(tensors["loss_mask"].tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(tensors["kv_valid"].tolist(), [1] * 5)
        self.assertEqual(tensors["logits_positions"].tolist(), [2, 3, 4])
        torch.testing.assert_close(
            tensors["teacher_topk_logits"][1:], fixture.logits[:2].topk(128).values
        )
        for name, source in fixture.sources.items():
            torch.testing.assert_close(
                tensors[name], source[[7, 3, 6, 1, 9]], rtol=0, atol=0
            )

    def test_verify_eos_truncates_later_correct_drafts_and_bonus(self):
        from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

        self.wait_until(lambda: len(self.coordinator.available) == 1)
        fixture = VerifyCaptureFixture(self.coordinator, self.request("eos"))
        fixture.request.eos_token_ids = {20}
        ticket = fixture.forward()
        fixture.accept(ticket, [[20, 30, 55, 0]], [3])
        fixture.finish([10, 20, 30, 55], 2, FINISH_MATCHED_TOKEN(20))
        manifest, tensors = read_snapshot(
            self.store, self.catalog.wait_publications(1)[0]
        )
        self.assertEqual(manifest.sequence.stop_reason, "eos")
        self.assertEqual(tensors["token_ids"].tolist(), [3, 4, 10, 20])
        self.assertEqual(tensors["kv_valid"].tolist(), [1] * 4)
        self.assertEqual(tensors["teacher_topk_logits"].shape[0], 2)

    def test_verify_crossing_host_capacity_trims_to_real_output(self):
        self.wait_until(lambda: len(self.coordinator.available) == 1)
        fixture = VerifyCaptureFixture(
            self.coordinator, CaptureTestRequest("capacity", 6)
        )
        ticket = fixture.forward()
        result = fixture.accept(ticket, [[20, 30, 40, 50]], [4])
        fixture.request.output_ids = [10, 20, 30, 40, 50]
        self.coordinator.after_result(result)
        ticket = fixture.forward(inputs=(50, 60, 70, 80), prefix=6)
        fixture.accept(ticket, [[60, 70, 80, 90]], [4])
        fixture.finish([10, 20, 30, 40, 50, 60, 70, 80, 90], 6)
        manifest, tensors = read_snapshot(
            self.store, self.catalog.wait_publications(1)[0]
        )
        self.assertEqual(manifest.sequence.total_length, 8)
        self.assertEqual(tensors["token_ids"].tolist(), [3, 4, 10, 20, 30, 40, 50, 60])
        self.assertEqual(tensors["kv_valid"].tolist(), [1] * 8)
        self.assertEqual(tensors["logits_positions"].tolist(), list(range(2, 8)))
        torch.testing.assert_close(
            tensors["teacher_topk_logits"][-1], fixture.logits[0].topk(128).values
        )

    def test_verify_selected_batch_row_maps_back_to_original_accept_counts(self):
        self.wait_until(lambda: len(self.coordinator.available) == 1)
        fixture = VerifyCaptureFixture(
            self.coordinator, self.request("selected-second")
        )
        ticket = fixture.forward(selected_row=1)
        self.assertEqual(ticket.steps[0].batch_row, 1)
        fixture.accept(ticket, [[20, 30, 40, 50], [55, 0, 0, 0]], [4, 1])
        fixture.finish([10, 55], 2)
        _, tensors = read_snapshot(self.store, self.catalog.wait_publications(1)[0])
        torch.testing.assert_close(
            tensors["teacher_topk_logits"][1], fixture.logits[4].topk(128).values
        )
        self.assertEqual(tensors["kv_valid"].tolist(), [1, 1, 1, 0])

    def test_verify_wrong_anchor_position_or_correct_prefix_never_publishes(self):
        for failure in ("anchor", "position", "correct_prefix"):
            with self.subTest(failure=failure):
                self.wait_until(lambda: len(self.coordinator.available) == 1)
                fixture = VerifyCaptureFixture(self.coordinator, self.request(failure))
                ticket = fixture.forward()
                if failure == "anchor":
                    ticket.input_tokens[0, 0] = 99
                elif failure == "position":
                    ticket.positions[0, 1] = 99
                outputs = [[99 if failure == "correct_prefix" else 20, 55, 0, 0]]
                fixture.accept(ticket, outputs, [2])
                self.wait_until(lambda fixture=fixture: fixture.record.state == "done")
                self.assertIsNone(fixture.request.training_capture_context)
                self.assertEqual(
                    self.catalog.captures[fixture.record.lease.capture_id]["reason"],
                    "verify_commit_failed",
                )
                self.assertFalse(self.catalog.publications)

    def test_admission_is_bounded_and_publication_runs_off_request_thread(self):
        self.wait_until(lambda: len(self.coordinator.available) == 1)
        first, overflow = self.request("first"), self.request("overflow")
        self.coordinator.before_forward([first, overflow])
        record = first.training_capture_context
        self.assertIsNotNone(record)
        self.assertIsNone(overflow.training_capture_context)
        self.assertEqual(
            self.coordinator.stats()["counters"]["admission_backpressure"], 1
        )
        context = record.context
        context.export_kv(self.coordinator.exporter, torch.tensor([7, 3]), end=2)
        context.record_teacher(
            capture_teacher(torch.arange(256).float()[None], 256), row=0, position=2
        )
        context.commit_token(position=2, token_id=10)
        context.seal("length")
        self.coordinator._detach(first, record)
        published = self.catalog.wait_publications(1)
        self.assertIn(published[0]["manifest_key"], self.sdk.data)
        self.assertFalse(self.catalog.errors)
        self.wait_until(lambda: self.coordinator.stats()["counters"].get("ready") == 1)

    def test_expired_spare_is_removed_and_retraction_never_publishes(self):
        self.wait_until(lambda: len(self.coordinator.available) == 1)
        self.coordinator.stop.set()
        self.coordinator.lease_thread.join(timeout=5)
        record = self.coordinator.available[0]
        record.invalid_reason = "capture_lease_expired"
        self.coordinator._queue_record(record)
        self.wait_until(lambda: record.state == "done")
        self.assertEqual(len(self.coordinator.available), 0)
        self.coordinator._reserve()
        req = self.request("retracted")
        self.coordinator.before_forward([req])
        record = req.training_capture_context
        req.is_retracted = True
        self.coordinator.before_forward([])
        self.wait_until(lambda: record.state == "done")
        self.assertIsNone(req.training_capture_context)
        self.assertFalse(self.catalog.publications)
        self.assertEqual(
            self.catalog.captures[record.lease.capture_id]["reason"],
            "request_retracted",
        )

    def test_shutdown_drains_spares_and_invalidates_active_before_unregister(self):
        self.wait_until(lambda: len(self.coordinator.available) == 1)
        req = self.request("shutdown")
        self.coordinator.before_forward([req])
        record = req.training_capture_context
        self.coordinator.close()
        self.assertEqual(record.state, "done")
        self.assertIsNone(req.training_capture_context)
        self.assertTrue(self.sdk.closed)
        self.assertFalse(self.coordinator.writer_thread.is_alive())
        self.assertFalse(self.coordinator.lease_thread.is_alive())
        self.assertEqual(
            self.catalog.captures[record.lease.capture_id]["state"], "FAILED"
        )

    def test_config_roundtrip_is_strict(self):
        path = self.directory.name + "/config.json"
        from pathlib import Path

        Path(path).write_bytes(msgspec.json.encode(self.coordinator.config))
        actual = CaptureConfig.load(path)
        self.assertEqual(actual.fingerprint, self.coordinator.config.fingerprint)
        with self.assertRaises(msgspec.ValidationError):
            msgspec.json.decode(b'{"unexpected": 1}', type=CaptureConfig)


if __name__ == "__main__":
    unittest.main()
