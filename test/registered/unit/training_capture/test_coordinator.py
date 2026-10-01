"""Bounded admission, background publication and shutdown without CUDA."""

import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import msgspec
import torch
from prometheus_client import CollectorRegistry
from sglang.srt.training_capture.admission import CaptureAdmission
from sglang.srt.training_capture.catalog import HTTPCaptureCatalog
from sglang.srt.training_capture.config import (
    AdaptiveCaptureConfig,
    CaptureConfig,
    CaptureLatencyConfig,
    StoreSetup,
)
from sglang.srt.training_capture.coordinator import CaptureCoordinator
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.metrics import CaptureMetrics
from sglang.srt.training_capture.mooncake_store import (
    MooncakeSnapshotStore,
    TransportError,
)
from sglang.srt.training_capture.teacher import capture_teacher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_catalog import TestCaptureCatalog
from sglang.test.training_capture_utils import (
    BufferStore,
    CaptureTestRequest,
    FakeReplicateConfig,
    OverlapCaptureFixture,
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
        self.metrics_registry = CollectorRegistry()
        metrics = CaptureMetrics({"model_name": "test"}, registry=self.metrics_registry)
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
            metrics=metrics,
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

    def metric(self, name, **labels):
        return self.metrics_registry.get_sample_value(
            "sglang:training_capture_" + name, {"model_name": "test", **labels}
        )

    def test_overlap_finalizes_previous_result_with_owned_lookahead_kv(self):
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH

        self.wait_until(lambda: len(self.coordinator.available) == 1)
        req = CaptureTestRequest("lookahead", 1)
        fixture = OverlapCaptureFixture(self.coordinator, req)
        previous = fixture.forward(2)
        lookahead = fixture.forward(3)
        req.output_ids = [10]
        req.finished_len = 1
        req.finished_reason = FINISH_LENGTH(1)
        self.coordinator.after_result(previous)
        for source in self.coordinator.exporter.buffers.values():
            source.zero_()
        self.coordinator.after_result(lookahead)
        manifest, tensors = read_snapshot(
            self.store, self.catalog.wait_publications(1)[0]
        )
        self.assertEqual(manifest.sequence.response_length, 1)
        self.assertEqual(tensors["token_ids"].tolist(), [3, 4, 10])
        self.assertEqual(tensors["kv_valid"].tolist(), [1, 1, 1])
        self.assertEqual(tensors["logits_positions"].tolist(), [2])
        torch.testing.assert_close(
            tensors["teacher_topk_logits"][0], torch.arange(255, 127, -1).float() + 2
        )
        for name, source in fixture.sources.items():
            torch.testing.assert_close(
                tensors[name], source[fixture.slots[:3]], rtol=0, atol=0
            )

    def test_overlap_at_exact_capacity_preserves_all_required_teacher_rows(self):
        from sglang.srt.managers.schedule_batch import FINISH_LENGTH

        self.wait_until(lambda: len(self.coordinator.available) == 1)
        req = CaptureTestRequest("capacity", 6)
        fixture = OverlapCaptureFixture(self.coordinator, req)
        previous = fixture.forward(2)
        for end in range(3, 9):
            current = fixture.forward(end)
            req.output_ids.append(end + 7)
            if end == 8:
                req.finished_len = 6
                req.finished_reason = FINISH_LENGTH(6)
            self.coordinator.after_result(previous)
            previous = current
        self.coordinator.after_result(previous)
        manifest, tensors = read_snapshot(
            self.store, self.catalog.wait_publications(1)[0]
        )
        self.assertEqual(manifest.sequence.total_length, 8)
        self.assertEqual(tensors["token_ids"].tolist(), [3, 4, 10, 11, 12, 13, 14, 15])
        self.assertEqual(tensors["logits_positions"].tolist(), list(range(2, 8)))
        self.assertEqual(tensors["kv_valid"].tolist(), [1] * 8)
        self.assertEqual(self.coordinator.stats()["counters"]["overlap_forwards"], 7)

    def test_overlap_abort_with_pending_forward_never_publishes(self):
        from sglang.srt.managers.schedule_batch import FINISH_ABORT

        self.wait_until(lambda: len(self.coordinator.available) == 1)
        req = self.request("aborted-lookahead")
        fixture = OverlapCaptureFixture(self.coordinator, req)
        previous = fixture.forward(2)
        lookahead = fixture.forward(3)
        req.finished_reason = FINISH_ABORT()
        self.coordinator.on_release(req)
        self.coordinator.after_result(previous)
        self.coordinator.after_result(lookahead)
        self.wait_until(lambda: fixture.record.state == "done")
        self.assertFalse(self.catalog.publications)
        self.assertEqual(
            self.catalog.captures[fixture.record.lease.capture_id]["reason"],
            "request_aborted_or_retracted",
        )

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

    def test_spec_overlap_confirms_pending_prefill_and_verify_tokens(self):
        self.wait_until(lambda: len(self.coordinator.available) == 1)
        req = CaptureTestRequest("spec-overlap", 5)
        fixture = VerifyCaptureFixture(self.coordinator, req, prefill_pending=True)
        first = fixture.accept(fixture.forward(), [[20, 30, 55, 0]], [3])
        self.assertEqual(fixture.record.context.token_ids, [3, 4])
        self.assertEqual(
            fixture.record.context.pending_tokens, {2: 10, 3: 20, 4: 30, 5: 55}
        )
        req.output_ids = [10]
        self.coordinator.after_result(fixture.prefill_result)
        second = fixture.accept(
            fixture.forward(inputs=(55, 60, 70, 80), prefix=5), [[90, 0, 0, 0]], [1]
        )
        req.output_ids = [10, 20, 30, 55]
        self.coordinator.after_result(first)
        self.assertEqual(fixture.record.context.pending_tokens, {6: 90})
        fixture.finish([10, 20, 30, 55, 90], 5)
        self.coordinator.after_result(second)
        _, tensors = read_snapshot(self.store, self.catalog.wait_publications(1)[0])
        self.assertEqual(tensors["token_ids"].tolist(), [3, 4, 10, 20, 30, 55, 90])
        self.assertEqual(tensors["logits_positions"].tolist(), list(range(2, 7)))
        self.assertEqual(tensors["kv_valid"].tolist(), [1] * 6 + [0])

    def test_spec_overlap_skips_lookahead_beyond_capacity_without_losing_sample(self):
        self.wait_until(lambda: len(self.coordinator.available) == 1)
        req = CaptureTestRequest("spec-capacity", 6)
        fixture = VerifyCaptureFixture(self.coordinator, req, prefill_pending=True)
        first = fixture.accept(fixture.forward(), [[20, 30, 40, 50]], [4])
        req.output_ids = [10]
        self.coordinator.after_result(fixture.prefill_result)
        second = fixture.accept(
            fixture.forward(inputs=(50, 60, 70, 80), prefix=6), [[60, 70, 80, 90]], [4]
        )
        req.output_ids = [10, 20, 30, 40, 50]
        self.coordinator.after_result(first)
        self.assertIsNone(fixture.forward(inputs=(90, 100, 110, 120), prefix=10))
        self.assertEqual(fixture.record.context.pending_tokens, {7: 60})
        fixture.finish([10, 20, 30, 40, 50, 60, 70, 80, 90], 6)
        self.coordinator.after_result(second)
        _, tensors = read_snapshot(self.store, self.catalog.wait_publications(1)[0])
        self.assertEqual(tensors["token_ids"].tolist(), [3, 4, 10, 20, 30, 40, 50, 60])
        self.assertEqual(tensors["kv_valid"].tolist(), [1] * 8)
        self.assertEqual(tensors["teacher_topk_logits"].shape[0], 6)

    def test_spec_overlap_cpu_token_mismatch_fails_capture(self):
        self.wait_until(lambda: len(self.coordinator.available) == 1)
        req = self.request("wrong-cpu-token")
        fixture = VerifyCaptureFixture(self.coordinator, req, prefill_pending=True)
        fixture.accept(fixture.forward(), [[55, 0, 0, 0]], [1])
        req.output_ids = [11]
        self.coordinator.after_result(fixture.prefill_result)
        self.wait_until(lambda: fixture.record.state == "done")
        self.assertEqual(
            self.catalog.captures[fixture.record.lease.capture_id]["reason"],
            "token_alignment_failed",
        )
        self.assertFalse(self.catalog.publications)

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

    def test_stalled_writer_pauses_admission_and_recovers_after_publication(self):
        self.wait_until(lambda: len(self.coordinator.available) == 1)
        self.coordinator.admission = CaptureAdmission(
            1.0,
            AdaptiveCaptureConfig(
                interval_seconds=0.01, writer_stall_seconds=0.03, cooldown_seconds=0.03
            ),
        )
        entered, release = threading.Event(), threading.Event()
        original_write = self.coordinator.writer.write

        def blocked_write(*args):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release writer")
            return original_write(*args)

        with patch.object(self.coordinator.writer, "write", side_effect=blocked_write):
            try:
                req = CaptureTestRequest("stalled", 1)
                fixture = OverlapCaptureFixture(self.coordinator, req)
                step = fixture.forward(2)
                from sglang.srt.managers.schedule_batch import FINISH_LENGTH

                req.output_ids = [10]
                req.finished_len = 1
                req.finished_reason = FINISH_LENGTH(1)
                self.coordinator.after_result(step)
                self.assertTrue(entered.wait(2))
                self.wait_until(lambda: self.coordinator._admission_ratio() == 0)
                skipped = self.request("while-stalled")
                self.coordinator.before_forward([skipped])
                self.assertIsNone(skipped.training_capture_context)
                self.assertEqual(fixture.record.state, "writing")
                self.assertEqual(fixture.record.slot.state, "filling")
                self.assertFalse(self.catalog.publications)
                self.assertEqual(
                    self.coordinator.stats()["admission"]["reason"], "writer_stall"
                )
                self.wait_until(
                    lambda: self.metric("sample_ratio", kind="effective") == 0
                )
                self.assertEqual(self.metric("reservations", state="writing"), 1)
                self.assertGreater(self.metric("writer_age_seconds"), 0)
            finally:
                release.set()
            manifest, tensors = read_snapshot(
                self.store, self.catalog.wait_publications(1)[0]
            )
            self.assertEqual(manifest.sequence.response_length, 1)
            self.assertEqual(tensors["token_ids"].tolist(), [3, 4, 10])
            self.wait_until(lambda: self.coordinator._admission_ratio() == 1)
            self.wait_until(lambda: len(self.coordinator.available) == 1)
            following = self.request("after-drain")
            self.coordinator.before_forward([following])
            self.assertIsNotNone(following.training_capture_context)
            self.assertEqual(self.coordinator.stats()["host_pool"]["quarantined"], 0)
            self.assertGreater(self.coordinator.stats()["admission"]["recoveries"], 0)

    def test_unsampled_results_supply_latency_and_health_checks_are_excluded(self):
        self.coordinator.admission = CaptureAdmission(
            1.0,
            AdaptiveCaptureConfig(
                interval_seconds=0.01,
                cooldown_seconds=0.03,
                latency=CaptureLatencyConfig(
                    ttft_seconds=0.1, min_observations=1, window_seconds=0.05
                ),
            ),
        )
        request = self.request("unsampled-slow")
        request.time_stats.scheduler_recv_time = time.perf_counter() - 1
        request.output_ids = [10]
        self.assertIsNone(request.training_capture_context)
        self.coordinator.after_result(None, requests=[request])
        self.wait_until(
            lambda: self.coordinator.stats()["admission"]["effective_ratio"] == 0
        )
        self.wait_until(
            lambda: self.coordinator.stats()["admission"]["latency"]["state"] == "stale"
        )
        from sglang.srt.training_capture.coordinator import HEALTH_CHECK_RID_PREFIX

        health = self.request(HEALTH_CHECK_RID_PREFIX + "latency")
        health.output_ids = [10]
        health.time_stats.scheduler_recv_time = time.perf_counter() - 10
        self.coordinator.after_result(None, requests=[health])
        self.assertIsNone(health.training_capture_latency)
        recovered = self.request("unsampled-healthy")
        recovered.output_ids = [10]
        recovered.time_stats.scheduler_recv_time = time.perf_counter() - 0.001
        self.coordinator.after_result(None, requests=[recovered])
        self.wait_until(lambda: self.coordinator._admission_ratio() > 0)
        self.assertEqual(self.coordinator.admission.latency.observation_ct["ttft"], 2)
        self.assertIsNone(recovered.training_capture_context)

    def test_uncertain_write_keeps_quarantine_during_admission_recovery(self):
        self.wait_until(lambda: len(self.coordinator.available) == 1)
        self.coordinator.admission = CaptureAdmission(
            1.0, AdaptiveCaptureConfig(interval_seconds=0.01, cooldown_seconds=0.05)
        )
        with patch.object(
            self.coordinator.writer, "write", side_effect=TransportError("uncertain")
        ):
            fixture = VerifyCaptureFixture(self.coordinator, self.request("uncertain"))
            result = fixture.accept(fixture.forward(), [[55, 0, 0, 0]], [1])
            fixture.finish([10, 55], 2)
            self.coordinator.after_result(result)
            self.wait_until(lambda: fixture.record.state == "done")
        self.assertEqual(self.coordinator.stats()["admission"]["failures"], 1)
        self.assertFalse(self.catalog.publications)
        self.wait_until(lambda: self.coordinator.pool.stats()["quarantined"] == 1)
        self.coordinator._admission_ratio()
        self.assertEqual(self.coordinator.stats()["admission"]["occupied_fraction"], 1)
        for index in range(20):
            request = self.request(f"after-error-{index}")
            self.coordinator.before_forward([request])
            self.assertIsNone(request.training_capture_context)
        self.assertEqual(fixture.record.slot.state, "quarantined")
        self.wait_until(lambda: self.metric("host_slots", state="quarantined") == 1)
        self.assertEqual(self.metric("events_total", event="writer_failed"), 1)

    def test_catalog_failure_pauses_reservation_retry_then_recovers(self):
        self.wait_until(lambda: len(self.coordinator.available) == 1)
        self.coordinator.admission = CaptureAdmission(
            1.0, AdaptiveCaptureConfig(interval_seconds=0.01, cooldown_seconds=0.2)
        )
        with patch.object(
            self.coordinator.catalog, "begin", side_effect=ConnectionError("offline")
        ):
            record = self.coordinator.available[0]
            record.invalid_reason = "test_recycle"
            self.coordinator._queue_record(record)
            self.wait_until(
                lambda: self.coordinator.stats()["admission"]["failures"] > 0
            )
            current = self.coordinator.stats()
            self.assertEqual(current["admission"]["effective_ratio"], 0)
            self.assertEqual(current["admission"]["reason"], "catalog_error")
            self.assertEqual(current["counters"].get("admitted", 0), 0)
        self.wait_until(lambda: len(self.coordinator.available) == 1)
        self.wait_until(lambda: self.coordinator._admission_ratio() == 1)
        req = self.request("catalog-restored")
        self.coordinator.before_forward([req])
        self.assertIsNotNone(req.training_capture_context)

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
        self.assertFalse(self.coordinator.metrics_thread.is_alive())
        self.assertEqual(
            self.catalog.captures[record.lease.capture_id]["state"], "FAILED"
        )

    def test_metrics_recover_from_export_error_while_catalog_is_blocked(self):
        self.wait_until(lambda: len(self.coordinator.available) == 1)
        entered, release = threading.Event(), threading.Event()
        original_begin = self.coordinator.catalog.begin
        original_update = self.coordinator.metrics.update
        failed_once = False

        def blocked_begin(*args):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("test did not release Catalog")
            return original_begin(*args)

        def flaky_update(stats):
            nonlocal failed_once
            if not failed_once:
                failed_once = True
                raise RuntimeError("test exporter failure")
            return original_update(stats)

        with (
            self.assertLogs("sglang.srt.training_capture.coordinator", level="ERROR"),
            patch.object(self.coordinator.catalog, "begin", side_effect=blocked_begin),
            patch.object(self.coordinator.metrics, "update", side_effect=flaky_update),
        ):
            try:
                initial = self.metric("metrics_update_timestamp_seconds") or 0
                record = self.coordinator.available[0]
                record.invalid_reason = "test_recycle"
                self.coordinator._queue_record(record)
                self.assertTrue(entered.wait(2))
                self.wait_until(
                    lambda: self.metric("metrics_update_timestamp_seconds") > initial
                )
                self.assertTrue(failed_once)
                self.assertTrue(self.coordinator.metrics_thread.is_alive())
                self.assertEqual(self.metric("events_total", event="writer_started"), 1)
                self.assertEqual(self.metric("disabled"), 0)
            finally:
                release.set()
        self.wait_until(lambda: len(self.coordinator.available) == 1)

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
