"""Bounded admission, background publication and shutdown without CUDA."""

import tempfile
import time
import unittest
from types import SimpleNamespace

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
    FakeReplicateConfig,
    make_snapshot,
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
        from sglang.srt.sampling.sampling_params import SamplingParams

        return SimpleNamespace(
            rid=rid,
            finished=lambda: False,
            is_retracted=False,
            output_ids=[],
            lora_id=None,
            multimodal_inputs=None,
            input_embeds=None,
            positional_embed_overrides=None,
            session=None,
            custom_logit_processor=None,
            sampling_params=SamplingParams(max_new_tokens=3),
            origin_input_ids=[3, 4],
            training_capture_attempted=False,
            training_capture_context=None,
            training_capture_finalize=None,
        )

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
