"""A failed per-request CUDA fence must not bypass the producer stop barrier."""

import json
import tempfile
import time
import unittest
from unittest.mock import patch

import torch
from sglang.srt.training_capture.catalog import HTTPCaptureCatalog
from sglang.srt.training_capture.config import CaptureConfig, StoreSetup
from sglang.srt.training_capture.coordinator import CaptureCoordinator
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_catalog import TestCaptureCatalog
from sglang.test.training_capture_utils import (
    BufferStore,
    CaptureTestRequest,
    FakeReplicateConfig,
    make_snapshot,
)

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestCudaCaptureShutdown(CustomTestCase):
    def test_failed_request_fence_drains_real_cuda_before_store_close(self):
        for staged in (False, True):
            with self.subTest(staged=staged), tempfile.TemporaryDirectory() as root:
                catalog = TestCaptureCatalog()
                manifest, _ = make_snapshot()
                client = BufferStore()
                store = MooncakeSnapshotStore(client, FakeReplicateConfig())
                sources = {
                    f"target_{component}.{layer.layer_id}": torch.arange(
                        16 * 2 * 4, device="cuda", dtype=torch.bfloat16
                    ).reshape(16, 2, 4)
                    for layer in manifest.kv.layers
                    for component in ("k", "v")
                }
                exporter = SelectedLayerKVExporter(manifest.kv, sources)
                config = CaptureConfig(
                    dataset_id=manifest.dataset_id,
                    model_id="fixture",
                    producer_revision="cuda-shutdown-test",
                    selected_layer_ids=manifest.kv.selected_layer_ids,
                    catalog_endpoint=catalog.endpoint,
                    journal_directory=root + "/journal",
                    store=StoreSetup(
                        local_hostname="localhost", master_server_addr="localhost:1"
                    ),
                    max_sample_tokens=8,
                    max_inflight_samples=1,
                    max_host_bytes=2 << 20,
                    sample_ratio=1.0,
                    kv_d2h_batch_tokens=3 if staged else 1,
                    max_device_bytes=1 << 20 if staged else 0,
                )
                coordinator = None
                try:
                    coordinator = CaptureCoordinator(
                        config=config,
                        teacher=manifest.teacher,
                        kv=manifest.kv,
                        exporter=exporter,
                        req_to_token=None,
                        store=store,
                        catalog=HTTPCaptureCatalog(catalog.endpoint),
                    )
                    deadline = time.monotonic() + 10
                    while coordinator.stats()["states"].get("available") != 1:
                        if time.monotonic() >= deadline:
                            self.fail(str(coordinator.stats()))
                        time.sleep(0.01)
                    request = CaptureTestRequest("unfenced-shutdown")
                    coordinator.before_forward([request])
                    context = request.training_capture_context.context
                    indices = torch.tensor([7, 3], dtype=torch.int32, device="cuda")
                    stream, completed = torch.cuda.Stream(), torch.cuda.Event()
                    stream.wait_stream(torch.cuda.current_stream())
                    destination = (
                        context.slot.device_tensors if staged else context.slot.tensors
                    )
                    # Warm the allocator on this stream before delaying the copy.
                    with torch.cuda.stream(stream):
                        exporter.export(indices, destination, 0, 2)
                    stream.synchronize()
                    for name in sources:
                        destination[name][:2].zero_()
                    torch.cuda.synchronize()
                    with torch.cuda.stream(stream):
                        torch.cuda._sleep(5_000_000_000)
                        with (
                            patch(
                                "torch.cuda.Event",
                                side_effect=RuntimeError("injected fence failure"),
                            ),
                            self.assertRaisesRegex(
                                RuntimeError, "injected fence failure"
                            ),
                        ):
                            context.export_kv(exporter, indices, end=2)
                        completed.record(stream)
                    self.assertTrue(context.transfer_uncertain)
                    self.assertFalse(completed.query())
                    coordinator.control("abort")
                    deadline = time.monotonic() + 10
                    while coordinator.pool.stats()["quarantined"] != 1:
                        if time.monotonic() >= deadline:
                            self.fail(str(coordinator.stats()))
                        time.sleep(0.01)
                    self.assertFalse(
                        completed.query(), "pending-copy precondition was not exercised"
                    )
                    completion_at_store_close = []
                    original_close = client.close

                    def close(
                        completion_at_store_close=completion_at_store_close,
                        completed=completed,
                        original_close=original_close,
                    ):
                        completion_at_store_close.append(completed.query())
                        return original_close()

                    with patch.object(client, "close", side_effect=close):
                        coordinator.close()
                    self.assertEqual(completion_at_store_close, [True])
                    self.assertTrue(coordinator.closed)
                    self.assertTrue(completed.query())
                    self.assertFalse(catalog.publications)
                    for name, source in sources.items():
                        torch.testing.assert_close(
                            destination[name][:2],
                            source[indices].to(destination[name].device),
                            rtol=0,
                            atol=0,
                        )
                    print(
                        json.dumps(
                            {
                                "cuda_shutdown": {
                                    "staged": staged,
                                    "store_closed_after_cuda": completion_at_store_close,
                                    "capture_quarantined_slots": coordinator.pool.stats()[
                                        "quarantined"
                                    ],
                                }
                            }
                        ),
                        flush=True,
                    )
                finally:
                    # The regression may fail before the stop barrier exists.
                    torch.cuda.synchronize()
                    if coordinator is not None:
                        coordinator.close()
                    else:
                        store.close()
                    catalog.close()


if __name__ == "__main__":
    unittest.main()
