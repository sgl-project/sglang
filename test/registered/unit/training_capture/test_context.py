"""Position and lifetime invariants for request-owned capture ledgers."""

import threading
import unittest
from unittest.mock import patch

import torch
from sglang.srt.training_capture.context import RequestCaptureContext
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.protocol import ContractError, validate_tensors
from sglang.srt.training_capture.teacher import capture_teacher
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_utils import Registrar, make_snapshot

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCaptureContext(CustomTestCase):
    def setUp(self):
        self.manifest, _ = make_snapshot()
        self.pool = HostBufferPool(
            kv=self.manifest.kv,
            max_tokens=8,
            slots=1,
            max_bytes=2 << 20,
            registrar=Registrar(),
            pin_memory=False,
        )
        self.slot = self.pool.acquire()
        self.context = RequestCaptureContext(
            slot=self.slot,
            prompt_ids=(3, 4),
            max_tokens=8,
            vocab_size=256,
        )
        self.buffers = {
            f"target_{c}.{layer.layer_id}": torch.arange(16 * 8)
            .reshape(16, 2, 4)
            .bfloat16()
            for layer in self.manifest.kv.layers
            for c in ("k", "v")
        }
        self.exporter = SelectedLayerKVExporter(self.manifest.kv, self.buffers)
        self.rows = capture_teacher(torch.arange(256).float()[None], 256)

    def tearDown(self):
        self.pool.release(self.slot, transfer_complete=True)
        self.pool.close()

    def snapshot(self, *, prepare=False):
        manifest = self.manifest
        method = self.context.prepare_snapshot if prepare else self.context.snapshot
        return method(
            dataset_id=manifest.dataset_id,
            sample_id=manifest.sample_id,
            generation_id=manifest.generation_id,
            teacher=manifest.teacher,
            kv=manifest.kv,
            provenance=manifest.provenance,
        )

    def test_chunked_prefill_then_single_response_retains_last_teacher(self):
        self.context.export_kv(self.exporter, torch.tensor([7]), end=1)
        self.context.record_positions(torch.tensor([0]), start=0)
        with self.assertRaises(ContractError):
            self.context.record_teacher(self.rows, row=0, position=1)
        self.context.export_kv(self.exporter, torch.tensor([2]), end=2)
        self.context.record_positions(torch.tensor([1]), start=1)
        self.context.record_teacher(self.rows, row=0, position=2)
        self.context.commit_token(position=2, token_id=255)
        self.context.seal("length")
        manifest, tensors = self.snapshot()
        validate_tensors(manifest, tensors)
        self.assertEqual(manifest.sequence.response_length, 1)
        self.assertEqual(self.slot.tensors["kv_valid"][:3].tolist(), [1, 1, 0])
        self.assertEqual(self.slot.tensors["logits_positions"][:1].tolist(), [2])
        self.assertEqual(self.slot.tensors["token_ids"][:3].tolist(), [3, 4, 255])
        self.assertEqual(self.slot.tensors["loss_mask"][:3].tolist(), [0, 0, 1])
        with self.assertRaises(ContractError):
            self.context.commit_token(position=3, token_id=1)

    def test_prefix_and_decode_append_exactly_once_with_independent_storage(self):
        slots = [7, 2, 12, 4]
        self.context.export_kv(self.exporter, torch.tensor(slots[:2]), end=2)
        for position, token in zip(range(2, 5), (10, 11, 12)):
            if position > 2:
                self.context.export_kv(
                    self.exporter, torch.tensor([slots[position - 1]]), end=position
                )
            self.context.record_teacher(self.rows, row=0, position=position)
            self.context.commit_token(position=position, token_id=token)
        original = {name: value[slots].clone() for name, value in self.buffers.items()}
        for value in self.buffers.values():
            value.zero_()
        self.context.seal("eos")
        manifest, tensors = self.snapshot()
        validate_tensors(manifest, tensors)
        self.assertEqual(self.slot.tensors["logits_positions"][:3].tolist(), [2, 3, 4])
        for name, expected in original.items():
            torch.testing.assert_close(self.slot.tensors[name][:4], expected)

    def test_missing_shifted_duplicate_and_aborted_data_cannot_publish(self):
        self.context.export_kv(self.exporter, torch.tensor([1, 3]), end=2)
        with self.assertRaises(ContractError):
            self.context.commit_token(position=2, token_id=5)
        with self.assertRaises(ContractError):
            self.context.record_teacher(self.rows, row=0, position=3)
        self.context.record_teacher(self.rows, row=0, position=2)
        with self.assertRaises(ContractError):
            self.context.record_teacher(self.rows, row=0, position=2)
        with self.assertRaises(ContractError):
            self.context.seal("length")
        self.context.abort("request_retracted")
        with self.assertRaises(ContractError):
            self.snapshot()

    def test_cancellation_after_seal_prevents_later_snapshot_preparation(self):
        """Sealed data still awaits publication and must remain cancellable."""
        self.context.export_kv(self.exporter, torch.tensor([1, 3]), end=2)
        self.context.record_teacher(self.rows, row=0, position=2)
        self.context.commit_token(position=2, token_id=5)
        self.context.seal("length")
        self.context.abort("peer_failed")
        with self.assertRaises(ContractError):
            self.snapshot()

    def test_cancel_during_copy_wait_never_returns_prepared_descriptors(self):
        """A writer already waiting for copies must observe a concurrent cancel."""
        for method_name in ("snapshot", "prepare_snapshot", "prepare_partition"):
            with self.subTest(method=method_name):
                partitioned = method_name == "prepare_partition"
                partition = (
                    plan_capture_layout(
                        self.manifest.kv, tp_size=1, pp_layer_ranges=[(0, 4)]
                    ).partitions[0]
                    if partitioned
                    else None
                )
                context = RequestCaptureContext(
                    slot=self.slot,
                    prompt_ids=(3, 4),
                    max_tokens=8,
                    vocab_size=256,
                    partition=partition,
                )
                context.export_kv(self.exporter, torch.tensor([1, 3]), end=2)
                context.record_teacher(self.rows, row=0, position=2)
                context.commit_token(position=2, token_id=5)
                context.seal("length")
                waiting, release = threading.Event(), threading.Event()
                returned, errors = [], []

                def wait_for_copies(waiting=waiting, release=release):
                    waiting.set()
                    if not release.wait(timeout=5):
                        raise TimeoutError("test did not release the copy wait")

                def prepare(
                    context=context,
                    returned=returned,
                    errors=errors,
                    method_name=method_name,
                ):
                    try:
                        method = getattr(context, method_name)
                        returned.append(
                            method(
                                dataset_id=self.manifest.dataset_id,
                                sample_id=self.manifest.sample_id,
                                generation_id=self.manifest.generation_id,
                                teacher=self.manifest.teacher,
                                kv=self.manifest.kv,
                                provenance=self.manifest.provenance,
                            )
                        )
                    except (ContractError, TimeoutError) as error:
                        errors.append(error)

                with patch.object(context, "wait_for_copies", wait_for_copies):
                    worker = threading.Thread(target=prepare, daemon=True)
                    worker.start()
                    try:
                        self.assertTrue(waiting.wait(timeout=5))
                        context.abort("peer_failed")
                    finally:
                        release.set()
                        worker.join(timeout=5)
                    self.assertFalse(worker.is_alive())
                self.assertEqual(len(returned), 0)
                self.assertEqual(len(errors), 1)
                self.assertIsInstance(errors[0], ContractError)

    def test_prepared_descriptors_do_not_certify_payload_contents(self):
        self.context.export_kv(self.exporter, torch.tensor([1, 3]), end=2)
        self.context.record_teacher(self.rows, row=0, position=2)
        self.context.commit_token(position=2, token_id=5)
        self.context.seal("length")
        self.slot.tensors["loss_mask"][0] = 1
        manifest, tensors = self.snapshot(prepare=True)
        with self.assertRaisesRegex(ContractError, "mask"):
            validate_tensors(manifest, tensors)
        with self.assertRaisesRegex(ContractError, "mask"):
            self.snapshot()

    def test_checked_snapshot_observes_cancel_during_content_validation(self):
        self.context.export_kv(self.exporter, torch.tensor([1, 3]), end=2)
        self.context.record_teacher(self.rows, row=0, position=2)
        self.context.commit_token(position=2, token_id=5)
        self.context.seal("length")

        def cancelled_validation(*args):
            validate_tensors(*args)
            self.context.abort("peer_failed")

        with (
            patch(
                "sglang.srt.training_capture.context.validate_tensors",
                side_effect=cancelled_validation,
            ),
            self.assertRaisesRegex(ContractError, "sealed"),
        ):
            self.snapshot()


if __name__ == "__main__":
    unittest.main()
