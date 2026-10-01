"""Ownership and exact transfer-result contracts for capture buffers."""

import unittest

import torch
from sglang.srt.training_capture.host_pool import HostBufferPool
from sglang.srt.training_capture.kv_exporter import SelectedLayerKVExporter
from sglang.srt.training_capture.mooncake_store import (
    MooncakeSnapshotStore,
    TransportError,
)
from sglang.srt.training_capture.protocol import (
    CaptureError,
    ContractError,
    digest_bytes,
    tensor_bytes,
)
from sglang.srt.training_capture.teacher import capture_teacher
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_utils import (
    BufferStore,
    FakeReplicateConfig,
    Registrar,
    make_kv_spec,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCaptureOwnership(CustomTestCase):
    def test_teacher_excludes_padding_and_survives_sampler_mutation(self):
        raw = torch.arange(2 * 300, dtype=torch.float32).reshape(2, 300) / 20
        raw[:, 256:] = 10000
        original = raw[:, :256].clone()
        rows = capture_teacher(raw, 256, torch.tensor([1, 0]))
        raw.fill_(-float("inf"))
        expected = original[[1, 0]].double()
        torch.testing.assert_close(
            rows.logits, expected.gather(1, rows.token_ids.long()).float()
        )
        torch.testing.assert_close(rows.logsumexp, expected.logsumexp(-1).float())
        self.assertTrue((rows.token_ids < 256).all())
        self.assertTrue(
            (torch.exp(rows.logits - rows.logsumexp[:, None]).sum(-1) < 1).all()
        )

    def test_pool_backpressure_lifetimes_and_noncontiguous_slots(self):
        registrar = Registrar()
        kv = make_kv_spec()
        pool = HostBufferPool(
            kv=kv,
            max_tokens=8,
            slots=1,
            max_bytes=2 << 20,
            registrar=registrar,
            pin_memory=False,
        )
        slot = pool.acquire()
        self.assertIsNone(pool.acquire())
        with self.assertRaises(CaptureError):
            pool.close()
        source = {
            f"target_{c}.{g.layer_id}": torch.arange(16 * 8)
            .reshape(16, 2, 4)
            .bfloat16()
            + g.layer_id
            for g in kv.layers
            for c in ("k", "v")
        }
        original = {name: t.clone() for name, t in source.items()}
        indices = torch.tensor([7, 0, 1, 0, 15, 0, 3, 0, 8, 0], dtype=torch.int32)[::2]
        SelectedLayerKVExporter(kv, source).export(indices, slot.tensors, 0, 5)
        for name in source:
            source[name].zero_()
            torch.testing.assert_close(slot.tensors[name][:5], original[name][indices])
        pool.release(slot, transfer_complete=True)
        self.assertIs(pool.acquire(), slot)
        self.assertEqual(registrar.registrations, 1)
        pool.release(slot, transfer_complete=False)
        self.assertIsNone(pool.acquire())
        with self.assertRaises(CaptureError):
            pool.close()
        with self.assertRaises(CaptureError):
            pool.release(slot, transfer_complete=True)

    def test_fractional_slots_are_rejected_before_any_host_write(self):
        kv = make_kv_spec()
        source = {
            f"target_{c}.{g.layer_id}": torch.ones(16, 2, 4, dtype=torch.bfloat16)
            for g in kv.layers
            for c in ("k", "v")
        }
        destination = {name: torch.zeros_like(value) for name, value in source.items()}
        with self.assertRaises(ContractError):
            SelectedLayerKVExporter(kv, source).export(
                torch.tensor([1.5]), destination, 0, 1
            )
        for value in destination.values():
            self.assertEqual(torch.count_nonzero(value).item(), 0)

    def test_quota_rejection_precedes_registration(self):
        registrar = Registrar()
        with self.assertRaises(ValueError):
            HostBufferPool(
                kv=make_kv_spec(),
                max_tokens=8,
                slots=1,
                max_bytes=1,
                registrar=registrar,
                pin_memory=False,
            )
        self.assertFalse(registrar.buffers)


class TestMooncakeTransferContract(CustomTestCase):
    def setUp(self):
        self.client = BufferStore()
        self.store = MooncakeSnapshotStore(
            self.client, FakeReplicateConfig(), max_receive_bytes=1 << 20
        )
        self.tensor = torch.arange(128, dtype=torch.int32)
        self.digest = digest_bytes(tensor_bytes(self.tensor))
        self.store.register(self.tensor)

    def tearDown(self):
        self.store.close()

    def test_immutable_idempotence_and_conflict(self):
        self.store.put_registered("test", self.tensor, self.digest)
        self.store.put_registered("test", self.tensor, self.digest)
        self.assertEqual(self.client.put_keys, ["test"])
        restored = self.store.get_tensor("test", [128], torch.int32, self.digest)
        torch.testing.assert_close(restored, self.tensor)
        self.client.data["test"] = b"\x00" * 512
        with self.assertRaises(ContractError):
            self.store.put_registered("test", self.tensor, self.digest)
        self.assertEqual(self.client.put_keys, ["test"])

    def test_put_requires_zero_status_and_quarantines_source(self):
        for rc in (-1, 512):
            with self.subTest(returncode=rc):
                client = BufferStore()
                client.put_status = rc
                store = MooncakeSnapshotStore(client, FakeReplicateConfig())
                store.register(self.tensor)
                with self.assertRaises(TransportError):
                    store.put_registered("test", self.tensor, self.digest)
                self.assertIn(self.tensor.data_ptr(), store.registered)
                with self.assertRaises(TransportError):
                    store.unregister(self.tensor)
                store.close()

    def test_short_read_is_not_success_and_failed_read_retains_destination(self):
        self.store.put_registered("test", self.tensor, self.digest)
        self.client.get_count = 0
        with self.assertRaisesRegex(ContractError, "short"):
            self.store.get_tensor("test", [128], torch.int32, self.digest)
        self.assertEqual(len(self.store.registered), 1)
        for index, status in enumerate((-1, -707)):
            with self.subTest(status=status):
                self.client.get_count = status
                with self.assertRaises(TransportError):
                    self.store.get_tensor("test", [128], torch.int32, self.digest)
                self.assertEqual(len(self.store.registered), index + 2)
                self.assertEqual(len(self.store.quarantined), index + 1)

    def test_unregistered_or_changed_source_never_reaches_transport(self):
        with self.assertRaises(ContractError):
            self.store.put_registered("test", self.tensor.clone(), self.digest)
        self.tensor[0] = 999
        with self.assertRaises(ContractError):
            self.store.put_registered("test", self.tensor, self.digest)
        self.assertFalse(self.client.put_keys)


if __name__ == "__main__":
    unittest.main()
