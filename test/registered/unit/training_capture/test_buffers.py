"""Ownership and exact transfer-result contracts for capture buffers."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

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

    def test_setup_exception_closes_the_partially_initialized_sdk_client(self):
        """A setup exception must not leak the client created before setup."""
        for outcome in (-1, OSError("setup transport failed")):
            with self.subTest(outcome=type(outcome).__name__):
                client = BufferStore()
                client.setup = Mock(
                    side_effect=outcome if isinstance(outcome, Exception) else None,
                    return_value=outcome,
                )
                sdk = SimpleNamespace(
                    MooncakeDistributedStore=Mock(return_value=client),
                    ReplicateConfig=FakeReplicateConfig,
                )
                with (
                    patch.dict("sys.modules", {"mooncake.store": sdk}),
                    self.assertRaises((TransportError, OSError)),
                ):
                    MooncakeSnapshotStore.connect({"protocol": "tcp"})
                self.assertTrue(client.closed)

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

    def batch_objects(self):
        second = torch.arange(16, dtype=torch.float32)
        self.store.register(second)
        return [
            ("first", self.tensor, self.digest),
            ("second", second, digest_bytes(tensor_bytes(second))),
        ]

    def test_batch_writes_and_mixed_immutable_retries(self):
        objects = self.batch_objects()
        self.store.put_registered_batch(objects[:1])
        self.store.put_registered_batch(objects)
        self.store.put_registered_batch(objects)
        self.assertEqual(self.client.put_batches, [["first"], ["second"]])
        self.assertEqual(
            self.client.exists_batches,
            [["first"], ["first", "second"], ["first", "second"]],
        )
        self.assertEqual(self.client.put_keys, ["first", "second"])
        for key, tensor, digest in objects:
            restored = self.store.get_tensor(
                key, list(tensor.shape), tensor.dtype, digest
            )
            torch.testing.assert_close(restored, tensor, rtol=0, atol=0)

    def test_batch_validates_all_sources_and_unique_keys_before_network(self):
        objects = self.batch_objects()
        for invalid in (
            [objects[0], ("second", objects[1][1], "0" * 64)],
            [objects[0], ("second", objects[1][1].clone(), objects[1][2])],
            [objects[0], objects[0]],
        ):
            with self.subTest(keys=[item[0] for item in invalid]):
                with self.assertRaises(ContractError):
                    self.store.put_registered_batch(invalid)
                self.assertFalse(self.client.exists_batches)
                self.assertFalse(self.client.put_keys)

    def test_batch_existing_conflict_prevents_missing_object_writes(self):
        objects = self.batch_objects()
        self.client.data["second"] = bytes(objects[1][1].numel() * 4)
        with self.assertRaisesRegex(ContractError, "digest"):
            self.store.put_registered_batch(objects)
        self.assertFalse(self.client.put_keys)
        self.assertFalse(self.store.quarantined)

    def test_batch_existence_failure_does_not_start_or_quarantine_transfers(self):
        objects = self.batch_objects()
        for result in (
            None,
            [],
            [0],
            [0, -1],
            [False, 0],
            [0, 0, 0],
            OSError("lookup"),
        ):
            with (
                self.subTest(result=result),
                patch.object(
                    self.client,
                    "batch_is_exist",
                    side_effect=result if isinstance(result, Exception) else None,
                    return_value=result,
                ),
            ):
                with self.assertRaises(TransportError):
                    self.store.put_registered_batch(objects)
                self.assertFalse(self.client.put_keys)
                self.assertFalse(self.store.quarantined)

    def test_batch_partial_failure_quarantines_only_failed_registrations(self):
        objects = self.batch_objects()
        for status in (-1, 64):
            with self.subTest(status=status):
                client = BufferStore()
                store = MooncakeSnapshotStore(client, FakeReplicateConfig())
                try:
                    for _, tensor, _ in objects:
                        store.register(tensor)
                    with (
                        patch.object(
                            client, "batch_put_from", return_value=[0, status]
                        ),
                        self.assertRaises(TransportError),
                    ):
                        store.put_registered_batch(objects)
                    self.assertEqual(store.quarantined, {objects[1][1].data_ptr()})
                    store.unregister(objects[0][1])
                    with self.assertRaises(TransportError):
                        store.unregister(objects[1][1])
                finally:
                    store.close()

    def test_batch_unknown_completion_retains_all_submitted_sources(self):
        objects = self.batch_objects()
        for result in (
            None,
            [],
            [0],
            [0, 0, 0],
            [0, None],
            [False, 0],
            OSError("write"),
        ):
            with self.subTest(result=result):
                client = BufferStore()
                store = MooncakeSnapshotStore(client, FakeReplicateConfig())
                try:
                    for _, tensor, _ in objects:
                        store.register(tensor)
                    with (
                        patch.object(
                            client,
                            "batch_put_from",
                            side_effect=(
                                result if isinstance(result, Exception) else None
                            ),
                            return_value=result,
                        ),
                        self.assertRaises(TransportError),
                    ):
                        store.put_registered_batch(objects)
                    self.assertEqual(
                        store.quarantined, {obj[1].data_ptr() for obj in objects}
                    )
                    for _, tensor, _ in objects:
                        with self.assertRaises(TransportError):
                            store.unregister(tensor)
                    with self.assertRaises(TransportError):
                        store.put_registered_batch(objects)
                finally:
                    store.close()

    def test_batch_failed_subview_quarantines_the_whole_registered_arena(self):
        objects = [
            (str(index), view, digest_bytes(tensor_bytes(view)))
            for index, view in enumerate(self.tensor.split(64))
        ]
        with (
            patch.object(self.client, "batch_put_from", return_value=[0, -1]),
            self.assertRaises(TransportError),
        ):
            self.store.put_registered_batch(objects)
        self.assertEqual(self.store.quarantined, {self.tensor.data_ptr()})
        with self.assertRaises(TransportError):
            self.store.put_registered_batch(objects[:1])

    def test_batch_falls_back_when_either_native_method_is_unavailable(self):
        objects = self.batch_objects()
        for missing in ("batch_is_exist", "batch_put_from"):
            with self.subTest(method=missing):
                self.client.data.clear()
                self.client.put_keys.clear()
                with patch.object(self.client, missing, None):
                    self.store.put_registered_batch(objects)
                    self.store.put_registered_batch(objects)
                self.assertEqual(self.client.put_keys, ["first", "second"])
                self.assertFalse(self.client.put_batches)
                self.assertFalse(self.client.exists_batches)

    def test_empty_batch_never_calls_the_sdk(self):
        self.store.put_registered_batch([])
        self.assertFalse(self.client.put_batches)
        self.assertFalse(self.client.exists_batches)

    def receive_objects(self):
        objects = self.batch_objects()
        self.store.put_registered_batch(objects)
        return [
            (key, list(tensor.shape), tensor.dtype, digest)
            for key, tensor, digest in objects
        ]

    def test_batch_reads_return_owned_ordered_tensors_including_duplicate_keys(self):
        objects = self.receive_objects()
        baseline = set(self.store.registered)
        outputs = self.store.get_tensors([objects[1], objects[0], objects[1]])
        self.assertEqual(self.client.get_batches, [["second", "first", "second"]])
        self.assertEqual(set(self.store.registered), baseline)
        self.assertEqual(len({out.data_ptr() for out in outputs}), 3)
        self.assertFalse(self.store.quarantined)
        torch.testing.assert_close(outputs[1], self.tensor)
        torch.testing.assert_close(outputs[0], outputs[2])
        outputs[0].zero_()
        self.assertGreater(outputs[2].sum().item(), 0)
        self.assertEqual(self.store.get_tensors([]), [])
        self.store.verify_tensors([])
        self.assertEqual(len(self.client.get_batches), 1)

    def test_batch_read_validates_all_shapes_and_aggregate_budget_before_sdk(self):
        objects = self.receive_objects()
        baseline = set(self.store.registered)
        invalid_shapes = ([], [0], [-1], [True], [1.5])
        for shape in invalid_shapes:
            with self.subTest(shape=shape), self.assertRaises(ContractError):
                self.store.get_tensors(
                    [objects[0], ("second", shape, torch.float32, "")]
                )
        self.store.max_receive_bytes = self.tensor.nbytes
        with self.assertRaisesRegex(ContractError, "budget"):
            self.store.get_tensors(objects)
        with (
            patch.object(self.client, "batch_get_into", None),
            self.assertRaisesRegex(ContractError, "budget"),
        ):
            self.store.get_tensors(objects)
        self.assertEqual(set(self.store.registered), baseline)
        self.assertFalse(self.client.get_batches)

    def test_batch_read_partial_failure_retains_only_failed_destinations(self):
        objects = self.receive_objects()
        baseline = set(self.store.registered)
        del self.client.data["second"]
        with self.assertRaises(TransportError):
            self.store.get_tensors(objects)
        self.assertEqual(len(self.store.quarantined), 1)
        self.assertEqual(set(self.store.registered) - baseline, self.store.quarantined)
        pointer = next(iter(self.store.quarantined))
        self.assertEqual(self.store.registered[pointer].nbytes, 64)
        with self.assertRaises(TransportError):
            self.store.unregister(self.store.registered[pointer])
        self.store.max_receive_bytes = self.tensor.nbytes
        with self.assertRaisesRegex(ContractError, "budget"):
            self.store.get_tensors(objects[:1])
        self.assertEqual(len(self.client.get_batches), 1)

    def test_batch_unknown_read_completion_retains_every_destination(self):
        objects = self.receive_objects()
        for result in (None, 0, [], [512], [512, None], [512, True], OSError("lost")):
            with self.subTest(result=result):
                client = BufferStore()
                store = MooncakeSnapshotStore(client, FakeReplicateConfig())
                try:
                    with (
                        patch.object(
                            client,
                            "batch_get_into",
                            side_effect=result
                            if isinstance(result, Exception)
                            else None,
                            return_value=result,
                        ) as read,
                        self.assertRaisesRegex(TransportError, "uncertain"),
                    ):
                        store.get_tensors(objects)
                    self.assertEqual(store.quarantined, set(read.call_args.args[1]))
                    self.assertEqual(set(store.registered), store.quarantined)
                    self.assertEqual(len(store.registered), 2)
                finally:
                    store.close()
                self.assertFalse(store.registered)

    def test_batch_completed_wrong_lengths_and_digests_release_all_destinations(self):
        objects = self.receive_objects()
        baseline = set(self.store.registered)
        for counts in ([0, 64], [511, 64], [513, 64]):
            with self.subTest(counts=counts):
                with (
                    patch.object(self.client, "batch_get_into", return_value=counts),
                    self.assertRaisesRegex(ContractError, "byte count"),
                ):
                    self.store.get_tensors(objects)
                self.assertEqual(set(self.store.registered), baseline)
                self.assertFalse(self.store.quarantined)
        self.client.data["second"] = b"\x00" * 64
        with self.assertRaisesRegex(ContractError, "digest"):
            self.store.get_tensors(objects)
        self.assertEqual(set(self.store.registered), baseline)
        self.assertFalse(self.store.quarantined)

    def test_batch_read_registration_failure_releases_unsubmitted_destinations(self):
        objects = self.receive_objects()
        baseline = set(self.store.registered)
        register = self.client.register_buffer
        calls = []

        def partial(pointer, size):
            calls.append(pointer)
            return register(pointer, size) if len(calls) == 1 else -1

        with (
            patch.object(self.client, "register_buffer", partial),
            self.assertRaises(TransportError),
        ):
            self.store.get_tensors(objects)
        self.assertFalse(self.client.get_batches)
        self.assertEqual(self.store.quarantined, {calls[1]})
        self.assertEqual(set(self.store.registered) - baseline, {calls[1]})

    def test_batch_read_cleanup_attempts_every_successful_destination(self):
        objects = self.receive_objects()
        baseline = set(self.store.registered)
        unregister = self.client.unregister_buffer
        calls = []

        def partial(pointer):
            calls.append(pointer)
            return -1 if len(calls) == 1 else unregister(pointer)

        with (
            patch.object(self.client, "unregister_buffer", partial),
            self.assertRaises(TransportError),
        ):
            self.store.get_tensors(objects)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.store.quarantined, {calls[0]})
        self.assertEqual(set(self.store.registered) - baseline, {calls[0]})

    def test_batch_read_fallback_preserves_order_and_quarantine_rules(self):
        objects = self.receive_objects()
        baseline = set(self.store.registered)
        with patch.object(self.client, "batch_get_into", None):
            outputs = self.store.get_tensors(objects)
            torch.testing.assert_close(outputs[0], self.tensor)
            self.assertEqual(set(self.store.registered), baseline)
            self.client.get_count = -1
            with self.assertRaises(TransportError):
                self.store.get_tensors(objects)
        self.assertFalse(self.client.get_batches)
        self.assertEqual(len(self.store.quarantined), 1)

    def test_verification_batches_respect_byte_and_object_limits(self):
        objects = self.receive_objects()
        self.store.max_receive_bytes = 512
        self.store.verify_tensors(objects)
        self.assertEqual(self.client.get_batches, [["first"], ["second"]])
        self.client.get_batches.clear()
        self.store.max_receive_bytes = 1 << 20
        self.store.verify_tensors(objects[1:2] * 65)
        self.assertEqual([len(batch) for batch in self.client.get_batches], [64, 1])
        self.assertEqual(len(self.store.registered), 2)
        self.store.max_receive_bytes = 63
        with self.assertRaisesRegex(ContractError, "budget"):
            self.store.verify_tensors(objects[1:2])
        self.assertEqual(len(self.client.get_batches), 2)


if __name__ == "__main__":
    unittest.main()
