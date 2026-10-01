"""Failure-boundary tests for manifest-last, fenced publication and replay."""

import base64
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import msgspec
import torch
from sglang.srt.training_capture.catalog import (
    CaptureLease,
    CatalogConflict,
    CatalogUnavailable,
    HTTPCaptureCatalog,
)
from sglang.srt.training_capture.mooncake_store import (
    MooncakeSnapshotStore,
    TransportError,
)
from sglang.srt.training_capture.protocol import (
    ContractError,
    canonical_bytes,
    decode_manifest,
    digest_bytes,
)
from sglang.srt.training_capture.snapshot_writer import (
    OwnerWriteReceipt,
    PublicationJournal,
    SnapshotWriter,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_partition import make_partitioned_snapshot
from sglang.test.training_capture_utils import (
    BufferStore,
    FakeReplicateConfig,
    make_snapshot,
    read_snapshot,
)

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class RecordingCatalog:
    def __init__(self, client):
        self.client = client
        self.events = []
        self.fail_seal = False
        self.fail_publish = False
        self.conflict_seal = False
        self.published = None

    def objects(self, lease, payload):
        self.events.append((payload["phase"], len(self.client.data)))
        return {
            "phase": payload["phase"],
            "accepted_object_ids": sorted(o["object_id"] for o in payload["objects"]),
        }

    def seal(self, lease, payload):
        self.events.append(("seal", len(self.client.data)))
        data = base64.b64decode(payload["manifest_base64"])
        decode_manifest(data)
        assert digest_bytes(data) == payload["manifest"]["sha256"]
        if self.conflict_seal:
            raise CatalogConflict("stale fence")
        if self.fail_seal:
            self.fail_seal = False
            raise CatalogUnavailable("seal response lost")
        return {"state": "PREPARED", "manifest_sha256": payload["manifest"]["sha256"]}

    def publish(self, payload):
        self.events.append(("publish", len(self.client.data)))
        assert payload["manifest_key"] in self.client.data
        assert (
            digest_bytes(self.client.data[payload["manifest_key"]])
            == payload["manifest_sha256"]
        )
        if self.published is not None:
            assert self.published == payload
        self.published = payload
        if self.fail_publish:
            self.fail_publish = False
            raise CatalogUnavailable("publication committed but response lost")
        return {
            "state": "AVAILABLE",
            "publication_id": "publication-1",
            "catalog_cursor": "cursor-1",
        }


class TestSnapshotPublication(CustomTestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.journal = PublicationJournal(self.directory.name)
        self.client = BufferStore()
        self.store = MooncakeSnapshotStore(self.client, FakeReplicateConfig())
        self.catalog = RecordingCatalog(self.client)
        self.writer = SnapshotWriter(self.store, self.catalog, self.journal)
        self.manifest, self.tensors = make_snapshot()
        self.lease = CaptureLease(
            capture_id="capture-1",
            fencing_token=1,
            dataset_id=self.manifest.dataset_id,
            sample_id=self.manifest.sample_id,
            generation_id=self.manifest.generation_id,
            expires_in_seconds=120.0,
            renew_after_seconds=20.0,
        )
        self.manifest_buffer = torch.empty(1 << 20, dtype=torch.uint8)
        self.store.register(self.manifest_buffer)
        for tensor in self.tensors.values():
            self.store.register(tensor)

    def tearDown(self):
        self.store.close()
        self.journal.close()
        self.directory.cleanup()

    def write(self):
        return self.writer.write(
            self.manifest, self.tensors, self.manifest_buffer, self.lease
        )

    def test_objects_registered_before_put_and_manifest_last(self):
        self.write()
        count = len(self.manifest.objects)
        self.assertEqual(
            self.catalog.events,
            [
                ("REGISTERED", 0),
                ("WRITTEN", count),
                ("seal", count),
                ("WRITTEN", count + 1),
                ("publish", count + 1),
            ],
        )
        self.assertEqual(
            self.client.put_keys[-1], self.manifest.key_prefix + "manifest"
        )
        self.assertFalse(list(self.journal.pending()))

    def test_tensor_put_failure_never_publishes_ready(self):
        self.client.put_status = -1
        with self.assertRaises(TransportError):
            self.write()
        self.assertIsNone(self.catalog.published)
        self.assertNotIn(self.manifest.key_prefix + "manifest", self.client.data)
        self.assertEqual(self.catalog.events, [("REGISTERED", 0)])

    def test_lost_publish_response_replays_without_tensor_sources(self):
        self.catalog.fail_publish = True
        with self.assertRaises(CatalogUnavailable):
            self.write()
        original_published = dict(self.catalog.published)
        original_puts = list(self.client.put_keys)
        self.assertEqual(len(list(self.journal.pending())), 1)
        # Simulate process restart after Host slots have been reused.
        for tensor in self.tensors.values():
            tensor.zero_()
        self.store.close()
        self.journal.close()
        self.store = MooncakeSnapshotStore(self.client, FakeReplicateConfig())
        self.journal = PublicationJournal(self.directory.name)
        self.writer = SnapshotWriter(self.store, self.catalog, self.journal)
        receipts = self.writer.recover()
        self.assertEqual(len(receipts), 1)
        self.assertEqual(self.catalog.published, original_published)
        self.assertEqual(self.client.put_keys, original_puts)
        self.assertFalse(list(self.journal.pending()))

    def test_lost_seal_response_can_recover_before_manifest_put(self):
        self.catalog.fail_seal = True
        with self.assertRaises(CatalogUnavailable):
            self.write()
        self.assertNotIn(self.manifest.key_prefix + "manifest", self.client.data)
        self.assertEqual(len(list(self.journal.pending())), 1)
        self.writer.recover()
        self.assertIsNotNone(self.catalog.published)
        self.assertEqual(len(self.client.put_keys), len(self.manifest.objects) + 1)

    def test_stale_fence_cannot_recover_or_erase_journal(self):
        self.catalog.fail_publish = True
        with self.assertRaises(CatalogUnavailable):
            self.write()
        puts = list(self.client.put_keys)
        self.catalog.conflict_seal = True
        with self.assertRaises(CatalogConflict):
            self.writer.recover()
        self.assertEqual(self.client.put_keys, puts)
        self.assertEqual(len(list(self.journal.pending())), 1)

    def test_recovery_does_not_publish_missing_or_corrupted_tensor(self):
        self.catalog.fail_seal = True
        with self.assertRaises(CatalogUnavailable):
            self.write()
        key = self.manifest.objects[0].key
        original = self.client.data.pop(key)
        with self.assertRaises(TransportError):
            self.writer.recover()
        self.assertFalse(self.store.quarantined)
        self.client.data[key] = bytes(len(original))
        with self.assertRaises(ContractError):
            self.writer.recover()
        self.assertIsNone(self.catalog.published)
        self.assertTrue(self.journal.has_pending(self.lease.capture_id))
        self.client.data[key] = original
        self.writer.recover()
        self.assertIsNotNone(self.catalog.published)

    def test_journal_ownership_and_namespace(self):
        with self.assertRaises(BlockingIOError):
            PublicationJournal(self.directory.name)
        with self.assertRaises(ContractError):
            self.journal.complete("../foreign")

    def prepare_partitions(self):
        self.manifest, self.tensors = make_partitioned_snapshot()
        self.lease = msgspec.structs.replace(
            self.lease, generation_id=self.manifest.generation_id
        )
        for tensor in self.tensors.values():
            self.store.register(tensor)
        return {
            owner: {
                obj.key: self.tensors[obj.key]
                for obj in self.manifest.objects
                if obj.owner_id == owner
            }
            for owner in self.manifest.topology.owners
        }

    def write_partitions(self):
        return [
            self.writer.write_partition(
                self.manifest, tensors, self.lease, owner_id=owner
            )
            for owner, tensors in self.prepare_partitions().items()
        ]

    def test_partition_payloads_cannot_omit_or_include_another_owners_objects(self):
        partitions = self.prepare_partitions()
        owner, tensors = next(iter(partitions.items()))
        other = next(values for key, values in partitions.items() if key != owner)
        for payload in ({}, tensors | other):
            with self.subTest(size=len(payload)), self.assertRaises(ContractError):
                self.writer.write_partition(
                    self.manifest, payload, self.lease, owner_id=owner
                )
        for invalid_owner in (None, "unknown"):
            with self.subTest(owner=invalid_owner), self.assertRaises(ContractError):
                self.writer.write_partition(
                    self.manifest, tensors, self.lease, owner_id=invalid_owner
                )
        self.assertFalse(self.catalog.events)
        self.assertFalse(self.client.data)

    def test_partition_aux_values_are_checked_before_store_writes(self):
        partitions = self.prepare_partitions()
        owner = self.manifest.topology.aux_owner
        obj = next(o for o in self.manifest.objects if o.name == "loss_mask")
        partitions[owner][obj.key][0] = 1
        replacement = msgspec.structs.replace(
            obj, sha256=digest_bytes(partitions[owner][obj.key].numpy().tobytes())
        )
        self.manifest = msgspec.structs.replace(
            self.manifest,
            objects=[
                replacement if item == obj else item for item in self.manifest.objects
            ],
        )
        with self.assertRaisesRegex(ContractError, "loss mask"):
            self.writer.write_partition(
                self.manifest, partitions[owner], self.lease, owner_id=owner
            )
        self.assertFalse(self.catalog.events)

    def test_publication_requires_exact_current_owner_receipts(self):
        receipts = self.write_partitions()
        before = list(self.catalog.events)
        for invalid in (
            receipts[:1],
            receipts + receipts[:1],
            [msgspec.structs.replace(receipts[0], fencing_token=2), receipts[1]],
            [msgspec.structs.replace(receipts[0], capture_id="other"), receipts[1]],
            [msgspec.structs.replace(receipts[0], owner_id="unknown"), receipts[1]],
            [
                msgspec.structs.replace(receipts[0], manifest_sha256="0" * 64),
                receipts[1],
            ],
        ):
            with self.subTest(receipts=invalid), self.assertRaises(ContractError):
                self.writer.publish_partitions(
                    self.manifest, invalid, self.manifest_buffer, self.lease
                )
        self.assertEqual(self.catalog.events, before)
        self.assertIsNone(self.catalog.published)
        self.assertNotIn(self.manifest.key_prefix + "manifest", self.client.data)
        receipts = [
            msgspec.json.decode(canonical_bytes(item), type=OwnerWriteReceipt)
            for item in reversed(receipts)
        ]
        self.writer.publish_partitions(
            self.manifest, receipts, self.manifest_buffer, self.lease
        )
        _, actual = read_snapshot(self.store, self.catalog.published)
        expected_manifest, expected = make_snapshot(response_length=4)
        for obj in expected_manifest.objects:
            tensor = actual[obj.name]
            if obj.kind == "kv":
                tensor = tensor[obj.token_range[0] : obj.token_range[1]]
            torch.testing.assert_close(tensor, expected[obj.key], rtol=0, atol=0)
        self.assertEqual(
            self.client.put_keys[-1], self.manifest.key_prefix + "manifest"
        )
        self.assertFalse(list(self.journal.pending()))

    def test_lost_partition_ack_requires_retry_before_receipt_and_publication(self):
        owner, tensors = next(iter(self.prepare_partitions().items()))
        original = self.catalog.objects

        def lose_written_ack(lease, payload):
            result = original(lease, payload)
            if payload["phase"] == "WRITTEN":
                raise CatalogUnavailable("owner WRITTEN response lost")
            return result

        with (
            patch.object(self.catalog, "objects", lose_written_ack),
            self.assertRaises(CatalogUnavailable),
        ):
            self.writer.write_partition(
                self.manifest, tensors, self.lease, owner_id=owner
            )
        puts = list(self.client.put_keys)
        receipt = self.writer.write_partition(
            self.manifest, tensors, self.lease, owner_id=owner
        )
        self.assertEqual(receipt.owner_id, owner)
        self.assertEqual(self.client.put_keys, puts)
        self.assertIsNone(self.catalog.published)
        self.assertFalse(list(self.journal.pending()))

    def test_partition_publication_recovers_without_any_owner_buffers(self):
        receipts = self.write_partitions()
        self.catalog.fail_seal = True
        with self.assertRaises(CatalogUnavailable):
            self.writer.publish_partitions(
                self.manifest, receipts, self.manifest_buffer, self.lease
            )
        for tensor in self.tensors.values():
            tensor.zero_()
        self.assertNotIn(self.manifest.key_prefix + "manifest", self.client.data)
        self.assertEqual(len(self.writer.recover()), 1)
        self.assertIsNotNone(self.catalog.published)
        read_snapshot(self.store, self.catalog.published)
        self.assertFalse(list(self.journal.pending()))


class TestCatalogHTTPContract(CustomTestCase):
    def test_timeout_retries_identical_body_and_object_ack_is_checked(self):
        catalog = HTTPCaptureCatalog("http://127.0.0.1:12345", attempts=2)
        response = MagicMock()
        response.__enter__.return_value = response
        response.read.return_value = json.dumps(
            {"phase": "REGISTERED", "accepted_object_ids": ["one"]}
        ).encode()
        catalog.opener = MagicMock()
        catalog.opener.open.side_effect = [TimeoutError(), response]
        lease = CaptureLease(
            capture_id="c1",
            fencing_token=1,
            dataset_id="d1",
            sample_id="s1",
            generation_id="g1",
            expires_in_seconds=120,
            renew_after_seconds=20,
        )
        payload = {
            "phase": "REGISTERED",
            "objects": [{"object_id": "one"}],
            "idempotency_key": "register-c1",
        }
        catalog.objects(lease, payload)
        calls = catalog.opener.open.call_args_list
        self.assertEqual(calls[0].args[0].data, calls[1].args[0].data)
        catalog.opener.open.side_effect = None
        catalog.opener.open.return_value = response
        response.read.return_value = b'{"phase":"REGISTERED","accepted_object_ids":[]}'
        with self.assertRaises(CatalogConflict):
            catalog.objects(lease, payload)


if __name__ == "__main__":
    unittest.main()
