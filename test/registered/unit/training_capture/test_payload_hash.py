"""Parallel hashing preserves exact bytes, validation and source lifetime."""

import hashlib
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import msgspec
import torch
from sglang.srt.training_capture.mooncake_store import MooncakeSnapshotStore
from sglang.srt.training_capture.payload_hash import PayloadHasher, _digest_group
from sglang.srt.training_capture.protocol import (
    ContractError,
    digest_bytes,
    tensor_bytes,
    validate_tensors,
)
from sglang.srt.training_capture.snapshot import (
    SnapshotMetadata,
    assemble_snapshot,
    prepare_snapshot,
    prepare_snapshot_partition,
)
from sglang.srt.training_capture.topology import plan_capture_layout
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.training_capture_utils import (
    BufferStore,
    FakeReplicateConfig,
    make_snapshot,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestPayloadHash(CustomTestCase):
    def hasher(self, workers):
        value = PayloadHasher(workers)
        self.addCleanup(value.close)
        return value

    def test_config_limits_serial_path_and_close(self):
        for workers in (0, 9, True, 1.5):
            with self.assertRaises(ValueError):
                PayloadHasher(workers)
        for workers in (1, 4):
            hasher = self.hasher(workers)
            self.assertEqual(hasher.digests([]), [])
            self.assertEqual(
                hasher.digests([memoryview(b"abc")]),
                [hashlib.sha256(b"abc").hexdigest()],
            )
            self.assertIsNone(hasher.executor)
            hasher.close()
            hasher.close()
            with self.assertRaises(RuntimeError):
                hasher.digests([])

    def test_order_offsets_empty_views_and_bounded_submissions(self):
        buffers = [
            memoryview(bytes([index]) * ((index + 1) << 16))[index:]
            for index in range(12)
        ]
        buffers.insert(3, memoryview(b""))
        expected = [hashlib.sha256(data).hexdigest() for data in buffers]
        for workers in (1, 2, 4, 8):
            hasher = self.hasher(workers)
            self.assertEqual(hasher.digests(buffers), expected)
            if workers > 1:
                with patch.object(
                    hasher.executor, "submit", wraps=hasher.executor.submit
                ) as submit:
                    self.assertEqual(
                        hasher.digests(list(reversed(buffers))),
                        list(reversed(expected)),
                    )
                self.assertEqual(submit.call_count, workers)
                threads = tuple(hasher.executor._threads)
                hasher.close()
                self.assertTrue(all(not thread.is_alive() for thread in threads))

    def check_failure_barrier(self, *, submission):
        hasher = self.hasher(2)
        hasher.executor = ThreadPoolExecutor(max_workers=2)
        entered, release, finished = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )
        errors = []

        def group_work(group):
            if group[0][0] == (0 if submission else 1):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("hash read gate")
                return _digest_group(group)
            self.assertTrue(entered.wait(5))
            raise ValueError("worker failed")

        submit = hasher.executor.submit
        calls = 0

        def submit_work(*args):
            nonlocal calls
            calls += 1
            if calls == 2 and submission:
                self.assertTrue(entered.wait(5))
                raise ValueError("submission failed")
            return submit(*args)

        def run():
            try:
                hasher.digests([memoryview(bytes(1 << 20)) for _ in range(2)])
            except ValueError as error:
                errors.append(error)
            finally:
                finished.set()

        with (
            patch("sglang.srt.training_capture.payload_hash._digest_group", group_work),
            patch.object(hasher.executor, "submit", submit_work),
        ):
            thread = threading.Thread(target=run)
            thread.start()
            try:
                self.assertTrue(entered.wait(5))
                self.assertFalse(finished.wait(0.1))
            finally:
                release.set()
                thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], ValueError)

    def test_worker_failure_waits_for_other_source_readers(self):
        self.check_failure_barrier(submission=False)

    def test_submission_failure_waits_for_started_source_readers(self):
        self.check_failure_barrier(submission=True)

    def fixture(self):
        manifest, tensors = make_snapshot(
            128, kv_heads=32, head_dim=128, storage_chunk_tokens=64
        )
        buffers = {
            obj.name: tensors[obj.key] for obj in manifest.objects if obj.kind == "aux"
        }
        for name in {obj.name for obj in manifest.objects if obj.kind == "kv"}:
            chunks = sorted(
                (obj for obj in manifest.objects if obj.name == name),
                key=lambda obj: obj.token_range,
            )
            buffers[name] = torch.cat([tensors[obj.key] for obj in chunks])
        metadata = SnapshotMetadata(
            **{
                name: getattr(manifest, name)
                for name in SnapshotMetadata.__struct_fields__
            }
        )
        return manifest, tensors, metadata, buffers

    def test_parallel_snapshot_and_validation_match_serial_digests(self):
        manifest, _, metadata, buffers = self.fixture()
        hasher = self.hasher(4)
        actual, actual_tensors = prepare_snapshot(
            metadata, buffers, valid_kv_tokens=129, payload_hasher=hasher
        )
        self.assertEqual(actual.objects, manifest.objects)
        self.assertIsNotNone(hasher.executor)
        validate_tensors(actual, actual_tensors, payload_hasher=hasher)
        obj = next(obj for obj in actual.objects if obj.kind == "kv")
        actual_tensors[obj.key][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ContractError, "checksum"):
            validate_tensors(actual, actual_tensors, payload_hasher=hasher)
        changed = msgspec.structs.replace(
            obj, sha256=digest_bytes(tensor_bytes(actual_tensors[obj.key]))
        )
        broken = msgspec.structs.replace(
            actual,
            objects=[
                changed if item.key == obj.key else item for item in actual.objects
            ],
        )
        with self.assertRaisesRegex(ContractError, "nonfinite"):
            validate_tensors(broken, actual_tensors, payload_hasher=hasher)

    def test_owner_local_hashes_preserve_partition_assembly(self):
        manifest, _, metadata, buffers = self.fixture()
        hasher = self.hasher(4)
        layout = plan_capture_layout(metadata.kv, tp_size=2, pp_layer_ranges=[(0, 4)])
        metadata = msgspec.structs.replace(metadata, topology=layout.topology)
        prepared, owner_tensors = [], {}
        for partition in layout.partitions:
            local = {
                name: value
                for name, value in buffers.items()
                if not name.startswith("target_")
            }
            for heads in partition.heads:
                for component in ("k", "v"):
                    name = f"target_{component}.{heads.layer_id}"
                    local[name] = buffers[name][:, heads.start : heads.end].contiguous()
            part, tensors = prepare_snapshot_partition(
                metadata,
                local,
                valid_kv_tokens=129,
                partition=partition,
                token_ids=buffers["token_ids"].tolist(),
                payload_hasher=hasher,
            )
            prepared.append(part)
            owner_tensors[partition.owner_id] = tensors
        actual = assemble_snapshot(metadata, prepared, layout=layout)
        for owner, tensors in owner_tensors.items():
            validate_tensors(actual, tensors, owner_id=owner, payload_hasher=hasher)
        self.assertEqual(actual.total_tensor_bytes, manifest.total_tensor_bytes)

    def test_store_batch_hashing_rejects_corruption_and_closes_workers(self):
        manifest, tensors, _, _ = self.fixture()
        client = BufferStore()
        store = MooncakeSnapshotStore(
            client, FakeReplicateConfig(), payload_hash_workers=4
        )
        try:
            for tensor in tensors.values():
                store.register(tensor)
            objects = [
                (obj.key, tensors[obj.key], obj.sha256) for obj in manifest.objects
            ]
            store.put_registered_batch(objects)
            self.assertIsNotNone(store.payload_hasher.executor)
            specs = [
                (obj.key, obj.shape, tensors[obj.key].dtype, obj.sha256)
                for obj in manifest.objects
            ]
            outputs = store.get_tensors(specs)
            for actual, obj in zip(outputs, manifest.objects, strict=True):
                torch.testing.assert_close(actual, tensors[obj.key], rtol=0, atol=0)
            tensors[manifest.objects[-1].key].zero_()
            before = list(client.put_keys)
            with self.assertRaises(ContractError):
                store.put_registered_batch(objects)
            self.assertEqual(client.put_keys, before)
            client.data[manifest.objects[0].key] = bytes(manifest.objects[0].nbytes)
            with self.assertRaises(ContractError):
                store.get_tensors(specs)
            self.assertFalse(store.quarantined)
        finally:
            store.close()
        self.assertTrue(store.payload_hasher.closed)
        self.assertTrue(client.closed)


if __name__ == "__main__":
    unittest.main()
