import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


MODULE = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/mem_cache/token_kv_snapshot_store.py"
)
spec = importlib.util.spec_from_file_location("token_kv_snapshot_store", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Store = module.TokenKVSnapshotStore
SnapshotError = module.TokenKVSnapshotError


class SnapshotStorageTests(CustomTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.store = Store(self.temporary.name)
        self.identity = {
            "weight_version": "step-7",
            "model_digest": "actual-model-content-digest",
            "kv_dtype": "bfloat16",
            "layout": "full_attention_and_gdn",
        }
        self.request = {
            "input_ids": [11, 22],
            "output_ids": [33, 44],
            "computed_token_count": 3,
            "next_input_token": 44,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, rank=0, size=2, **changes):
        values = {
            "tp_rank": rank,
            "tp_size": size,
            "identity": self.identity,
            "request_metadata": self.request,
            "tensor_metadata": {"components": ["attention_kv", "conv", "recurrent"]},
            "payload": b"actual-exported-bytes-for-unit-storage-test-" + bytes([rank]),
        }
        values.update(changes)
        return self.store.write_rank("snapshot-1", **values)

    def read(self, rank=0, size=2, **changes):
        values = {"tp_rank": rank, "tp_size": size, "expected_identity": self.identity}
        values.update(changes)
        return self.store.read_rank("snapshot-1", **values)

    def complete(self):
        self.write()
        self.write(rank=1)
        return self.store.commit("snapshot-1", tp_size=2)

    def test_read_requires_all_rank_commit(self):
        self.write()
        with self.assertRaises(SnapshotError):
            self.read()
        with self.assertRaises(SnapshotError):
            self.store.commit("snapshot-1", tp_size=2)
        self.assertFalse(
            (Path(self.temporary.name) / "snapshot-1/COMMITTED.json").exists()
        )

    def test_real_file_roundtrip_and_idempotent_commit(self):
        manifest = self.complete()
        metadata, payload = self.read(rank=1)
        self.assertEqual(metadata["request_metadata"], self.request)
        self.assertTrue(payload.endswith(bytes([1])))
        self.assertEqual(manifest, self.store.commit("snapshot-1", tp_size=2))
        self.assertEqual(self.write(rank=1), metadata)

    def test_conflicting_late_write_is_rejected(self):
        self.complete()
        with self.assertRaises(SnapshotError):
            self.write(payload=b"late-mutated-state")
        self.read()

    def test_tp_ranks_must_agree_on_computed_prefix(self):
        self.write()
        self.write(rank=1, request_metadata={**self.request, "computed_token_count": 4})
        with self.assertRaises(SnapshotError):
            self.store.commit("snapshot-1", tp_size=2)

    def test_tp_ranks_must_agree_on_model_version(self):
        self.write()
        self.write(rank=1, identity={**self.identity, "weight_version": "step-8"})
        with self.assertRaises(SnapshotError):
            self.store.commit("snapshot-1", tp_size=2)

    def test_receiver_model_or_dtype_mismatch_is_rejected(self):
        self.complete()
        for change in [{"weight_version": "step-8"}, {"kv_dtype": "float8_e4m3fn"}]:
            with self.assertRaises(SnapshotError):
                self.read(expected_identity={**self.identity, **change})

    def test_payload_corruption_is_rejected_before_deserialization(self):
        self.complete()
        payload = Path(self.temporary.name) / "snapshot-1/rank-0.bin"
        data = payload.read_bytes()
        payload.write_bytes(data[:-1] + b"x")
        with self.assertRaises(SnapshotError):
            self.read()

    def test_deleted_payload_is_rejected(self):
        self.complete()
        (Path(self.temporary.name) / "snapshot-1/rank-1.bin").unlink()
        with self.assertRaises(SnapshotError):
            self.read(rank=1)

    def test_metadata_corruption_is_rejected(self):
        self.complete()
        path = Path(self.temporary.name) / "snapshot-1/rank-0.json"
        value = json.loads(path.read_text())
        value["request_metadata"]["next_input_token"] = 99
        path.write_text(json.dumps(value))
        with self.assertRaises(SnapshotError):
            self.read()

    def test_receiver_request_history_is_bound(self):
        self.complete()
        self.read(expected_request_metadata=self.request)
        with self.assertRaises(SnapshotError):
            self.read(
                expected_request_metadata={**self.request, "output_ids": [33, 55]}
            )

    def test_snapshot_id_cannot_escape_root(self):
        for value in ["../outside", "/tmp/outside", "a/b", "", "a" * 129]:
            with self.assertRaises(SnapshotError):
                self.store._directory(value)

    def test_wrong_world_size_or_rank_is_rejected(self):
        self.complete()
        with self.assertRaises(SnapshotError):
            self.read(size=1)
        with self.assertRaises(SnapshotError):
            self.read(rank=2)
        with self.assertRaises(SnapshotError):
            self.read(rank=True)

    def test_concurrent_rank_writer_is_rejected(self):
        self.write()
        lock = Path(self.temporary.name) / "snapshot-1/rank-0.lock"
        lock.write_text("")
        with self.assertRaises(SnapshotError):
            self.write()

    def test_commit_receipt_binds_actual_manifest_file(self):
        self.complete()
        receipt = self.store.receipt("snapshot-1")
        reference = receipt["manifest_ref"]
        raw = Path(reference["path"]).read_bytes()
        self.assertEqual(reference["bytes"], len(raw))
        self.assertEqual(reference["sha256"], hashlib.sha256(raw).hexdigest())
        self.assertTrue(receipt["snapshot_committed"])


if __name__ == "__main__":
    unittest.main()
