import json
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
    GlobalMetadataState,
    Hf3fsMetadataServer,
    RankMetadata,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHf3fsMetadataSnapshot(unittest.TestCase):
    @contextmanager
    def saving_snapshot(self, state):
        ready = threading.Event()
        release = threading.Event()
        dump = json.dump

        def gated_dump(*args, **kwargs):
            ready.set()
            if not release.wait(5):
                raise TimeoutError("metadata snapshot gate was not released")
            return dump(*args, **kwargs)

        with patch(
            "sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server.json.dump",
            side_effect=gated_dump,
        ):
            saver = threading.Thread(target=state.save_to_disk, daemon=True)
            saver.start()
            try:
                self.assertTrue(ready.wait(5), "metadata snapshot was not captured")
                yield
            finally:
                release.set()
                saver.join(5)
            self.assertFalse(saver.is_alive(), "metadata save did not finish")

    def test_http_delete_after_snapshot_cannot_free_a_persisted_key_page(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            server = Hf3fsMetadataServer(str(path))
            with TestClient(server.app) as client:
                self.assertEqual(
                    client.post("/0/initialize", json={"num_pages": 2}).status_code,
                    204,
                )
                reservation = client.post(
                    "/0/reserve_and_allocate_page_indices", json={"keys": [["a", ""]]}
                ).json()["indices"]
                page = reservation[0][1]
                self.assertEqual(
                    client.post(
                        "/0/confirm_write",
                        json={"written_keys_to_confirm": [["a", page]]},
                    ).status_code,
                    204,
                )
                # Periodic persistence runs on a separate thread from HTTP writes.
                with self.saving_snapshot(server.state):
                    self.assertEqual(
                        client.post("/0/delete_keys", json={"keys": ["a"]}).status_code,
                        204,
                    )

            restored = GlobalMetadataState(str(path), save_interval=60)
            restored.load_from_disk()
            metadata = restored.ranks["0:kv"]
            self.assertEqual(metadata.get_page_indices(["a"]), [page])
            self.assertNotIn(page, metadata.free_pages)
            new_page = metadata.reserve_and_allocate_page_indices([("b", "")])[0][1]
            metadata.confirm_write([("b", new_page)], [])
            self.assertNotEqual(metadata.get_page_indices(["a"]), [new_page])

    def test_reservation_after_snapshot_does_not_remove_saved_free_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            state = GlobalMetadataState(str(path), save_interval=60)
            metadata = RankMetadata(3)
            state.ranks["0:kv"] = metadata
            metadata.reserve_and_allocate_page_indices([("a", "")])
            metadata.confirm_write([("a", 2)], [])
            with self.saving_snapshot(state):
                self.assertEqual(
                    metadata.reserve_and_allocate_page_indices([("b", "")]),
                    [(False, 1)],
                )
                metadata.confirm_write([("b", 1)], [])

            restored = GlobalMetadataState(str(path), save_interval=60)
            restored.load_from_disk()
            saved = restored.ranks["0:kv"]
            self.assertEqual(saved.free_pages, [0, 1])
            self.assertEqual(list(saved.key_to_index.items()), [("a", 2)])
            self.assertEqual(
                saved.reserve_and_allocate_page_indices([("c", "")]), [(False, 1)]
            )

    def test_roundtrip_preserves_each_rank_namespace_and_lru_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.json"
            state = GlobalMetadataState(str(path), save_interval=60)
            for key in ("0:kv", "1:kv", "0:indexer"):
                metadata = RankMetadata(4)
                state.ranks[key] = metadata
                metadata.reserve_and_allocate_page_indices([("a", ""), ("b", "")])
                metadata.confirm_write([("a", 3), ("b", 2)], [])
                metadata.get_page_indices(["a"])
            state.save_to_disk()
            restored = GlobalMetadataState(str(path), save_interval=60)
            restored.load_from_disk()
            self.assertEqual(set(restored.ranks), set(state.ranks))
            for metadata in restored.ranks.values():
                self.assertEqual(metadata.free_pages, [0, 1])
                self.assertEqual(
                    list(metadata.key_to_index.items()), [("b", 2), ("a", 3)]
                )


if __name__ == "__main__":
    unittest.main()
