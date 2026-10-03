import unittest

from fastapi.testclient import TestClient

from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
    Hf3fsMetadataServer,
    RankMetadata,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHf3fsMetadataCapacity(unittest.TestCase):
    def test_oversized_batch_preserves_allocations_for_confirmation(self):
        metadata = RankMetadata(2)
        indices = metadata.reserve_and_allocate_page_indices(
            [("a", ""), ("b", ""), ("c", "")]
        )
        self.assertEqual(indices, [(False, 1), (False, 0), (False, -1)])

        metadata.confirm_write([("a", 1), ("b", 0)], [])
        self.assertEqual(metadata.get_page_indices(["a", "b", "c"]), [1, 0, None])
        self.assertEqual(
            metadata.reserve_and_allocate_page_indices([("c", "")]),
            [(False, 1)],
        )
        metadata.confirm_write([("c", 1)], [])
        self.assertEqual(metadata.get_page_indices(["b", "c"]), [0, 1])

    def test_overlapping_batches_preserve_remaining_free_page(self):
        metadata = RankMetadata(3)
        first = metadata.reserve_and_allocate_page_indices([("a", ""), ("b", "")])
        second = metadata.reserve_and_allocate_page_indices([("c", ""), ("d", "")])
        self.assertEqual(first, [(False, 2), (False, 1)])
        self.assertEqual(second, [(False, 0), (False, -1)])

        # The first writer can still finish while the second writer releases
        # its failed write. The refused key never owned a page to release.
        metadata.confirm_write([("a", 2), ("b", 1)], [])
        metadata.confirm_write([], [0])
        self.assertEqual(
            metadata.reserve_and_allocate_page_indices([("d", "")]),
            [(False, 0)],
        )
        metadata.confirm_write([("d", 0)], [])
        self.assertEqual(metadata.get_page_indices(["a", "b", "d"]), [2, 1, 0])

    def test_all_pages_in_flight_can_be_released_and_reused(self):
        metadata = RankMetadata(1)
        self.assertEqual(
            metadata.reserve_and_allocate_page_indices([("a", "")]),
            [(False, 0)],
        )
        for _ in range(2):
            self.assertEqual(
                metadata.reserve_and_allocate_page_indices([("b", "")]),
                [(False, -1)],
            )
        metadata.confirm_write([], [0])
        self.assertEqual(
            metadata.reserve_and_allocate_page_indices([("b", "")]),
            [(False, 0)],
        )

    def test_mixed_hits_and_lru_eviction(self):
        metadata = RankMetadata(3)
        metadata.reserve_and_allocate_page_indices([("a", ""), ("b", "")])
        metadata.confirm_write([("a", 2), ("b", 1)], [])
        indices = metadata.reserve_and_allocate_page_indices(
            [("b", ""), ("c", ""), ("d", "")]
        )
        self.assertEqual(indices, [(True, 1), (False, 0), (False, 2)])
        metadata.confirm_write([("c", 0), ("d", 2)], [])
        self.assertEqual(
            metadata.get_page_indices(["a", "b", "c", "d"]), [None, 1, 0, 2]
        )

    def test_empty_batch_and_zero_capacity(self):
        metadata = RankMetadata(0)
        self.assertEqual(metadata.reserve_and_allocate_page_indices([]), [])
        self.assertEqual(
            metadata.reserve_and_allocate_page_indices([("a", ""), ("b", "")]),
            [(False, -1), (False, -1)],
        )

    def test_http_partial_reservation_can_be_confirmed(self):
        server = Hf3fsMetadataServer()
        with TestClient(server.app, raise_server_exceptions=False) as client:
            response = client.post("/0/initialize", json={"num_pages": 2})
            self.assertEqual(response.status_code, 204)
            response = client.post(
                "/0/reserve_and_allocate_page_indices",
                json={"keys": [["a", ""], ["b", ""], ["c", ""]]},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json()["indices"], [[False, 1], [False, 0], [False, -1]]
            )
            response = client.post(
                "/0/confirm_write",
                json={"written_keys_to_confirm": [["a", 1]], "pages_to_release": [0]},
            )
            self.assertEqual(response.status_code, 204)
            response = client.post(
                "/0/reserve_and_allocate_page_indices", json={"keys": [["c", ""]]}
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["indices"], [[False, 0]])


if __name__ == "__main__":
    unittest.main()
