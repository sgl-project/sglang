"""Unit tests for HiCacheController incremental prefix_keys tracking across batches."""

import unittest
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.cache_controller import (
    STORAGE_BATCH_SIZE,
    HiCacheController,
    PrefetchOperation,
    StorageOperation,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHiCachePrefixKeysBatching(unittest.TestCase):
    def _make_controller(self) -> HiCacheController:
        cc = HiCacheController.__new__(HiCacheController)
        cc.page_size = 1
        cc.prefetch_sync_queue = Queue()
        cc.storage_backend = Mock()
        return cc

    def test_storage_hit_query_advances_prefix_keys_without_mutating_operation(self):
        cc = self._make_controller()
        num_batches = 3
        total_pages = STORAGE_BATCH_SIZE * num_batches
        initial_prefix = ["root_p0", "root_p1"]
        op = SimpleNamespace(
            last_hash="root_p1",
            token_ids=list(range(total_pages)),
            prefix_keys=list(initial_prefix),
            all_hash_values=None,
        )

        observed_prefix_lens = []

        def fake_batch_exists(batch_hashes, extra_info):
            observed_prefix_lens.append(len(extra_info.prefix_keys))
            return len(batch_hashes)

        cc.storage_backend.batch_exists.side_effect = fake_batch_exists
        expected_hashes = [f"h_{i}" for i in range(total_pages)]
        with patch(
            "sglang.srt.managers.cache_controller.get_storage_hash_str",
            return_value=expected_hashes,
        ):
            hits, hit_tokens = cc._storage_hit_query(op)

        self.assertEqual(hits, expected_hashes)
        self.assertEqual(hit_tokens, total_pages)
        self.assertEqual(
            observed_prefix_lens,
            [
                2,
                2 + STORAGE_BATCH_SIZE,
                2 + 2 * STORAGE_BATCH_SIZE,
            ],
        )
        self.assertEqual(op.prefix_keys, initial_prefix)

    def test_page_transfer_advances_prefix_keys_without_mutating_operation(self):
        cc = self._make_controller()
        num_batches = 3
        total_pages = STORAGE_BATCH_SIZE * num_batches
        initial_prefix = ["p_init"]
        hashes = [f"h_{i}" for i in range(total_pages)]
        op = PrefetchOperation(
            request_id="req-1",
            token_ids=list(range(total_pages)),
            last_hash="p_init",
            prefix_keys=list(initial_prefix),
        )
        op.host_indices = torch.arange(total_pages)
        op.hash_value = hashes

        observed_prefixes = []

        def fake_page_get(operation, batch_hashes, batch_host_indices, extra_info):
            observed_prefixes.append(list(extra_info.prefix_keys))
            return len(batch_hashes)

        cc.page_get_func = fake_page_get
        completed = cc._page_transfer(op)

        self.assertEqual(completed, total_pages)
        self.assertEqual(len(observed_prefixes), num_batches)
        self.assertEqual(observed_prefixes[0], initial_prefix)
        self.assertEqual(
            observed_prefixes[1], initial_prefix + hashes[:STORAGE_BATCH_SIZE]
        )
        self.assertEqual(
            observed_prefixes[2], initial_prefix + hashes[: 2 * STORAGE_BATCH_SIZE]
        )
        self.assertEqual(op.prefix_keys, initial_prefix)

    def test_page_backup_advances_prefix_keys_without_mutating_operation(self):
        cc = self._make_controller()
        num_batches = 3
        total_pages = STORAGE_BATCH_SIZE * num_batches
        initial_prefix = ["backup_p0"]
        hashes = [f"h_{i}" for i in range(total_pages)]
        op = StorageOperation(
            host_indices=torch.arange(total_pages),
            token_ids=list(range(total_pages)),
            hash_value=hashes,
            prefix_keys=list(initial_prefix),
        )

        observed_prefix_lens = []

        def fake_page_set(batch_hashes, batch_host_indices, extra_info):
            observed_prefix_lens.append(len(extra_info.prefix_keys))
            return True

        cc.page_set_func = fake_page_set
        cc._page_backup(op)

        self.assertEqual(op.completed_tokens, total_pages)
        self.assertEqual(
            observed_prefix_lens,
            [
                1,
                1 + STORAGE_BATCH_SIZE,
                1 + 2 * STORAGE_BATCH_SIZE,
            ],
        )
        self.assertEqual(op.prefix_keys, initial_prefix)


if __name__ == "__main__":
    unittest.main()
