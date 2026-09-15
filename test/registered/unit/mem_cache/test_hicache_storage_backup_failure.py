"""Unit tests for HiCache storage backup failure handling."""

import threading
import unittest
from queue import Queue
from unittest import mock

import torch

from sglang.srt.managers.cache_controller import (
    STORAGE_BACKUP_MAX_ATTEMPTS,
    HiCacheController,
    StorageOperation,
)
from sglang.srt.mem_cache.hicache_storage import STORAGE_BATCH_SIZE
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHiCacheStorageBackupFailure(CustomTestCase):
    def _make_controller(self, side_effect):
        controller = HiCacheController.__new__(HiCacheController)
        controller.page_size = 1
        controller.backup_skip = False
        controller.page_set_func = mock.Mock(side_effect=side_effect)
        controller.backup_queue = Queue()
        controller.ack_backup_queue = Queue()
        controller.storage_stop_event = threading.Event()
        return controller

    def _make_operation(self, num_pages):
        return StorageOperation(
            host_indices=torch.arange(num_pages, dtype=torch.int64),
            token_ids=list(range(num_pages)),
            hash_value=[f"hash-{i}" for i in range(num_pages)],
        )

    def test_all_batches_success_completes_operation(self):
        num_pages = STORAGE_BATCH_SIZE * 2 + 3
        controller = self._make_controller([True, True, True])
        operation = self._make_operation(num_pages)

        success = controller._page_backup(operation)

        self.assertTrue(success)
        self.assertFalse(operation.backup_failed)
        self.assertEqual(operation.completed_tokens, num_pages)
        self.assertEqual(controller.page_set_func.call_count, 3)

    def test_failed_batch_retries_and_marks_terminal_failure(self):
        num_pages = STORAGE_BATCH_SIZE * 2 + 3
        controller = self._make_controller(
            [True] + [False] * STORAGE_BACKUP_MAX_ATTEMPTS
        )
        operation = self._make_operation(num_pages)

        success = controller._page_backup(operation)

        self.assertFalse(success)
        self.assertTrue(operation.backup_failed)
        self.assertEqual(operation.completed_tokens, STORAGE_BATCH_SIZE)
        self.assertEqual(
            controller.page_set_func.call_count, 1 + STORAGE_BACKUP_MAX_ATTEMPTS
        )

    def test_transient_failure_is_retried_successfully(self):
        num_pages = STORAGE_BATCH_SIZE * 2 + 3
        controller = self._make_controller([False, True, True, True])
        operation = self._make_operation(num_pages)

        success = controller._page_backup(operation)

        self.assertTrue(success)
        self.assertFalse(operation.backup_failed)
        self.assertEqual(operation.completed_tokens, num_pages)
        self.assertEqual(controller.page_set_func.call_count, 4)

    def test_hybrid_controller_propagates_backup_failure(self):
        num_pages = STORAGE_BATCH_SIZE + 1
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.page_size = 1
        controller.backup_skip = False
        controller.page_set_func = mock.Mock(
            side_effect=[True] + [False] * STORAGE_BACKUP_MAX_ATTEMPTS
        )
        operation = self._make_operation(num_pages)
        operation.pool_transfers = []

        success = controller._page_backup(operation)

        self.assertFalse(success)
        self.assertTrue(operation.backup_failed)
        self.assertEqual(operation.completed_tokens, STORAGE_BATCH_SIZE)
        self.assertEqual(
            controller.page_set_func.call_count, 1 + STORAGE_BACKUP_MAX_ATTEMPTS
        )

    def test_backup_ack_exposes_terminal_failure(self):
        num_pages = STORAGE_BATCH_SIZE + 1
        controller = self._make_controller(
            [True] + [False] * STORAGE_BACKUP_MAX_ATTEMPTS
        )
        operation = self._make_operation(num_pages)
        controller.backup_queue.put(operation)

        thread = threading.Thread(target=controller.backup_thread_func, daemon=True)
        thread.start()
        acked_operation = controller.ack_backup_queue.get(timeout=10)
        controller.storage_stop_event.set()
        thread.join(timeout=5)

        self.assertIs(acked_operation, operation)
        self.assertTrue(acked_operation.backup_failed)
        self.assertEqual(acked_operation.completed_tokens, STORAGE_BATCH_SIZE)
        self.assertTrue(controller.ack_backup_queue.empty())


if __name__ == "__main__":
    unittest.main()
