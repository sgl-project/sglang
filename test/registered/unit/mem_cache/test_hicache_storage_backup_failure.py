"""Unit tests for HiCache storage backup failure handling."""

import threading
import unittest
from queue import Queue
from types import SimpleNamespace
from unittest import mock

import torch
from prometheus_client import CollectorRegistry, Counter

from sglang.srt.managers.cache_controller import (
    STORAGE_BACKUP_MAX_ATTEMPTS,
    HiCacheController,
    StorageOperation,
)
from sglang.srt.mem_cache.hicache_storage import (
    STORAGE_BATCH_SIZE,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    StorageOperation as HybridStorageOperation,
)
from sglang.srt.mem_cache.radix_cache import TreeNode
from sglang.srt.observability.metrics_collector import StorageMetricsCollector
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestHiCacheStorageBackupFailure(CustomTestCase):
    def setUp(self):
        super().setUp()
        retry_delay = mock.patch(
            "sglang.srt.managers.cache_controller.STORAGE_BACKUP_RETRY_DELAY", 0
        )
        retry_delay.start()
        self.addCleanup(retry_delay.stop)

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
        controller.storage_stop_event = threading.Event()
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
        try:
            acked_operation = controller.ack_backup_queue.get(timeout=10)
        finally:
            controller.storage_stop_event.set()
            controller.backup_queue.put(None)
            thread.join(timeout=5)

        self.assertIs(acked_operation, operation)
        self.assertTrue(acked_operation.backup_failed)
        self.assertEqual(acked_operation.completed_tokens, STORAGE_BATCH_SIZE)
        self.assertTrue(controller.ack_backup_queue.empty())

    def test_middle_batch_retry_preserves_data_and_prefix_keys(self):
        for failure in (False, TimeoutError("temporary outage")):
            with self.subTest(failure=type(failure).__name__):
                controller = self._make_controller(None)
                controller.page_size = 4
                pages = STORAGE_BATCH_SIZE * 2 + 3
                operation = StorageOperation(
                    host_indices=torch.arange(pages * 4),
                    token_ids=list(range(pages * 4)),
                    hash_value=[f"hash-{i}" for i in range(pages)],
                    prefix_keys=["parent"],
                )
                persisted = {}
                attempts = []
                failed = False

                def write(keys, indices, extra_info):
                    nonlocal failed
                    attempts.append((list(keys), list(extra_info.prefix_keys)))
                    if keys[0] == f"hash-{STORAGE_BATCH_SIZE}" and not failed:
                        failed = True
                        if isinstance(failure, Exception):
                            raise failure
                        return False
                    for index, key in enumerate(keys):
                        persisted[key] = indices[index * 4 : (index + 1) * 4].tolist()
                    return True

                controller.page_set_func = write
                self.assertTrue(controller._page_backup(operation))
                self.assertEqual(operation.completed_tokens, pages * 4)
                self.assertFalse(operation.backup_failed)
                self.assertEqual(operation.prefix_keys, ["parent"])
                self.assertEqual(attempts[1], attempts[2])
                self.assertEqual(len(attempts), 4)
                self.assertEqual(
                    persisted,
                    {
                        f"hash-{i}": list(range(i * 4, (i + 1) * 4))
                        for i in range(pages)
                    },
                )

    def test_worker_survives_terminal_failure_and_processes_next_operation(self):
        for controller_type in (HiCacheController, HybridCacheController):
            for failure in (False, OSError("backend unavailable")):
                with self.subTest(
                    controller=controller_type.__name__, failure=type(failure).__name__
                ):
                    controller = self._make_controller(
                        [failure] * STORAGE_BACKUP_MAX_ATTEMPTS + [True]
                    )
                    controller.__class__ = controller_type
                    operations = [self._make_operation(2), self._make_operation(2)]
                    for operation in operations:
                        operation.pool_transfers = []
                        controller.backup_queue.put(operation)
                    thread = threading.Thread(
                        target=controller.backup_thread_func, daemon=True
                    )
                    thread.start()
                    try:
                        first = controller.ack_backup_queue.get(timeout=5)
                        second = controller.ack_backup_queue.get(timeout=5)
                        self.assertTrue(thread.is_alive())
                    finally:
                        controller.storage_stop_event.set()
                        controller.backup_queue.put(None)
                        thread.join(timeout=5)
                    self.assertIs(first, operations[0])
                    self.assertTrue(first.backup_failed)
                    self.assertEqual(first.completed_tokens, 0)
                    self.assertIs(second, operations[1])
                    self.assertFalse(second.backup_failed)
                    self.assertEqual(second.completed_tokens, 2)
                    self.assertFalse(thread.is_alive())

    def test_ack_waits_until_write_finishes(self):
        controller = self._make_controller(None)
        started, release = threading.Event(), threading.Event()

        def write(*args):
            started.set()
            self.assertTrue(release.wait(timeout=5))
            return True

        controller.page_set_func = write
        operation = self._make_operation(2)
        controller.backup_queue.put(operation)
        thread = threading.Thread(target=controller.backup_thread_func, daemon=True)
        thread.start()
        try:
            self.assertTrue(started.wait(timeout=5))
            self.assertTrue(controller.ack_backup_queue.empty())
            release.set()
            ack = controller.ack_backup_queue.get(timeout=5)
            self.assertEqual(ack.completed_tokens, 2)
            self.assertFalse(ack.backup_failed)
        finally:
            release.set()
            controller.storage_stop_event.set()
            controller.backup_queue.put(None)
            thread.join(timeout=5)

    def test_shutdown_interrupts_retry_wait(self):
        controller = self._make_controller([False])
        operation = self._make_operation(2)

        def stop(timeout):
            controller.storage_stop_event.set()
            return True

        with mock.patch.object(controller.storage_stop_event, "wait", side_effect=stop):
            self.assertFalse(controller._page_backup(operation))
        self.assertTrue(operation.backup_failed)
        self.assertEqual(operation.completed_tokens, 0)
        self.assertEqual(controller.page_set_func.call_count, 1)

    def test_auxiliary_write_failures_are_retried_on_all_ranks(self):
        for backup_skip in (False, True):
            for terminal in (False, True):
                for failure in (
                    {"mamba": [True, False]},
                    TimeoutError("auxiliary pool"),
                ):
                    with self.subTest(
                        backup_skip=backup_skip,
                        terminal=terminal,
                        failure=type(failure).__name__,
                    ):
                        controller = self._make_controller([True])
                        controller.__class__ = HybridCacheController
                        controller.backup_skip = backup_skip
                        results = (
                            [failure] * STORAGE_BACKUP_MAX_ATTEMPTS
                            if terminal
                            else [failure, {"mamba": [True, True]}]
                        )
                        controller.storage_backend = SimpleNamespace(
                            batch_set_v2=mock.Mock(side_effect=results)
                        )
                        operation = HybridStorageOperation(
                            host_indices=torch.arange(2),
                            token_ids=[0, 1],
                            hash_value=["a", "b"],
                            pool_transfers=[
                                PoolTransfer(
                                    name=PoolName.MAMBA,
                                    keys=["a", "b"],
                                    host_indices=torch.arange(2),
                                )
                            ],
                        )
                        self.assertEqual(
                            controller._page_backup(operation), not terminal
                        )
                        self.assertEqual(operation.backup_failed, terminal)
                        self.assertEqual(
                            operation.completed_tokens, 0 if terminal else 2
                        )
                        self.assertEqual(
                            controller.page_set_func.call_count,
                            int(not terminal and not backup_skip),
                        )

    def test_replicated_primary_skip_is_not_a_failure(self):
        for controller_type in (HiCacheController, HybridCacheController):
            with self.subTest(controller=controller_type.__name__):
                controller = self._make_controller([])
                controller.__class__ = controller_type
                controller.backup_skip = True
                operation = self._make_operation(2)
                operation.pool_transfers = []
                self.assertTrue(controller._page_backup(operation))
                self.assertFalse(operation.backup_failed)
                self.assertEqual(operation.completed_tokens, 0)

    def test_terminal_ack_releases_host_lock_and_accounts_partial_backup(self):
        controller = self._make_controller(
            [True] + [False] * STORAGE_BACKUP_MAX_ATTEMPTS
        )
        controller.page_size = 4
        pages = STORAGE_BATCH_SIZE + 2
        operation = StorageOperation(
            host_indices=torch.arange(pages * 4),
            token_ids=list(range(pages * 4)),
            hash_value=[f"hash-{i}" for i in range(pages)],
        )
        node = TreeNode()
        node.protect_host()
        self.assertFalse(controller._page_backup(operation))
        self.assertEqual(node.host_ref_counter, 1)
        controller.ack_backup_queue.put(operation)
        for name in (
            "prefetch_hit_queue",
            "ack_prefetch_queue",
            "host_mem_release_queue",
        ):
            setattr(controller, name, Queue())
        registry = CollectorRegistry()
        collector = StorageMetricsCollector.__new__(StorageMetricsCollector)
        collector.labels = {"test": "backup"}
        collector.backuped_tokens_total = Counter(
            "completed", "Completed tokens", ["test"], registry=registry
        )
        collector.backup_failed_tokens_total = Counter(
            "failed", "Failed tokens", ["test"], registry=registry
        )
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.cache_controller = controller
        cache.page_size = 4
        cache.ongoing_backup = {operation.id: node}
        cache.enable_storage_metrics = True
        cache.storage_metrics_collector = collector
        for _ in range(2):
            cache._drain_storage_control_queues_impl(0, 0, None, 0, True)
        self.assertEqual(node.host_ref_counter, 0)
        self.assertEqual(cache.ongoing_backup, {})
        self.assertEqual(
            registry.get_sample_value("completed_total", collector.labels),
            STORAGE_BATCH_SIZE * 4,
        )
        self.assertEqual(registry.get_sample_value("failed_total", collector.labels), 8)


if __name__ == "__main__":
    unittest.main()
