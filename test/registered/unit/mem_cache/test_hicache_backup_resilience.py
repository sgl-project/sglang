"""HiCache backup worker resilience.

The backup thread only caught ``queue.Empty``; any exception raised by the
storage backend (I/O error, timeout, malformed response) escaped the loop and
killed the daemon thread. Every subsequent backup operation then stayed in the
queue forever, its host memory was never released, and the failure was only
visible as host-pool exhaustion much later. These tests exercise the production
``backup_thread_func`` / ``_page_backup`` and assert that failed operations are
still acked and annotated with what went wrong.

Usage:
    python -m pytest test/registered/unit/mem_cache/test_hicache_backup_resilience.py -v
"""

import threading
import unittest
from queue import Queue
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

from sglang.srt.managers.cache_controller import HiCacheController, StorageOperation
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    StorageOperation as HybridStorageOperation,
)


def _make_op(num_pages: int, page_size: int = 16) -> StorageOperation:
    return StorageOperation(
        host_indices=torch.zeros(num_pages * page_size),
        token_ids=list(range(num_pages * page_size)),
        hash_value=[f"h{i}" for i in range(num_pages)],
    )


def _run_worker(cls, controller) -> threading.Thread:
    thread = threading.Thread(
        target=cls.backup_thread_func, args=(controller,), daemon=True
    )
    thread.start()
    return thread


def _stop_worker(controller, thread: threading.Thread) -> None:
    controller.storage_stop_event.set()
    controller.backup_queue.put(None)
    thread.join(timeout=5)


class TestStorageOperationFailureFields(CustomTestCase):
    def test_default_values(self):
        op = _make_op(2)
        self.assertFalse(op.failed)
        self.assertIsNone(op.failure_kind)
        self.assertEqual(op.unwritten_pages, 0)
        self.assertEqual(op.sidecar_unwritten_by_pool, {})


class TestBackupThreadExceptionHandling(CustomTestCase):
    """backup_thread_func survives exceptions and still acks the operation."""

    def _make_controller(self, page_backup_impl, backup_skip=False):
        controller = mock.MagicMock()
        controller.storage_stop_event = threading.Event()
        controller.backup_skip = backup_skip
        controller.page_size = 16
        controller.backup_queue = Queue()
        controller.ack_backup_queue = Queue()
        controller._page_backup = page_backup_impl
        return controller

    def test_exception_does_not_kill_thread(self):
        def failing_page_backup(operation):
            raise RuntimeError("storage IO error")

        controller = self._make_controller(failing_page_backup)
        op = _make_op(4)
        controller.backup_queue.put(op)

        thread = _run_worker(HiCacheController, controller)
        acked_op = controller.ack_backup_queue.get(timeout=5)
        _stop_worker(controller, thread)

        self.assertFalse(thread.is_alive())
        self.assertIs(acked_op, op)
        self.assertTrue(acked_op.failed)
        self.assertEqual(acked_op.failure_kind, "exception")
        self.assertEqual(acked_op.unwritten_pages, 4)

    def test_exception_after_partial_progress_counts_remaining_pages(self):
        def partial_then_raise(operation):
            operation.completed_tokens = 16 * 3
            raise RuntimeError("storage IO error")

        controller = self._make_controller(partial_then_raise)
        op = _make_op(4)
        controller.backup_queue.put(op)

        thread = _run_worker(HiCacheController, controller)
        acked_op = controller.ack_backup_queue.get(timeout=5)
        _stop_worker(controller, thread)

        self.assertTrue(acked_op.failed)
        self.assertEqual(acked_op.unwritten_pages, 1)

    def test_backend_false_marks_failure(self):
        controller = mock.MagicMock()
        controller.page_size = 16
        controller.backup_skip = False
        controller.page_set_func = mock.MagicMock(return_value=False)
        op = _make_op(4)

        HiCacheController._page_backup(controller, op)

        self.assertTrue(op.failed)
        self.assertEqual(op.failure_kind, "backend_false")
        self.assertEqual(op.unwritten_pages, 4)
        self.assertEqual(op.completed_tokens, 0)

    def test_backend_success_leaves_operation_clean(self):
        controller = mock.MagicMock()
        controller.page_size = 16
        controller.backup_skip = False
        controller.page_set_func = mock.MagicMock(return_value=True)
        op = _make_op(4)

        HiCacheController._page_backup(controller, op)

        self.assertFalse(op.failed)
        self.assertIsNone(op.failure_kind)
        self.assertEqual(op.unwritten_pages, 0)
        self.assertEqual(op.completed_tokens, 64)

    def test_thread_continues_after_exception(self):
        first_call = [True]

        def fail_on_first_then_succeed(operation):
            if first_call[0]:
                first_call[0] = False
                raise RuntimeError("transient error")
            operation.completed_tokens = 16 * len(operation.hash_value)

        controller = self._make_controller(fail_on_first_then_succeed)
        op1 = _make_op(2)
        op2 = _make_op(2)
        controller.backup_queue.put(op1)
        controller.backup_queue.put(op2)

        thread = _run_worker(HiCacheController, controller)
        ack1 = controller.ack_backup_queue.get(timeout=5)
        ack2 = controller.ack_backup_queue.get(timeout=5)
        _stop_worker(controller, thread)

        self.assertIs(ack1, op1)
        self.assertTrue(ack1.failed)
        self.assertEqual(ack1.failure_kind, "exception")
        self.assertIs(ack2, op2)
        self.assertFalse(ack2.failed)
        self.assertEqual(ack2.completed_tokens, 32)

    def test_backup_skip_does_not_call_page_backup(self):
        controller = self._make_controller(
            mock.MagicMock(side_effect=AssertionError("should not be called")),
            backup_skip=True,
        )
        op = _make_op(2)
        controller.backup_queue.put(op)

        thread = _run_worker(HiCacheController, controller)
        acked_op = controller.ack_backup_queue.get(timeout=5)
        _stop_worker(controller, thread)

        self.assertFalse(acked_op.failed)
        self.assertIs(acked_op, op)


class TestHybridBackupResilience(CustomTestCase):
    """HybridCacheController: worker survives, sidecar failures are recorded."""

    def _make_hybrid_op(self, num_pages: int, pool_transfers=None):
        return HybridStorageOperation(
            host_indices=torch.zeros(num_pages * 16),
            token_ids=list(range(num_pages * 16)),
            hash_value=[f"h{i}" for i in range(num_pages)],
            pool_transfers=pool_transfers,
        )

    def test_hybrid_backup_worker_survives_exception(self):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.storage_stop_event = threading.Event()
        controller.backup_skip = False
        controller.page_size = 16
        controller.backup_queue = Queue()
        controller.ack_backup_queue = Queue()
        controller._page_backup = mock.MagicMock(
            side_effect=RuntimeError("sidecar IO error")
        )
        op = self._make_hybrid_op(2)
        controller.backup_queue.put(op)

        thread = _run_worker(HybridCacheController, controller)
        acked_op = controller.ack_backup_queue.get(timeout=5)
        _stop_worker(controller, thread)

        self.assertFalse(thread.is_alive())
        self.assertIs(acked_op, op)
        self.assertTrue(op.failed)
        self.assertEqual(op.failure_kind, "exception")
        self.assertEqual(op.unwritten_pages, 2)

    def _make_sidecar_controller(self, sidecar_results):
        controller = HybridCacheController.__new__(HybridCacheController)
        controller.page_size = 16
        controller.backup_skip = False
        controller.storage_backend_type = "test"
        controller.storage_backend = mock.MagicMock()
        controller.storage_backend.batch_set_v2 = mock.MagicMock(
            return_value=sidecar_results
        )
        controller._resolve_sidecar_kv_derived_pool_transfers = lambda op: None
        controller._resolve_sidecar_nonkv_derived_pool_transfers = lambda op: None
        return controller

    def _indexer_transfer(self, num_pages: int) -> PoolTransfer:
        return PoolTransfer(
            name=PoolName.INDEXER,
            host_indices=torch.zeros(num_pages),
            keys=[f"i{i}" for i in range(num_pages)],
        )

    def test_sidecar_failure_marks_operation(self):
        # 2 failures out of 3 indexer pages; primary KV write succeeds.
        controller = self._make_sidecar_controller(
            {PoolName.INDEXER: [True, False, False]}
        )
        op = self._make_hybrid_op(3, pool_transfers=[self._indexer_transfer(3)])

        with mock.patch.object(
            HiCacheController, "_page_backup", autospec=True, return_value=None
        ):
            HybridCacheController._page_backup(controller, op)

        self.assertTrue(op.failed)
        self.assertEqual(op.failure_kind, "sidecar_backend_false")
        self.assertEqual(op.sidecar_unwritten_by_pool, {PoolName.INDEXER: 2})
        self.assertEqual(op.unwritten_pages, 0)

    def test_sidecar_success_leaves_operation_clean(self):
        controller = self._make_sidecar_controller(
            {PoolName.INDEXER: [True, True, True]}
        )
        op = self._make_hybrid_op(3, pool_transfers=[self._indexer_transfer(3)])

        with mock.patch.object(
            HiCacheController, "_page_backup", autospec=True, return_value=None
        ):
            HybridCacheController._page_backup(controller, op)

        self.assertFalse(op.failed)
        self.assertIsNone(op.failure_kind)
        self.assertEqual(op.sidecar_unwritten_by_pool, {})


if __name__ == "__main__":
    unittest.main()
