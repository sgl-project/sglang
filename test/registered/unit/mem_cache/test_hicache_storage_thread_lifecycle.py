"""Unit tests for HiCache storage thread lifecycle — no server, no model loading."""

import threading
import time
import unittest
from queue import Empty
from types import SimpleNamespace

import torch

from sglang.srt.managers.cache_controller import HiCacheController
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

WORKER_NAME_PREFIX = "hicache-prefetch-io"


def _operation(request_id: str, completed_tokens: int = 0):
    op = SimpleNamespace(
        request_id=request_id,
        host_indices=torch.arange(8, dtype=torch.int64),
        completed_tokens=completed_tokens,
        terminated=False,
    )
    op.mark_terminate = lambda: setattr(op, "terminated", True)
    return op


def _live_workers():
    return [t for t in threading.enumerate() if t.name.startswith(WORKER_NAME_PREFIX)]


class _LifecycleController(HiCacheController):
    """Controller stripped down to the storage thread lifecycle.

    The real __init__ needs device/host pools and a model config; none of that is
    reachable from the lifecycle methods, so this double sets exactly the state
    _start_storage_threads / _stop_storage_threads / reset touch.
    """

    def __init__(self):
        self.enable_storage = True
        self.storage_stop_event = threading.Event()
        self.prefetch_thread = None
        self.prefetch_io_aux_thread = None
        self.backup_thread = None
        self.prefetch_queue = None
        self.prefetch_buffer = None
        self.backup_queue = None
        self.write_queue = []
        self.load_queue = []
        self.ack_write_queue = []
        self.ack_load_queue = []
        self.prefetch_tokens_occupied = 7
        self.transferred = []
        self.released = []
        self.fail_next_transfer = False

    def _idle_loop(self, queue_name):
        while not self.storage_stop_event.is_set():
            try:
                getattr(self, queue_name).get(block=True, timeout=0.05)
            except Empty:
                continue

    def prefetch_thread_func(self):
        self._idle_loop("prefetch_queue")

    def backup_thread_func(self):
        self._idle_loop("backup_queue")

    def _page_transfer(self, operation):
        if self.fail_next_transfer:
            self.fail_next_transfer = False
            raise RuntimeError("injected transfer failure")
        self.transferred.append(operation.request_id)

    def append_host_mem_release(self, host_indices):
        self.released.append(host_indices)


class TestHiCacheStorageThreadLifecycle(CustomTestCase):
    def setUp(self):
        self.controller = _LifecycleController()
        self.addCleanup(self._force_stop)
        self.workers_before = set(_live_workers())

    def _force_stop(self):
        try:
            self.controller._stop_storage_threads()
        except Exception:
            pass

    def _new_workers(self):
        return [t for t in _live_workers() if t not in self.workers_before]

    def test_reset_leaves_exactly_one_live_prefetch_io_worker(self):
        """Reset must not orphan the previous IO worker.

        The worker is only reachable through controller.prefetch_io_aux_thread. If
        reset restarts the storage threads without joining it first, the old worker
        keeps running while the attribute points at the new one, so nothing can ever
        join it and every reset adds one more thread competing for prefetch_buffer.
        """
        self.controller._start_storage_threads()
        first_worker = self.controller.prefetch_io_aux_thread
        self.assertEqual(len(self._new_workers()), 1)

        for _ in range(3):
            self.controller.reset()

        self.assertEqual(len(self._new_workers()), 1)
        self.assertFalse(first_worker.is_alive())
        self.assertIs(self._new_workers()[0], self.controller.prefetch_io_aux_thread)

    def test_reset_rebuilds_queues_and_clears_occupancy(self):
        """Reset goes through the same start path as attach.

        Reset used to re-create the threads inline instead of calling
        _start_storage_threads(), so anything added to that method (queues, a
        subclass hook) was silently missing after a reset.
        """
        self.controller._start_storage_threads()
        stale_queue = self.controller.prefetch_buffer
        stale_queue.put(_operation("stale"))

        self.controller.reset()

        self.assertIsNot(self.controller.prefetch_buffer, stale_queue)
        self.assertTrue(self.controller.prefetch_buffer.empty())
        self.assertEqual(self.controller.prefetch_tokens_occupied, 0)

    def test_worker_survives_transfer_failure_and_still_releases_memory(self):
        """A failing transfer must not kill the worker or leak host pages.

        _page_transfer used to run inside the same try that only caught Empty, so
        any other exception escaped the thread function: the single IO worker died
        and every later prefetch could only end in a timeout. The release call sat
        after it, so the failed operation's host pages were never returned either.
        """
        self.controller._start_storage_threads()
        worker = self.controller.prefetch_io_aux_thread

        self.controller.fail_next_transfer = True
        self.controller.prefetch_buffer.put(_operation("fails"))
        self.controller.prefetch_buffer.put(_operation("succeeds"))

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and len(self.controller.released) < 2:
            time.sleep(0.01)

        self.assertTrue(worker.is_alive())
        self.assertEqual(self.controller.transferred, ["succeeds"])
        self.assertEqual(len(self.controller.released), 2)

    def test_stop_joins_every_storage_thread(self):
        self.controller._start_storage_threads()
        threads = self.controller._storage_threads()
        self.assertEqual(len(threads), 3)

        self.controller._stop_storage_threads()

        self.assertEqual([t.is_alive() for t in threads], [False, False, False])


if __name__ == "__main__":
    unittest.main()
