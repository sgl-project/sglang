"""Unit tests for HiCache prefetch I/O worker concurrency.

Tests cover:
- _parse_prefetch_io_workers() input validation
- _prefetch_io_worker_count() thread count resolution
- prefetch_io_aux_func() exception handling and normal completion
- _stop_storage_threads() multi-worker sentinel and join behavior
- _release_prefetch_remainder() memory release on failure and normal completion
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import threading
import time
import unittest
from queue import Queue
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation


def make_controller():
    """Create a HiCacheController without calling __init__ (avoids GPU deps)."""
    return HiCacheController.__new__(HiCacheController)


class TestParsePrefetchIoWorkers(unittest.TestCase):
    """Tests for _parse_prefetch_io_workers()."""

    def setUp(self):
        self.controller = make_controller()

    def test_valid_positive_integers(self):
        for value in [1, 2, 4, 8, 100]:
            with self.subTest(value=value):
                result = self.controller._parse_prefetch_io_workers(value)
                self.assertEqual(result, value)

    def test_bool_rejected(self):
        for value in [True, False]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as ctx:
                    self.controller._parse_prefetch_io_workers(value)
                self.assertIn("must be a positive integer", str(ctx.exception))

    def test_zero_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.controller._parse_prefetch_io_workers(0)
        self.assertIn("must be >= 1", str(ctx.exception))

    def test_negative_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.controller._parse_prefetch_io_workers(-1)
        self.assertIn("must be >= 1", str(ctx.exception))

    def test_float_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.controller._parse_prefetch_io_workers(1.5)
        self.assertIn("must be a positive integer", str(ctx.exception))

    def test_string_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.controller._parse_prefetch_io_workers("2")
        self.assertIn("must be a positive integer", str(ctx.exception))

    def test_none_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.controller._parse_prefetch_io_workers(None)
        self.assertIn("must be a positive integer", str(ctx.exception))


class TestPrefetchIoWorkerCount(unittest.TestCase):
    """Tests for _prefetch_io_worker_count()."""

    def setUp(self):
        self.controller = make_controller()

    def test_returns_thread_list_length_when_non_empty(self):
        self.controller.prefetch_io_aux_threads = [
            MagicMock(),
            MagicMock(),
            MagicMock(),
        ]
        self.assertEqual(self.controller._prefetch_io_worker_count(), 3)

    def test_falls_back_to_prefetch_io_workers_when_empty(self):
        self.controller.prefetch_io_aux_threads = []
        self.controller.prefetch_io_workers = 4
        self.assertEqual(self.controller._prefetch_io_worker_count(), 4)

    def test_defaults_to_one_when_neither_set(self):
        # Neither attribute is set on the bare __new__ instance.
        # getattr defaults: [] and 1.
        self.assertEqual(self.controller._prefetch_io_worker_count(), 1)

    def test_empty_list_with_workers_set(self):
        self.controller.prefetch_io_aux_threads = []
        self.controller.prefetch_io_workers = 2
        self.assertEqual(self.controller._prefetch_io_worker_count(), 2)


class TestPrefetchIoAuxFunc(unittest.TestCase):
    """Tests for prefetch_io_aux_func() exception handling."""

    def setUp(self):
        self.controller = make_controller()
        self.controller.storage_stop_event = threading.Event()
        self.controller.prefetch_buffer = Queue()

    def _run_and_join(self, timeout=10):
        """Run prefetch_io_aux_func in a thread, then signal stop and join."""
        thread = threading.Thread(
            target=self.controller.prefetch_io_aux_func, daemon=True
        )
        thread.start()
        # Give the worker time to process queued items.
        time.sleep(0.5)
        self.controller.storage_stop_event.set()
        thread.join(timeout=timeout)
        self.assertFalse(thread.is_alive(), "Worker thread did not stop in time.")

    def test_normal_completion_releases_remainder(self):
        self.controller._page_transfer = MagicMock()
        self.controller._release_prefetch_remainder = MagicMock()

        op = PrefetchOperation("req-normal", [1, 2, 3])
        op.host_indices = torch.tensor([0, 1, 2])
        self.controller.prefetch_buffer.put(op)

        self._run_and_join()

        self.controller._page_transfer.assert_called_once_with(op)
        self.controller._release_prefetch_remainder.assert_called_once_with(op)
        self.assertFalse(op.is_terminated())

    def test_exception_marks_terminate_and_releases(self):
        self.controller._page_transfer = MagicMock(
            side_effect=RuntimeError("simulated transfer failure")
        )
        self.controller._release_prefetch_remainder = MagicMock()

        op = PrefetchOperation("req-fail", [1, 2, 3])
        op.host_indices = torch.tensor([0, 1, 2])
        self.controller.prefetch_buffer.put(op)

        self._run_and_join()

        self.controller._page_transfer.assert_called_once_with(op)
        self.assertTrue(op.is_terminated())
        self.controller._release_prefetch_remainder.assert_called_once_with(op)

    def test_none_sentinel_does_not_crash(self):
        self.controller._page_transfer = MagicMock()
        self.controller._release_prefetch_remainder = MagicMock()

        self.controller.prefetch_buffer.put(None)

        self._run_and_join()

        self.controller._page_transfer.assert_not_called()
        self.controller._release_prefetch_remainder.assert_not_called()

    def test_worker_continues_after_exception(self):
        """Worker should process subsequent operations after one fails."""
        call_count = [0]

        def side_effect(op):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first operation fails")

        self.controller._page_transfer = MagicMock(side_effect=side_effect)
        self.controller._release_prefetch_remainder = MagicMock()

        op1 = PrefetchOperation("req-fail", [1, 2])
        op1.host_indices = torch.tensor([0, 1])
        op2 = PrefetchOperation("req-ok", [3, 4])
        op2.host_indices = torch.tensor([2, 3])

        self.controller.prefetch_buffer.put(op1)
        self.controller.prefetch_buffer.put(op2)

        self._run_and_join(timeout=15)

        self.assertEqual(self.controller._page_transfer.call_count, 2)
        self.assertTrue(op1.is_terminated())
        self.assertFalse(op2.is_terminated())


class TestStopStorageThreadsMultiWorker(unittest.TestCase):
    """Tests for _stop_storage_threads() with multiple IO workers."""

    def setUp(self):
        self.controller = make_controller()
        self.controller.storage_stop_event = threading.Event()

    def _make_mock_thread(self):
        t = MagicMock()
        t.is_alive.return_value = False
        return t

    def _setup_controller(self, num_workers):
        """Set up a controller with the given number of mock IO workers."""
        mock_threads = [self._make_mock_thread() for _ in range(num_workers)]
        self.controller.prefetch_io_aux_threads = mock_threads
        self.controller.prefetch_io_aux_thread = mock_threads[0]
        self.controller.prefetch_queue = Queue()
        self.controller.backup_queue = Queue()
        self.controller.prefetch_buffer = Queue()
        self.controller.prefetch_thread = self._make_mock_thread()
        self.controller.backup_thread = self._make_mock_thread()

    def test_sends_correct_number_of_sentinels(self):
        num_workers = 3
        self._setup_controller(num_workers)

        self.controller._stop_storage_threads()

        sentinel_count = 0
        while not self.controller.prefetch_buffer.empty():
            item = self.controller.prefetch_buffer.get_nowait()
            if item is None:
                sentinel_count += 1
        self.assertEqual(sentinel_count, num_workers)

    def test_single_worker_sends_one_sentinel(self):
        self._setup_controller(1)

        self.controller._stop_storage_threads()

        sentinel_count = 0
        while not self.controller.prefetch_buffer.empty():
            item = self.controller.prefetch_buffer.get_nowait()
            if item is None:
                sentinel_count += 1
        self.assertEqual(sentinel_count, 1)

    def test_clears_prefetch_io_aux_threads(self):
        self._setup_controller(2)

        self.controller._stop_storage_threads()

        self.assertEqual(self.controller.prefetch_io_aux_threads, [])

    def test_joins_all_io_worker_threads(self):
        num_workers = 4
        mock_threads = [self._make_mock_thread() for _ in range(num_workers)]
        self.controller.prefetch_io_aux_threads = mock_threads
        self.controller.prefetch_io_aux_thread = mock_threads[0]
        self.controller.prefetch_queue = Queue()
        self.controller.backup_queue = Queue()
        self.controller.prefetch_buffer = Queue()
        self.controller.prefetch_thread = self._make_mock_thread()
        self.controller.backup_thread = self._make_mock_thread()

        self.controller._stop_storage_threads()

        for t in mock_threads:
            t.join.assert_called_once_with(timeout=10)

    def test_raises_runtime_error_if_thread_still_alive(self):
        self._setup_controller(2)
        # Make one thread appear still alive after join.
        self.controller.prefetch_io_aux_threads[0].is_alive.return_value = True

        with self.assertRaises(RuntimeError) as ctx:
            self.controller._stop_storage_threads()
        self.assertIn("Failed to stop", str(ctx.exception))


class TestReleasePrefetchRemainder(unittest.TestCase):
    """Tests for _release_prefetch_remainder()."""

    def setUp(self):
        self.controller = make_controller()
        self.controller.host_mem_release_queue = Queue()
        self.controller.mem_pool_host = MagicMock()
        self.controller.mem_pool_host.page_size = 1

    def test_releases_unfilled_host_memory(self):
        op = PrefetchOperation("req-1", [1, 2, 3, 4])
        op.host_indices = torch.tensor([10, 20, 30, 40])
        op.completed_tokens = 2

        self.controller._release_prefetch_remainder(op)

        released = []
        while not self.controller.host_mem_release_queue.empty():
            released.append(self.controller.host_mem_release_queue.get_nowait())
        self.assertEqual(len(released), 2)
        self.assertTrue(torch.equal(released[0], torch.tensor([30])))
        self.assertTrue(torch.equal(released[1], torch.tensor([40])))

    def test_no_release_when_all_completed(self):
        op = PrefetchOperation("req-1", [1, 2])
        op.host_indices = torch.tensor([10, 20])
        op.completed_tokens = 2

        self.controller._release_prefetch_remainder(op)

        self.assertTrue(self.controller.host_mem_release_queue.empty())

    def test_logs_but_does_not_raise_on_failure(self):
        op = PrefetchOperation("req-1", [1, 2])
        # host_indices is None by default from PrefetchOperation.__init__,
        # which will cause a TypeError when slicing.
        self.assertIsNone(op.host_indices)

        with patch("sglang.srt.managers.cache_controller.logger") as mock_logger:
            self.controller._release_prefetch_remainder(op)
            mock_logger.exception.assert_called_once()


if __name__ == "__main__":
    unittest.main()
