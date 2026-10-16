"""CPU regressions for simulator partial L3 transfers and cancellation."""

import sys
import unittest
from pathlib import Path
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[4] / "tools/sglang-simulator/src")
)

from sglang_simulator.simulation.manager import StateManager
from sglang_simulator.simulation.sglang.cache_controller import C_HiCacheController
from sglang_simulator.simulation.sglang.req_stats_manager import request_stats_manager

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# The dedicated simulator CPU job installs this component's extra dependencies.
register_cpu_ci(
    est_time=5,
    suite="base-a-test-cpu",
    disabled="Run by _pr-test-simulator-cpu.yml with simulator dependencies",
)


class TestPrefetchProgress(CustomTestCase):
    def setUp(self):
        StateManager.reset()
        request_stats_manager.reset()
        self.addCleanup(StateManager.reset)
        self.addCleanup(request_stats_manager.reset)
        for name, value in (("KV_CACHE_BYTES", 1), ("DISK_READ_BANDWIDTH_BYTES", 8)):
            p = patch.object(C_HiCacheController, name, value)
            p.start()
            self.addCleanup(p.stop)

        class Controller:
            def __init__(self):
                self.enable_storage = True
                self.page_size = 4
                self.prefetch_queue = Queue()
                self.prefetch_hit_queue = Queue()
                self.released = []

            def prefetch_io_aux_func(self):
                raise AssertionError("Real IO worker must not run in the simulator")

            def terminate_prefetch(self, op):
                op.mark_terminate()
                return op.completed_tokens, op.hash_value

            def _storage_hit_query(self, op):
                return op.hash_value, len(op.host_indices)

            def append_host_mem_release(self, indices):
                self.released.extend(indices)

        C_HiCacheController.hook(Controller)
        self.controller = Controller()

    def enqueue(self, rid="request", count=8, offset=0):
        op = SimpleNamespace(
            request_id=rid,
            completed_tokens=0,
            host_indices=list(range(offset, offset + count)),
            hash_value=list(range(count // self.controller.page_size)),
            _terminated_flag=False,
        )
        op.mark_terminate = lambda: setattr(op, "_terminated_flag", True)
        self.controller.prefetch_buffer.put(op)
        return op

    def poll(self, duration):
        StateManager.set_current_inference_dur(duration)
        self.controller.handle_prefetch_operation()

    def test_fractional_progress_accumulates_without_exposing_partial_pages(self):
        op = self.enqueue()
        for i in range(1, 33):
            self.poll(1 / 32)  # 0.25 tokens per poll, 4 tokens per page.
            self.assertIsInstance(op.completed_tokens, int)
            self.assertEqual(op.completed_tokens, (i // 16) * 4)
        self.assertTrue(op._terminated_flag)
        self.assertIsNone(self.controller.chunked_prefetch_operation)

    def test_zero_budget_does_not_advance_an_inflight_transfer(self):
        op = self.enqueue()
        self.poll(0.25)  # Half a page.
        for _ in range(16):
            self.poll(0)
        self.assertEqual(op.completed_tokens, 0)
        self.poll(0.25)
        self.assertEqual(op.completed_tokens, 4)

    def test_early_stop_reports_only_complete_pages_and_releases_tail_once(self):
        for duration, expected in ((0.125, 0), (0.75, 4)):
            with self.subTest(duration=duration):
                self.controller.released.clear()
                op = self.enqueue(rid=str(duration))
                self.poll(duration)
                completed, _ = self.controller.terminate_prefetch(op)
                self.assertEqual(completed, expected)
                self.assertEqual(op.completed_tokens, expected)
                self.assertEqual(
                    request_stats_manager.get_req_stats(
                        op.request_id
                    ).final_storage_hit_len,
                    expected,
                )
                self.poll(0)
                self.poll(0)
                self.assertEqual(self.controller.released, list(range(expected, 8)))

    def test_completion_uses_remaining_budget_for_next_transfer(self):
        first = self.enqueue("first")
        second = self.enqueue("second", offset=8)
        self.poll(1.75)  # 8 tokens + 6 tokens, only 4 of the latter are usable.
        self.assertEqual(first.completed_tokens, 8)
        self.assertEqual(second.completed_tokens, 4)
        self.poll(0.25)
        self.assertEqual(second.completed_tokens, 8)
        self.assertTrue(second._terminated_flag)

    def test_real_io_worker_cannot_steal_a_pending_transfer(self):
        op = self.enqueue()
        self.controller.prefetch_io_aux_func()
        self.poll(1)
        self.assertEqual(op.completed_tokens, 8)

    def test_stop_before_host_allocation(self):
        op = self.enqueue()
        op.host_indices = None
        completed, _ = self.controller.terminate_prefetch(op)
        self.assertEqual(completed, 0)
        self.controller.append_host_mem_release(None)
        self.assertEqual(self.controller.released, [])

    def test_page_size_one_keeps_subtoken_progress_private(self):
        self.controller.page_size = 1
        op = self.enqueue(count=2)
        for i in range(1, 9):
            self.poll(1 / 32)
            self.assertEqual(op.completed_tokens, i // 4)


if __name__ == "__main__":
    unittest.main()
