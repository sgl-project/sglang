"""Unit tests for managers/scheduler_input_blocker.py — no server, no model loading."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cpu_ci(est_time=6, suite="base-b-test-cpu")

import unittest
from unittest.mock import MagicMock, patch

from sglang.srt.managers.io_struct import BlockReqInput, BlockReqType
from sglang.srt.managers.scheduler_input_blocker import (
    SchedulerInputBlocker,
    _State,
    input_blocker_guard_region,
)
from sglang.test.test_utils import CustomTestCase

_BARRIER_PATH = "sglang.srt.managers.scheduler_input_blocker.PollBasedBarrier"


def _block():
    return BlockReqInput(type=BlockReqType.BLOCK)


def _unblock():
    return BlockReqInput(type=BlockReqType.UNBLOCK)


class TestSchedulerInputBlocker(CustomTestCase):
    def setUp(self):
        self._barrier_patch = patch(_BARRIER_PATH)
        self.mock_barrier_cls = self._barrier_patch.start()
        self.addCleanup(self._barrier_patch.stop)

    def _make_blocker(self, noop=False):
        self.mock_barrier_cls.return_value.poll_global_arrived.return_value = False
        return SchedulerInputBlocker(noop=noop)

    def _arrive(self, blocker):
        blocker._global_unblock_barrier.poll_global_arrived.return_value = True

    def test_initial_state_is_unblocked(self):
        blocker = self._make_blocker()
        self.assertEqual(blocker._state, _State.UNBLOCKED)

    def test_plain_request_passes_through_when_unblocked(self):
        blocker = self._make_blocker()
        result = blocker.handle([{"req": 1}])
        self.assertEqual(result, [{"req": 1}])

    def test_block_req_changes_state_to_blocked(self):
        blocker = self._make_blocker()
        blocker.handle([_block()])
        self.assertEqual(blocker._state, _State.BLOCKED)

    def test_requests_are_queued_when_blocked(self):
        blocker = self._make_blocker()
        blocker.handle([_block()])
        result = blocker.handle([{"req": 1}, {"req": 2}])
        self.assertEqual(result, [])
        self.assertEqual(len(blocker._pending_reqs), 2)

    def test_double_block_asserts(self):
        blocker = self._make_blocker()
        blocker.handle([_block()])
        with self.assertRaises(AssertionError):
            blocker.handle([_block()])

    def test_unblock_without_block_asserts(self):
        blocker = self._make_blocker()
        with self.assertRaises(AssertionError):
            blocker.handle([_unblock()])

    def test_unblock_transitions_to_barrier_state(self):
        blocker = self._make_blocker()
        blocker.handle([_block()])
        blocker.handle([_unblock()])
        self.assertEqual(blocker._state, _State.GLOBAL_UNBLOCK_BARRIER)

    def test_unblock_calls_local_arrive(self):
        blocker = self._make_blocker()
        blocker.handle([_block()])
        blocker.handle([_unblock()])
        self.mock_barrier_cls.return_value.local_arrive.assert_called_once()

    def test_barrier_not_arrived_does_not_drain(self):
        blocker = self._make_blocker()
        blocker.handle([_block()])
        blocker.handle([_unblock()])
        blocker.handle([{"req": 1}])
        self.assertEqual(blocker._state, _State.GLOBAL_UNBLOCK_BARRIER)
        self.assertEqual(len(blocker._pending_reqs), 1)

    def test_barrier_arrived_drains_pending_requests(self):
        blocker = self._make_blocker()
        blocker.handle([_block()])
        blocker.handle([_unblock()])
        blocker.handle([{"req": 1}])
        self._arrive(blocker)
        result = blocker.handle([])
        self.assertEqual(result, [{"req": 1}])

    def test_barrier_arrived_resets_state_to_unblocked(self):
        blocker = self._make_blocker()
        blocker.handle([_block()])
        blocker.handle([_unblock()])
        self._arrive(blocker)
        blocker.handle([])
        self.assertEqual(blocker._state, _State.UNBLOCKED)

    def test_full_cycle_restores_normal_operation(self):
        blocker = self._make_blocker()
        blocker.handle([_block()])
        blocker.handle([_unblock()])
        self._arrive(blocker)
        blocker.handle([])
        result = blocker.handle([{"req": 99}])
        self.assertEqual(result, [{"req": 99}])

    def test_noop_accepts_none(self):
        blocker = self._make_blocker(noop=True)
        blocker.handle(None)

    def test_noop_rejects_non_none(self):
        blocker = self._make_blocker(noop=True)
        with self.assertRaises(AssertionError):
            blocker.handle([{"req": 1}])

    def test_noop_still_polls_barrier(self):
        blocker = self._make_blocker(noop=True)
        blocker.handle(None)
        self.mock_barrier_cls.return_value.poll_global_arrived.assert_called_once()

    def test_pending_reqs_preserve_insertion_order(self):
        blocker = self._make_blocker()
        blocker.handle([_block()])
        blocker.handle(["a", "b", "c"])
        blocker.handle([_unblock()])
        self._arrive(blocker)
        result = blocker.handle([])
        self.assertEqual(result, ["a", "b", "c"])


class TestInputBlockerGuardRegion(CustomTestCase):
    def test_sends_block_before_yield_and_unblock_after(self):
        send_to_scheduler = MagicMock()
        with input_blocker_guard_region(send_to_scheduler):
            pass
        self.assertEqual(
            send_to_scheduler.send_pyobj.mock_calls,
            [unittest.mock.call(_block()), unittest.mock.call(_unblock())],
        )

    def test_sends_unblock_even_on_exception(self):
        send_to_scheduler = MagicMock()
        with self.assertRaises(ValueError):
            with input_blocker_guard_region(send_to_scheduler):
                raise ValueError("boom")
        self.assertEqual(
            send_to_scheduler.send_pyobj.mock_calls,
            [unittest.mock.call(_block()), unittest.mock.call(_unblock())],
        )


if __name__ == "__main__":
    unittest.main()
