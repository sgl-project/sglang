"""Unit tests for srt/distributed/pp_sync_window."""

import unittest

from sglang.srt.distributed.pp_sync_window import PPSyncSendWindow
from sglang.srt.distributed.vpp_layout import VPPLayout
from sglang.srt.distributed.vpp_schedule import VPPFixedSchedule, VPPTask
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeWork:
    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True


def _round(window: PPSyncSendWindow, sends: int) -> list[_FakeWork]:
    """One scheduler round: open it, then issue this round's sends."""
    window.begin_round()
    works = [_FakeWork() for _ in range(sends)]
    window.add(works)
    return works


class TestPPSyncSendWindow(CustomTestCase):
    """Sends inside the window are never waited on, so a rank is not tied to the
    next rank's progress; sends older than the window are, so the backlog stays
    bounded (the leak #28916 fixed with the single-round drain)."""

    def test_rounds_within_window_are_not_waited(self):
        window = PPSyncSendWindow(rounds=3)
        rounds = [_round(window, sends=2) for _ in range(3)]
        self.assertFalse(any(w.waited for works in rounds for w in works))
        self.assertEqual(len(window), 3)

        _round(window, sends=1)
        self.assertTrue(all(w.waited for w in rounds[0]))
        self.assertFalse(any(w.waited for works in rounds[1:] for w in works))
        self.assertEqual(len(window), 3)

    def test_window_one_is_the_single_round_drain(self):
        window = PPSyncSendWindow(rounds=1)
        first = _round(window, sends=2)
        self.assertFalse(any(w.waited for w in first))

        second = _round(window, sends=1)
        self.assertTrue(all(w.waited for w in first))
        self.assertFalse(any(w.waited for w in second))
        self.assertEqual(len(window), 1)

    def test_non_positive_depth_still_keeps_the_open_round(self):
        """A window of zero rounds would pop the round it just opened and leave
        nowhere to record the sends."""
        window = PPSyncSendWindow(rounds=0)
        first = _round(window, sends=1)
        self.assertFalse(first[0].waited)
        _round(window, sends=1)
        self.assertTrue(first[0].waited)


class TestVPPFixedSchedule(CustomTestCase):
    """FIFO properties that prevent TP lanes from choosing different forwards."""

    def test_full_wave_drains_before_new_admissions(self):
        schedule = VPPFixedSchedule(VPPLayout(8, 2), 0, max_inflight=2)
        schedule.enqueue(VPPTask(0, 0))
        schedule.enqueue(VPPTask(1, 0))
        for task in (VPPTask(0, 0), VPPTask(1, 0)):
            self.assertEqual(schedule.next_task(), task)
            schedule.record_dispatch(task)
        with self.assertRaisesRegex(RuntimeError, "window is full"):
            schedule.enqueue(VPPTask(2, 0))
        schedule.record_dispatch(VPPTask(0, 2))
        schedule.enqueue(VPPTask(2, 0))
        self.assertEqual(schedule.next_task(), VPPTask(1, 2))
        schedule.record_dispatch(VPPTask(1, 2))
        self.assertEqual(schedule.next_task(), VPPTask(2, 0))

    def test_partial_wave_drain_is_sticky_while_head_input_is_delayed(self):
        schedule = VPPFixedSchedule(VPPLayout(8, 2), 0, max_inflight=3)
        schedule.enqueue(VPPTask(0, 0))
        schedule.record_dispatch(VPPTask(0, 0))
        # Choosing second pass commits to waiting for its input. A new
        # admission must not make a slower lane switch back to first pass.
        head = schedule.next_task()
        self.assertEqual(head, VPPTask(0, 2))
        schedule.enqueue(VPPTask(1, 0))
        self.assertEqual(schedule.next_task(), head)
        schedule.record_dispatch(head)
        self.assertEqual(schedule.next_task(), VPPTask(1, 0))

    def test_remote_rank_cannot_bypass_received_head(self):
        schedule = VPPFixedSchedule(
            VPPLayout(8, 2, virtual_stages=3), 1, max_inflight=3
        )
        tasks = (VPPTask(0, 1), VPPTask(0, 3), VPPTask(0, 5))
        for task in tasks:
            schedule.enqueue(task)
        with self.assertRaisesRegex(RuntimeError, "out of FIFO order"):
            schedule.record_dispatch(tasks[1])
        for task in tasks:
            self.assertEqual(schedule.next_task(), task)
            schedule.record_dispatch(task)
        self.assertIsNone(schedule.next_task())
        with self.assertRaisesRegex(ValueError, "must increase"):
            schedule.enqueue(tasks[0])

    def test_multiple_passes_keep_wave_credit_until_final_pass(self):
        """An intermediate pass cannot release a batch's wave slot or allow
        another wave to interrupt the remaining virtual stages."""
        for virtual_stages in (3, 4):
            for physical_size in (1, 2, 4):
                with self.subTest(
                    virtual_stages=virtual_stages, physical_size=physical_size
                ):
                    schedule = VPPFixedSchedule(
                        VPPLayout(
                            physical_size * virtual_stages,
                            physical_size,
                            virtual_stages=virtual_stages,
                        ),
                        0,
                        max_inflight=2,
                    )
                    for seq in (0, 1):
                        schedule.enqueue(VPPTask(seq, 0))
                    for seq in (0, 1):
                        schedule.record_dispatch(VPPTask(seq, 0))
                    for pass_index in range(1, virtual_stages):
                        for seq in (0, 1):
                            task = VPPTask(seq, pass_index * physical_size)
                            self.assertEqual(schedule.next_task(), task)
                            schedule.record_dispatch(task)
                            if pass_index < virtual_stages - 1:
                                with self.assertRaisesRegex(
                                    RuntimeError, "window is full"
                                ):
                                    schedule.enqueue(VPPTask(2, 0))
                            elif seq == 0:
                                schedule.enqueue(VPPTask(2, 0))
                    self.assertEqual(schedule.next_task(), VPPTask(2, 0))
                    schedule.record_dispatch(VPPTask(2, 0))
                    # The new partial wave starts again at the second pass.
                    for pass_index in range(1, virtual_stages):
                        task = VPPTask(2, pass_index * physical_size)
                        self.assertEqual(schedule.next_task(), task)
                        schedule.record_dispatch(task)
                    self.assertIsNone(schedule.next_task())

    def test_partial_multi_pass_wave_stays_closed_to_new_admissions(self):
        schedule = VPPFixedSchedule(
            VPPLayout(8, 2, virtual_stages=4), 0, max_inflight=3
        )
        schedule.enqueue(VPPTask(0, 0))
        schedule.record_dispatch(VPPTask(0, 0))
        self.assertEqual(schedule.next_task(), VPPTask(0, 2))
        schedule.enqueue(VPPTask(1, 0))
        for stage_id in (2, 4, 6):
            task = VPPTask(0, stage_id)
            self.assertEqual(schedule.next_task(), task)
            schedule.record_dispatch(task)
        self.assertEqual(schedule.next_task(), VPPTask(1, 0))

    def test_single_virtual_stage_has_no_local_return_pass(self):
        schedule = VPPFixedSchedule(
            VPPLayout(4, 2, virtual_stages=1), 0, max_inflight=1
        )
        for seq in range(3):
            task = VPPTask(seq, 0)
            schedule.enqueue(task)
            self.assertEqual(schedule.next_task(), task)
            schedule.record_dispatch(task)
            self.assertIsNone(schedule.next_task())


if __name__ == "__main__":
    unittest.main()
