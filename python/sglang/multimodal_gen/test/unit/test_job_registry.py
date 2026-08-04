"""Unit tests for runtime/managers/job_registry (job control)."""

import unittest
from unittest.mock import patch

from sglang.multimodal_gen.runtime.managers.job_registry import (
    _FINISHED_HARD_CAP,
    CANCELLED,
    COMPLETED,
    FAILED,
    QUEUED,
    RUNNING,
    JobRegistry,
    RequestCancelledError,
    check_current_step,
    clear_current_jobs,
    set_current_jobs,
)


class _Output:
    def __init__(
        self,
        error=None,
        cancelled=False,
        output=None,
        raw_frame_batches=None,
        trajectory_latents=None,
        rollout_trajectory_data=None,
    ):
        self.error = error
        self.cancelled = cancelled
        self.output = output
        self.raw_frame_batches = raw_frame_batches
        self.trajectory_latents = trajectory_latents
        self.rollout_trajectory_data = rollout_trajectory_data


class TestJobRegistry(unittest.TestCase):
    def test_admit_dedupes_and_replays(self):
        registry = JobRegistry()
        verdict, handle = registry.admit("a", b"c1")
        self.assertEqual(verdict, "new")
        self.assertEqual(handle.status, QUEUED)

        verdict, same = registry.admit("a", b"c2")
        self.assertEqual(verdict, "wait")
        self.assertEqual(same.waiters, [b"c2"])

        output = _Output()
        waiters = registry.finish("a", output)
        self.assertEqual(waiters, [b"c2"])
        self.assertEqual(registry.status("a")["status"], COMPLETED)

        verdict, cached = registry.admit("a", b"c3")
        self.assertEqual(verdict, "replay")
        self.assertIs(cached, output)

    def test_finish_classifies_from_output(self):
        registry = JobRegistry()
        registry.admit("ok", None)
        registry.finish("ok", _Output())
        self.assertEqual(registry.status("ok")["status"], COMPLETED)

        registry.admit("bad", None)
        registry.finish("bad", _Output(error="exploded"))
        self.assertEqual(registry.status("bad")["status"], FAILED)

        registry.admit("halted", None)
        registry.cancel("halted")
        registry.finish(
            "halted", _Output(error="request cancelled: step 3", cancelled=True)
        )
        self.assertEqual(registry.status("halted")["status"], CANCELLED)

    def test_classification_is_typed_not_text_matched(self):
        """The CANCELLED/FAILED split must come from the typed `cancelled`
        flag: matching on error text would misclassify user prompts or error
        messages that merely mention the word "cancelled"."""
        registry = JobRegistry()
        registry.admit("worded", None)
        registry.finish(
            "worded", _Output(error="upstream said: cancelled is a nice word")
        )
        self.assertEqual(registry.status("worded")["status"], FAILED)

        registry.admit("typed", None)
        registry.finish("typed", _Output(error="step aborted", cancelled=True))
        self.assertEqual(registry.status("typed")["status"], CANCELLED)

    def test_unreplayable_payload_is_dropped_but_status_kept(self):
        """Bulk terminal payloads (raw realtime frames, trajectory tensors)
        must not be pinned for the retention window: finish() keeps the
        terminal status but drops the payload, so a duplicate replays
        "not replayable" instead of the payload."""
        registry = JobRegistry()
        _, handle = registry.admit("bulky", None)
        registry.finish("bulky", _Output(raw_frame_batches=[[b"frame"]]))
        self.assertEqual(registry.status("bulky")["status"], COMPLETED)
        self.assertIsNone(handle.output)

        verdict, payload = registry.admit("bulky", b"dup")
        self.assertEqual(verdict, "replay")
        self.assertIsNone(payload)

    def test_waiters_receive_live_reply_even_when_payload_is_dropped(self):
        """Waiters attached before the terminal transition are owed the real
        first reply. Only the retained replay copy may be dropped."""
        registry = JobRegistry()
        registry.admit("bulky", b"c1")
        registry.admit("bulky", b"c2")
        waiters = registry.finish("bulky", _Output(trajectory_latents=object()))
        self.assertEqual(waiters, [b"c2"])

    def test_hard_cap_bounds_young_finished_jobs(self):
        """The TTL keeps young terminal jobs replayable, but sustained load
        must still be bounded by the hard cap on retained entries."""
        registry = JobRegistry()
        total = _FINISHED_HARD_CAP + 8
        for index in range(total):
            request_id = f"job-{index}"
            registry.admit(request_id, None)
            registry.finish(request_id, _Output())
        self.assertEqual(len(registry._finished), _FINISHED_HARD_CAP)
        self.assertEqual(registry.status("job-0")["status"], "unknown")
        self.assertEqual(registry.status(f"job-{total - 1}")["status"], COMPLETED)

    def test_late_cancel_of_successful_output_is_completed(self):
        registry = JobRegistry()
        registry.admit("late", None)
        registry.mark_running("late")
        # cancel lands after the last denoise step: the forward finished
        self.assertTrue(registry.cancel("late")["cancelled"])
        registry.finish("late", _Output())
        self.assertEqual(registry.status("late")["status"], COMPLETED)

    def test_cancel_transitions(self):
        registry = JobRegistry()
        registry.admit("job", None)
        handle = registry.mark_running("job")
        self.assertEqual(handle.status, RUNNING)
        result = registry.cancel("job")
        self.assertTrue(result["cancelled"])
        self.assertTrue(registry.is_cancelled("job"))

        registry.finish("job", _Output(error="request cancelled", cancelled=True))
        # terminal jobs are not cancellable again
        self.assertFalse(registry.cancel("job")["cancelled"])

    def test_precancel_tombstone_honored_at_admission(self):
        registry = JobRegistry()
        acked = registry.cancel("not-yet-arrived")
        self.assertTrue(acked["cancelled"])
        self.assertEqual(acked["status"], "unknown")

        verdict, payload = registry.admit("not-yet-arrived", b"c1")
        self.assertEqual(verdict, "cancelled")
        self.assertIsNone(payload)
        self.assertEqual(registry.status("not-yet-arrived")["status"], CANCELLED)

        # the scheduler records a typed tombstone via finish(): finish on an
        # already-terminal handle must still retain the payload so every later
        # duplicate replays a typed cancel instead of "not replayable"
        tombstone = _Output(error="request cancelled before dispatch", cancelled=True)
        registry.finish("not-yet-arrived", tombstone)
        verdict, payload = registry.admit("not-yet-arrived", b"c2")
        self.assertEqual(verdict, "replay")
        self.assertIs(payload, tombstone)
        self.assertTrue(payload.cancelled)

    def test_precancel_tombstone_expires(self):
        registry = JobRegistry()
        registry.cancel("stale-id")
        future = unittest.mock.MagicMock(return_value=1e12)
        with patch(
            "sglang.multimodal_gen.runtime.managers.job_registry.time.time", future
        ):
            verdict, _ = registry.admit("stale-id", b"c1")
        self.assertEqual(verdict, "new")

    def test_check_current_step_updates_progress_and_aborts(self):
        registry = JobRegistry()
        _, first = registry.admit("m1", None)
        _, second = registry.admit("m2", None)
        set_current_jobs([first, second])
        try:
            check_current_step(3, 10)
            self.assertEqual(first.step, 3)
            self.assertEqual(second.total_steps, 10)

            # a merged batch aborts only when every member is cancelled
            first.cancel_event.set()
            check_current_step(4, 10)
            second.cancel_event.set()
            with self.assertRaises(RequestCancelledError):
                check_current_step(5, 10)
        finally:
            clear_current_jobs()
        # no current jobs: the checkpoint is inert
        check_current_step(6, 10)

    def test_precancel_overflow_still_tombstones(self):
        """DELETE promises the id will not run. With the table full, cancel
        used to return cancelled=True without storing the tombstone, so the
        later submit dispatched on GPU anyway."""
        from sglang.multimodal_gen.runtime.managers.job_registry import (
            _PRECANCEL_CAP,
            JobRegistry,
        )

        registry = JobRegistry()
        for i in range(_PRECANCEL_CAP):
            registry.cancel(f"filler-{i}")
        self.assertEqual(len(registry._precancelled), _PRECANCEL_CAP)

        ack = registry.cancel("overflow")
        self.assertTrue(ack["cancelled"])
        status, _ = registry.admit("overflow", None)
        self.assertEqual(status, "cancelled")
        self.assertLessEqual(len(registry._precancelled), _PRECANCEL_CAP)


if __name__ == "__main__":
    unittest.main()
