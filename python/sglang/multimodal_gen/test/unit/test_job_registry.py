"""Unit tests for runtime/managers/job_registry (job control)."""

import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from sglang.multimodal_gen.runtime.managers.job_registry import (
    _FINISHED_HARD_CAP,
    _WAITER_CAP,
    CANCELLED,
    COMPLETED,
    FAILED,
    QUEUED,
    RUNNING,
    JobRegistry,
    RequestCancelledError,
    _value_nbytes,
    check_current_step,
    clear_current_job,
    set_current_job,
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
        output_file_paths=None,
        audio=None,
        action_pred=None,
        noise_pred=None,
    ):
        self.error = error
        self.cancelled = cancelled
        self.output = output
        self.raw_frame_batches = raw_frame_batches
        self.trajectory_latents = trajectory_latents
        self.rollout_trajectory_data = rollout_trajectory_data
        self.output_file_paths = output_file_paths
        self.audio = audio
        self.action_pred = action_pred
        self.noise_pred = noise_pred


class TestJobRegistry(unittest.TestCase):
    def test_admit_dedupes_and_replays(self):
        registry = JobRegistry()
        verdict, handle = registry.admit("a", b"c1", "fingerprint")
        self.assertEqual(verdict, "new")
        self.assertEqual(handle.status, QUEUED)

        verdict, same = registry.admit("a", b"c2", "fingerprint")
        self.assertEqual(verdict, "wait")
        self.assertEqual(same.waiters, [b"c2"])

        output = _Output()
        waiters = registry.finish("a", output)
        self.assertEqual(waiters, [b"c2"])
        self.assertEqual(registry.status("a")["status"], COMPLETED)

        verdict, cached = registry.admit("a", b"c3", "fingerprint")
        self.assertEqual(verdict, "replay")
        self.assertIs(cached, output)

    def test_reused_id_with_different_fingerprint_conflicts(self):
        registry = JobRegistry()
        verdict, _ = registry.admit("a", b"c1", "fingerprint-a")
        self.assertEqual(verdict, "new")

        verdict, payload = registry.admit("a", b"c2", "fingerprint-b")
        self.assertEqual(verdict, "conflict")
        self.assertIsNone(payload)

    def test_missing_fingerprint_never_dedupes(self):
        for first_fingerprint, retry_fingerprint in (
            (None, None),
            ("fingerprint", None),
            (None, "fingerprint"),
        ):
            with self.subTest(
                first=first_fingerprint,
                retry=retry_fingerprint,
            ):
                registry = JobRegistry()
                _, handle = registry.admit("same", b"first", first_fingerprint)
                verdict, payload = registry.admit("same", b"second", retry_fingerprint)
                self.assertEqual(verdict, "conflict")
                self.assertIsNone(payload)
                self.assertEqual(handle.waiters, [])

                registry.finish("same", _Output())
                verdict, payload = registry.admit("same", b"third", retry_fingerprint)
                self.assertEqual(verdict, "conflict")
                self.assertIsNone(payload)

    def test_duplicate_waiters_are_bounded(self):
        registry = JobRegistry()
        registry.admit("a", b"original", "fingerprint")
        for index in range(_WAITER_CAP):
            verdict, _ = registry.admit("a", str(index).encode(), "fingerprint")
            self.assertEqual(verdict, "wait")
        verdict, payload = registry.admit("a", b"overflow", "fingerprint")
        self.assertEqual(verdict, "overloaded")
        self.assertIsNone(payload)

    def test_path_backed_output_keeps_status_but_not_replay_payload(self):
        registry = JobRegistry()
        _, handle = registry.admit("path-backed", None, "fingerprint")
        registry.finish(
            "path-backed", _Output(output_file_paths=["/temporary/result.png"])
        )
        self.assertEqual(registry.status("path-backed")["status"], COMPLETED)
        self.assertIsNone(handle.output)

        verdict, payload = registry.admit("path-backed", b"duplicate", "fingerprint")
        self.assertEqual(verdict, "replay")
        self.assertIsNone(payload)

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
        _, handle = registry.admit("bulky", None, "fingerprint")
        registry.finish("bulky", _Output(raw_frame_batches=[[b"frame"]]))
        self.assertEqual(registry.status("bulky")["status"], COMPLETED)
        self.assertIsNone(handle.output)

        verdict, payload = registry.admit("bulky", b"dup", "fingerprint")
        self.assertEqual(verdict, "replay")
        self.assertIsNone(payload)

    def test_waiters_receive_live_reply_even_when_payload_is_dropped(self):
        """Waiters attached before the terminal transition are owed the real
        first reply. Only the retained replay copy may be dropped."""
        registry = JobRegistry()
        registry.admit("bulky", b"c1", "fingerprint")
        registry.admit("bulky", b"c2", "fingerprint")
        waiters = registry.finish("bulky", _Output(trajectory_latents=object()))
        self.assertEqual(waiters, [b"c2"])

    def test_live_jobs_are_bounded_and_capacity_is_released(self):
        with patch(
            "sglang.multimodal_gen.runtime.managers.job_registry._LIVE_JOB_CAP",
            2,
        ):
            registry = JobRegistry()
            registry.admit("first", b"first", "fingerprint")
            registry.admit("second", b"second", "fingerprint")

            verdict, payload = registry.admit("third", b"third", "fingerprint")
            self.assertEqual(verdict, "capacity")
            self.assertIsNone(payload)
            self.assertEqual(registry.status("third")["status"], "unknown")

            verdict, handle = registry.admit("first", b"waiter", "fingerprint")
            self.assertEqual(verdict, "wait")
            self.assertEqual(handle.waiters, [b"waiter"])

            registry.finish("first", _Output())
            verdict, _ = registry.admit("third", b"third", "fingerprint")
            self.assertEqual(verdict, "new")
            self.assertEqual(registry._live_jobs, 2)

            registry.finish("first", _Output())
            self.assertEqual(registry._live_jobs, 2)

    def test_concurrent_admission_respects_live_job_cap(self):
        with patch(
            "sglang.multimodal_gen.runtime.managers.job_registry._LIVE_JOB_CAP",
            8,
        ):
            registry = JobRegistry()
            with ThreadPoolExecutor(max_workers=16) as executor:
                results = list(
                    executor.map(
                        lambda index: registry.admit(
                            f"job-{index}", None, f"fingerprint-{index}"
                        )[0],
                        range(64),
                    )
                )

            self.assertEqual(results.count("new"), 8)
            self.assertEqual(results.count("capacity"), 56)
            self.assertEqual(registry._live_jobs, 8)

    def test_replay_payloads_are_byte_bounded(self):
        small = np.zeros(8, dtype=np.uint8)
        cap = _value_nbytes(_Output(output=small))
        self.assertIsInstance(cap, int)
        with patch(
            "sglang.multimodal_gen.runtime.managers.job_registry._REPLAY_BYTES_CAP",
            cap,
        ):
            registry = JobRegistry()
            _, first = registry.admit("first", None, "fingerprint")
            registry.finish("first", _Output(output=small))
            self.assertIsNotNone(first.output)

            _, second = registry.admit("second", None, "fingerprint")
            registry.finish("second", _Output(output=np.zeros(1, dtype=np.uint8)))
            self.assertIsNone(first.output)
            self.assertIsNotNone(second.output)
            self.assertLessEqual(registry._replay_bytes, cap)

            _, oversized = registry.admit("oversized", None, "fingerprint")
            registry.finish(
                "oversized", _Output(output=np.zeros(cap + 1, dtype=np.uint8))
            )
            self.assertIsNone(oversized.output)
            self.assertIsNotNone(second.output)
            self.assertLessEqual(registry._replay_bytes, cap)

    def test_replay_cap_counts_error_and_metadata(self):
        base_size = _value_nbytes(_Output())
        self.assertIsInstance(base_size, int)
        payloads = {
            "error": "x" * 1024,
            "raw_frame_metadata": {"blob": "x" * 1024},
            "metrics": SimpleNamespace(blob="x" * 1024),
            "metrics_list": [SimpleNamespace(blob="x" * 1024)],
        }
        for field, payload in payloads.items():
            with self.subTest(field=field), patch(
                "sglang.multimodal_gen.runtime.managers.job_registry._REPLAY_BYTES_CAP",
                base_size + 128,
            ):
                output = _Output()
                setattr(output, field, payload)
                registry = JobRegistry()
                _, handle = registry.admit(field, None, "fingerprint")
                registry.finish(field, output)
                self.assertIsNone(handle.output)
                self.assertEqual(registry._replay_bytes, 0)

    def test_replay_does_not_retain_device_tensors(self):
        payload = SimpleNamespace(device="cuda:0", nbytes=1)
        registry = JobRegistry()
        _, handle = registry.admit("device-output", None, "fingerprint")
        registry.finish("device-output", _Output(output=payload))
        self.assertIsNone(handle.output)

    def test_replay_rejects_secondary_device_payloads_and_large_containers(self):
        payload = SimpleNamespace(device="cuda:0", nbytes=1)
        for field in ("action_pred", "noise_pred"):
            with self.subTest(field=field):
                registry = JobRegistry()
                _, handle = registry.admit(field, None, "fingerprint")
                registry.finish(field, _Output(**{field: payload}))
                self.assertIsNone(handle.output)

        with patch(
            "sglang.multimodal_gen.runtime.managers.job_registry._REPLAY_BYTES_CAP",
            128,
        ):
            registry = JobRegistry()
            _, handle = registry.admit("large-list", None, "fingerprint")
            registry.finish("large-list", _Output(output=[0] * 100))
            self.assertIsNone(handle.output)

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

        verdict, payload = registry.admit("not-yet-arrived", b"c1", "fingerprint")
        self.assertEqual(verdict, "cancelled")
        self.assertIsNone(payload)
        self.assertEqual(registry.status("not-yet-arrived")["status"], CANCELLED)

        # the scheduler records a typed tombstone via finish(): finish on an
        # already-terminal handle must still retain the payload so every later
        # duplicate replays a typed cancel instead of "not replayable"
        tombstone = _Output(error="request cancelled before dispatch", cancelled=True)
        registry.finish("not-yet-arrived", tombstone)
        verdict, payload = registry.admit("not-yet-arrived", b"c2", "fingerprint")
        self.assertEqual(verdict, "replay")
        self.assertIs(payload, tombstone)
        self.assertTrue(payload.cancelled)

    def test_precancel_tombstone_expires(self):
        registry = JobRegistry()
        registry.cancel("stale-id")
        future = unittest.mock.MagicMock(return_value=1e12)
        with patch(
            "sglang.multimodal_gen.runtime.managers.job_registry.time.monotonic", future
        ):
            verdict, _ = registry.admit("stale-id", b"c1")
        self.assertEqual(verdict, "new")

    def test_check_current_step_updates_progress_and_aborts(self):
        registry = JobRegistry()
        _, handle = registry.admit("job", None)
        set_current_job(handle)
        try:
            check_current_step(3, 10)
            self.assertEqual(handle.step, 3)
            self.assertEqual(handle.total_steps, 10)

            handle.cancel_event.set()
            with self.assertRaises(RequestCancelledError):
                check_current_step(4, 10)
        finally:
            clear_current_job()
        # no current jobs: the checkpoint is inert
        check_current_step(5, 10)

    def test_precancel_overflow_preserves_acknowledged_tombstones(self):
        from sglang.multimodal_gen.runtime.managers.job_registry import (
            _PRECANCEL_CAP,
            JobRegistry,
        )

        registry = JobRegistry()
        for i in range(_PRECANCEL_CAP):
            registry.cancel(f"filler-{i}")
        self.assertEqual(len(registry._precancelled), _PRECANCEL_CAP)

        ack = registry.cancel("overflow")
        self.assertFalse(ack["cancelled"])
        self.assertTrue(ack["overloaded"])

        status, _ = registry.admit("filler-0", None)
        self.assertEqual(status, "cancelled")
        status, _ = registry.admit("overflow", None)
        self.assertEqual(status, "new")
        self.assertLessEqual(len(registry._precancelled), _PRECANCEL_CAP)

    def test_job_control_bind_failure_is_fatal(self):
        from unittest.mock import MagicMock

        import zmq

        from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.context = MagicMock()
        socket = scheduler.context.socket.return_value
        socket.bind.side_effect = zmq.ZMQError(zmq.EADDRINUSE)
        scheduler.server_args = SimpleNamespace(
            scheduler_cancel_endpoint="tcp://127.0.0.1:5601"
        )

        with self.assertRaisesRegex(RuntimeError, "job-control channel failed to bind"):
            scheduler._start_job_control()
        socket.close.assert_called_once_with(linger=0)

    def test_scheduler_returns_typed_admission_conflicts(self):
        from unittest.mock import MagicMock

        from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
        from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.jobs = JobRegistry()
        scheduler._try_return = MagicMock()

        first = Req(sampling_params=SamplingParams(prompt="first", request_id="same"))
        first.extra["job_request_fingerprint"] = "fingerprint-a"
        self.assertEqual(
            scheduler._admit_new_reqs([(b"first", first)]), [(b"first", first)]
        )

        second = Req(sampling_params=SamplingParams(prompt="second", request_id="same"))
        second.extra["job_request_fingerprint"] = "fingerprint-b"
        self.assertEqual(scheduler._admit_new_reqs([(b"second", second)]), [])
        output, identity = scheduler._try_return.call_args.args
        self.assertEqual(identity, b"second")
        self.assertTrue(output.idempotency_conflict)

        scheduler.jobs._jobs["same"].waiters = [b"waiter"] * _WAITER_CAP
        first.extra["job_request_fingerprint"] = "fingerprint-a"
        self.assertEqual(scheduler._admit_new_reqs([(b"overflow", first)]), [])
        output, identity = scheduler._try_return.call_args.args
        self.assertEqual(identity, b"overflow")
        self.assertIn("too many duplicate waiters", output.error)

    def test_scheduler_returns_typed_capacity_overload(self):
        from unittest.mock import MagicMock

        from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
        from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler
        from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req

        scheduler = Scheduler.__new__(Scheduler)
        scheduler.jobs = JobRegistry()
        scheduler._try_return = MagicMock()

        first = Req(sampling_params=SamplingParams(prompt="first", request_id="first"))
        second = Req(
            sampling_params=SamplingParams(prompt="second", request_id="second")
        )
        first.extra["job_request_fingerprint"] = "first-fingerprint"
        second.extra["job_request_fingerprint"] = "second-fingerprint"

        with patch(
            "sglang.multimodal_gen.runtime.managers.job_registry._LIVE_JOB_CAP",
            1,
        ):
            self.assertEqual(
                scheduler._admit_new_reqs([(b"first", first)]),
                [(b"first", first)],
            )
            self.assertEqual(scheduler._admit_new_reqs([(b"second", second)]), [])

        output, identity = scheduler._try_return.call_args.args
        self.assertEqual(identity, b"second")
        self.assertTrue(output.overloaded)
        self.assertFalse(output.idempotency_conflict)


if __name__ == "__main__":
    unittest.main()
