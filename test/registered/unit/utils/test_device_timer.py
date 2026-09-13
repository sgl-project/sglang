"""CPU-only tests for the CUDA event timer state machines."""

import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

from sglang.srt.utils.device_timer import DeviceTimer, GapTimer, device_timer_ctx
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeEvent:
    def __init__(self, timestamp_ms: float, *, ready: bool = True):
        self.timestamp_ms = timestamp_ms
        self.ready = ready
        self.recorded = False

    def record(self):
        self.recorded = True

    def query(self):
        return self.ready

    def elapsed_time(self, end_event):
        return end_event.timestamp_ms - self.timestamp_ms


class _EventFactory:
    def __init__(self, timestamps: list[float], pending_indices=()):
        self.timestamps = iter(timestamps)
        self.pending_indices = set(pending_indices)
        self.events = []

    def __call__(self, *, enable_timing: bool):
        assert enable_timing
        event = _FakeEvent(
            next(self.timestamps), ready=len(self.events) not in self.pending_indices
        )
        self.events.append(event)
        return event


class TestDeviceTimer(unittest.TestCase):
    def test_context_helper_noop_and_delegation(self):
        with device_timer_ctx(None, "decode") as value:
            self.assertIsNone(value)

        timer = Mock()
        timer.wrap.return_value = nullcontext("wrapped")
        with device_timer_ctx(timer, "prefill") as value:
            self.assertEqual(value, "wrapped")
        timer.wrap.assert_called_once_with(metadata={"category": "prefill"})

    def test_reports_completed_intervals_in_fifo_order(self):
        reports = []
        second_reporter = []
        timer = DeviceTimer(lambda **values: reports.append(values))
        timer.add_reporter(lambda **values: second_reporter.append(values))
        events = _EventFactory([0.0, 50.0, 100.0, 130.0], pending_indices={1})

        with patch(
            "sglang.srt.utils.device_timer.torch.cuda.Event", side_effect=events
        ):
            with timer.wrap({"category": "prefill"}):
                pass
            with timer.wrap({"category": "decode"}):
                pass

            self.assertEqual(reports, [])
            events.events[1].ready = True
            timer._report()

        expected = [
            {"t": 0.05, "category": "prefill"},
            {"t": 0.03, "category": "decode"},
        ]
        self.assertEqual(reports, expected)
        self.assertEqual(second_reporter, expected)
        self.assertTrue(all(event.recorded for event in events.events))

    def test_wrap_cleans_up_after_exception(self):
        reports = []
        timer = DeviceTimer(lambda **values: reports.append(values))
        events = _EventFactory([10.0, 35.0, 40.0, 50.0])

        with patch(
            "sglang.srt.utils.device_timer.torch.cuda.Event", side_effect=events
        ):
            with self.assertRaisesRegex(RuntimeError, "forward failed"):
                with timer.wrap({"category": "failed"}):
                    raise RuntimeError("forward failed")

            with timer.wrap({"category": "next"}):
                pass

        self.assertEqual(
            reports,
            [
                {"t": 0.025, "category": "failed"},
                {"t": 0.01, "category": "next"},
            ],
        )

    def test_wrap_rejects_nested_contexts(self):
        reports = []
        timer = DeviceTimer(lambda **values: reports.append(values))
        events = _EventFactory([20.0, 45.0])

        with patch(
            "sglang.srt.utils.device_timer.torch.cuda.Event", side_effect=events
        ):
            with timer.wrap({"category": "outer"}):
                with self.assertRaisesRegex(AssertionError, "not re-entrant"):
                    with timer.wrap({"category": "inner"}):
                        pass

        self.assertEqual(reports, [{"t": 0.025, "category": "outer"}])


class TestGapTimer(unittest.TestCase):
    def test_reports_gaps_and_cancel_discards_pending_gap(self):
        reports = []
        timer = GapTimer(lambda **values: reports.append(values))
        events = _EventFactory([10.0, 30.0, 50.0, 80.0, 100.0, 130.0])

        with patch(
            "sglang.srt.utils.device_timer.torch.cuda.Event", side_effect=events
        ):
            with timer.wrap({"category": "first"}):
                pass
            with timer.wrap({"category": "second"}):
                pass

            timer.cancel()
            with timer.wrap({"category": "third"}):
                pass
            with timer.wrap({"category": "fourth"}):
                pass

        self.assertEqual(
            reports,
            [
                {"t": 0.02, "category": "second"},
                {"t": 0.02, "category": "fourth"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
