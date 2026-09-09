"""CPU-only timing ownership tests; fake events never synchronize or use a GPU."""

import unittest
from unittest.mock import Mock, patch

from sglang.srt.utils.device_timer import DeviceTimer, _TimingInterval, device_timer_ctx
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FakeInterval:
    def __init__(self, milliseconds, ready=False):
        self.milliseconds = milliseconds
        self.ready = ready
        self.end_event = self
        self.capture = None
        self.metadata = None

    def end(self, metadata):
        self.metadata = metadata

    def query(self):
        return self.ready

    def elapsed_time(self):
        return self.milliseconds


class TestDeviceTimerCapture(unittest.TestCase):
    def test_overlap_groups_do_not_share_completed_times(self):
        reporter = Mock()
        timer = DeviceTimer(reporter)
        intervals = [
            FakeInterval(2),
            FakeInterval(5),
            FakeInterval(3),
            FakeInterval(20),
        ]
        with patch.object(_TimingInterval, "create", side_effect=intervals) as create:
            with timer.capture() as first:
                for stage in ("draft", "verify", "draft_extend"):
                    with device_timer_ctx(timer, stage):
                        pass
            with timer.capture() as second:
                with device_timer_ctx(timer, "decode"):
                    pass
        # Exactly the existing events: capture adds no timing intervals.
        self.assertEqual(create.call_count, 4)
        first_result, second_result = Mock(), Mock()
        timer._report()
        first_result.assert_not_called()
        second_result.assert_not_called()
        # Both iterations become ready before CPU consumes either result.
        for interval in intervals:
            interval.ready = True
        timer._report()
        self.assertEqual(len(timer._intervals), 0)
        # FPM attaches its CPU snapshot only after both groups have been drained.
        first.when_ready(first_result)
        second.when_ready(second_result)
        first_result.assert_called_once_with(0.010)
        second_result.assert_called_once_with(0.020)
        self.assertEqual(reporter.call_count, 4)
        timer._report()
        first_result.assert_called_once()
        second_result.assert_called_once()

    def test_partial_completion_waits_for_entire_closed_group(self):
        timer = DeviceTimer()
        intervals = [FakeInterval(2, ready=True), FakeInterval(5)]
        result = Mock()
        with patch.object(_TimingInterval, "create", side_effect=intervals):
            with timer.capture() as timing:
                timing.when_ready(result)
                with timer.wrap({}):
                    pass
                result.assert_not_called()  # Scope still open: more stages can arrive.
                with timer.wrap({}):
                    pass
        result.assert_not_called()
        intervals[1].ready = True
        timer._report()
        result.assert_called_once_with(0.007)

    def test_completed_time_survives_later_unscoped_work(self):
        timer = DeviceTimer()
        with patch.object(
            _TimingInterval,
            "create",
            side_effect=[FakeInterval(4, True), FakeInterval(99, True)],
        ):
            with timer.capture() as timing:
                with timer.wrap({}):
                    pass
            with timer.wrap({}):
                pass
        result = Mock()
        timing.when_ready(result)
        result.assert_called_once_with(0.004)

    def test_empty_capture_does_not_create_gpu_events(self):
        timer = DeviceTimer()
        with patch.object(_TimingInterval, "create") as create:
            with timer.capture() as timing:
                pass
        self.assertEqual(timing.num_intervals, 0)
        create.assert_not_called()

    def test_capture_cleans_up_after_exception(self):
        timer = DeviceTimer()
        with self.assertRaisesRegex(ValueError, "forward failed"):
            with timer.capture():
                raise ValueError("forward failed")
        with timer.capture() as timing:
            pass
        self.assertEqual(timing.num_intervals, 0)

    def test_zero_elapsed_interval_is_not_a_missing_interval(self):
        timer = DeviceTimer()
        with patch.object(
            _TimingInterval, "create", return_value=FakeInterval(0, True)
        ):
            with timer.capture() as timing:
                with timer.wrap({}):
                    pass
        result = Mock()
        timing.when_ready(result)
        result.assert_called_once_with(0.0)
        self.assertEqual(timing.num_intervals, 1)


if __name__ == "__main__":
    unittest.main()
