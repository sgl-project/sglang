"""CPU-only FPM timing tests; fake events never synchronize or use a GPU."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.observability.fpm_timing import (
    capture_fpm_timing,
    wrap_forward_with_fpm,
)
from sglang.srt.utils.device_timer import DeviceTimer, _TimingInterval, device_timer_ctx
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class FakeEvent:
    def __init__(self, timestamp):
        self.timestamp = timestamp

    def elapsed_time(self, end):
        return end.timestamp - self.timestamp


class FakeInterval:
    def __init__(self, milliseconds, ready=False, start=0, stream=0):
        self.milliseconds = milliseconds
        self.ready = ready
        self.end_event = self
        self.start_event = FakeEvent(start)
        self.timestamp = start + milliseconds
        self.stream = stream
        self.observer = None
        self.metadata = None

    def end(self, metadata):
        self.metadata = metadata

    def query(self):
        return self.ready

    def elapsed_time(self):
        return self.milliseconds


class TestDeviceTimerCapture(unittest.TestCase):
    def test_unobserved_timer_keeps_original_event_recording(self):
        with (
            patch("sglang.srt.utils.device_timer.torch.cuda.Event") as event,
            patch("sglang.srt.utils.device_timer.torch.cuda.current_stream") as current,
        ):
            _TimingInterval.create()
        current.assert_not_called()
        event.assert_called_once_with(enable_timing=True)
        event.return_value.record.assert_called_once_with()

    def test_enabled_wrapper_preserves_forward_arguments_and_result(self):
        timer = DeviceTimer()
        result = SimpleNamespace()
        calls = []

        def forward(batch, *, pp_proxy_tensors):
            calls.append((batch, pp_proxy_tensors))
            return result

        wrapped = wrap_forward_with_fpm(forward, timer)
        self.assertIs(wrapped("batch", pp_proxy_tensors="proxy"), result)
        self.assertEqual(calls, [("batch", "proxy")])
        self.assertIs(wrapped.__wrapped__, forward)
        self.assertEqual(result.fpm_timing.num_intervals, 0)
        self.assertIsNone(timer._observer)

    def test_overlap_groups_do_not_share_completed_times(self):
        reporter = Mock()
        timer = DeviceTimer(reporter)
        intervals = [
            FakeInterval(2),
            FakeInterval(5, start=4),
            FakeInterval(3, start=12),
            FakeInterval(20, start=20),
        ]
        with patch.object(_TimingInterval, "create", side_effect=intervals) as create:
            with capture_fpm_timing(timer) as first:
                for stage in ("draft", "verify", "draft_extend"):
                    with device_timer_ctx(timer, stage):
                        pass
            with capture_fpm_timing(timer) as second:
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
        self.assertTrue(all(interval.observer is None for interval in intervals))
        # FPM attaches its CPU snapshot only after both groups have been drained.
        first.when_ready(first_result)
        second.when_ready(second_result)
        first_result.assert_called_once_with(0.015)
        second_result.assert_called_once_with(0.020)
        self.assertEqual(reporter.call_count, 4)
        timer._report()
        first_result.assert_called_once()
        second_result.assert_called_once()

    def test_partial_completion_waits_for_entire_closed_group(self):
        timer = DeviceTimer()
        intervals = [FakeInterval(2, ready=True), FakeInterval(5, start=4)]
        result = Mock()
        with patch.object(_TimingInterval, "create", side_effect=intervals):
            with capture_fpm_timing(timer) as timing:
                timing.when_ready(result)
                with timer.wrap({}):
                    pass
                result.assert_not_called()  # Scope still open: more stages can arrive.
                with timer.wrap({}):
                    pass
        result.assert_not_called()
        intervals[1].ready = True
        timer._report()
        result.assert_called_once_with(0.009)

    def test_different_streams_do_not_claim_an_ordered_span(self):
        timer = DeviceTimer()
        intervals = [FakeInterval(2, True, stream=1), FakeInterval(5, True, stream=2)]
        with patch.object(_TimingInterval, "create", side_effect=intervals):
            with capture_fpm_timing(timer) as timing:
                with timer.wrap({}):
                    pass
                with timer.wrap({}):
                    pass
        result = Mock()
        timing.when_ready(result)
        result.assert_called_once_with(None)

    def test_completed_time_survives_later_unscoped_work(self):
        timer = DeviceTimer()
        with patch.object(
            _TimingInterval,
            "create",
            side_effect=[FakeInterval(4, True), FakeInterval(99, True)],
        ):
            with capture_fpm_timing(timer) as timing:
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
            with capture_fpm_timing(timer) as timing:
                pass
        self.assertEqual(timing.num_intervals, 0)
        create.assert_not_called()

    def test_capture_cleans_up_after_exception(self):
        timer = DeviceTimer()
        with self.assertRaisesRegex(ValueError, "forward failed"):
            with capture_fpm_timing(timer):
                raise ValueError("forward failed")
        with capture_fpm_timing(timer) as timing:
            pass
        self.assertEqual(timing.num_intervals, 0)

    def test_zero_elapsed_interval_is_not_a_missing_interval(self):
        timer = DeviceTimer()
        with patch.object(
            _TimingInterval, "create", return_value=FakeInterval(0, True)
        ):
            with capture_fpm_timing(timer) as timing:
                with timer.wrap({}):
                    pass
        result = Mock()
        timing.when_ready(result)
        result.assert_called_once_with(0.0)
        self.assertEqual(timing.num_intervals, 1)


if __name__ == "__main__":
    unittest.main()
