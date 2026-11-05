"""CPU control-flow tests; CUDA ordering is validated separately on device."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.utils import overlap_events
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestOverlapEventReuse(unittest.TestCase):
    def test_join_reuses_only_matching_stream_pair(self):
        dst = SimpleNamespace(cuda_stream=1, wait_stream=Mock(), wait_event=Mock())
        src = SimpleNamespace(cuda_stream=2)
        other = SimpleNamespace(cuda_stream=3)
        with (
            patch.object(overlap_events, "WAIT_EVENTS", True),
            patch.object(overlap_events, "_join_events", {}),
            patch("torch.cuda.Event", side_effect=lambda: Mock()) as create,
        ):
            overlap_events.wait_stream(dst, src)
            first = dst.wait_event.call_args.args[0]
            overlap_events.wait_stream(dst, src)
            self.assertIs(dst.wait_event.call_args.args[0], first)
            self.assertEqual(first.record.call_count, 2)
            first.record.assert_called_with(src)
            overlap_events.wait_stream(dst, other)
            self.assertIsNot(dst.wait_event.call_args.args[0], first)
            self.assertEqual(create.call_count, 2)
            dst.wait_stream.assert_not_called()

    def test_disabled_path_uses_native_wait_and_fresh_event(self):
        dst = Mock()
        src = Mock()
        device = SimpleNamespace(Event=Mock(side_effect=lambda: object()))
        with patch.object(overlap_events, "WAIT_EVENTS", False):
            overlap_events.wait_stream(dst, src)
            dst.wait_stream.assert_called_once_with(src)
            first = overlap_events.copy_done_event(device)
            second = overlap_events.copy_done_event(device)
            self.assertIsNot(first, second)
            self.assertEqual(device.Event.call_count, 2)

    def test_completion_ring_wraps_after_four_events(self):
        device = SimpleNamespace(Event=Mock(side_effect=lambda: object()))
        with (
            patch.object(overlap_events, "WAIT_EVENTS", True),
            patch.object(overlap_events, "_copy_done_ring", []),
            patch.object(overlap_events, "_copy_done_pos", 0),
        ):
            events = [overlap_events.copy_done_event(device) for _ in range(12)]
            self.assertEqual(device.Event.call_count, 4)
            self.assertEqual(len({id(event) for event in events[:4]}), 4)
            for index, event in enumerate(events):
                self.assertIs(event, events[index % 4])


if __name__ == "__main__":
    unittest.main()
