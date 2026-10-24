"""The overlap scheduler's schedule stream: distinct from the forward stream
by default, the forward stream itself with SGLANG_SHARE_SCHEDULE_FORWARD_STREAM."""

import itertools
import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.managers.utils import select_schedule_stream
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeDeviceModule:
    """Hands out streams from a fixed handle sequence, like a round-robin pool."""

    def __init__(self, handles):
        self._handles = itertools.cycle(handles)
        self.drawn = 0

    def Stream(self, priority=0):
        self.drawn += 1
        return SimpleNamespace(cuda_stream=next(self._handles))


class TestSelectScheduleStream(CustomTestCase):
    def test_keeps_a_distinct_schedule_stream(self):
        device = _FakeDeviceModule([3])
        schedule, forward = (
            SimpleNamespace(cuda_stream=2),
            SimpleNamespace(cuda_stream=1),
        )
        chosen = select_schedule_stream(
            device, schedule, forward, share_forward_stream=False
        )
        self.assertIs(chosen, schedule)
        self.assertEqual(device.drawn, 0)

    def test_redraws_an_aliasing_schedule_stream(self):
        device = _FakeDeviceModule([1, 1, 4])
        schedule, forward = (
            SimpleNamespace(cuda_stream=1),
            SimpleNamespace(cuda_stream=1),
        )
        chosen = select_schedule_stream(
            device, schedule, forward, share_forward_stream=False
        )
        self.assertEqual(chosen.cuda_stream, 4)
        self.assertEqual(device.drawn, 3)

    def test_shares_the_forward_stream_when_asked(self):
        device = _FakeDeviceModule([3])
        schedule, forward = (
            SimpleNamespace(cuda_stream=2),
            SimpleNamespace(cuda_stream=1),
        )
        chosen = select_schedule_stream(
            device, schedule, forward, share_forward_stream=True
        )
        self.assertIs(chosen, forward)
        self.assertEqual(device.drawn, 0)

    def test_sharing_is_off_by_default(self):
        self.assertFalse(envs.SGLANG_SHARE_SCHEDULE_FORWARD_STREAM.get())


if __name__ == "__main__":
    unittest.main()
