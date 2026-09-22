"""Unit tests for the HiCache load-back duration metric."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=9, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestLoadBackDurationMetric(CustomTestCase):
    def setUp(self):
        from sglang.srt.managers import cache_controller as cc
        from sglang.srt.mem_cache import l2_transfer as transfer

        transfer._timing_events_supported.cache_clear()
        self.cc = cc
        self.transfer = transfer

    def _completed_pair(self, payload_floats=1024 * 1024):
        start, finish, timing_enabled = self.transfer.make_timing_event_pair()
        self.assertTrue(timing_enabled)
        stream = torch.cuda.Stream()
        start.record()
        with torch.cuda.stream(stream):
            start.wait(stream)
            torch.empty(payload_floats, device="cuda").fill_(0)
            finish.record()
        torch.cuda.synchronize()
        return start, finish

    def test_elapsed_time_works(self):
        start, finish = self._completed_pair()
        self.assertGreater(start.elapsed_time(finish), 0.0)

    def test_timing_fallback_uses_dedicated_events(self):
        events = []

        def create_event(*, enable_timing=False):
            if enable_timing:
                raise TypeError
            event = MagicMock()
            events.append(event)
            return event

        with patch.object(
            self.transfer.device_module, "Event", side_effect=create_event
        ):
            self.transfer._timing_events_supported.cache_clear()
            start, finish, timing_enabled = self.transfer.make_timing_event_pair()

        self.assertFalse(timing_enabled)
        self.assertIs(start, events[0])
        self.assertIs(finish, events[1])
        self.assertIsNot(start, finish)


if __name__ == "__main__":
    unittest.main()
