"""Unit tests for the HiCache load-back duration metric."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


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

    def test_loading_check_observes_duration_and_tokens(self):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        start, finish = self._completed_pair()
        ack = self.cc.HiCacheAck(
            start,
            finish,
            node_ids=[1, 2],
            num_tokens=1024,
            timing_enabled=True,
            num_tokens_by_pool={"kv": 1024},
        )
        stub = object.__new__(HiRadixCache)
        stub.cache_controller = SimpleNamespace(ack_load_queue=[ack])
        stub.ongoing_load_back = {1: object(), 2: object()}
        stub.dec_lock_ref = MagicMock()
        stub.metrics_collector = MagicMock()
        stub.pp_rank = 0
        stub._all_reduce = MagicMock()

        stub.loading_check()

        stub.metrics_collector.increment_load_back_num_tokens.assert_called_once_with(
            num_tokens=1024, pool="kv"
        )
        stub.metrics_collector.observe_load_back_duration.assert_called_once()
        (observed,), _ = stub.metrics_collector.observe_load_back_duration.call_args
        self.assertGreater(observed, 0.0)
        self.assertEqual(stub.cache_controller.ack_load_queue, [])

    def test_loading_check_fallback_when_timing_unsupported(self):
        """On backends without enable_timing, count tokens but skip duration."""
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        start = torch.cuda.Event()
        finish = torch.cuda.Event()
        start.record()
        finish.record()
        torch.cuda.synchronize()

        ack = self.cc.HiCacheAck(
            start_event=start,
            finish_event=finish,
            node_ids=[7],
            num_tokens=512,
            timing_enabled=False,
            num_tokens_by_pool={"kv": 512},
        )
        stub = object.__new__(HiRadixCache)
        stub.cache_controller = SimpleNamespace(ack_load_queue=[ack])
        stub.ongoing_load_back = {7: object()}
        stub.dec_lock_ref = MagicMock()
        stub.metrics_collector = MagicMock()
        stub.pp_rank = 0
        stub._all_reduce = MagicMock()

        stub.loading_check()

        stub.metrics_collector.increment_load_back_num_tokens.assert_called_once_with(
            num_tokens=512, pool="kv"
        )
        stub.metrics_collector.observe_load_back_duration.assert_not_called()
        self.assertEqual(stub.cache_controller.ack_load_queue, [])


if __name__ == "__main__":
    unittest.main()
