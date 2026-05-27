"""Unit tests for the HiCache load-back duration metric."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestLoadBackDurationMetric(CustomTestCase):
    def setUp(self):
        from sglang.srt.managers import cache_controller as cc

        cc._timing_event_supported = None
        self.cc = cc

    def _completed_pair(self, payload_floats=1024 * 1024):
        start = self.cc.make_timing_event()
        finish = self.cc.make_timing_event()
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

    def test_loading_check_observes_duration_and_tokens(self):
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        start, finish = self._completed_pair()
        ack = self.cc.HiCacheAck(start, finish, node_ids=[1, 2], num_tokens=1024)
        stub = SimpleNamespace(
            cache_controller=SimpleNamespace(ack_load_queue=[ack]),
            ongoing_load_back={1: object(), 2: object()},
            dec_lock_ref=MagicMock(),
            metrics_collector=MagicMock(),
            pp_rank=0,
            _all_reduce=MagicMock(),
        )

        HiRadixCache.loading_check(stub)

        stub.metrics_collector.increment_load_back_num_tokens.assert_called_once_with(
            1024
        )
        stub.metrics_collector.observe_load_back_duration.assert_called_once()
        (observed,), _ = stub.metrics_collector.observe_load_back_duration.call_args
        self.assertGreater(observed, 0.0)
        self.assertEqual(stub.cache_controller.ack_load_queue, [])

    def test_loading_check_fallback_when_timing_unsupported(self):
        """On backends without enable_timing, count tokens but skip duration."""
        from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

        finish = torch.cuda.Event()
        finish.record()
        torch.cuda.synchronize()

        ack = self.cc.HiCacheAck(
            start_event=None,
            finish_event=finish,
            node_ids=[7],
            num_tokens=512,
        )
        stub = SimpleNamespace(
            cache_controller=SimpleNamespace(ack_load_queue=[ack]),
            ongoing_load_back={7: object()},
            dec_lock_ref=MagicMock(),
            metrics_collector=MagicMock(),
            pp_rank=0,
            _all_reduce=MagicMock(),
        )

        HiRadixCache.loading_check(stub)

        stub.metrics_collector.increment_load_back_num_tokens.assert_called_once_with(
            512
        )
        stub.metrics_collector.observe_load_back_duration.assert_not_called()
        self.assertEqual(stub.cache_controller.ack_load_queue, [])


if __name__ == "__main__":
    unittest.main()
