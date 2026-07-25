import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.overlap_utils import ConfidenceRelay, FutureMap
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")


_STREAM_DELAY_CYCLES = 50_000_000


def _delay_current_stream(multiplier=1):
    torch.cuda._sleep(_STREAM_DELAY_CYCLES * multiplier)


def _make_relay(device, req_pool_size=2, gamma=2):
    relay = ConfidenceRelay(
        device=device,
        req_pool_size=req_pool_size,
        pool=SimpleNamespace(
            req_generation=torch.arange(req_pool_size, dtype=torch.int64)
        ),
    )
    relay._lazy_init(torch.empty((req_pool_size, gamma), device=device))
    relay.confidence_buf.fill_(-101)
    relay.confidence_snapshot_ring.fill_(-202)
    relay.conf_ring.fill_(-303)
    return relay


@unittest.skipUnless(torch.cuda.is_available(), "CUDA or ROCm required")
class TestOverlapFutureMapStreams(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda", 0)
        self.indices = torch.arange(2, dtype=torch.int64, device=self.device)

    def test_confidence_snapshot_waits_for_publish_stream(self):
        relay = _make_relay(self.device)
        publish_stream = torch.cuda.Stream(device=self.device)
        d2h_stream = torch.cuda.Stream(device=self.device)
        expected = torch.tensor(
            [[11.0, 12.0], [13.0, 14.0]], device=self.device
        )
        setup_ready = torch.cuda.Event()
        setup_ready.record()
        publish_stream.wait_event(setup_ready)
        d2h_stream.wait_event(setup_ready)

        with torch.cuda.stream(publish_stream):
            relay.scatter(self.indices, expected)
            _delay_current_stream()
            relay.issue_ring_copy(
                stream=d2h_stream, publish_stream=publish_stream
            )

        torch.cuda.synchronize(self.device)
        self.assertTrue(torch.equal(relay.conf_ring[0], expected.cpu()))

    def test_confidence_reset_rewind_fences_scatter_and_slot_reuse(self):
        relay = _make_relay(self.device)
        first_publish_stream = torch.cuda.Stream(device=self.device)
        second_publish_stream = torch.cuda.Stream(device=self.device)
        d2h_stream = torch.cuda.Stream(device=self.device)
        first = torch.tensor([[21.0, 22.0], [23.0, 24.0]], device=self.device)
        second = torch.tensor([[31.0, 32.0], [33.0, 34.0]], device=self.device)
        setup_ready = torch.cuda.Event()
        first_scatter_ready = torch.cuda.Event()
        setup_ready.record()

        # Hold the snapshot and its D2H reader independently across the rewind.
        for stream in (first_publish_stream, second_publish_stream, d2h_stream):
            stream.wait_event(setup_ready)
        with torch.cuda.stream(d2h_stream):
            _delay_current_stream(multiplier=8)
        with torch.cuda.stream(first_publish_stream):
            relay.scatter(self.indices, first)
            first_scatter_ready.record()
            _delay_current_stream()
            relay.issue_ring_copy(
                stream=d2h_stream, publish_stream=first_publish_stream
            )

        first_host_ring = relay.conf_ring
        relay.conf_ring = torch.empty(
            first_host_ring.shape, dtype=first_host_ring.dtype, pin_memory=True
        )
        relay.reset()

        with torch.cuda.stream(second_publish_stream):
            second_publish_stream.wait_event(first_scatter_ready)
            relay.wait_for_previous_snapshot(second_publish_stream)
            relay.scatter(self.indices, second)
            relay.issue_ring_copy(
                stream=d2h_stream, publish_stream=second_publish_stream
            )

        torch.cuda.synchronize(self.device)
        self.assertTrue(torch.equal(first_host_ring[0], first.cpu()))
        self.assertTrue(torch.equal(relay.conf_ring[0], second.cpu()))

    def test_future_map_resolves_off_publish_stream_on_device(self):
        future_map = object.__new__(FutureMap)
        future_map.device = self.device
        future_map.spec_algo = SimpleNamespace(is_some=lambda: True)
        future_map.needs_cpu_seq_lens = False
        future_map.needs_confidence_relay = False
        future_map.new_seq_lens_buf = torch.full(
            (4,), -1, dtype=torch.int64, device=self.device
        )
        future_map.publish_ready = None
        future_map._publish_stream = None
        future_map._publish_fresh = False

        publish_stream = torch.cuda.Stream(device=self.device)
        resolve_stream = torch.cuda.Stream(device=self.device)
        expected = torch.tensor([41, 42], dtype=torch.int64, device=self.device)
        batch = SimpleNamespace(
            spec_info=SimpleNamespace(future_indices=self.indices),
            seq_lens=None,
        )
        setup_ready = torch.cuda.Event()
        setup_ready.record()
        publish_stream.wait_event(setup_ready)
        resolve_stream.wait_event(setup_ready)

        with torch.cuda.stream(publish_stream):
            _delay_current_stream()
            future_map.publish(self.indices, expected)
        with torch.cuda.stream(resolve_stream):
            future_map.resolve_seq_lens_device(batch)
            observed = batch.seq_lens.clone()

        torch.cuda.synchronize(self.device)
        self.assertTrue(torch.equal(observed.cpu(), expected.cpu()))


if __name__ == "__main__":
    unittest.main()
