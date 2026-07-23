"""Unit tests for the SWA-capture completion-event handshake on
DeepSeekV4PagedHostPool (memory_pool_host).

SWA-window capture (both prefill window capture and decode-source capture) runs
its D2H on the compute/forward stream. That is same-stream-ordered vs the ring
write/overwrite (hazard H1), but a CROSS-stream consumer of the captured host
page -- restore H2D, L3 write-through, the device-landing check -- must still
order strictly after the page is fully written (hazard H2). record_capture_done
records a completion event on the compute stream and wait_capture_done awaits it.

These tests lock that gate. Cross-stream byte-exactness itself is proven on real
hardware (evidence doc device-landing check), not here.
"""

import unittest

import torch

from sglang.srt.mem_cache.memory_pool_host import DeepSeekV4PagedHostPool


def _bare_pool(gpu_device):
    """A DeepSeekV4PagedHostPool with only the attributes the capture-done event
    facility touches, avoiding the heavy host-memory allocation of __init__."""
    p = object.__new__(DeepSeekV4PagedHostPool)
    p.gpu_device = gpu_device
    p._capture_done_event = None
    return p


class TestCaptureDoneEventCPU(unittest.TestCase):
    def test_cpu_device_record_is_noop(self):
        p = _bare_pool("cpu")
        p.record_capture_done()
        self.assertIsNone(p._capture_done_event)
        p.wait_capture_done()  # must not raise

    def test_none_device_is_noop(self):
        p = _bare_pool(None)
        p.record_capture_done()
        self.assertIsNone(p._capture_done_event)
        p.wait_capture_done()


@unittest.skipUnless(torch.cuda.is_available(), "needs GPU")
class TestCaptureDoneEventGPU(unittest.TestCase):
    """The capture keeps its D2H on the compute/forward stream (no side stream);
    the completion event is created lazily and gates a cross-stream consumer."""

    def test_records_event_and_orders_cross_stream_consumer(self):
        p = _bare_pool(torch.device("cuda", 0))
        self.assertIsNone(p._capture_done_event)
        # emulate capture: non_blocking D2H on the current stream, then record.
        src = torch.ones(8, device="cuda")
        dst = torch.empty(8, device="cpu", pin_memory=True)
        dst.copy_(src, non_blocking=True)
        p.record_capture_done()
        self.assertIsInstance(p._capture_done_event, torch.cuda.Event)
        # a consumer on a DIFFERENT stream is ordered strictly after the D2H.
        consumer = torch.cuda.Stream()
        p.wait_capture_done(consumer)
        consumer.synchronize()
        self.assertTrue(bool((dst == 1).all().item()))

    def test_event_is_reused_across_calls(self):
        p = _bare_pool(torch.device("cuda", 0))
        p.record_capture_done()
        first = p._capture_done_event
        p.record_capture_done()
        self.assertIs(p._capture_done_event, first)

    def test_wait_before_any_capture_is_noop(self):
        p = _bare_pool(torch.device("cuda", 0))
        # no capture recorded yet -> no event -> wait is a no-op, must not raise.
        p.wait_capture_done()
        self.assertIsNone(p._capture_done_event)


if __name__ == "__main__":
    unittest.main()
