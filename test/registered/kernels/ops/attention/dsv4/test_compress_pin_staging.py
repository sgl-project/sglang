"""Unit tests for the c128 online prefill staging buffer in
sglang.kernels.ops.attention.dsv4.compress.

Pins two contracts the fix relies on:
- staging-buffer lifetime: cuda-graph use returns the SAME object across calls
  (so the host pointer baked into the captured H2D stays valid), while the
  eager path hands back a fresh per-call buffer;
- the write-after-read barrier that keeps the next forward's host refill behind
  the previous forward's async H2D read of that shared buffer.
"""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.compress import (
    _PIN_STAGING_CACHE,
    _PIN_STAGING_READ_DONE,
    _get_pin_staging_buffer,
    _record_pin_staging_read_done,
    _wait_prev_pin_staging_read,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestPinStagingBuffer(CustomTestCase):
    def setUp(self):
        _PIN_STAGING_CACHE.clear()

    def tearDown(self):
        _PIN_STAGING_CACHE.clear()

    def test_shape_dtype_pinned(self):
        buf = _get_pin_staging_buffer(17, torch.device("cuda:0"), persistent=False)
        self.assertEqual(tuple(buf.shape), (2, 17, 16))
        self.assertEqual(buf.dtype, torch.uint8)
        self.assertTrue(buf.is_pinned())

    def test_eager_path_allocates_per_call(self):
        # persistent=False must never cache: each call is a fresh buffer and
        # nothing is retained in the module-level cache.
        dev = torch.device("cuda:0")
        a = _get_pin_staging_buffer(8, dev, persistent=False)
        b = _get_pin_staging_buffer(8, dev, persistent=False)
        self.assertIsNot(a, b)
        self.assertEqual(len(_PIN_STAGING_CACHE), 0)

    def test_graph_path_returns_same_object(self):
        # persistent=True must return the SAME tensor for a repeated
        # (device, num_q_tokens) so the pointer captured in the CUDA graph
        # stays valid across replays.
        dev = torch.device("cuda:0")
        a = _get_pin_staging_buffer(8, dev, persistent=True)
        b = _get_pin_staging_buffer(8, dev, persistent=True)
        self.assertIs(a, b)
        self.assertEqual(tuple(a.shape), (2, 8, 16))

    def test_graph_path_keys_on_num_q_tokens(self):
        # A different bucket size is a different cache entry and object.
        dev = torch.device("cuda:0")
        a = _get_pin_staging_buffer(8, dev, persistent=True)
        c = _get_pin_staging_buffer(16, dev, persistent=True)
        self.assertIsNot(a, c)
        self.assertEqual(len(_PIN_STAGING_CACHE), 2)


class TestPinStagingReadBarrier(CustomTestCase):
    def setUp(self):
        _PIN_STAGING_READ_DONE.clear()

    def tearDown(self):
        _PIN_STAGING_READ_DONE.clear()

    def test_wait_without_event_is_noop(self):
        # No prior forward recorded for this key: the barrier is a no-op, not an
        # error.
        _wait_prev_pin_staging_read((torch.device("cuda:0"), 123))

    def test_record_then_wait_roundtrip(self):
        # Record a read-done after some stream work; the next forward's wait
        # must complete (not hang), and the event is observed done afterwards.
        key = (torch.device("cuda:0"), 32)
        torch.empty((2, 32, 16), dtype=torch.uint8, device="cuda").mul_(1)
        _record_pin_staging_read_done(key)
        self.assertIn(key, _PIN_STAGING_READ_DONE)
        _wait_prev_pin_staging_read(key)
        self.assertTrue(_PIN_STAGING_READ_DONE[key].query())

    def test_record_reuses_same_event_object(self):
        # The per-key event is created once and re-recorded, not re-allocated.
        key = (torch.device("cuda:0"), 64)
        _record_pin_staging_read_done(key)
        first = _PIN_STAGING_READ_DONE[key]
        _record_pin_staging_read_done(key)
        self.assertIs(_PIN_STAGING_READ_DONE[key], first)


if __name__ == "__main__":
    unittest.main()
