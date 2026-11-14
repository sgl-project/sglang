"""Post-verify late join on real CUDA streams.

A stream ordered by wait(), and a slot the free-list guard hands out from the run freed under a pending commit, run
after the pending side-stream commit; sync() finishes it on the host. The commit here is a delayed write on a side
stream, so a reader that is not ordered after it would see the old values.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative import post_verify_late_join as late
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# Long enough that the commit is still running when the reader is enqueued.
_SLEEP_CYCLES = 500_000_000


def _pending_commit(state: torch.Tensor) -> SimpleNamespace:
    """The graphs' late-join view of a side-stream commit that writes 1s into ``state`` after a delay."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    done = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(side):
        torch.cuda._sleep(_SLEEP_CYCLES)
        state.fill_(1)
        done.record(side)
    return SimpleNamespace(_side_stream=side, _commit_done=done, _commit_pending=True)


class TestPostVerifyLateJoinGpu(CustomTestCase):
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("needs CUDA")
        # A nonzero fill_ loads the fill kernel now (zeros is a memset); loaded lazily inside
        # _pending_commit, its first launch blocks the host until the side-stream sleep ends.
        self.state = torch.empty(1 << 20, device="cuda").fill_(-1)
        self.host = torch.empty(self.state.shape, pin_memory=True)
        self.reader = torch.cuda.Stream()
        self.start = torch.cuda.Event(enable_timing=True)

    def _read(self):
        # On the reader stream (already ordered or not by the caller): mark the start, copy the state out.
        with torch.cuda.stream(self.reader):
            self.start.record(self.reader)
            self.host.copy_(self.state, non_blocking=True)

    def _assert_read_after(self, graphs):
        torch.cuda.synchronize()
        self.assertTrue(bool((self.host == 1).all()))
        self.assertGreaterEqual(graphs._commit_done.elapsed_time(self.start), 0.0)

    def test_wait_orders_a_reader_stream_after_the_pending_commit(self):
        graphs = _pending_commit(self.state)
        with patch.object(late, "_graphs", graphs):
            self.assertFalse(graphs._commit_done.query())
            self.assertTrue(late.wait("hicache_write", stream=self.reader))
            # The forward stream still joins the commit.
            self.assertTrue(graphs._commit_pending)
            self._read()
        self._assert_read_after(graphs)

    def test_sync_finishes_the_commit_on_the_host(self):
        graphs = _pending_commit(self.state)
        with patch.object(late, "_graphs", graphs):
            self.assertTrue(late.sync("idle_pass"))
            self.assertTrue(graphs._commit_done.query())
            self.assertFalse(graphs._commit_pending)
            self.assertFalse(late.sync("idle_pass"))
        self.assertTrue(bool((self.state.cpu() == 1).all()))

    def test_a_dirty_slot_is_handed_out_after_the_pending_commit(self):
        graphs = _pending_commit(self.state)
        with patch.object(late, "_graphs", graphs):
            guard = late.FreeListGuard()
            # Three clean slots at the front; everything freed from here on may still be written.
            guard.freed(free_len_before=3)
            first_dirty = guard.taken(5)
            self.assertEqual(first_dirty, 3)
            with torch.cuda.stream(self.reader):
                guard.handing_out(first_dirty=first_dirty, index=3)
            self._read()
        self._assert_read_after(graphs)

    def test_inactive_late_join_orders_nothing(self):
        with patch.object(late, "_graphs", None):
            self.assertFalse(late.wait("hicache_write", stream=self.reader))
            self.assertFalse(late.sync("idle_pass"))
            guard = late.FreeListGuard()
            guard.freed(free_len_before=0)
            self.assertIsNone(guard.taken(4))


if __name__ == "__main__":
    unittest.main()
