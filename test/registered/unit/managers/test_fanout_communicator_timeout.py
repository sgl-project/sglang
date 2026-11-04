"""Deadline regressions for FanOutCommunicator.queueing_call (issue #45512).

A silent scheduler rank must not leave a queueing control call waiting
without a deadline: the call fails fast with FanOutTimeoutError, while the
slot stays owned by the timed-out call until its rank replies drain, so a
late reply can never be consumed by -- or complete -- a later call.
"""

import asyncio
import unittest

from sglang.srt.managers.communicator import (
    DEFAULT_QUEUEING_TIMEOUT,
    FanOutCommunicator,
    FanOutTimeoutError,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQueueingCallTimeout(CustomTestCase):
    def test_default_timeout_is_explicit(self):
        # The deadline is on by default; None opts back into the legacy
        # unbounded wait.
        self.assertEqual(DEFAULT_QUEUEING_TIMEOUT, 300.0)
        comm = FanOutCommunicator(send=lambda obj: None, fan_out=1)
        self.assertEqual(comm._queueing_timeout, 300.0)
        comm = FanOutCommunicator(
            send=lambda obj: None, fan_out=1, queueing_timeout=None
        )
        self.assertIsNone(comm._queueing_timeout)

    def test_silent_rank_fails_fast_and_slot_frees_after_drain(self):
        """fan-out 2, only one rank replies: the first call raises
        FanOutTimeoutError at the deadline (not partial results). The slot
        stays owned by the timed-out call, so a later caller cannot dispatch
        until the late reply drains -- and that late reply is consumed by the
        orphaned collector, never by the later call."""

        async def scenario():
            sent = []
            comm = FanOutCommunicator(
                send=sent.append, fan_out=2, mode="queueing", queueing_timeout=1.0
            )

            first = asyncio.create_task(comm("A"))
            await asyncio.sleep(0)
            comm.handle_recv("A-0")  # only rank 0 replies; rank 1 is silent

            start = asyncio.get_running_loop().time()
            with self.assertRaises(FanOutTimeoutError):
                await asyncio.wait_for(first, timeout=10)
            elapsed = asyncio.get_running_loop().time() - start
            # Fails fast at the deadline instead of hanging without one ...
            self.assertLess(elapsed, 10)
            # ... and partial results are never reported as success.
            self.assertTrue(first.done() and first.cancelled() is False)

            # A later caller cannot dispatch while the orphaned slot drains.
            second = asyncio.create_task(comm("B"))
            await asyncio.sleep(0.2)
            self.assertEqual(sent, ["A"])
            self.assertFalse(second.done())

            # The silent rank recovers: its late reply drains into the
            # orphaned collector, freeing the slot for the next call.
            comm.handle_recv("A-1")
            for _ in range(1000):
                if sent == ["A", "B"]:
                    break
                await asyncio.sleep(0)
            self.assertEqual(sent, ["A", "B"])

            # The late A-1 reply must not leak into B's results.
            comm.handle_recv("B-0")
            comm.handle_recv("B-1")
            self.assertEqual(await asyncio.wait_for(second, timeout=10), ["B-0", "B-1"])

        asyncio.run(scenario())

    def test_timeout_before_dispatch_never_sends(self):
        """A caller that times out while queued behind an abandoned slot
        never dispatches its request."""

        async def scenario():
            sent = []
            comm = FanOutCommunicator(
                send=sent.append, fan_out=1, mode="queueing", queueing_timeout=0.5
            )
            first = asyncio.create_task(comm("A"))
            await asyncio.sleep(0)
            self.assertEqual(sent, ["A"])
            # No rank ever replies: the first call's deadline expires after
            # dispatch (orphan keeps the slot), while the queued second
            # caller's own deadline expires first -- without ever sending.
            second = asyncio.create_task(comm("B"))
            with self.assertRaises(FanOutTimeoutError):
                await asyncio.wait_for(first, timeout=5)
            with self.assertRaises(FanOutTimeoutError):
                await asyncio.wait_for(second, timeout=5)
            self.assertEqual(sent, ["A"])

        asyncio.run(scenario())

    def test_call_within_deadline_succeeds(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(
                send=sent.append, fan_out=2, mode="queueing", queueing_timeout=5
            )
            task = asyncio.create_task(comm("A"))
            await asyncio.sleep(0)
            comm.handle_recv("A-0")
            comm.handle_recv("A-1")
            self.assertEqual(await asyncio.wait_for(task, timeout=5), ["A-0", "A-1"])
            self.assertEqual(sent, ["A"])

        asyncio.run(scenario())

    def test_timeout_none_preserves_unbounded_wait(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(
                send=sent.append, fan_out=1, mode="queueing", queueing_timeout=None
            )
            task = asyncio.create_task(comm("A"))
            await asyncio.sleep(0)
            # No deadline: still pending after a grace period, then the
            # reply completes it normally.
            await asyncio.sleep(0.2)
            self.assertFalse(task.done())
            comm.handle_recv("A-0")
            self.assertEqual(await asyncio.wait_for(task, timeout=5), ["A-0"])

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
