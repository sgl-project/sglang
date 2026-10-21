"""Unit tests for FanOutCommunicator -- no server, no model loading."""

import asyncio
import unittest

from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestQueueingCall(CustomTestCase):
    def test_correlated_late_response_after_cancellation_is_ignored(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(sent.append, 1, correlation_key=lambda obj: obj)
            cancelled = asyncio.create_task(comm("old"))
            await asyncio.sleep(0)
            cancelled.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await cancelled
            current = asyncio.create_task(comm("new"))
            await asyncio.sleep(0)
            comm.handle_recv("old")
            await asyncio.sleep(0)
            self.assertFalse(current.done())
            comm.handle_recv("new")
            self.assertEqual(await current, ["new"])
            self.assertEqual(sent, ["old", "new"])

        asyncio.run(scenario())

    def test_synchronous_reply_and_send_exception_cleanup(self):
        async def scenario():
            comm = FanOutCommunicator(
                lambda obj: comm.handle_recv(obj), 1, correlation_key=lambda obj: obj
            )
            self.assertEqual(await comm("inline"), ["inline"])

            def fail(obj):
                raise RuntimeError("send failed")

            comm._send = fail
            with self.assertRaisesRegex(RuntimeError, "send failed"):
                await comm("failed")
            self.assertIsNone(comm._result_event)
            comm._send = lambda obj: comm.handle_recv(obj)
            self.assertEqual(await comm("next"), ["next"])

        asyncio.run(scenario())

    def test_concurrent_caller_cannot_bypass_queue(self):
        """A new caller arriving in the wakeup window must not overtake a
        queued caller (this interleaving used to raise AssertionError and
        return 500 on concurrent /server_info requests)."""

        async def scenario():
            sent = []
            comm = FanOutCommunicator(send=sent.append, fan_out=1, mode="queueing")

            # A in-flight, B queued behind it.
            task_a = asyncio.create_task(comm("A"))
            await asyncio.sleep(0)
            task_b = asyncio.create_task(comm("B"))
            await asyncio.sleep(0)

            # Complete A, then create C before A's wakeup runs, so C's first
            # step lands between A's cleanup and B's wakeup.
            comm.handle_recv("resp-A")
            task_c = asyncio.create_task(comm("C"))

            # Drive to completion: feed a response whenever one is in flight.
            tasks = [task_a, task_b, task_c]
            for _ in range(100):
                if all(t.done() for t in tasks):
                    break
                if comm._result_event is not None and not comm._result_event.is_set():
                    comm.handle_recv(f"resp-{len(sent)}")
                await asyncio.sleep(0)

            # All callers complete without error, in strict FIFO order.
            await asyncio.gather(*tasks)
            self.assertEqual(sent, ["A", "B", "C"])

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
