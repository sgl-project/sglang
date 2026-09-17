"""Unit tests for FanOutCommunicator -- no server, no model loading."""

import asyncio
import unittest
from types import SimpleNamespace

from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestQueueingCall(CustomTestCase):
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

    def test_cancelled_call_drops_late_correlated_response(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(
                send=sent.append,
                fan_out=1,
                mode="queueing",
                correlation_attr="submission_id",
            )

            abandoned = asyncio.create_task(
                comm(SimpleNamespace(submission_id="submission-1"))
            )
            await asyncio.sleep(0)
            abandoned.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await abandoned

            retry = asyncio.create_task(
                comm(SimpleNamespace(submission_id="submission-2"))
            )
            await asyncio.sleep(0)
            comm.handle_recv(SimpleNamespace(submission_id="submission-1"))
            await asyncio.sleep(0)
            self.assertFalse(retry.done())

            response = SimpleNamespace(submission_id="submission-2")
            comm.handle_recv(response)
            self.assertEqual(await retry, [response])

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
