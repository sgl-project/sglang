"""Unit tests for FanOutCommunicator -- no server, no model loading."""

import asyncio
import unittest

from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQueueingCall(CustomTestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_serial_calls(self):
        async def scenario():
            sent = []
            comm = FanOutCommunicator(send=sent.append, fan_out=2, mode="queueing")

            async def respond():
                comm.handle_recv("r1")
                comm.handle_recv("r2")

            task = asyncio.create_task(comm("req"))
            await asyncio.sleep(0)
            await respond()
            result = await task
            self.assertEqual(sent, ["req"])
            self.assertEqual(result, ["r1", "r2"])

        self._run(scenario())

    def test_concurrent_caller_cannot_bypass_queue(self):
        """A new caller arriving in the wakeup window must not overtake a
        queued caller (this interleaving used to raise AssertionError and
        return 500 on concurrent /server_info requests)."""

        async def scenario():
            sent = []
            comm = FanOutCommunicator(send=sent.append, fan_out=1, mode="queueing")

            async def call(name):
                return await comm(name)

            # A in-flight, B queued behind it.
            task_a = asyncio.create_task(call("A"))
            await asyncio.sleep(0)
            task_b = asyncio.create_task(call("B"))
            await asyncio.sleep(0)

            # Complete A, then create C before A's wakeup runs, so C's first
            # step lands between A's cleanup and B's wakeup.
            comm.handle_recv("resp-A")
            task_c = asyncio.create_task(call("C"))

            # Drive to completion: feed a response whenever one is in flight.
            tasks = [task_a, task_b, task_c]
            for _ in range(100):
                if all(t.done() for t in tasks):
                    break
                if comm._result_event is not None and not comm._result_event.is_set():
                    comm.handle_recv(f"resp-{len(sent)}")
                await asyncio.sleep(0)

            results = await asyncio.gather(*tasks)
            # All callers complete without error, in strict FIFO order.
            self.assertEqual(sent, ["A", "B", "C"])
            self.assertEqual(len(results), 3)

        self._run(scenario())

    def test_stress_concurrent_callers(self):
        """Mirrors test_get_server_info_concurrent without a server: many
        concurrent callers with jittered responses must all complete."""

        async def scenario():
            import random

            n_callers = 30
            sent = []
            comm = FanOutCommunicator(send=sent.append, fan_out=1, mode="queueing")
            done = 0

            async def responder():
                while done < n_callers:
                    event = comm._result_event
                    if event is not None and not event.is_set():
                        comm.handle_recv("resp")
                    await asyncio.sleep(random.random() * 1e-4)

            async def one_call(i):
                nonlocal done
                await asyncio.sleep(random.random() * 1e-4)
                await comm(f"req-{i}")
                done += 1

            responder_task = asyncio.create_task(responder())
            await asyncio.wait_for(
                asyncio.gather(*[one_call(i) for i in range(n_callers)]),
                timeout=30,
            )
            await responder_task
            self.assertEqual(len(sent), n_callers)

        self._run(scenario())


if __name__ == "__main__":
    unittest.main()
