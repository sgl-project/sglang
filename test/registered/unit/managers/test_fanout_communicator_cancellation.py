"""Cancellation regressions for the scheduler control-response collector."""

import asyncio
import unittest

from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFanOutCancellation(unittest.IsolatedAsyncioTestCase):
    async def start_call(self, comm, request):
        task = asyncio.create_task(comm(request))
        self.addAsyncCleanup(self.cancel_call, task)
        await asyncio.sleep(0)
        return task

    async def cancel_call(self, task):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_cancelled_call_drains_all_ranks_before_next_send(self):
        for fan_out in (1, 2, 4):
            with self.subTest(fan_out=fan_out):
                sent = []
                comm = FanOutCommunicator(sent.append, fan_out)
                first = await self.start_call(comm, "A")
                for rank in range(fan_out - 1):
                    comm.handle_recv(f"A-{rank}")
                await self.cancel_call(first)

                second = await self.start_call(comm, "B")
                self.assertEqual(sent, ["A"])
                comm.handle_recv(f"A-{fan_out - 1}")
                await asyncio.sleep(0)
                self.assertEqual(sent, ["A", "B"])
                for rank in range(fan_out):
                    comm.handle_recv(f"B-{rank}")
                self.assertEqual(
                    await asyncio.wait_for(second, 1),
                    [f"B-{rank}" for rank in range(fan_out)],
                )

    async def test_cancelled_queued_call_is_never_sent(self):
        sent = []
        comm = FanOutCommunicator(sent.append, 1)
        first = await self.start_call(comm, "A")
        queued = await self.start_call(comm, "B")
        last = await self.start_call(comm, "C")
        await self.cancel_call(queued)

        comm.handle_recv("A-result")
        self.assertEqual(await asyncio.wait_for(first, 1), ["A-result"])
        await asyncio.sleep(0)
        self.assertEqual(sent, ["A", "C"])
        comm.handle_recv("C-result")
        self.assertEqual(await asyncio.wait_for(last, 1), ["C-result"])

    async def test_cancel_after_final_response_releases_slot(self):
        sent = []
        comm = FanOutCommunicator(sent.append, 1)
        first = await self.start_call(comm, "A")
        comm.handle_recv("A-result")
        await self.cancel_call(first)

        second = await self.start_call(comm, "B")
        self.assertEqual(sent, ["A", "B"])
        comm.handle_recv("B-result")
        self.assertEqual(await asyncio.wait_for(second, 1), ["B-result"])

    async def test_send_failure_does_not_block_next_call(self):
        sent = []

        def send(request):
            if request == "A":
                raise RuntimeError("send failed")
            sent.append(request)

        comm = FanOutCommunicator(send, 1)
        with self.assertRaisesRegex(RuntimeError, "send failed"):
            await comm("A")
        second = await self.start_call(comm, "B")
        self.assertEqual(sent, ["B"])
        comm.handle_recv("B-result")
        self.assertEqual(await asyncio.wait_for(second, 1), ["B-result"])

    async def test_fan_out_change_does_not_shorten_cancelled_call(self):
        sent = []
        comm = FanOutCommunicator(sent.append, 2)
        first = await self.start_call(comm, "A")
        await self.cancel_call(first)
        comm.set_fan_out(1)
        second = await self.start_call(comm, "B")

        comm.handle_recv("A-0")
        await asyncio.sleep(0)
        self.assertEqual(sent, ["A"])
        comm.handle_recv("A-1")
        await asyncio.sleep(0)
        self.assertEqual(sent, ["A", "B"])
        comm.handle_recv("B-0")
        self.assertEqual(await asyncio.wait_for(second, 1), ["B-0"])

    async def test_cancelled_watcher_preserves_shared_response(self):
        sent = []
        comm = FanOutCommunicator(sent.append, 2, mode="watching")
        first = await self.start_call(comm, "A")
        second = await self.start_call(comm, "B")
        comm.handle_recv("A-0")
        await self.cancel_call(first)
        comm.handle_recv("A-1")
        self.assertEqual(await asyncio.wait_for(second, 1), ["A-0", "A-1"])
        self.assertEqual(sent, ["A"])


if __name__ == "__main__":
    unittest.main()
