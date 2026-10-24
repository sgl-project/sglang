"""A reply owed to a call ``set_fan_out`` cut short must not reach the next call."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import asyncio
import unittest

from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.test.test_utils import CustomTestCase

LAUNCH_FAN_OUT = 4


def _comm(send=lambda obj: None):
    return FanOutCommunicator(send=send, fan_out=LAUNCH_FAN_OUT, mode="queueing")


async def _call(comm, obj, replies):
    """Start a call and feed it ``replies``: ints, which like ReqOutput carry no id."""
    call = asyncio.ensure_future(comm(obj))
    await asyncio.sleep(0)
    for reply in replies:
        comm.handle_recv(reply)
    return call


class TestFanOutStraggler(CustomTestCase):
    def test_two_sequential_calls_stay_separate(self):
        """set_fan_out is only called by a resize, so the drain must be unreachable."""

        async def body():
            comm, results = _comm(), []
            for base in (0, 10):
                call = await _call(comm, "req", range(base, base + LAUNCH_FAN_OUT))
                results.append(await asyncio.wait_for(call, timeout=1.0))
            return results

        self.assertEqual(asyncio.run(body()), [[0, 1, 2, 3], [10, 11, 12, 13]])

    def test_straggler_is_absorbed_not_handed_to_the_next_call(self):
        """The leak itself; without the drain the next call returns [99, 10, 11, 12]."""

        async def body():
            comm = _comm()
            first = await _call(comm, "a", range(3))
            asyncio.get_running_loop().call_later(0.01, comm.handle_recv, 99)
            # Queued on the lock before the first call finishes, which is what puts
            # its bucket in the straggler's path.
            second = asyncio.ensure_future(comm("b"))
            comm.set_fan_out(3)
            first_out = await asyncio.wait_for(first, timeout=3.0)
            await asyncio.sleep(0.02)
            for i in range(3):
                comm.handle_recv(10 + i)
            return first_out, await asyncio.wait_for(second, timeout=3.0)

        self.assertEqual(asyncio.run(body()), ([0, 1, 2], [10, 11, 12]))

    def test_a_straggler_that_never_arrives_is_bounded(self):
        """A retiree that exited without answering must not hold the lock open."""

        async def body():
            comm = _comm()
            call = await _call(comm, "req", range(3))
            comm.set_fan_out(3)
            loop = asyncio.get_running_loop()
            started = loop.time()
            out = await asyncio.wait_for(call, timeout=2.0)
            return out, loop.time() - started

        out, elapsed = asyncio.run(body())
        self.assertEqual(out, [0, 1, 2])
        self.assertLess(elapsed, 1.0)

    def test_the_next_call_still_runs_after_a_short_one(self):
        sent = []

        async def body():
            comm = _comm(sent.append)
            first = await _call(comm, "a", range(3))
            comm.set_fan_out(3)
            await asyncio.wait_for(first, timeout=2.0)
            second = await _call(comm, "b", range(20, 23))
            return await asyncio.wait_for(second, timeout=2.0)

        self.assertEqual(asyncio.run(body()), [20, 21, 22])
        self.assertEqual(sent, ["a", "b"])

    def test_cancelled_caller_does_not_poison_the_next_call(self):
        async def body():
            comm = _comm()
            call = await _call(comm, "req", [0])
            call.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await call
            second = await _call(comm, "req", range(30, 34))
            return await asyncio.wait_for(second, timeout=1.0)

        self.assertEqual(asyncio.run(body()), [30, 31, 32, 33])


if __name__ == "__main__":
    unittest.main()
