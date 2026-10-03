"""Unit tests for the fan-out reply leak across a shrink.

``set_fan_out`` lowers the expected reply count under an in-flight call, because a
retired rank has already exited and will never answer. That lets the call finish, but
it can still be owed a reply from a slow survivor. If that reply lands after the next
call has started it is appended to the next call's results, which both corrupts them
and can complete that call early.

The fix keeps the finished call's bucket open for a bounded window, under the same lock
the request is sent from, so anything arriving belongs to the call that just finished.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import asyncio
import unittest

from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.test.test_utils import CustomTestCase

LAUNCH_FAN_OUT = 4


class Reply:
    """Stands in for a control ReqOutput. These carry no id, which is the problem."""

    def __init__(self, tag):
        self.tag = tag

    def __repr__(self):
        return f"Reply({self.tag})"


def _comm(fan_out=LAUNCH_FAN_OUT):
    sent = []
    comm = FanOutCommunicator(send=sent.append, fan_out=fan_out, mode="queueing")
    return comm, sent


class TestNoBehaviourChangeWithoutAResize(CustomTestCase):
    """set_fan_out is only called by a resize, so the drain must be unreachable."""

    def test_full_fan_out_returns_every_reply(self):
        async def body():
            comm, sent = _comm()
            call = asyncio.ensure_future(comm("req"))
            await asyncio.sleep(0)
            for i in range(LAUNCH_FAN_OUT):
                comm.handle_recv(Reply(i))
            return await asyncio.wait_for(call, timeout=1.0)

        out = asyncio.run(body())
        self.assertEqual([r.tag for r in out], [0, 1, 2, 3])

    def test_two_sequential_calls_stay_separate(self):
        async def body():
            comm, sent = _comm()
            results = []
            for base in (0, 10):
                call = asyncio.ensure_future(comm("req"))
                await asyncio.sleep(0)
                for i in range(LAUNCH_FAN_OUT):
                    comm.handle_recv(Reply(base + i))
                results.append(await asyncio.wait_for(call, timeout=1.0))
            return results

        first, second = asyncio.run(body())
        self.assertEqual([r.tag for r in first], [0, 1, 2, 3])
        self.assertEqual([r.tag for r in second], [10, 11, 12, 13])


class TestStragglerDoesNotLeakIntoTheNextCall(CustomTestCase):
    def test_short_completion_returns_only_what_arrived(self):
        async def body():
            comm, _ = _comm()
            call = asyncio.ensure_future(comm("req"))
            await asyncio.sleep(0)
            for i in range(3):
                comm.handle_recv(Reply(i))
            # The 4th rank retired, so the resize lowers the target and the call ends.
            comm.set_fan_out(3)
            return await asyncio.wait_for(call, timeout=1.0)

        out = asyncio.run(body())
        self.assertEqual([r.tag for r in out], [0, 1, 2])

    def test_straggler_is_absorbed_not_handed_to_the_next_call(self):
        """The leak itself, with the next call already queued when it arrives.

        Before the drain this returned ``[99, 10, 11, 12]`` for the second call: the
        foreign reply counted toward its target of 3, so it completed on a reply that
        answered the previous request and one of its own arrived too late to count.
        """

        async def body():
            comm, _ = _comm()
            first = asyncio.ensure_future(comm("a"))
            await asyncio.sleep(0)
            for i in range(3):
                comm.handle_recv(Reply(i))

            async def straggler():
                await asyncio.sleep(0.01)
                comm.handle_recv(Reply(99))

            strag = asyncio.ensure_future(straggler())
            # Queued on the lock before the first call finishes, which is what puts
            # its bucket in the straggler's path.
            second = asyncio.ensure_future(comm("b"))

            comm.set_fan_out(3)
            first_out = await asyncio.wait_for(first, timeout=3.0)
            await asyncio.sleep(0.02)
            for i in range(3):
                comm.handle_recv(Reply(10 + i))
            second_out = await asyncio.wait_for(second, timeout=3.0)
            await strag
            return first_out, second_out

        first_out, second_out = asyncio.run(body())
        self.assertEqual([r.tag for r in first_out], [0, 1, 2])
        self.assertNotIn(99, [r.tag for r in second_out])
        self.assertEqual([r.tag for r in second_out], [10, 11, 12])

    def test_a_straggler_that_never_arrives_is_bounded(self):
        """A retiree that exited without answering must not hold the lock open."""

        async def body():
            comm, _ = _comm()
            call = asyncio.ensure_future(comm("req"))
            await asyncio.sleep(0)
            for i in range(3):
                comm.handle_recv(Reply(i))
            comm.set_fan_out(3)
            started = asyncio.get_running_loop().time()
            out = await asyncio.wait_for(call, timeout=2.0)
            return out, asyncio.get_running_loop().time() - started

        out, elapsed = asyncio.run(body())
        self.assertEqual(len(out), 3)
        self.assertLess(elapsed, 1.0)

    def test_the_next_call_still_runs_after_a_short_one(self):
        async def body():
            comm, sent = _comm()
            first = asyncio.ensure_future(comm("a"))
            await asyncio.sleep(0)
            for i in range(3):
                comm.handle_recv(Reply(i))
            comm.set_fan_out(3)
            await asyncio.wait_for(first, timeout=2.0)

            second = asyncio.ensure_future(comm("b"))
            await asyncio.sleep(0)
            for i in range(3):
                comm.handle_recv(Reply(20 + i))
            return await asyncio.wait_for(second, timeout=2.0), sent

        out, sent = asyncio.run(body())
        self.assertEqual([r.tag for r in out], [20, 21, 22])
        self.assertEqual(sent, ["a", "b"])


class TestStateIsLeftCleanOnCancellation(CustomTestCase):
    def test_cancelled_caller_does_not_poison_the_next_call(self):
        async def body():
            comm, _ = _comm()
            call = asyncio.ensure_future(comm("req"))
            await asyncio.sleep(0)
            comm.handle_recv(Reply(0))
            call.cancel()
            try:
                await call
            except asyncio.CancelledError:
                pass
            second = asyncio.ensure_future(comm("req"))
            await asyncio.sleep(0)
            for i in range(LAUNCH_FAN_OUT):
                comm.handle_recv(Reply(30 + i))
            return await asyncio.wait_for(second, timeout=1.0)

        out = asyncio.run(body())
        self.assertEqual([r.tag for r in out], [30, 31, 32, 33])


if __name__ == "__main__":
    unittest.main()
