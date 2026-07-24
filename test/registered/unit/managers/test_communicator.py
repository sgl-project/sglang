"""Unit tests for FanOutCommunicator (srt/managers/communicator.py).

FanOutCommunicator has no zmq/torch/CUDA dependency of its own -- it's a
generic fan-out/fan-in coordination primitive over an injected `send`
callable -- so these tests drive it directly with a fake `send` and manually
scheduled `handle_recv` deliveries instead of a real socket.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import asyncio
import unittest

from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.test.test_utils import CustomTestCase


def _respond_after_yield(comm: FanOutCommunicator, responses):
    """Build a `send` callable that schedules `handle_recv(response)` for
    every response, one asyncio tick later -- mimicking a reply that arrives
    after the caller has finished setting up its wait state, the same way a
    real zmq reply would.
    """
    sent = []

    def send(obj):
        sent.append(obj)

        async def deliver():
            await asyncio.sleep(0)
            for response in responses:
                comm.handle_recv(response)

        asyncio.create_task(deliver())

    return send, sent


class TestQueueingMode(CustomTestCase):
    def test_single_call_sends_and_returns_fan_out_responses(self):
        async def drive():
            comm = FanOutCommunicator(send=None, fan_out=2, mode="queueing")
            send, sent = _respond_after_yield(comm, ["r1", "r2"])
            comm._send = send
            result = await comm.queueing_call("req")
            return sent, result

        sent, result = asyncio.run(drive())
        self.assertEqual(sent, ["req"])
        self.assertEqual(result, ["r1", "r2"])

    def test_concurrent_callers_are_served_strictly_in_order(self):
        # A second caller must not send its request until the first caller's
        # round has fully completed -- that's what "queueing" means here.
        async def drive():
            comm = FanOutCommunicator(send=None, fan_out=1, mode="queueing")
            sent = []

            def send(obj):
                sent.append(obj)

                async def deliver():
                    await asyncio.sleep(0)
                    comm.handle_recv(f"resp-{obj}")

                asyncio.create_task(deliver())

            comm._send = send
            first, second = await asyncio.gather(
                comm.queueing_call("a"), comm.queueing_call("b")
            )
            return sent, first, second

        sent, first, second = asyncio.run(drive())
        self.assertEqual(sent, ["a", "b"])
        self.assertEqual(first, ["resp-a"])
        self.assertEqual(second, ["resp-b"])


class TestWatchingMode(CustomTestCase):
    def test_concurrent_callers_share_one_inflight_request(self):
        # Both callers arrive while nothing is in flight yet, but only the
        # first should actually send -- the second "watches" the same result.
        async def drive():
            comm = FanOutCommunicator(send=None, fan_out=1, mode="watching")
            send, sent = _respond_after_yield(comm, ["shared-response"])
            comm._send = send
            first, second = await asyncio.gather(
                comm.watching_call("req"), comm.watching_call("req")
            )
            return sent, first, second

        sent, first, second = asyncio.run(drive())
        self.assertEqual(sent, ["req"])
        self.assertEqual(first, ["shared-response"])
        self.assertEqual(second, ["shared-response"])

    def test_late_joiner_after_completion_starts_a_new_request(self):
        # Once a watching round has fully resolved, the next call is a fresh
        # round and must send again.
        async def drive():
            comm = FanOutCommunicator(send=None, fan_out=1, mode="watching")
            send, sent = _respond_after_yield(comm, ["first-response"])
            comm._send = send
            first = await comm.watching_call("req")

            send2, sent2 = _respond_after_yield(comm, ["second-response"])
            comm._send = send2
            second = await comm.watching_call("req")
            return sent, sent2, first, second

        sent, sent2, first, second = asyncio.run(drive())
        self.assertEqual(sent, ["req"])
        self.assertEqual(sent2, ["req"])
        self.assertEqual(first, ["first-response"])
        self.assertEqual(second, ["second-response"])


class TestConstructorValidation(CustomTestCase):
    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(AssertionError):
            FanOutCommunicator(send=lambda obj: None, fan_out=1, mode="bogus")


class _FakeResult:
    def __init__(self, success, message):
        self.success = success
        self.message = message


class TestMergeResults(CustomTestCase):
    def test_all_success_joins_messages_with_pipe(self):
        results = [_FakeResult(True, "ok-1"), _FakeResult(True, "ok-2")]
        success, message = FanOutCommunicator.merge_results(results)
        self.assertTrue(success)
        self.assertEqual(message, "ok-1 | ok-2")

    def test_any_failure_makes_overall_success_false(self):
        results = [_FakeResult(True, "ok"), _FakeResult(False, "boom")]
        success, message = FanOutCommunicator.merge_results(results)
        self.assertFalse(success)
        self.assertEqual(message, "ok | boom")


if __name__ == "__main__":
    unittest.main()
