import asyncio
import unittest

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ModuleNotFoundError:
    # Allows running this unit test in minimal local environments.
    def register_cpu_ci(*args, **kwargs):
        return


register_cpu_ci(est_time=10, suite="default", nightly=True)


from sglang.srt.managers.communicator import FanOutCommunicator


class _DummySender:
    def __init__(self):
        self.sent = []

    def send_pyobj(self, obj):
        self.sent.append(obj)


class TestTokenizerCommunicator(unittest.IsolatedAsyncioTestCase):
    async def test_handle_recv_ignores_stale_message_after_watch_completes(self):
        sender = _DummySender()
        communicator = FanOutCommunicator(sender=sender, fan_out=1, mode="watching")

        watch_task = asyncio.create_task(communicator.watching_call(obj={"op": "x"}))
        await asyncio.sleep(0)

        # Simulate normal response completing the active watch.
        communicator.handle_recv({"result": 1})
        result = await watch_task

        self.assertEqual(result, [{"result": 1}])
        self.assertIsNone(communicator._result_values)
        self.assertIsNone(communicator._result_event)

        # Simulate a stale late-arriving message after cleanup.
        communicator.handle_recv({"stale": True})

        # No crash and no state reactivation.
        self.assertIsNone(communicator._result_values)
        self.assertIsNone(communicator._result_event)


if __name__ == "__main__":
    unittest.main(verbosity=3)
