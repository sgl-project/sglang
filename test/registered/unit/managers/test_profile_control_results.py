"""Profiling control must report failures from every data-parallel worker."""

import asyncio
import unittest
from types import SimpleNamespace

from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.io_struct import ProfileReq, ProfileReqOutput
from sglang.srt.managers.tokenizer_control_mixin import TokenizerControlMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestProfileControlResults(CustomTestCase):
    def test_profile_failure_does_not_depend_on_reply_order(self):
        async def scenario(success_flags):
            communicator = FanOutCommunicator(lambda req: None, len(success_flags))
            manager = SimpleNamespace(profile_communicator=communicator)
            task = asyncio.create_task(
                TokenizerControlMixin._execute_profile(manager, ProfileReq())
            )
            await asyncio.sleep(0)
            for success in success_flags:
                communicator.handle_recv(
                    ProfileReqOutput(
                        success=success,
                        message="ok" if success else "profiler could not start",
                    )
                )
            with self.assertRaisesRegex(RuntimeError, "profiler could not start"):
                await asyncio.wait_for(task, timeout=1)

        for flags in ([False], [False, True], [True, False], [True, False, True]):
            with self.subTest(replies=flags):
                asyncio.run(scenario(flags))

    def test_success_preserves_the_existing_response_object(self):
        async def scenario(worker_count):
            communicator = FanOutCommunicator(lambda req: None, worker_count)
            manager = SimpleNamespace(profile_communicator=communicator)
            task = asyncio.create_task(
                TokenizerControlMixin._execute_profile(manager, ProfileReq())
            )
            await asyncio.sleep(0)
            replies = [
                ProfileReqOutput(success=True, message=f"trace-{i}")
                for i in range(worker_count)
            ]
            for reply in replies:
                communicator.handle_recv(reply)
            self.assertIs(await asyncio.wait_for(task, timeout=1), replies[0])

        for count in (1, 2, 4):
            with self.subTest(workers=count):
                asyncio.run(scenario(count))


if __name__ == "__main__":
    unittest.main()
