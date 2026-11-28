from __future__ import annotations

import asyncio
from typing import Any, Dict, List

from sglang.srt.managers.communicator import FanOutCommunicator
from sglang.srt.managers.io_struct import GetInternalStateReqOutput
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _manager(communicator: FanOutCommunicator) -> TokenizerManager:
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.get_internal_state_communicator = communicator
    manager.auto_create_handle_loop = lambda: None
    return manager


def _reply(rank: int) -> GetInternalStateReqOutput:
    return GetInternalStateReqOutput(internal_state={"rank": rank})


class TestInternalStateReuse(CustomTestCase):
    def test_late_replies_are_reused_by_next_caller(self):
        async def scenario() -> None:
            sends: List[Any] = []
            send_event = asyncio.Event()

            def send(req) -> None:
                sends.append(req)
                send_event.set()

            communicator = FanOutCommunicator(send=send, fan_out=2)
            manager = _manager(communicator)

            # /server_info times out while the DP replies are still in flight.
            try:
                await asyncio.wait_for(manager.get_internal_state(), timeout=0.01)
            except asyncio.TimeoutError:
                pass
            else:
                self.fail("expected the first query to time out")

            await asyncio.wait_for(send_event.wait(), timeout=1)

            # The late reply belongs to the first query, not the next one.
            communicator.handle_recv(_reply(0))
            second = asyncio.create_task(manager.get_internal_state())
            await asyncio.sleep(0)
            communicator.handle_recv(_reply(1))
            states: List[Dict[str, Any]] = await second

            self.assertEqual(len(sends), 1)
            self.assertEqual([state["rank"] for state in states], [0, 1])
            self.assertIsNone(getattr(manager, "_internal_state_task", None))

        asyncio.run(scenario())

    def test_concurrent_waiters_share_one_query(self):
        async def scenario() -> None:
            sends: List[Any] = []
            send_event = asyncio.Event()

            def send(req) -> None:
                sends.append(req)
                send_event.set()

            communicator = FanOutCommunicator(send=send, fan_out=2)
            manager = _manager(communicator)

            try:
                await asyncio.wait_for(manager.get_internal_state(), timeout=0.01)
            except asyncio.TimeoutError:
                pass
            else:
                self.fail("expected the first query to time out")

            await asyncio.wait_for(send_event.wait(), timeout=1)

            first = asyncio.create_task(manager.get_internal_state())
            second = asyncio.create_task(manager.get_internal_state())
            await asyncio.sleep(0)
            communicator.handle_recv(_reply(0))
            communicator.handle_recv(_reply(1))

            results = await asyncio.gather(first, second)
            for states in results:
                self.assertEqual([state["rank"] for state in states], [0, 1])

            self.assertEqual(len(sends), 1)
            self.assertIsNone(getattr(manager, "_internal_state_task", None))

        asyncio.run(scenario())
