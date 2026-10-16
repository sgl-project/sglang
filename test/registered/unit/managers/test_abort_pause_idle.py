import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    GetSchedulerIdleReqInput,
    GetSchedulerIdleReqOutput,
    PauseGenerationReqInput,
)
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.tokenizer_manager import TokenizerManager

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class TestAbortPauseIdle(unittest.IsolatedAsyncioTestCase):
    def manager(self):
        return SimpleNamespace(
            is_pause_cond=asyncio.Condition(),
            model_update_lock=SimpleNamespace(is_locked=AsyncMock(return_value=False)),
            abort_request=Mock(),
            auto_create_handle_loop=Mock(),
            get_scheduler_idle_communicator=AsyncMock(),
            _async_dispatch_to_scheduler=AsyncMock(),
        )

    async def test_disconnected_handler_does_not_imply_scheduler_idle(self):
        manager = self.manager()
        probe_started = asyncio.Event()
        scheduler_drained = asyncio.Event()

        async def probe(req):
            self.assertIsInstance(req, GetSchedulerIdleReqInput)
            probe_started.set()
            await scheduler_drained.wait()
            return [GetSchedulerIdleReqOutput(is_idle=True)]

        manager.get_scheduler_idle_communicator.side_effect = probe
        task = asyncio.create_task(
            TokenizerManager.pause_generation(manager, PauseGenerationReqInput())
        )
        try:
            await asyncio.wait_for(probe_started.wait(), timeout=2)
            self.assertFalse(task.done())
            manager.abort_request.assert_called_once_with(abort_all=True)
            scheduler_drained.set()
            await asyncio.wait_for(task, timeout=2)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def test_all_scheduler_replicas_must_drain(self):
        manager = self.manager()
        manager.get_scheduler_idle_communicator.side_effect = [
            [
                GetSchedulerIdleReqOutput(is_idle=True),
                GetSchedulerIdleReqOutput(is_idle=False),
            ],
            [
                GetSchedulerIdleReqOutput(is_idle=True),
                GetSchedulerIdleReqOutput(is_idle=True),
            ],
        ]
        await TokenizerManager.pause_generation(manager, PauseGenerationReqInput())
        self.assertEqual(manager.get_scheduler_idle_communicator.await_count, 2)

    async def test_tokenizer_reader_must_also_finish(self):
        manager = self.manager()
        manager.model_update_lock.is_locked.side_effect = [True, False]
        manager.get_scheduler_idle_communicator.return_value = [
            GetSchedulerIdleReqOutput(is_idle=True)
        ]
        await TokenizerManager.pause_generation(manager, PauseGenerationReqInput())
        self.assertEqual(manager.get_scheduler_idle_communicator.await_count, 2)

    async def test_other_pause_modes_do_not_wait_for_idle(self):
        for mode in ("in_place", "retract"):
            manager = self.manager()
            req = PauseGenerationReqInput(mode=mode)
            await TokenizerManager.pause_generation(manager, req)
            manager._async_dispatch_to_scheduler.assert_awaited_once_with(req)
            manager.get_scheduler_idle_communicator.assert_not_awaited()

    def test_scheduler_reports_full_idle_predicate(self):
        scheduler = SimpleNamespace(is_fully_idle=Mock(return_value=False))
        self.assertFalse(
            Scheduler.get_scheduler_idle(scheduler, GetSchedulerIdleReqInput()).is_idle
        )
        scheduler.is_fully_idle.return_value = True
        self.assertTrue(
            Scheduler.get_scheduler_idle(scheduler, GetSchedulerIdleReqInput()).is_idle
        )


if __name__ == "__main__":
    unittest.main()
