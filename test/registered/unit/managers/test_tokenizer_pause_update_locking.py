import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    PauseContinueBroadcast,
    PauseGenerationReqInput,
)
from sglang.srt.managers.multi_tokenizer_mixin import TokenizerWorker
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.utils.aio_rwlock import RWLock

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestTokenizerPauseUpdateLocking(unittest.IsolatedAsyncioTestCase):
    def _new_tokenizer_manager(self, paused: bool = False) -> TokenizerManager:
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.is_pause = paused
        manager.is_pause_cond = asyncio.Condition()
        manager._pause_notify = asyncio.Event()
        if paused:
            manager._pause_notify.set()
        manager.model_update_lock = RWLock()
        manager.send_to_scheduler = MagicMock()
        manager.send_to_scheduler.send_pyobj = AsyncMock()
        return manager

    async def test_update_guard_acquires_writer_when_not_paused(self):
        manager = self._new_tokenizer_manager(paused=False)

        async with manager._ensure_paused_or_model_locked():
            self.assertTrue(await manager.model_update_lock.is_locked())

        self.assertFalse(await manager.model_update_lock.is_locked())

    async def test_pause_generation_wakes_update_guard_blocked_on_writer(self):
        manager = self._new_tokenizer_manager(paused=False)
        await manager.model_update_lock.acquire_reader()

        entered_update = asyncio.Event()
        release_update = asyncio.Event()

        async def run_update_guard():
            async with manager._ensure_paused_or_model_locked():
                entered_update.set()
                await release_update.wait()

        update_task = asyncio.create_task(run_update_guard())
        await asyncio.sleep(0)

        self.assertFalse(entered_update.is_set())

        await manager.pause_generation(PauseGenerationReqInput(mode="in_place"))
        await asyncio.wait_for(entered_update.wait(), timeout=1.0)

        # The reader is still held, so the update guard must have entered
        # through the paused path instead of waiting for the writer lock.
        self.assertTrue(await manager.model_update_lock.is_locked())

        release_update.set()
        await asyncio.wait_for(update_task, timeout=1.0)
        await manager.model_update_lock.release_reader()
        self.assertFalse(await manager.model_update_lock.is_locked())

    async def test_pause_broadcast_wakes_update_guard_blocked_on_writer(self):
        worker = TokenizerWorker.__new__(TokenizerWorker)
        worker.is_pause = False
        worker.is_pause_cond = asyncio.Condition()
        worker._pause_notify = asyncio.Event()
        worker.model_update_lock = RWLock()
        worker._pause_continue_future = None

        await worker.model_update_lock.acquire_reader()

        entered_update = asyncio.Event()
        release_update = asyncio.Event()

        async def run_update_guard():
            async with worker._ensure_paused_or_model_locked():
                entered_update.set()
                await release_update.wait()

        update_task = asyncio.create_task(run_update_guard())
        await asyncio.sleep(0)

        self.assertFalse(entered_update.is_set())

        await worker._apply_pause_continue_broadcast(
            PauseContinueBroadcast(is_pause=True)
        )
        await asyncio.wait_for(entered_update.wait(), timeout=1.0)

        release_update.set()
        await asyncio.wait_for(update_task, timeout=1.0)
        await worker.model_update_lock.release_reader()
        self.assertFalse(await worker.model_update_lock.is_locked())


if __name__ == "__main__":
    unittest.main()
