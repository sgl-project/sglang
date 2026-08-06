import asyncio
import unittest
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightFromDiskReqOutput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _SchedulerBoundary:
    def __init__(self, manager: TokenizerManager):
        self.manager = manager
        self.requests = []

    def dispatch(self, request: UpdateWeightFromDiskReqInput):
        self.requests.append(request)

    async def wait_for_requests(self, count: int):
        async def wait():
            while len(self.requests) < count:
                await asyncio.sleep(0)

        await asyncio.wait_for(wait(), timeout=1)

    def complete(self, message: str):
        self.manager._handle_update_weights_from_disk_req_output(
            UpdateWeightFromDiskReqOutput(success=True, message=message)
        )


def _paused_manager() -> tuple[TokenizerManager, _SchedulerBoundary]:
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.server_args = SimpleNamespace(
        checkpoint_engine_wait_weights_before_ready=False,
    )
    manager.elastic_worker_count = 1
    manager._config_updates = []
    manager.event_loop = asyncio.get_running_loop()
    manager.asyncio_tasks = set()
    manager.init_weight_update()
    manager.is_pause = True

    scheduler = _SchedulerBoundary(manager)
    manager._dispatch_to_scheduler = scheduler.dispatch
    return manager, scheduler


class TestTokenizerManagerWeightUpdates(unittest.IsolatedAsyncioTestCase):
    async def test_paused_update_blocks_resume_until_completion(self):
        manager, scheduler = _paused_manager()
        await manager.model_update_operation_lock.acquire()

        update = asyncio.create_task(
            manager.update_weights_from_disk(
                UpdateWeightFromDiskReqInput(model_path="first", load_format="dummy")
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        async def resume_generation():
            async with manager.is_pause_cond:
                manager.is_pause = False
                manager.is_pause_cond.notify_all()

        resume = asyncio.create_task(resume_generation())
        await asyncio.sleep(0)
        resumed_before_update_owned_completion = resume.done()

        manager.model_update_operation_lock.release()
        await scheduler.wait_for_requests(1)
        scheduler.complete("first complete")
        self.assertEqual(
            await asyncio.wait_for(update, timeout=1),
            (True, "first complete", 0),
        )
        await asyncio.wait_for(resume, timeout=1)

        self.assertFalse(
            resumed_before_update_owned_completion,
            "generation resumed while a paused update was waiting to dispatch",
        )

    async def test_unpaused_updates_preserve_writer_preference(self):
        manager, scheduler = _paused_manager()
        manager.is_pause = False
        await manager.model_update_lock.acquire_reader()

        first = asyncio.create_task(
            manager.update_weights_from_disk(
                UpdateWeightFromDiskReqInput(model_path="first", load_format="dummy")
            )
        )
        while manager.model_update_lock._waiting_writers < 1:
            await asyncio.sleep(0)

        second = asyncio.create_task(
            manager.update_weights_from_disk(
                UpdateWeightFromDiskReqInput(model_path="second", load_format="dummy")
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        late_reader_acquired = asyncio.Event()
        release_late_reader = asyncio.Event()

        async def run_late_reader():
            async with manager.model_update_lock.reader_lock:
                late_reader_acquired.set()
                await release_late_reader.wait()

        late_reader = asyncio.create_task(run_late_reader())
        await manager.model_update_lock.release_reader()

        await scheduler.wait_for_requests(1)
        scheduler.complete("first complete")

        second_dispatched = asyncio.create_task(scheduler.wait_for_requests(2))
        late_reader_arrived = asyncio.create_task(late_reader_acquired.wait())
        done, pending = await asyncio.wait(
            {second_dispatched, late_reader_arrived},
            return_when=asyncio.FIRST_COMPLETED,
            timeout=1,
        )
        second_kept_priority = second_dispatched in done

        if second_kept_priority:
            scheduler.complete("second complete")
        else:
            release_late_reader.set()
            await scheduler.wait_for_requests(2)
            scheduler.complete("second complete")

        await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
        release_late_reader.set()
        await asyncio.wait_for(late_reader, timeout=1)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        self.assertTrue(
            second_kept_priority,
            "a generation reader overtook an update that was already queued",
        )

    async def test_paused_concurrent_updates_each_receive_own_completion(self):
        manager, scheduler = _paused_manager()
        first = asyncio.create_task(
            manager.update_weights_from_disk(
                UpdateWeightFromDiskReqInput(model_path="first", load_format="dummy")
            )
        )
        second = asyncio.create_task(
            manager.update_weights_from_disk(
                UpdateWeightFromDiskReqInput(model_path="second", load_format="dummy")
            )
        )

        try:
            await scheduler.wait_for_requests(1)
            scheduler.complete("first complete")

            done, _ = await asyncio.wait({first}, timeout=1)
            self.assertIn(first, done, "the first update lost its completion")
            self.assertEqual(
                first.result(),
                (True, "first complete", 0),
            )

            await scheduler.wait_for_requests(2)
            scheduler.complete("second complete")
            self.assertEqual(
                await asyncio.wait_for(second, timeout=1),
                (True, "second complete", 0),
            )
        finally:
            for task in (first, second):
                if not task.done():
                    task.cancel()
            await asyncio.gather(first, second, return_exceptions=True)

    async def test_cancellation_keeps_completion_ownership_until_response(self):
        manager, scheduler = _paused_manager()
        first = asyncio.create_task(
            manager.update_weights_from_disk(
                UpdateWeightFromDiskReqInput(model_path="first", load_format="dummy")
            )
        )
        await scheduler.wait_for_requests(1)

        first.cancel()
        await asyncio.sleep(0)
        second = asyncio.create_task(
            manager.update_weights_from_disk(
                UpdateWeightFromDiskReqInput(model_path="second", load_format="dummy")
            )
        )

        try:
            await asyncio.sleep(0)
            self.assertEqual(
                len(scheduler.requests),
                1,
                "a canceled update released ownership before its response arrived",
            )

            scheduler.complete("first complete")
            with self.assertRaises(asyncio.CancelledError):
                await first

            await scheduler.wait_for_requests(2)
            scheduler.complete("second complete")
            self.assertEqual(
                await asyncio.wait_for(second, timeout=1),
                (True, "second complete", 0),
            )
        finally:
            for task in (first, second):
                if not task.done():
                    task.cancel()
            await asyncio.gather(first, second, return_exceptions=True)

    async def test_cancelled_successful_update_still_publishes_weight_version(self):
        manager, scheduler = _paused_manager()
        update = asyncio.create_task(
            manager.update_weights_from_disk(
                UpdateWeightFromDiskReqInput(
                    model_path="first",
                    load_format="dummy",
                    weight_version="version-1",
                )
            )
        )
        await scheduler.wait_for_requests(1)

        update.cancel()
        await asyncio.sleep(0)
        scheduler.complete("first complete")

        with self.assertRaises(asyncio.CancelledError):
            await update
        self.assertEqual(
            manager.config_value("weight_version"),
            "version-1",
            "a completed update must publish metadata even if its caller cancels",
        )

    async def test_repeated_cancellation_cannot_interrupt_response_drain(self):
        manager, scheduler = _paused_manager()
        update = asyncio.create_task(
            manager.update_weights_from_disk(
                UpdateWeightFromDiskReqInput(model_path="first", load_format="dummy")
            )
        )
        await scheduler.wait_for_requests(1)

        update.cancel()
        await asyncio.sleep(0)
        update.cancel()
        await asyncio.sleep(0)
        released_before_response = update.done()

        scheduler.complete("first complete")
        with self.assertRaises(asyncio.CancelledError):
            await update

        self.assertFalse(
            released_before_response,
            "repeated cancellation released response ownership before completion",
        )


if __name__ == "__main__":
    unittest.main()
