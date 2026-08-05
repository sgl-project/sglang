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


if __name__ == "__main__":
    unittest.main()
