"""Cancelled or concurrent acquisitions must not strand LoRA reference counts."""

import asyncio
import unittest
from contextlib import suppress

from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLoRARegistryCancellation(CustomTestCase, unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        super().setUp()
        self.ref = LoRARef(lora_name="adapter", lora_path="/unused/local/adapter")
        self.registry = LoRARegistry([self.ref])
        self.counter = self.registry._counters[self.ref.lora_id]

    async def _cancel_task(self, task):
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _assert_unload_completes(self):
        lora_id = await self.registry.unregister("adapter")
        await asyncio.wait_for(self.registry.wait_for_unload(lora_id), timeout=1)
        self.assertNotIn(lora_id, self.registry._counters)

    async def _start_paused_acquire(self, names, pause_at):
        entered = asyncio.Event()
        proceed = asyncio.Event()
        original = self.counter.increment
        calls = 0

        async def increment(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == pause_at:
                entered.set()
                await proceed.wait()
            await original(*args, **kwargs)

        self.counter.increment = increment
        task = asyncio.create_task(self.registry.acquire(names))
        self.addAsyncCleanup(self._cancel_task, task)
        await asyncio.wait_for(entered.wait(), timeout=1)
        return task, proceed

    async def test_cancellation_after_counter_tasks_complete_does_not_leak(self):
        # With gather(), increments can finish before acquire resumes. Cancelling
        # at that point used to lose the returned IDs while keeping their counts.
        task = asyncio.create_task(self.registry.acquire(["adapter", "adapter"]))
        self.addAsyncCleanup(self._cancel_task, task)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        try:
            ids = await task
        except asyncio.CancelledError:
            pass
        else:
            # An acquisition that already returned transfers ownership normally.
            await self.registry.release(ids)
        self.assertEqual(self.counter.value(), 0)
        await self._assert_unload_completes()

    async def test_cancel_partial_batch_rolls_back_before_unload(self):
        task, _ = await self._start_paused_acquire(["adapter"] * 3, pause_at=2)
        self.assertGreater(self.counter.value(), 0)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.counter.value(), 0)
        await self._assert_unload_completes()

    async def test_repeated_cancellation_cannot_interrupt_rollback(self):
        task, _ = await self._start_paused_acquire(["adapter"] * 3, pause_at=2)
        rollback_entered = asyncio.Event()
        finish_rollback = asyncio.Event()
        self.addCleanup(finish_rollback.set)
        original = self.counter.decrement

        async def decrement(*args, **kwargs):
            rollback_entered.set()
            await finish_rollback.wait()
            await original(*args, **kwargs)

        self.counter.decrement = decrement
        task.cancel()
        await asyncio.wait_for(rollback_entered.wait(), timeout=1)
        task.cancel()
        finish_rollback.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.counter.value(), 0)
        await self._assert_unload_completes()

    async def test_unregister_waits_for_acquisition_to_increment(self):
        task, proceed = await self._start_paused_acquire(["adapter"], pause_at=1)
        unregister_entered = asyncio.Event()

        async def unregister():
            unregister_entered.set()
            return await self.registry.unregister("adapter")

        unregister_task = asyncio.create_task(unregister())
        self.addAsyncCleanup(self._cancel_task, unregister_task)
        await asyncio.wait_for(unregister_entered.wait(), timeout=1)
        self.assertFalse(unregister_task.done())
        proceed.set()
        ids = await task
        lora_id = await unregister_task
        self.assertEqual(ids, [lora_id])
        self.assertEqual(self.counter.value(), 1)
        unload = asyncio.create_task(self.registry.wait_for_unload(lora_id))
        self.addAsyncCleanup(self._cancel_task, unload)
        await asyncio.sleep(0)
        self.assertFalse(unload.done())
        await self.registry.release(ids)
        await asyncio.wait_for(unload, timeout=1)
        self.assertNotIn(lora_id, self.registry._counters)

    async def test_invalid_batch_does_not_acquire_partial_references(self):
        with self.assertRaises(ValueError):
            await self.registry.acquire(["adapter", "missing"])
        self.assertEqual(self.counter.value(), 0)
        await self._assert_unload_completes()

    async def test_duplicate_names_keep_one_reference_per_request(self):
        ids = await self.registry.acquire(["adapter", None, "adapter"])
        self.assertEqual(ids, [self.ref.lora_id, None, self.ref.lora_id])
        self.assertEqual(self.counter.value(), 2)
        await self.registry.release(ids)
        self.assertEqual(self.counter.value(), 0)
        await self._assert_unload_completes()

    async def test_empty_batch_does_not_acquire_references(self):
        self.assertEqual(await self.registry.acquire([]), [])
        self.assertEqual(self.counter.value(), 0)
        await self._assert_unload_completes()

    async def test_single_acquisition_preserves_existing_contract(self):
        lora_id = await self.registry.acquire("adapter")
        self.assertEqual(lora_id, self.ref.lora_id)
        self.assertEqual(self.counter.value(), 1)
        await self.registry.release(lora_id)
        await self._assert_unload_completes()


if __name__ == "__main__":
    unittest.main()
