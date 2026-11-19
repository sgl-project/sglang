"""Batch acquire must hold the adapter until release.

Looking up ids and bumping usage used to be split across the registry lock.
A queued unload ran in the gap, saw a zero count, and dropped the adapter.
release() then raised KeyError.
"""

import asyncio
import unittest

from sglang.srt.lora.lora_registry import LoRARef, LoRARegistry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestLoRARegistryAcquire(unittest.IsolatedAsyncioTestCase, CustomTestCase):
    def setUp(self):
        super().setUp()
        self.ref = LoRARef(lora_name="a", lora_path="/x")
        self.reg = LoRARegistry([self.ref])

    async def test_queued_unload_waits_for_acquire(self):
        # single name already waited. a list used to lose the race.
        for names in ("a", ["a"]):
            reg = LoRARegistry([LoRARef(lora_name="a", lora_path="/x")])
            unloaded = asyncio.Event()
            await reg._registry_lock.acquire_writer()

            async def request():
                ids = await reg.acquire(names)
                await asyncio.sleep(0)
                self.assertFalse(unloaded.is_set())
                await reg.release(ids)

            async def unload():
                lora_id = await reg.unregister("a")
                await reg.wait_for_unload(lora_id)
                unloaded.set()

            req = asyncio.create_task(request())
            await asyncio.sleep(0)
            drop = asyncio.create_task(unload())
            await asyncio.sleep(0)
            await reg._registry_lock.release_writer()
            await asyncio.wait_for(asyncio.gather(req, drop), timeout=1)
            self.assertTrue(unloaded.is_set())

    async def test_cancel_rolls_back_bumped_counters(self):
        counter = self.reg._counters[self.ref.lora_id]
        started = asyncio.Event()
        rollback_started = asyncio.Event()
        finish_dec = asyncio.Event()
        calls = 0
        original_inc = counter.increment
        original_dec = counter.decrement

        async def increment(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                started.set()
                await asyncio.Event().wait()
            await original_inc(*args, **kwargs)

        async def decrement(*args, **kwargs):
            rollback_started.set()
            await finish_dec.wait()
            await original_dec(*args, **kwargs)

        counter.increment = increment
        counter.decrement = decrement
        task = asyncio.create_task(self.reg.acquire(["a", "a"]))
        await asyncio.wait_for(started.wait(), timeout=1)
        self.assertEqual(counter.value(), 1)
        task.cancel()
        await asyncio.wait_for(rollback_started.wait(), timeout=1)
        # cancel again while the rollback decrement is waiting
        task.cancel()
        finish_dec.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(counter.value(), 0)
        lora_id = await self.reg.unregister("a")
        await asyncio.wait_for(self.reg.wait_for_unload(lora_id), timeout=1)

    async def test_release_missing_counter(self):
        await self.reg.release("missing")
        await self.reg.release(["missing", None])
        lora_id = await self.reg.acquire("a")
        await self.reg.release([lora_id, "missing"])
        self.assertEqual(self.reg._counters[lora_id].value(), 0)


if __name__ == "__main__":
    unittest.main()
