import asyncio
import unittest

from sglang.srt.utils.aio_rwlock import RWLock
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRWLock(CustomTestCase):
    def test_readers_can_acquire_lock_together(self):
        async def run():
            lock = RWLock()
            entered = asyncio.Event()

            async def reader():
                async with lock.reader_lock:
                    entered.set()

            async with lock.reader_lock:
                task = asyncio.create_task(reader())
                await asyncio.wait_for(entered.wait(), timeout=1)
                self.assertTrue(await lock.is_locked())
            await task
            self.assertFalse(await lock.is_locked())

        asyncio.run(run())

    def test_writer_waits_for_active_reader(self):
        async def run():
            lock = RWLock()
            entered = asyncio.Event()

            async def writer():
                async with lock.writer_lock:
                    entered.set()

            async with lock.reader_lock:
                task = asyncio.create_task(writer())
                await asyncio.sleep(0)
                self.assertFalse(entered.is_set())
            await asyncio.wait_for(entered.wait(), timeout=1)
            await task

        asyncio.run(run())

    def test_waiting_writer_has_priority_over_later_reader(self):
        async def run():
            lock = RWLock()
            writer_entered = asyncio.Event()
            release_writer = asyncio.Event()
            reader_entered = asyncio.Event()

            async def writer():
                async with lock.writer_lock:
                    writer_entered.set()
                    await release_writer.wait()

            async def reader():
                async with lock.reader_lock:
                    reader_entered.set()

            async with lock.reader_lock:
                writer_task = asyncio.create_task(writer())
                await asyncio.sleep(0)
                reader_task = asyncio.create_task(reader())
                await asyncio.sleep(0)
                self.assertFalse(reader_entered.is_set())

            await asyncio.wait_for(writer_entered.wait(), timeout=1)
            self.assertFalse(reader_entered.is_set())
            release_writer.set()
            await asyncio.wait_for(reader_entered.wait(), timeout=1)
            await asyncio.gather(writer_task, reader_task)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
