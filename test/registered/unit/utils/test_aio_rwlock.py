import asyncio
import unittest

from sglang.srt.utils.aio_rwlock import RWLock
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestRWLock(CustomTestCase):
    def test_cancelled_waiting_writer_unblocks_waiting_readers(self):
        async def run():
            rwlock = RWLock()
            await rwlock.acquire_reader()

            writer_task = asyncio.create_task(rwlock.acquire_writer())
            await asyncio.sleep(0)
            self.assertEqual(rwlock._waiting_writers, 1)

            reader2_acquired = False

            async def reader2():
                nonlocal reader2_acquired
                async with rwlock.reader_lock:
                    reader2_acquired = True

            reader2_task = asyncio.create_task(reader2())
            await asyncio.sleep(0)
            self.assertFalse(reader2_acquired)

            writer_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await writer_task

            await asyncio.wait_for(reader2_task, timeout=1.0)
            self.assertTrue(reader2_acquired)
            await rwlock.release_reader()
            self.assertFalse(await rwlock.is_locked())

        asyncio.run(run())

    def test_cancelled_writer_after_writer_release_does_not_deadlock_reader(self):
        async def run():
            rwlock = RWLock()
            await rwlock.acquire_writer()

            reader_acquired = False

            async def reader():
                nonlocal reader_acquired
                async with rwlock.reader_lock:
                    reader_acquired = True

            # Queue reader first, then waiting writer, while writer0 holds lock.
            reader_task = asyncio.create_task(reader())
            await asyncio.sleep(0)

            writer2_task = asyncio.create_task(rwlock.acquire_writer())
            await asyncio.sleep(0)
            self.assertEqual(rwlock._waiting_writers, 1)

            # Release writer0 and cancel writer2 in the same turn so reader wakes
            # while _waiting_writers == 1 before writer2 runs its finally block.
            release_task = asyncio.create_task(rwlock.release_writer())
            await asyncio.sleep(0)
            writer2_task.cancel()
            await release_task
            with self.assertRaises(asyncio.CancelledError):
                await writer2_task

            await asyncio.wait_for(reader_task, timeout=1.0)
            self.assertTrue(reader_acquired)
            self.assertFalse(await rwlock.is_locked())

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
