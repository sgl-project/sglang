import asyncio

from sglang.srt.utils.aio_rwlock import RWLock
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAioRWLock(CustomTestCase):
    def test_cancelled_waiting_writer_wakes_queued_readers(self):
        async def _run():
            rwlock = RWLock()

            await rwlock.acquire_reader()

            writer_task = asyncio.create_task(rwlock.acquire_writer())
            await asyncio.sleep(0)
            self.assertEqual(rwlock._waiting_writers, 1)

            reader_acquired = False

            async def _queued_reader():
                nonlocal reader_acquired
                async with rwlock.reader_lock:
                    reader_acquired = True

            reader_task = asyncio.create_task(_queued_reader())
            await asyncio.sleep(0)
            self.assertFalse(reader_acquired)

            writer_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await writer_task

            self.assertEqual(rwlock._waiting_writers, 0)
            await asyncio.wait_for(reader_task, timeout=1.0)
            self.assertTrue(reader_acquired)

            await rwlock.release_reader()
            self.assertFalse(await rwlock.is_locked())

        asyncio.run(_run())

    def test_cancelled_writer_after_last_reader_release_wakes_reader(self):
        async def _run():
            rwlock = RWLock()

            await rwlock.acquire_reader()

            writer_task = asyncio.create_task(rwlock.acquire_writer())
            await asyncio.sleep(0)

            reader_task = asyncio.create_task(rwlock.acquire_reader())
            await asyncio.sleep(0)

            # Release the initial reader and cancel the waiting writer in the
            # same event-loop turn before the writer resumes.
            release_task = asyncio.create_task(rwlock.release_reader())
            await asyncio.sleep(0)
            writer_task.cancel()

            await release_task
            with self.assertRaises(asyncio.CancelledError):
                await writer_task

            await asyncio.wait_for(reader_task, timeout=1.0)
            self.assertEqual(rwlock._readers, 1)
            await rwlock.release_reader()
            self.assertFalse(await rwlock.is_locked())

        asyncio.run(_run())
