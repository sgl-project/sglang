import asyncio
import unittest

from sglang.srt.utils.aio_rwlock import RWLock
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestRWLockCancellation(CustomTestCase):
    def test_cancelled_waiting_writer_wakes_blocked_reader(self):
        async def run():
            lock = RWLock()
            reader_entered = asyncio.Event()

            async def writer():
                async with lock.writer_lock:
                    pass

            async def reader():
                async with lock.reader_lock:
                    reader_entered.set()

            async with lock.reader_lock:
                writer_task = asyncio.create_task(writer())
                await asyncio.sleep(0)
                reader_task = asyncio.create_task(reader())
                await asyncio.sleep(0)

                writer_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await writer_task

                await asyncio.wait_for(reader_entered.wait(), timeout=1)

            await reader_task

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
