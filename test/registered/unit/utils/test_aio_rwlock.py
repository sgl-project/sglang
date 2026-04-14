import asyncio
import unittest
from typing import List

from sglang.srt.utils.aio_rwlock import RWLock
from sglang.test.test_utils import CustomTestCase, register_cpu_ci


class TestAIORWLock(CustomTestCase):
    def setUp(self):
        self.rwlock = RWLock()

    def test_init_state(self):
        """Test the initial state of the RWLock."""
        self.assertEqual(self.rwlock._readers, 0)
        self.assertFalse(self.rwlock._writer_active)
        self.assertEqual(self.rwlock._waiting_writers, 0)

        async def check_lock():
            return await self.rwlock.is_locked()

        self.assertFalse(asyncio.run(check_lock()))

    def test_multiple_readers_concurrently(self):
        """Multiple readers should be able to acquire the lock concurrently without blocking."""
        active_readers = 0
        max_concurrent = 0

        async def reader_task():
            nonlocal active_readers, max_concurrent
            async with self.rwlock.reader_lock:
                active_readers += 1
                max_concurrent = max(max_concurrent, active_readers)
                await asyncio.sleep(0)  # yield to let other readers acquire
                active_readers -= 1

        async def main():
            tasks = [asyncio.create_task(reader_task()) for _ in range(5)]
            await asyncio.gather(*tasks)
            self.assertEqual(self.rwlock._readers, 0)
            self.assertFalse(await self.rwlock.is_locked())

        asyncio.run(main())
        # At least 2 readers should have been active concurrently
        self.assertGreaterEqual(max_concurrent, 2)

    def test_writer_exclusivity(self):
        """A writer should have exclusive access, blocking both readers and other writers."""
        events: List[str] = []

        async def main():
            writer1_acquired = asyncio.Event()
            writer1_release = asyncio.Event()

            async def writer1():
                async with self.rwlock.writer_lock:
                    events.append("w1_start")
                    writer1_acquired.set()
                    await writer1_release.wait()
                    events.append("w1_end")

            async def reader():
                await writer1_acquired.wait()  # ensure w1 holds the lock
                async with self.rwlock.reader_lock:
                    events.append("reader_start")
                    events.append("reader_end")

            async def writer2():
                await writer1_acquired.wait()  # ensure w1 holds the lock
                async with self.rwlock.writer_lock:
                    events.append("w2_start")
                    events.append("w2_end")

            t1 = asyncio.create_task(writer1())
            t2 = asyncio.create_task(reader())
            t3 = asyncio.create_task(writer2())

            await asyncio.sleep(0)  # let tasks start
            await writer1_acquired.wait()

            # At this point w1 holds lock, reader and w2 are blocked
            self.assertEqual(events, ["w1_start"])

            # Release w1
            writer1_release.set()
            await asyncio.gather(t1, t2, t3)

        asyncio.run(main())

        # w1 must complete before any others start
        self.assertEqual(events[0], "w1_start")
        self.assertEqual(events[1], "w1_end")

        # w2's start/end must be contiguous (exclusive)
        w2_start = events.index("w2_start")
        self.assertEqual(events[w2_start + 1], "w2_end")

        # reader's start/end must be contiguous
        r_start = events.index("reader_start")
        self.assertEqual(events[r_start + 1], "reader_end")

    def test_writer_priority_over_new_readers(self):
        """When a writer is waiting, new readers shouldn't jump the queue."""
        events = []

        async def main():
            r1_acquired = asyncio.Event()
            r1_release = asyncio.Event()
            w1_waiting = asyncio.Event()

            async def active_reader():
                async with self.rwlock.reader_lock:
                    events.append("r1_start")
                    r1_acquired.set()
                    await r1_release.wait()
                    events.append("r1_end")

            async def waiting_writer():
                await r1_acquired.wait()  # wait until r1 holds the lock
                w1_waiting.set()
                async with self.rwlock.writer_lock:
                    events.append("w1_start")
                    events.append("w1_end")

            async def queueing_reader():
                await w1_waiting.wait()  # wait until writer is queued
                await asyncio.sleep(0)   # yield so writer registers _waiting_writers
                async with self.rwlock.reader_lock:
                    events.append("r2_start")
                    events.append("r2_end")

            r1 = asyncio.create_task(active_reader())
            w1 = asyncio.create_task(waiting_writer())
            r2 = asyncio.create_task(queueing_reader())

            # Let r1 acquire, then w1 queue, then r2 queue
            await r1_acquired.wait()
            await w1_waiting.wait()
            await asyncio.sleep(0)  # yield for r2 to register

            # Release r1 — writer should go before r2
            r1_release.set()
            await asyncio.gather(r1, w1, r2)

        asyncio.run(main())

        expected = ["r1_start", "r1_end", "w1_start", "w1_end", "r2_start", "r2_end"]
        self.assertEqual(events, expected)

    def test_is_locked(self):
        """Test is_locked reflects the current state correctly."""
        async def main():
            self.assertFalse(await self.rwlock.is_locked())

            async with self.rwlock.reader_lock:
                self.assertTrue(await self.rwlock.is_locked())
            self.assertFalse(await self.rwlock.is_locked())

            async with self.rwlock.writer_lock:
                self.assertTrue(await self.rwlock.is_locked())
            self.assertFalse(await self.rwlock.is_locked())

        asyncio.run(main())


register_cpu_ci(TestAIORWLock)

if __name__ == "__main__":
    unittest.main()
