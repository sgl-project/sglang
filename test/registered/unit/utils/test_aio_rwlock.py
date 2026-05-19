import asyncio
import unittest

from sglang.srt.utils.aio_rwlock import RWLock
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(1.0, "stage-a-test-cpu")


class TestAIORWLock(CustomTestCase):
    """Tests for async reader-writer lock.

    Each test creates its own RWLock inside the async context to avoid
    cross-event-loop issues: asyncio.run() creates a fresh loop, and
    asyncio primitives (Lock, Condition) are bound to their creation loop.
    """

    def test_init_state(self):
        """Test the initial state of the RWLock."""

        async def main():
            rwlock = RWLock()
            self.assertEqual(rwlock._readers, 0)
            self.assertFalse(rwlock._writer_active)
            self.assertEqual(rwlock._waiting_writers, 0)
            self.assertFalse(await rwlock.is_locked())

        asyncio.run(main())

    def test_multiple_readers_concurrently(self):
        """Multiple readers should be able to acquire the lock concurrently."""

        async def main():
            rwlock = RWLock()
            active_readers = 0
            max_concurrent = 0

            async def reader_task():
                nonlocal active_readers, max_concurrent
                async with rwlock.reader_lock:
                    active_readers += 1
                    max_concurrent = max(max_concurrent, active_readers)
                    await asyncio.sleep(0)  # yield to let other readers acquire
                    active_readers -= 1

            tasks = [asyncio.create_task(reader_task()) for _ in range(5)]
            await asyncio.gather(*tasks)
            self.assertEqual(rwlock._readers, 0)
            self.assertFalse(await rwlock.is_locked())
            # At least 2 readers held the lock concurrently
            self.assertGreaterEqual(max_concurrent, 2)

        asyncio.run(main())

    def test_writer_exclusivity(self):
        """A writer has exclusive access, blocking readers and other writers."""

        async def main():
            rwlock = RWLock()
            events = []

            writer1_acquired = asyncio.Event()
            writer1_release = asyncio.Event()

            async def writer1():
                async with rwlock.writer_lock:
                    events.append("w1_start")
                    writer1_acquired.set()
                    await writer1_release.wait()
                    events.append("w1_end")

            async def reader():
                await writer1_acquired.wait()
                async with rwlock.reader_lock:
                    events.append("reader_start")
                    events.append("reader_end")

            async def writer2():
                await writer1_acquired.wait()
                async with rwlock.writer_lock:
                    events.append("w2_start")
                    events.append("w2_end")

            t1 = asyncio.create_task(writer1())
            t2 = asyncio.create_task(reader())
            t3 = asyncio.create_task(writer2())

            await writer1_acquired.wait()
            # w1 holds lock; reader and w2 must be blocked
            self.assertEqual(events, ["w1_start"])

            writer1_release.set()
            await asyncio.gather(t1, t2, t3)

            # w1 must complete before any others start
            self.assertEqual(events[0], "w1_start")
            self.assertEqual(events[1], "w1_end")

            # w2 start/end must be contiguous (exclusive)
            w2_start = events.index("w2_start")
            self.assertEqual(events[w2_start + 1], "w2_end")

            # reader start/end must be contiguous
            r_start = events.index("reader_start")
            self.assertEqual(events[r_start + 1], "reader_end")

        asyncio.run(main())

    def test_writer_priority_over_new_readers(self):
        """When a writer is waiting, new readers wait behind it."""

        async def main():
            rwlock = RWLock()
            events = []

            r1_acquired = asyncio.Event()
            r1_release = asyncio.Event()
            w1_waiting = asyncio.Event()

            async def active_reader():
                async with rwlock.reader_lock:
                    events.append("r1_start")
                    r1_acquired.set()
                    await r1_release.wait()
                    events.append("r1_end")

            async def waiting_writer():
                await r1_acquired.wait()
                w1_waiting.set()
                async with rwlock.writer_lock:
                    events.append("w1_start")
                    events.append("w1_end")

            async def queueing_reader():
                await w1_waiting.wait()
                await asyncio.sleep(0)  # yield so writer registers _waiting_writers
                async with rwlock.reader_lock:
                    events.append("r2_start")
                    events.append("r2_end")

            r1 = asyncio.create_task(active_reader())
            w1 = asyncio.create_task(waiting_writer())
            r2 = asyncio.create_task(queueing_reader())

            await r1_acquired.wait()
            await w1_waiting.wait()
            await asyncio.sleep(0)

            r1_release.set()
            await asyncio.gather(r1, w1, r2)

            expected = [
                "r1_start",
                "r1_end",
                "w1_start",
                "w1_end",
                "r2_start",
                "r2_end",
            ]
            self.assertEqual(events, expected)

        asyncio.run(main())

    def test_is_locked(self):
        """is_locked reflects the current state correctly."""

        async def main():
            rwlock = RWLock()
            self.assertFalse(await rwlock.is_locked())

            async with rwlock.reader_lock:
                self.assertTrue(await rwlock.is_locked())
            self.assertFalse(await rwlock.is_locked())

            async with rwlock.writer_lock:
                self.assertTrue(await rwlock.is_locked())
            self.assertFalse(await rwlock.is_locked())

        asyncio.run(main())


if __name__ == "__main__":
    unittest.main()
