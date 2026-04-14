import asyncio
import os
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
        shared_resource = 0

        async def reader_task():
            async with self.rwlock.reader_lock:
                # simulating some read operation
                await asyncio.sleep(0.01)
                return shared_resource

        async def main():
            # Run 5 readers concurrently
            tasks = [asyncio.create_task(reader_task()) for _ in range(5)]
            await asyncio.gather(*tasks)
            # Check internal state after readers release
            self.assertEqual(self.rwlock._readers, 0)
            self.assertFalse(await self.rwlock.is_locked())

        asyncio.run(main())

    def test_writer_exclusivity(self):
        """A writer should have exclusive access, blocking both readers and other writers."""
        events: List[str] = []

        async def reader():
            async with self.rwlock.reader_lock:
                events.append("reader_start")
                await asyncio.sleep(0.05)
                events.append("reader_end")

        async def writer():
            async with self.rwlock.writer_lock:
                events.append("writer_start")
                await asyncio.sleep(0.1)
                events.append("writer_end")

        async def main():
            # Start writer first
            t1 = asyncio.create_task(writer())
            await asyncio.sleep(0.01)  # Ensure writer acquires lock
            # Start reader and another writer while lock is held
            t2 = asyncio.create_task(reader())
            t3 = asyncio.create_task(writer())

            await asyncio.gather(t1, t2, t3)

        asyncio.run(main())

        # If writer gets lock first, reader_start/writer_start must happen after writer_end
        self.assertEqual(events[0], "writer_start")
        self.assertEqual(events[1], "writer_end")
        
        # Afterwards, the order of reader and the second writer isn't strictly defined globally,
        # but their own start/end pairs cannot overlap since writers are exclusive
        writer_2_indices = (events.index("writer_start", 2), events.index("writer_end", 2))
        self.assertEqual(writer_2_indices[1], writer_2_indices[0] + 1)
        
        reader_indices = (events.index("reader_start", 2), events.index("reader_end", 2))
        # Ensure they execute in contiguous blocks of exclusion
        self.assertEqual(reader_indices[1], reader_indices[0] + 1)

    def test_writer_priority_over_new_readers(self):
        """When a writer is waiting, new readers shouldn't jump the queue, to prevent writer starvation."""
        events = []

        async def active_reader():
            async with self.rwlock.reader_lock:
                events.append("r1_start")
                await asyncio.sleep(0.1)
                events.append("r1_end")
                
        async def waiting_writer():
            async with self.rwlock.writer_lock:
                events.append("w1_start")
                events.append("w1_end")

        async def queueing_reader():
            async with self.rwlock.reader_lock:
                events.append("r2_start")
                events.append("r2_end")

        async def main():
            # Start active reader
            r1 = asyncio.create_task(active_reader())
            await asyncio.sleep(0.01)  # ensure r1 is active
            
            # Start writer (will be queued because reader is active)
            w1 = asyncio.create_task(waiting_writer())
            await asyncio.sleep(0.01) # ensure w1 is next in queue
            
            # Start another reader (should wait because a writer is waiting)
            r2 = asyncio.create_task(queueing_reader())
            
            await asyncio.gather(r1, w1, r2)
            
        asyncio.run(main())
        
        # Expectation: r1 finishes, then writer handles, then new reader handles.
        expected_events = ["r1_start", "r1_end", "w1_start", "w1_end", "r2_start", "r2_end"]
        self.assertEqual(events, expected_events)

    def test_is_locked(self):
        async def main():
            # Unlocked
            self.assertFalse(await self.rwlock.is_locked())
            
            # Locked by reader
            async with self.rwlock.reader_lock:
                self.assertTrue(await self.rwlock.is_locked())
            self.assertFalse(await self.rwlock.is_locked())
            
            # Locked by writer
            async with self.rwlock.writer_lock:
                self.assertTrue(await self.rwlock.is_locked())
            self.assertFalse(await self.rwlock.is_locked())

        asyncio.run(main())

register_cpu_ci(TestAIORWLock)

if __name__ == "__main__":
    unittest.main()
