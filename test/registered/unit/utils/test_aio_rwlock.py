"""Regression tests for the cancellation behavior of the async reader-writer lock."""

import asyncio
import contextlib
import unittest

from sglang.srt.utils.aio_rwlock import RWLock
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


class TestRWLock(CustomTestCase):
    def test_cancelled_writer_unblocks_waiting_reader(self):
        asyncio.run(self._cancelled_writer_unblocks_waiting_reader())

    async def _cancelled_writer_unblocks_waiting_reader(self):
        lock = RWLock()
        await lock.acquire_reader()

        writer = asyncio.create_task(lock.acquire_writer())
        while lock._waiting_writers != 1:
            await asyncio.sleep(0)

        blocked_reader = asyncio.create_task(lock.acquire_reader())
        await asyncio.sleep(0)

        writer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await writer

        try:
            await asyncio.wait_for(asyncio.shield(blocked_reader), timeout=0.1)
        except asyncio.TimeoutError:
            unblocked = False
        else:
            unblocked = True
        finally:
            await lock.release_reader()
            await blocked_reader
            await lock.release_reader()

        self.assertTrue(unblocked)


if __name__ == "__main__":
    unittest.main()
