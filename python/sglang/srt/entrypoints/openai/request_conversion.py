"""Bound CPU request conversion without blocking the HTTP event loop."""

import asyncio
from functools import partial


class RequestConversionExecutor:
    def __init__(self, concurrency: int):
        if concurrency <= 0:
            raise ValueError("request conversion concurrency must be positive")
        self._slots = asyncio.Semaphore(concurrency)

    async def run(self, function, *args):
        await self._slots.acquire()
        try:
            worker = asyncio.create_task(asyncio.to_thread(partial(function, *args)))
        except BaseException:
            self._slots.release()
            raise

        def finished(task):
            # Cancellation of the HTTP waiter cannot stop an executing thread.
            # Keep its slot until it actually ends and consume detached failures.
            self._slots.release()
            if not task.cancelled():
                task.exception()

        worker.add_done_callback(finished)
        return await asyncio.shield(worker)
