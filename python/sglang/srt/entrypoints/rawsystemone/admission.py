"""FIFO, weighted batch admission, shared by all parents in the HTTP process."""

import asyncio
from collections import deque
from contextlib import asynccontextmanager


class BatchAdmission:
    def __init__(self, batches: int, candidates: int, tokens: int):
        self.limits = (batches, candidates, tokens)
        self.used = [0, 0, 0]
        self.waiters = deque()
        self.condition = asyncio.Condition()

    @asynccontextmanager
    async def lease(self, candidates: int, tokens: int):
        weight = (1, candidates, tokens)
        if any(n > limit for n, limit in zip(weight, self.limits)):
            raise ValueError("Batch cannot fit admission limits")
        ticket = object()
        acquired = False
        try:
            async with self.condition:
                self.waiters.append(ticket)
                await self.condition.wait_for(
                    lambda: (
                        self.waiters[0] is ticket
                        and all(
                            n + w <= lim
                            for n, w, lim in zip(self.used, weight, self.limits)
                        )
                    )
                )
                self.waiters.popleft()
                self.used = [n + w for n, w in zip(self.used, weight)]
                acquired = True
                self.condition.notify_all()
            yield
        finally:
            async with self.condition:
                if acquired:
                    self.used = [n - w for n, w in zip(self.used, weight)]
                elif ticket in self.waiters:
                    self.waiters.remove(ticket)
                self.condition.notify_all()
