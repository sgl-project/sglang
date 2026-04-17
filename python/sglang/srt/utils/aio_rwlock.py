import asyncio
import collections


class RWLock:
    """asyncio readers-writer lock with writer preference. Non-reetrant."""

    def __init__(self):
        self._cond = asyncio.Condition()
        self._readers = 0
        self._writer_active = False
        self._waiting_writers = 0

    @property
    def reader_lock(self):
        return _ReaderLock(self)

    @property
    def writer_lock(self):
        return _WriterLock(self)

    async def acquire_reader(self):
        async with self._cond:
            await self._cond.wait_for(
                lambda: not self._writer_active and self._waiting_writers == 0
            )
            self._readers += 1

    async def release_reader(self):
        async with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    async def acquire_writer(self):
        async with self._cond:
            self._waiting_writers += 1
            try:
                await self._cond.wait_for(
                    lambda: not self._writer_active and self._readers == 0
                )
                self._writer_active = True
            finally:
                self._waiting_writers -= 1
                self._cond.notify_all()

    async def release_writer(self):
        async with self._cond:
            self._writer_active = False
            self._cond.notify_all()

    def is_locked(self):
        return self._writer_active or self._readers > 0


class RWCondition:
    """Condition variable built on an RWLock. Copies the CPython implementation:
    https://github.com/python/cpython/blob/446edda20919447fdc8b5a43f2f2ae686df82e6a/Lib/asyncio/locks.py#L219
    """

    def __init__(self, rwlock: RWLock | None = None):
        self._rwlock = rwlock if rwlock is not None else RWLock()
        self._waiters = collections.deque()

    @property
    def reader_lock(self):
        return _ReaderLock(self._rwlock)

    @property
    def writer_lock(self):
        return _WriterLock(self._rwlock)

    async def acquire_reader(self):
        await self._rwlock.acquire_reader()

    async def release_reader(self):
        await self._rwlock.release_reader()

    async def acquire_writer(self):
        await self._rwlock.acquire_writer()

    async def release_writer(self):
        await self._rwlock.release_writer()

    def is_locked(self):
        return self._rwlock.is_locked()

    def _notify(self, n):
        idx = 0
        for fut in self._waiters:
            if idx >= n:
                break
            if not fut.done():
                idx += 1
                fut.set_result(False)

    def notify_all(self):
        self._notify(len(self._waiters))

    async def _wait(self, release_fn, acquire_fn):
        fut = asyncio.get_running_loop().create_future()
        await release_fn()
        try:
            try:
                self._waiters.append(fut)
                try:
                    await fut
                    return True
                finally:
                    self._waiters.remove(fut)
            finally:
                err = None
                while True:
                    try:
                        await acquire_fn()
                        break
                    except asyncio.CancelledError as e:
                        err = e
                if err is not None:
                    try:
                        raise err
                    finally:
                        err = None
        except BaseException:
            self._notify(1)
            raise

    async def wait_reader(self):
        return await self._wait(
            self._rwlock.release_reader, self._rwlock.acquire_reader
        )

    async def wait_writer(self):
        return await self._wait(
            self._rwlock.release_writer, self._rwlock.acquire_writer
        )

    async def wait_for_reader(self, predicate):
        result = predicate()
        while not result:
            await self.wait_reader()
            result = predicate()
        return result

    async def wait_for_writer(self, predicate):
        result = predicate()
        while not result:
            await self.wait_writer()
            result = predicate()
        return result


class _ReaderLock:
    def __init__(self, rwlock: RWLock):
        self._rwlock = rwlock

    async def __aenter__(self):
        await self._rwlock.acquire_reader()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._rwlock.release_reader()


class _WriterLock:
    def __init__(self, rwlock: RWLock):
        self._rwlock = rwlock

    async def __aenter__(self):
        await self._rwlock.acquire_writer()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._rwlock.release_writer()
