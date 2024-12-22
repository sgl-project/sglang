import asyncio


class RWLock:
    """
    A Read-Write Lock for asyncio:
        - Multiple readers can hold the lock in parallel if no writer holds it.
        - A writer has exclusive access.
    """

    def __init__(self):
        self._readers = 0  # How many readers currently hold the lock
        self._writer_active = False
        self._lock = asyncio.Lock()  # Internal mutex to protect state
        # Conditions associated with _lock:
        self._readers_ok = asyncio.Condition(self._lock)  # Notify blocked readers
        self._writers_ok = asyncio.Condition(self._lock)  # Notify blocked writers

        # Expose two async context-manager helpers:
        self.reader_lock = self._ReaderLock(self)
        self.writer_lock = self._WriterLock(self)

    async def _acquire_reader(self):
        """
        Wait until there is no active writer.
        Then increment the count of active readers.
        """
        async with self._lock:
            # If a writer is active, wait until it's done.
            while self._writer_active:
                await self._readers_ok.wait()
            self._readers += 1

    async def _release_reader(self):
        """
        Decrement the count of active readers.
        If this was the last active reader, wake up a possible waiting writer.
        """
        async with self._lock:
            self._readers -= 1
            # If no more readers, a writer could proceed.
            if self._readers == 0:
                self._writers_ok.notify()

    async def _acquire_writer(self):
        """
        Wait until there is no active writer and no active readers.
        Then mark a writer as active.
        """
        async with self._lock:
            while self._writer_active or self._readers > 0:
                await self._writers_ok.wait()
            self._writer_active = True

    async def _release_writer(self):
        """
        Mark the writer as done and notify readers and writers.
        """
        async with self._lock:
            self._writer_active = False
            # Allow any waiting readers to proceed:
            self._readers_ok.notify_all()
            # Allow next waiting writer to proceed:
            self._writers_ok.notify()

    class _ReaderLock:
        """
        A simple async context manager that acquires a reader lock
        on entering and releases it on exit.
        """

        def __init__(self, parent: "RWLock"):
            self._parent = parent

        async def __aenter__(self):
            await self._parent._acquire_reader()

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self._parent._release_reader()

    class _WriterLock:
        """
        A simple async context manager that acquires a writer lock
        on entering and releases it on exit.
        """

        def __init__(self, parent: "RWLock"):
            self._parent = parent

        async def __aenter__(self):
            await self._parent._acquire_writer()

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            await self._parent._release_writer()
